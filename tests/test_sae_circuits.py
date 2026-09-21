import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    EEGLens,
    SAEFeatureAblation,
    SAEFeatureSteering,
    SAETrainingConfig,
    SignalBatch,
    TopKSAE,
    path_patch,
    sae_metrics,
    train_sae,
)


def identity_sae():
    sae = TopKSAE(2, 2, 2).double().eval()
    with torch.no_grad():
        sae.encoder_weight.copy_(torch.eye(2, dtype=torch.float64))
        sae.encoder_bias.zero_()
        sae.decoder_weight.copy_(torch.eye(2, dtype=torch.float64))
        sae.decoder_bias.zero_()
    return sae


def test_sae_feature_edits_preserve_residual_and_record_parameters():
    batch = SignalBatch(
        torch.tensor([[[[3.0, 4.0]]]], dtype=torch.float64),
        ("trial",),
        ("C3",),
        200,
        "fixture",
    )
    lens = EEGLens(
        nn.Sequential(nn.Identity()).double().eval(), Adapter([ActivationSite("x", "0")])
    )
    sae = identity_sae()
    ablated = lens.run_with_interventions(batch, interventions=[SAEFeatureAblation("x", sae, (0,))])
    torch.testing.assert_close(ablated.output, torch.tensor([[[[0.0, 4.0]]]], dtype=torch.float64))
    assert ablated.metadata["interventions"][0]["parameters"]["features"] == [0]
    steered = lens.run_with_interventions(
        batch, interventions=[SAEFeatureSteering("x", sae, (1,), (2.5,))]
    )
    torch.testing.assert_close(steered.output, torch.tensor([[[[3.0, 6.5]]]], dtype=torch.float64))


def test_sae_training_reduces_reconstruction_error_and_normalizes_directions():
    torch.manual_seed(21)
    activations = torch.randn(96, 3, dtype=torch.float64)
    sae = TopKSAE(3, 8, 3).double().eval()
    initial = sae_metrics(sae, activations)
    result = train_sae(
        sae,
        activations,
        config=SAETrainingConfig(epochs=25, batch_size=24, learning_rate=3e-3, seed=5),
    )
    assert result.final_metrics.mean_squared_error < initial.mean_squared_error
    assert result.losses[-1] < result.losses[0]
    assert not sae.training
    torch.testing.assert_close(
        torch.linalg.vector_norm(sae.decoder_weight, dim=1),
        torch.ones(8, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )


class PathModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.source = nn.Identity()
        self.mediator = nn.Identity()

    def forward(self, data):
        hidden = self.source(data)
        return self.mediator(hidden * 2) + 1


def test_path_patch_recovers_source_effect_through_known_mediator():
    recipient = SignalBatch(torch.zeros(2, 1, 1, 1), ("a", "b"), ("C3",), 100, "fixture")
    donor = SignalBatch(torch.tensor([[[[3.0]]], [[[2.0]]]]), ("b", "a"), ("C3",), 100, "fixture")
    lens = EEGLens(
        PathModel().eval(),
        Adapter([ActivationSite("source", "source"), ActivationSite("mediator", "mediator")]),
    )
    result = path_patch(
        lens,
        donor,
        recipient,
        lambda output, batch: output[:, 0, 0, 0],
        source_site="source",
        mediator_site="mediator",
    )
    torch.testing.assert_close(result.source_patched_score, torch.tensor([5.0, 7.0]))
    torch.testing.assert_close(result.clean_score, torch.tensor([5.0, 7.0]))
    torch.testing.assert_close(result.mediator_patched_score, result.source_patched_score)
    torch.testing.assert_close(result.mediated_effect, torch.tensor([4.0, 6.0]))
