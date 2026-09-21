import pytest
import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    AxisSelection,
    EEGLens,
    SAEFeatureClamping,
    SignalBatch,
    TopKSAE,
    attribute,
    fit_concept_direction,
    fit_sae_code_reference,
    sae_concept_profile,
    tcav_permutation_test,
    tcav_score,
)
from eegfmlens.errors import ValidationError


def concept_fixture():
    generator = torch.Generator().manual_seed(611)
    shared = torch.randn(100, 3, generator=generator, dtype=torch.float64)
    positive = torch.cat((torch.full((100, 1), 2.0), shared), dim=1)
    negative = torch.cat((torch.full((100, 1), -2.0), shared), dim=1)
    features = torch.stack((positive, negative), dim=1).reshape(200, 4)
    labels = torch.tensor([True, False] * 100)
    train = torch.zeros(200, dtype=torch.bool)
    train[:160] = True
    test = ~train
    return features, labels, train, test


class QuadraticRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.representation = nn.Identity()

    def forward(self, data):
        hidden = self.representation(data)
        return hidden.square().flatten(1).sum(1)


def test_held_out_concept_direction_and_native_gradient_tcav_are_known_answer():
    features, labels, train, test = concept_fixture()
    concept = fit_concept_direction(
        features,
        labels,
        train_mask=train,
        test_mask=test,
        alpha=1e-3,
    )
    assert concept.validation_balanced_accuracy == 1
    torch.testing.assert_close(
        concept.direction,
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )

    generator = torch.Generator().manual_seed(997)
    evaluation = torch.cat(
        (
            torch.full((80, 1), 2.0, dtype=torch.float64),
            4 * torch.randn(80, 3, generator=generator, dtype=torch.float64),
        ),
        dim=1,
    )
    batch = SignalBatch(
        evaluation.reshape(80, 1, 1, 4),
        tuple(f"positive-{index}" for index in range(80)),
        ("C3",),
        200,
        "tcav-known-answer",
    )
    lens = EEGLens(
        QuadraticRepresentation().double().eval(),
        Adapter([ActivationSite("representation", "representation")]),
    )
    attributed = attribute(
        lens,
        batch,
        lambda output, current: output,
        method="gradient",
        sites=("representation",),
    )
    score = tcav_score(attributed, "representation", concept)
    assert score.score == 1
    torch.testing.assert_close(score.sensitivities, torch.full((80,), 4.0, dtype=torch.float64))

    tested = tcav_permutation_test(
        attributed,
        "representation",
        features,
        labels,
        train_mask=train,
        test_mask=test,
        alpha=1e-3,
        permutations=31,
        seed=19,
    )
    assert tested.observed.score == 1
    assert tested.p_value <= 0.1
    assert tested.null_scores.shape == (31,)


def identity_sae():
    sae = TopKSAE(3, 3, 3).double().eval()
    with torch.no_grad():
        sae.encoder_weight.copy_(torch.eye(3, dtype=torch.float64))
        sae.encoder_bias.zero_()
        sae.decoder_weight.copy_(torch.eye(3, dtype=torch.float64))
        sae.decoder_bias.zero_()
    return sae


def test_sae_concept_profile_and_target_centroid_clamping_preserve_residual():
    sae = identity_sae()
    positive = torch.tensor([[3.0, 0.0, -1.0], [2.0, 0.0, -2.0]], dtype=torch.float64)
    negative = torch.tensor([[0.0, 2.0, -1.0], [0.0, 4.0, -3.0]], dtype=torch.float64)
    profile = sae_concept_profile(
        sae,
        positive,
        negative,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        profile.firing_rate_difference,
        torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        profile.decoder_alignment,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )

    reference = fit_sae_code_reference(sae, negative)
    batch = SignalBatch(
        torch.tensor([[[[3.0, 0.5, -4.0]]]], dtype=torch.float64),
        ("trial",),
        ("C3",),
        200,
        "sae-clamping-known-answer",
    )
    lens = EEGLens(
        nn.Sequential(nn.Identity()).double().eval(),
        Adapter([ActivationSite("representation", "0")]),
    )
    clamped = lens.run_with_interventions(
        batch,
        interventions=(SAEFeatureClamping("representation", sae, (0,), reference),),
    )
    torch.testing.assert_close(
        clamped.output,
        torch.tensor([[[[0.0, 0.5, -4.0]]]], dtype=torch.float64),
    )
    parameters = clamped.metadata["interventions"][0]["parameters"]
    assert parameters["reference_fit_rows"] == 2
    assert len(parameters["reference_sha256"]) == 64


def test_concept_and_clamping_contracts_reject_leakage_or_invalid_partial_edits():
    features, labels, train, test = concept_fixture()
    with pytest.raises(ValidationError, match="disjoint"):
        fit_concept_direction(
            features,
            labels,
            train_mask=train,
            test_mask=train,
        )

    sae = identity_sae()
    reference = fit_sae_code_reference(sae, features[:4, :3])
    batch = SignalBatch(
        features[:2, :3].reshape(2, 1, 1, 3),
        ("a", "b"),
        ("C3",),
        200,
        "contract",
    )
    partial = SAEFeatureClamping(
        "representation",
        sae,
        (0,),
        reference,
        AxisSelection(axis=3, indices=(0,)),
    )
    with pytest.raises(ValidationError, match="decoded feature axis"):
        partial.apply(batch.data, batch, "bcpd", "model")

    with torch.no_grad():
        sae.decoder_bias.add_(1)
    stale = SAEFeatureClamping("representation", sae, (0,), reference)
    with pytest.raises(ValidationError, match="changed after fitting"):
        stale.apply(batch.data, batch, "bcpd", "model")
