import pytest
import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    EEGLens,
    SignalBatch,
    TopKSAE,
    fit_sae_code_reference,
    sae_feature_sweep,
)
from eegfmlens.errors import ValidationError


class TwoObjectiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Identity()

    def forward(self, data):
        features = self.hidden(data)[:, 0, 0]
        return torch.stack((features[:, 0], features[:, 1]), dim=1)


def fixture():
    lens = EEGLens(
        TwoObjectiveModel().double().eval(),
        Adapter([ActivationSite("hidden", "hidden")]),
        model_id="two-objective",
    )
    batch = SignalBatch(
        torch.tensor(
            [
                [[[3.0, 5.0, 2.0]]],
                [[[4.0, 6.0, 1.0]]],
            ],
            dtype=torch.float64,
        ),
        ("trial-a", "trial-b"),
        ("C3",),
        200,
        "known-answer",
    )
    sae = TopKSAE(3, 3, 3).double().eval()
    with torch.no_grad():
        sae.encoder_weight.copy_(torch.eye(3, dtype=torch.float64))
        sae.encoder_bias.zero_()
        sae.decoder_weight.copy_(torch.eye(3, dtype=torch.float64))
        sae.decoder_bias.zero_()
    metrics = {
        "target": lambda output, batch: output[:, 0],
        "off_target": lambda output, batch: output[:, 1],
    }
    return lens, batch, sae, metrics


def test_ranked_feature_ablation_has_known_effect_and_seeded_random_controls():
    lens, batch, sae, metrics = fixture()
    result = sae_feature_sweep(
        lens,
        batch,
        "hidden",
        sae,
        (0, 2),
        (0, 1, 2),
        metrics,
        random_draws=3,
        seed=19,
    )

    torch.testing.assert_close(
        result.scores["target"],
        torch.tensor([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.scores["off_target"],
        torch.tensor([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.mean_delta("target"),
        torch.tensor([0.0, -3.5, -3.5], dtype=torch.float64),
    )
    assert result.integrated_mean_delta("target") == pytest.approx(-1.75)
    assert result.random_scores["target"].shape == (3, 2, 3)
    assert result.random_integrated_mean_delta("target").shape == (3,)
    assert result.random_area_between("target", "off_target").shape == (3,)
    torch.testing.assert_close(
        result.random_scores["target"][:, :, 0],
        result.scores["target"][:, 0].expand(3, -1),
    )
    assert all(len(set(ranking)) == sae.n_features for ranking in result.random_rankings)
    repeated = sae_feature_sweep(
        lens,
        batch,
        "hidden",
        sae,
        (0, 2),
        (0, 1, 2),
        metrics,
        random_draws=3,
        seed=19,
    )
    assert repeated.random_rankings == result.random_rankings
    assert result.trial_ids == batch.trial_ids
    assert result.metadata["execution_batch_size"] == 1
    assert result.metadata["output_metric_names"] == ("target", "off_target")
    assert result.metadata["run_metric_names"] == ()
    assert len(result.metadata["input_sha256"]) == 64
    assert len(result.metadata["sae"]["decoder_weight_sha256"]) == 64
    assert not lens.model.hidden._forward_hooks


def test_ranked_feature_clamping_uses_held_out_reference_codes():
    lens, batch, sae, metrics = fixture()
    reference = fit_sae_code_reference(
        sae,
        torch.tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 2.0]], dtype=torch.float64),
    )
    result = sae_feature_sweep(
        lens,
        batch,
        "hidden",
        sae,
        (0,),
        (0, 1),
        metrics,
        mode="clamp",
        reference=reference,
    )

    torch.testing.assert_close(
        result.scores["target"],
        torch.tensor([[3.0, 1.0], [4.0, 1.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.scores["off_target"],
        torch.tensor([[5.0, 5.0], [6.0, 6.0]], dtype=torch.float64),
    )
    assert len(result.metadata["reference_sha256"]) == 64
    assert result.random_scores["target"].shape == (0, 2, 2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"feature_ranking": (0, 0)}, "ranking"),
        ({"feature_counts": (1, 2)}, "counts"),
        ({"feature_counts": (0, 2, 1)}, "counts"),
        ({"feature_counts": (0, 3)}, "counts"),
        ({"mode": "clamp"}, "SAECodeReference"),
        ({"random_draws": -1}, "random_draws"),
        ({"execution_kwargs": []}, "execution_kwargs"),
    ],
)
def test_feature_sweep_rejects_invalid_experiment_contracts(overrides, message):
    lens, batch, sae, metrics = fixture()
    arguments = {
        "feature_ranking": (0, 1),
        "feature_counts": (0, 1, 2),
        "metrics": metrics,
    }
    arguments.update(overrides)
    with pytest.raises(ValidationError, match=message):
        sae_feature_sweep(lens, batch, "hidden", sae, **arguments)


def test_feature_sweep_rejects_reference_for_ablation_and_invalid_metric_output():
    lens, batch, sae, metrics = fixture()
    reference = fit_sae_code_reference(sae, torch.ones(2, 3, dtype=torch.float64))
    with pytest.raises(ValidationError, match="only for clamp"):
        sae_feature_sweep(
            lens,
            batch,
            "hidden",
            sae,
            (0,),
            (0, 1),
            metrics,
            reference=reference,
        )
    with pytest.raises(ValidationError, match="one finite floating value"):
        sae_feature_sweep(
            lens,
            batch,
            "hidden",
            sae,
            (0,),
            (0, 1),
            {"bad": lambda output, batch: output},
        )
    assert not lens.model.hidden._forward_hooks


def test_feature_sweep_rejects_read_only_sites_before_model_execution():
    model = TwoObjectiveModel().double().eval()
    calls = []
    handle = model.hidden.register_forward_hook(lambda *args: calls.append(1))
    try:
        lens = EEGLens(model, Adapter([ActivationSite("hidden", "hidden", writable=False)]))
        _, batch, sae, metrics = fixture()
        with pytest.raises(ValidationError, match="writable"):
            sae_feature_sweep(
                lens,
                batch,
                "hidden",
                sae,
                (0,),
                (0, 1),
                metrics,
            )
        assert calls == []
    finally:
        handle.remove()


def test_feature_sweep_run_metric_reads_explicit_post_intervention_cache():
    lens, batch, sae, _ = fixture()
    result = sae_feature_sweep(
        lens,
        batch,
        "hidden",
        sae,
        (0,),
        (0, 1),
        {},
        run_metrics={"cached_feature": lambda run, current: run.cache["hidden"].tensor[:, 0, 0, 0]},
        cache_sites=("hidden",),
    )
    torch.testing.assert_close(
        result.scores["cached_feature"],
        torch.tensor([[3.0, 0.0], [4.0, 0.0]], dtype=torch.float64),
    )
    assert result.metadata["output_metric_names"] == ()
    assert result.metadata["run_metric_names"] == ("cached_feature",)
    assert result.metadata["metric_names"] == ("cached_feature",)
    assert result.metadata["cache_sites"] == ("hidden",)
