import pytest
import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    AxisSelection,
    EEGLens,
    LEACEAblation,
    SignalBatch,
    SubspaceAblation,
    categorical_targets,
    fit_leace_eraser,
    fit_random_subspace_control,
)
from eegfmlens.errors import ValidationError


def correlated_concept_fixture():
    generator = torch.Generator().manual_seed(71)
    labels = tuple(f"class-{index % 3}" for index in range(240))
    concepts = categorical_targets(labels, dtype=torch.float64)
    noise = torch.randn(240, 5, generator=generator, dtype=torch.float64)
    mixing = torch.tensor(
        [
            [2.0, 1.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 1.7, 0.0, 0.0],
            [0.0, 0.0, 0.3, 1.2, 0.0],
            [0.4, 0.0, 0.0, 0.2, 1.0],
            [0.0, 0.6, 0.0, 0.0, 0.8],
        ],
        dtype=torch.float64,
    )
    offsets = torch.tensor(
        [[3.0, -1.0, 0.5, 0.0, 1.0], [-2.0, 2.0, 0.0, 1.0, -0.5], [0.0] * 5],
        dtype=torch.float64,
    )
    features = noise @ mixing + concepts @ offsets
    return features, concepts


def cross_covariance(features, concepts):
    return (features - features.mean(0)).T @ (concepts - concepts.mean(0))


def test_leace_is_affine_idempotent_and_linearly_erases_the_concept():
    features, concepts = correlated_concept_fixture()
    eraser = fit_leace_eraser(features, concepts)
    erased = eraser(features)
    assert eraser.rank == 2
    torch.testing.assert_close(erased.mean(0), features.mean(0), atol=1e-12, rtol=0)
    torch.testing.assert_close(eraser(erased), erased, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        cross_covariance(erased, concepts),
        torch.zeros(5, 3, dtype=torch.float64),
        atol=2e-10,
        rtol=0,
    )
    assert not torch.allclose(
        eraser.proj_left @ eraser.proj_right,
        (eraser.proj_left @ eraser.proj_right).T,
    )


def test_leace_intervention_uses_the_fixed_fit_and_records_provenance():
    features, concepts = correlated_concept_fixture()
    features = features[:12]
    concepts = concepts[:12]
    eraser = fit_leace_eraser(features, concepts, covariance_shrinkage=0.05)
    batch = SignalBatch(
        features.reshape(12, 1, 1, 5),
        tuple(f"trial-{index}" for index in range(12)),
        ("C3",),
        200,
        "leace-known-answer",
    )
    model = nn.Sequential(nn.Identity()).eval()
    lens = EEGLens(model, Adapter([ActivationSite("hidden", "0")]))
    result = lens.run_with_interventions(
        batch,
        sites=("hidden",),
        interventions=(LEACEAblation("hidden", eraser),),
    )
    torch.testing.assert_close(result.output, eraser(batch.data), atol=1e-12, rtol=1e-12)
    record = result.metadata["interventions"][0]
    assert record["parameters"]["rank"] == eraser.rank
    assert record["parameters"]["fit_rows"] == 12
    assert len(record["parameters"]["proj_left_sha256"]) == 64
    assert not model[0]._forward_hooks


def test_leace_intervention_rejects_partial_feature_axis_selection():
    features, concepts = correlated_concept_fixture()
    eraser = fit_leace_eraser(features, concepts)
    batch = SignalBatch(
        features[:2].reshape(2, 1, 1, 5),
        ("trial-0", "trial-1"),
        ("C3",),
        200,
        "leace-known-answer",
    )
    intervention = LEACEAblation(
        "hidden",
        eraser,
        AxisSelection(axis=3, indices=(0,)),
    )
    with pytest.raises(ValidationError, match="feature axis"):
        intervention.apply(batch.data, batch, "bcpd", "model")


def test_same_rank_random_control_is_reproducible_and_intervention_ready():
    features, concepts = correlated_concept_fixture()
    eraser = fit_leace_eraser(features, concepts)
    first = fit_random_subspace_control(features, rank=eraser.rank, seed=19)
    second = fit_random_subspace_control(features, rank=eraser.rank, seed=19)
    torch.testing.assert_close(first.basis, second.basis, atol=0, rtol=0)
    torch.testing.assert_close(
        first.basis.T @ first.basis,
        torch.eye(first.rank, dtype=features.dtype),
        atol=1e-12,
        rtol=1e-12,
    )
    controlled = SubspaceAblation("hidden", first.basis, first.center).apply(
        features[:, None, None],
        SignalBatch(
            torch.zeros(240, 1, 1, 5, dtype=torch.float64),
            tuple(f"trial-{index}" for index in range(240)),
            ("C3",),
            200,
            "control",
        ),
        "bcpd",
        "model",
    )
    assert controlled.shape == (240, 1, 1, 5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"covariance_shrinkage": -0.1},
        {"covariance_shrinkage": 1.1},
        {"relative_tolerance": 0},
    ],
)
def test_leace_rejects_invalid_fit_settings(kwargs):
    features, concepts = correlated_concept_fixture()
    with pytest.raises(ValidationError, match="LEACE"):
        fit_leace_eraser(features, concepts, **kwargs)


def test_leace_rejects_a_concept_with_no_cross_covariance():
    features = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    concepts = torch.ones(2, 1, dtype=torch.float64)
    with pytest.raises(ValidationError, match="no nonzero cross-covariance"):
        fit_leace_eraser(features, concepts)
