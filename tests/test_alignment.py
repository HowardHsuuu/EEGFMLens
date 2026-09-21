import pytest
import torch

from eegfmlens import (
    Activation,
    center_within_groups,
    cross_model_similarity,
    linear_cka,
    rsa_correlation,
)


def activation(tensor, site, model, trial_ids):
    return Activation(
        tensor,
        site,
        model,
        trial_ids,
        ("C3",),
        "fixture",
        100,
        4,
        4,
        "scaled",
        "batch",
        tuple(tensor.shape),
    )


def test_linear_cka_is_rotation_scale_and_feature_count_invariant():
    generator = torch.Generator().manual_seed(4)
    left = torch.randn(20, 3, generator=generator, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    right = 7 * left @ rotation
    assert linear_cka(left, right) == pytest.approx(1)
    unrelated = torch.randn(20, 5, generator=generator, dtype=torch.float64)
    assert linear_cka(left, unrelated) < 0.5


def test_group_centering_separates_identity_from_within_group_geometry():
    groups = tuple(f"subject-{index // 4}" for index in range(16))
    identity = torch.repeat_interleave(torch.arange(4, dtype=torch.float64), 4) * 100
    left_within = torch.tensor([-1.0, -0.3, 0.3, 1.0], dtype=torch.float64).repeat(4)
    right_within = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64).repeat(4)
    left = torch.column_stack((identity, left_within))
    right = torch.column_stack((identity, right_within))
    assert linear_cka(left, right) > 0.999
    assert linear_cka(left, right, groups=groups) < 1e-12
    centered = center_within_groups(left, groups)
    for group in set(groups):
        rows = torch.tensor([value == group for value in groups])
        torch.testing.assert_close(centered[rows].mean(0), torch.zeros(2, dtype=left.dtype))


def test_rsa_and_cross_model_matrix_align_trials_by_identity():
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    transformed = 3 * base + 5
    assert rsa_correlation(base, transformed, metric="euclidean") == pytest.approx(1)
    trials = tuple(f"trial-{index}" for index in range(8))
    reversed_trials = tuple(reversed(trials))
    left = {
        "early": activation(base, "early", "left-model", trials),
        "late": activation(base.square(), "late", "left-model", trials),
    }
    right = {
        "aligned": activation(
            transformed.flip(0),
            "aligned",
            "right-model",
            reversed_trials,
        )
    }
    result = cross_model_similarity(left, right)
    assert result.values.shape == (2, 1)
    assert result.values[0, 0] == pytest.approx(1)
    assert result.left_model_id == "left-model"
    assert result.right_model_id == "right-model"
    assert result.trial_ids == trials
