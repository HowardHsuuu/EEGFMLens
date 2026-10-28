import torch

from eegfmlens import (
    categorical_targets,
    fit_cross_covariance_subspace,
    fit_ridge_probe,
    group_variance_decomposition,
    layerwise_ridge_probe,
    r2_score,
    within_group_contrast_consistency,
)


def test_ridge_probe_and_layerwise_split_recover_held_out_signal():
    generator = torch.Generator().manual_seed(12)
    features = torch.randn(40, 5, generator=generator, dtype=torch.float64)
    targets = 3 * features[:, 0] - 2 * features[:, 2] + 0.5
    probe = fit_ridge_probe(features[:30], targets[:30], alpha=1e-6)
    assert r2_score(targets[30:], probe.predict(features[30:])).mean_r2 > 0.999
    train = torch.arange(40) < 30
    test = ~train
    results = layerwise_ridge_probe(
        {"early": features[:, None, :], "late": features},
        targets,
        train_mask=train,
        test_mask=test,
        alpha=1e-6,
    )
    assert {result.site for result in results} == {"early", "late"}
    assert all(result.test.mean_r2 > 0.999 for result in results)


def test_cross_covariance_subspace_names_and_removes_concept_direction():
    labels = ("s1", "s1", "s2", "s2", "s3", "s3")
    concepts = categorical_targets(labels, dtype=torch.float64)
    features = torch.column_stack(
        (concepts[:, 0] - concepts[:, 2], concepts[:, 1] - concepts[:, 2], torch.arange(6))
    )
    fitted = fit_cross_covariance_subspace(features, concepts, rank=2)
    assert fitted.basis.shape == (3, 2)
    erased = features - ((features - fitted.center) @ fitted.basis) @ fitted.basis.T
    cross_covariance = (erased - erased.mean(0)).T @ (concepts - concepts.mean(0))
    torch.testing.assert_close(
        cross_covariance, torch.zeros_like(cross_covariance), atol=1e-10, rtol=0
    )


def test_subject_variance_and_condition_direction_consistency_are_exact():
    embeddings = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [10.0, 1.0], [12.0, 1.0]], dtype=torch.float64
    )
    groups = ("a", "a", "b", "b")
    variance = group_variance_decomposition(embeddings, groups)
    assert (
        variance.total_sum_squares
        == variance.between_group_sum_squares + variance.within_group_sum_squares
    )
    result = within_group_contrast_consistency(
        embeddings, groups, torch.tensor([0, 1, 0, 1], dtype=torch.float64)
    )
    assert result.mean_pairwise_cosine == 1.0
    assert result.contrast_signal_to_noise > 1e12
