import numpy as np
import torch
from correct import Correction
from fit_corrections import ridge_candidates

from eeglens import Selection, SignalBatch


def test_affine_control_norm_is_per_trial_and_cls_stays_fixed():
    batch = SignalBatch(torch.ones(2, 1, 2, 2), ("a", "b"), ("C3",), 200, "test")
    current = torch.tensor(
        [[[99.0, 99.0], [1.0, 2.0], [3.0, 4.0]], [[88.0, 88.0], [3.0, 7.0], [4.0, 8.0]]]
    )
    edit = Correction(
        "site",
        Selection(patches=(0, 1)),
        torch.tensor([1.0, -2.0]),
        torch.eye(2) * 0.5,
        reference_bias=torch.zeros(2),
        reference_matrix=torch.eye(2) * 2,
    )
    output = edit.apply(current, batch, "tokens", "model")
    torch.testing.assert_close(output[:, 0], current[:, 0])
    for i in range(2):
        expected = torch.linalg.vector_norm(current[i, 1:])
        torch.testing.assert_close(torch.linalg.vector_norm(output[i] - current[i]), expected)
        assert edit.effects[i]["valid"]
    other = Correction(
        "site",
        Selection(patches=(0, 1)),
        torch.zeros(2),
        torch.eye(2),
        reference_bias=torch.zeros(2),
        reference_matrix=torch.eye(2) * 2,
    )
    torch.testing.assert_close(other.apply(current, batch, "tokens", "model"), current)
    assert not any(r["valid"] for r in other.effects)


def test_ridge_raw_coordinates_match_standardized_solution():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 3)) * [1, 3, 7] + [4, -5, 10]
    y = x @ np.array([[1, 2], [3, 4], [5, 6]]) + [8, -2]
    for alpha, matrix, bias, ys in ridge_candidates(x, y):
        xx = (x - x.mean(0)) / x.std(0)
        yy = (y - y.mean(0)) / ys
        b = np.linalg.solve(xx.T @ xx / len(x) + alpha * np.eye(3), xx.T @ yy / len(x))
        np.testing.assert_allclose(x @ matrix + bias, (xx @ b) * ys + y.mean(0), atol=1e-10)
