"""Run explicitly: pytest research/spindle/test_experiment.py."""

import numpy as np
import pytest
import torch
from baselines import fit, parameters, splits
from intervene import MatchedErasure
from prepare import voltage_correction

from eeglens import Selection, SignalBatch


def test_uppercase_uv_reader_scale_and_misaligned_signal():
    physical = np.array([10.0, -100.0, 25.3])
    assert voltage_correction(physical, physical) == 1e-6
    assert voltage_correction(physical * 1e-6, physical) == 1.0
    with pytest.raises(ValueError, match="align"):
        voltage_correction(physical[::-1], physical)


def test_subject_partitions_and_test_data_cannot_change_fitted_readout():
    subjects = np.repeat(np.arange(1, 9), 6)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(48, 3))
    y = np.tile([0, 1, 0, 1, 0, 1], 8)
    for held in range(1, 9):
        train, val, test = splits(subjects, held)
        assert len(set(subjects[train])) == 5
        assert len(set(subjects[val])) == 2
        _, _, model = fit(x, y, train, val)
        changed = x.copy()
        changed[test] *= 10000
        changed_y = y.copy()
        changed_y[test] = 1 - changed_y[test]
        _, _, second = fit(changed, changed_y, train, val)
        w, b = parameters(model)
        w2, b2 = parameters(second)
        np.testing.assert_allclose(w, w2)
        assert b == b2
        np.testing.assert_allclose(x @ w + b, model.decision_function(x))


def test_control_norm_matching_and_cls_preservation():
    batch = SignalBatch(torch.randn(2, 1, 15, 200), ("a", "b"), ("CZ",), 200, "test")
    current = torch.randn(2, 16, 3)
    reference = torch.tensor([[1.0], [0.0], [0.0]])
    direction = torch.tensor([[0.0], [1.0], [0.0]])
    edit = MatchedErasure(
        "site", direction, torch.zeros(3), reference, Selection(patches=tuple(range(15)))
    )
    changed = edit.apply(current, batch, "tokens", "test")
    torch.testing.assert_close(changed[:, 0], current[:, 0])
    torch.testing.assert_close(changed[..., 0], current[..., 0])
    torch.testing.assert_close(changed[..., 2], current[..., 2])
    torch.testing.assert_close(edit.last_norm, edit.last_reference_norm)
    torch.testing.assert_close(
        torch.linalg.vector_norm(current - changed, dim=(1, 2)), edit.last_norm
    )
