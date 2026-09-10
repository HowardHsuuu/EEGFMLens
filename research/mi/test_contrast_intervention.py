"""Independent known-answer geometry and minimum-energy checks."""

import pytest
import torch
from contrast_intervention import contrast_donor

from eeglens import Activation, SignalBatch


@pytest.mark.parametrize("layout", ["bcpd", "tokens"])
def test_preserves_common_mode_temporal_deviations_and_untouched_values(layout):
    generator = torch.Generator().manual_seed(52)
    x = torch.randn(2, 3, 4, 6, generator=generator, dtype=torch.float64)
    cls = torch.randn(2, 1, 6, generator=generator, dtype=torch.float64)
    tensor = x if layout == "bcpd" else torch.cat([cls, x.flatten(1, 2)], dim=1)
    batch = SignalBatch(torch.zeros(2, 3, 4, 1), ("a", "b"), ("C3", "CZ", "C4"), 200, "fixture")
    activation = Activation(
        tensor,
        "middle",
        "fixture",
        batch.trial_ids,
        batch.channels,
        "fixture",
        200,
        1,
        1,
        "model_scaled",
        layout,
        tuple(tensor.shape),
    )
    q = torch.linalg.qr(torch.randn(6, 2, generator=generator, dtype=torch.float64)).Q
    center = torch.arange(6, dtype=torch.float64) / 10
    donor = contrast_donor(activation, batch, q, center)
    result = donor.tensor if layout == "bcpd" else donor.tensor[:, 1:].reshape_as(x)
    delta = result - x
    v = (x[:, 2].mean(1) - x[:, 0].mean(1) - center) @ q @ q.T
    torch.testing.assert_close(
        result[:, 2].mean(1) - result[:, 0].mean(1),
        x[:, 2].mean(1) - x[:, 0].mean(1) - v,
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(result[:, 0] + result[:, 2], x[:, 0] + x[:, 2], atol=1e-12, rtol=0)
    torch.testing.assert_close(
        result - result.mean(2, keepdim=True), x - x.mean(2, keepdim=True), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(result[:, 1], x[:, 1], atol=0, rtol=0)
    # For P equal-weight patches, the constrained minimum squared norm is P/2 ||v||².
    torch.testing.assert_close(
        delta.square().sum((1, 2, 3)), 2 * v.square().sum(1), atol=1e-12, rtol=0
    )
    alternative = delta.clone()
    alternative[:, 0] += 1
    alternative[:, 2] += 1
    assert (alternative.square().sum((1, 2, 3)) > delta.square().sum((1, 2, 3))).all()
    if layout == "tokens":
        torch.testing.assert_close(donor.tensor[:, :1], cls, atol=0, rtol=0)
    torch.testing.assert_close(activation.tensor, tensor, atol=0, rtol=0)
