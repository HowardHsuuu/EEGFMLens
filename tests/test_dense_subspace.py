"""Geometric invariants beyond erasing coordinate axes."""

import torch

from eeglens import Selection, SignalBatch, SubspaceAblation


def test_dense_erasure_is_basis_invariant_idempotent_and_local():
    generator = torch.Generator().manual_seed(6274)
    x = torch.randn(2, 3, 4, 11, generator=generator, dtype=torch.float64)
    batch = SignalBatch(torch.ones(2, 3, 4, 200), ("a", "b"), ("C3", "CZ", "C4"), 200, "test")
    q = torch.linalg.qr(torch.randn(11, 3, generator=generator, dtype=x.dtype)).Q
    rotation = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=x.dtype)).Q
    center = torch.randn(11, generator=generator, dtype=x.dtype)
    selection = Selection(sensors=("C4",), patches=(1,))
    edit = SubspaceAblation("site", q, center, selection)
    y = edit.apply(x, batch, "bcpd", "model")
    rotated = SubspaceAblation("site", q @ rotation, center, selection)
    torch.testing.assert_close(y, rotated.apply(x, batch, "bcpd", "model"), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(y, edit.apply(y, batch, "bcpd", "model"), atol=1e-12, rtol=1e-12)
    # Independent complementary coordinates from a complete QR factorization.
    complement = torch.linalg.qr(q, mode="complete").Q[:, 3:]
    torch.testing.assert_close(
        (y[:, 2, 1] - center) @ q, torch.zeros(2, 3, dtype=x.dtype), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(y @ complement, x @ complement, atol=1e-12, rtol=1e-12)
    mask = torch.ones(x.shape[:-1], dtype=torch.bool)
    mask[:, 2, 1] = False
    torch.testing.assert_close(y[mask], x[mask], atol=0, rtol=0)
