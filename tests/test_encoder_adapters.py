from types import SimpleNamespace

import pytest
import torch

from eeglens import EEGPTAdapter, SignalBatch
from eeglens.adapters.continuous import _ContinuousAdapter
from eeglens.errors import ValidationError


def eegpt_contract():
    return EEGPTAdapter(
        SimpleNamespace(
            patch_embed=SimpleNamespace(patch_size=64, patch_stride=None),
            num_patches=(2, 3),
            embed_num=4,
            blocks=[None],
        )
    )


def test_window_fold_preserves_trial_identity_and_native_roundtrip():
    adapter = eegpt_contract()
    batch = SignalBatch(torch.ones(2, 2, 3, 64), ("a", "b"), ("C3", "C4"), 256, "fixture")
    adapter.validate(batch)
    native = torch.arange(2 * 3 * 6 * 5).reshape(6, 6, 5).float()
    site = adapter.require("blocks.0.output")
    exposed = adapter.expose(native, site, batch)
    assert exposed.shape == (2, 3, 6, 5)
    torch.testing.assert_close(exposed[1], native[3:])
    torch.testing.assert_close(adapter.restore(exposed, site, native.shape), native)
    with pytest.raises(ValidationError, match="geometry"):
        adapter.expose(native[:3], site, batch)


def test_continuous_inputs_reject_channel_reordering_and_overlap():
    adapter = _ContinuousAdapter([], ("C3", "C4"), 256)
    batch = SignalBatch(torch.ones(1, 2, 3, 64), ("a",), ("C4", "C3"), 256, "fixture")
    with pytest.raises(ValidationError, match="channels/order"):
        adapter.validate(batch)
    batch = SignalBatch(
        torch.ones(1, 2, 3, 64), ("a",), ("C3", "C4"), 256, "fixture", patch_stride_samples=32
    )
    with pytest.raises(ValidationError, match="nonoverlapping"):
        adapter.validate(batch)
