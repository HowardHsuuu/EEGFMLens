from types import SimpleNamespace

import pytest
import torch

from eegfmlens import SignalBatch
from eegfmlens.adapters.csbrain import CSBrainAdapter
from eegfmlens.errors import ValidationError


class NativeMaskFixture:
    sorted_indices = (2, 0, 1)
    encoder = SimpleNamespace(num_layers=1)

    def __call__(self, x, mask=None):
        output = x[:, self.sorted_indices].clone()
        if mask is not None:
            output[mask] = 0
        return output


def test_physical_mask_and_hidden_input_independence():
    model = NativeMaskFixture()
    adapter = CSBrainAdapter(model, channels=("C3", "C4", "F3"))
    x = torch.arange(2 * 3 * 4 * 200).reshape(2, 3, 4, 200).float()
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    mask[:, 0, 1] = True
    batch = SignalBatch(x, ("a", "b"), adapter.channels, 200, "fixture")
    observed = adapter.forward(model, batch, mask=mask.tolist())
    expected = model(x)
    expected[:, 1, 1] = 0  # native position of physical C3
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    x[:, 0, 1] = -123456
    torch.testing.assert_close(
        adapter.forward(model, batch, mask=mask.tolist()), expected, atol=0, rtol=0
    )
    torch.testing.assert_close(adapter.forward(model, batch), model(x), atol=0, rtol=0)


@pytest.mark.parametrize("mask", [torch.zeros(2, 3, 4), torch.zeros(2, 4, 3, dtype=torch.bool)])
def test_invalid_masks_rejected(mask):
    model = NativeMaskFixture()
    adapter = CSBrainAdapter(model, channels=("C3", "C4", "F3"))
    batch = SignalBatch(torch.ones(2, 3, 4, 200), ("a", "b"), adapter.channels, 200, "fixture")
    with pytest.raises(ValidationError, match="boolean"):
        adapter.forward(model, batch, mask=mask)


@pytest.mark.parametrize("order", ["native_sorted", "input"])
def test_projection_target_order_is_separate_from_encoder_order(order):
    model = NativeMaskFixture()
    adapter = CSBrainAdapter(model, channels=("C3", "C4", "F3"), projection_order=order)
    native = torch.arange(2 * 3 * 4 * 200).reshape(2, 3, 4, 200).float()
    batch = SignalBatch(native, ("a", "b"), adapter.channels, 200, "fixture")
    hidden_site = adapter.require("blocks.0.output")
    hidden = adapter.expose(native, hidden_site, batch)
    torch.testing.assert_close(hidden[:, 0], native[:, 1], rtol=0, atol=0)
    site = adapter.require("projection.output")
    exposed = adapter.expose(native, site, batch)
    expected_index = 0 if order == "input" else 1
    torch.testing.assert_close(exposed[:, 0], native[:, expected_index], rtol=0, atol=0)
    edited = exposed.clone()
    edited[:, 0, 1] = -42
    restored = adapter.restore(edited, site, native.shape)
    expected = native.clone()
    expected[:, expected_index, 1] = -42
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    torch.testing.assert_close(adapter.forward(model, batch), model(native), rtol=0, atol=0)


def test_projection_order_requires_explicit_supported_mapping():
    with pytest.raises(ValidationError, match="projection_order"):
        CSBrainAdapter(NativeMaskFixture(), channels=("C3", "C4", "F3"), projection_order="guess")
