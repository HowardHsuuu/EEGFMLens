from types import SimpleNamespace

import pytest
import torch

from eegfmlens import SignalBatch
from eegfmlens.adapters.brainomni import BrainOmniAdapter
from eegfmlens.adapters.csbrain import CSBrainAdapter
from eegfmlens.errors import ValidationError


def test_csbrain_native_sort_maps_back_to_declared_sensors():
    model = SimpleNamespace(sorted_indices=[2, 0, 1], encoder=SimpleNamespace(num_layers=1))
    adapter = CSBrainAdapter(model, channels=("C3", "C4", "F3"))
    batch = SignalBatch(torch.ones(2, 3, 2, 200), ("a", "b"), adapter.channels, 200, "fixture")
    raw = torch.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4).float()
    site = adapter.require("blocks.0.output")
    exposed = adapter.expose(raw, site, batch)
    torch.testing.assert_close(exposed[:, 0], raw[:, 1])
    torch.testing.assert_close(adapter.restore(exposed, site, raw.shape), raw)


def test_brainomni_geometry_cannot_change_silently():
    model = SimpleNamespace(
        window_length=512,
        overlap_ratio=0.25,
        blocks=[None, None],
        tokenizer=SimpleNamespace(
            window_length=512,
            sensor_embed=SimpleNamespace(sensor_embedding_layer=SimpleNamespace(num_embeddings=3)),
        ),
    )
    pos = torch.zeros(2, 6)
    types = torch.zeros(2, dtype=torch.long)
    adapter = BrainOmniAdapter(model, channels=("C3", "C4"), positions=pos, sensor_types=types)
    batch = SignalBatch(torch.ones(1, 2, 2, 512), ("a",), adapter.channels, 256, "fixture")
    pos.add_(1)  # caller mutation must not change adapter-owned coordinates
    adapter.validate(batch)
    assert len(adapter.sites) == 2  # projection and first block; final block is not in encode
    adapter.positions.add_(1)
    with pytest.raises(ValidationError, match="geometry changed"):
        adapter.validate(batch)


def test_bendr_context_sequence_axis_roundtrip_preserves_trial_edits():
    from eegfmlens.adapters.bendr import BENDRAdapter

    model = [
        SimpleNamespace(in_features=2, encoder_h=4),
        SimpleNamespace(in_features=4, transformer_layers=[None]),
    ]
    adapter = BENDRAdapter(model, channels=("C3", "C4"))
    batch = SignalBatch(torch.ones(2, 2, 1, 256), ("a", "b"), adapter.channels, 256, "fixture")
    raw = torch.arange(5 * 2 * 12).reshape(5, 2, 12).float()
    site = adapter.require("context.blocks.0.output")
    exposed = adapter.expose(raw, site, batch).clone()
    assert exposed.shape == (2, 5, 12)
    exposed[1] = 0
    edited = adapter.restore(exposed, site, raw.shape)
    assert edited.is_contiguous()
    torch.testing.assert_close(edited[:, 0], raw[:, 0])
    assert edited[:, 1].count_nonzero() == 0
    conditioning = adapter.require("context.input")
    native_conditioning = torch.randn(2, 12, 5).permute(2, 0, 1)
    exposed = adapter.expose(native_conditioning, conditioning, batch).contiguous()
    restored = adapter.restore(exposed, conditioning, native_conditioning.shape)
    assert restored.stride() == native_conditioning.stride()
    torch.testing.assert_close(restored, native_conditioning, rtol=0, atol=0)


def test_signaljepa_rejects_short_input_and_changed_channel_mapping():
    from eegfmlens import SignalJEPAAdapter

    model = SimpleNamespace(
        chs_info=[{"ch_name": "C3"}, {"ch_name": "C4"}],
        sfreq=128,
        transformer=SimpleNamespace(encoder=SimpleNamespace(layers=[None])),
        pos_encoder=SimpleNamespace(default_ch_idxs=torch.tensor([12, 14])),
        feature_encoder=SimpleNamespace(
            conv_layers_spec=[(8, 32, 8), (16, 2, 2), (32, 2, 2), (64, 2, 2), (64, 2, 2)]
        ),
    )
    adapter = SignalJEPAAdapter(model)
    short = SignalBatch(torch.ones(1, 2, 1, 128), ("a",), adapter.channels, 128, "fixture")
    with pytest.raises(ValidationError, match="receptive field"):
        adapter.validate(short)
    valid = SignalBatch(torch.ones(1, 2, 3, 128), ("a",), adapter.channels, 128, "fixture")
    adapter.validate(valid)
    model.pos_encoder.default_ch_idxs[:] = torch.tensor([14, 12])
    with pytest.raises(ValidationError, match="mapping changed"):
        adapter.forward(model, valid)
