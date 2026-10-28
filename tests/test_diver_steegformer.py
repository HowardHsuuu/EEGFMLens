from types import SimpleNamespace

import pytest
import torch

from eegfmlens import SignalBatch, STEEGFormerAdapter
from eegfmlens.adapters.diver import DIVERAdapter
from eegfmlens.errors import ValidationError


def test_steegformer_channel_lookup_and_patch_contract():
    model = SimpleNamespace(
        global_pool=False,
        patch_embed=SimpleNamespace(p=16),
        blocks=[None],
        enc_channel_emd=SimpleNamespace(channel_transformation=SimpleNamespace(num_embeddings=145)),
        enc_temporal_emd=SimpleNamespace(pe=torch.zeros(1, 512, 4)),
    )
    adapter = STEEGFormerAdapter(model, channels=("C4", "C3"), channel_mapping={"C3": 41, "C4": 2})
    assert adapter.channel_indices == (2, 41)
    valid = SignalBatch(torch.ones(1, 2, 48, 16), ("a",), adapter.channels, 128, "fixture")
    adapter.validate(valid)
    short = SignalBatch(torch.ones(1, 2, 1, 15), ("a",), adapter.channels, 128, "fixture")
    with pytest.raises(ValidationError, match="complete"):
        adapter.validate(short)
    with pytest.raises(ValidationError, match="absent"):
        STEEGFormerAdapter(model, channels=("unknown",), channel_mapping={})
    model.global_pool = True
    with pytest.raises(ValidationError, match="configuration changed"):
        adapter.forward(model, valid)


def test_diver_sensor_coordinates_are_owned_and_guarded():
    model = SimpleNamespace(
        patcher=SimpleNamespace(patch_len=500, stride=500),
        encoder=SimpleNamespace(encoder=SimpleNamespace(layers=[None])),
    )
    positions = torch.zeros(2, 3)
    adapter = DIVERAdapter(model, channels=("C3", "C4"), positions=positions)
    batch = SignalBatch(torch.ones(1, 2, 2, 500), ("a",), adapter.channels, 500, "fixture")
    positions.add_(1)
    adapter.validate(batch)
    assert adapter.data_info(batch)[0]["xyz_id"].count_nonzero() == 0
    assert adapter.data_info(batch)[0]["modality"] == "EEG"
    adapter.positions.add_(1)
    with pytest.raises(ValidationError, match="positions changed"):
        adapter.validate(batch)
