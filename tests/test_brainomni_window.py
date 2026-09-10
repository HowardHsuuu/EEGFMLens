from types import SimpleNamespace

import pytest
import torch

from eeglens import BrainOmniAdapter, SignalBatch
from eeglens.errors import ValidationError


def fixture():
    calls = []
    model = SimpleNamespace(
        window_length=512,
        overlap_ratio=0.25,
        blocks=[None, None],
        tokenizer=SimpleNamespace(
            window_length=512,
            sensor_embed=SimpleNamespace(sensor_embedding_layer=SimpleNamespace(num_embeddings=2)),
        ),
        encode=lambda *args: calls.append(1),
    )
    adapter = BrainOmniAdapter(
        model,
        channels=("C3",),
        positions=torch.zeros(1, 6),
        sensor_types=torch.zeros(1, dtype=torch.long),
    )
    return model, adapter, calls


@pytest.mark.parametrize("field", ["window_length", "overlap_ratio", "tokenizer_window"])
def test_window_configuration_change_is_rejected_before_encode(field):
    model, adapter, calls = fixture()
    target, attr = (
        (model.tokenizer, "window_length") if field == "tokenizer_window" else (model, field)
    )
    original = getattr(target, attr)
    setattr(target, attr, original + 1)
    batch = SignalBatch(torch.ones(1, 1, 1, 1), ("a",), ("C3",), 256, "fixture")
    with pytest.raises(ValidationError, match="configuration changed"):
        adapter.forward(model, batch)
    assert not calls
    setattr(target, attr, original)
    adapter.validate(batch)  # Short input remains allowed; native tokenizer pads it.
    adapter.forward(model, batch)
    assert calls == [1]
    assert adapter.metadata()["native_stride_samples"] == 384


@pytest.mark.parametrize("overlap", [-1, 1, float("nan"), 0.99999])
def test_invalid_window_stride_is_rejected(overlap):
    model, _, _ = fixture()
    model.overlap_ratio = overlap
    with pytest.raises(ValidationError, match="valid overlap"):
        BrainOmniAdapter(
            model,
            channels=("C3",),
            positions=torch.zeros(1, 6),
            sensor_types=torch.zeros(1, dtype=torch.long),
        )
