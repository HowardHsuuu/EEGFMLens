from types import SimpleNamespace

import pytest
import torch

from eeglens import NeuroRVQAdapter, SignalBatch
from eeglens.errors import ValidationError


def test_channel_aliases_must_not_duplicate_native_electrodes():
    model = SimpleNamespace(pos_embed=torch.zeros(3, 4), time_embed=torch.zeros(8, 4), blocks=[])
    adapter = NeuroRVQAdapter(model, channel_vocabulary=("c3", "c4"))
    duplicate = SignalBatch(torch.ones(1, 2, 1, 200), ("a",), ("C3", "c3"), 200, "fixture")
    with pytest.raises(ValidationError, match="distinct native electrodes"):
        adapter.validate(duplicate)
    valid = SignalBatch(duplicate.data, ("a",), ("C3", "c4"), 200, "fixture")
    adapter.validate(valid)
    _, space = adapter.indices(valid)
    assert space.tolist() == [[0, 1]]
