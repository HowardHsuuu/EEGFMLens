from dataclasses import replace

import pytest
import torch
from torch import nn

from eegfmlens import ActivationSite, Adapter, EEGLens, Replacement, SignalBatch
from eegfmlens.errors import ValidationError


def test_mutated_weights_and_duplicate_wrapper_are_rejected():
    model = nn.Sequential(nn.Linear(2, 2)).eval()
    adapter = Adapter([ActivationSite("layer", "0")])
    lens = EEGLens(model, adapter)
    with pytest.raises(ValidationError, match="live"):
        EEGLens(model, adapter)
    batch = SignalBatch(torch.ones(1, 1, 1, 2), ("a",), ("C3",), 200, "test")
    lens.run_with_cache(batch)
    with torch.no_grad():
        model[0].weight.add_(1)
    with pytest.raises(ValidationError, match="state changed"):
        lens.run_with_cache(batch)
    assert not model[0]._forward_hooks


def test_configuration_mismatch_and_input_mutation_are_rejected():
    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Identity()

        def forward(self, x, scale=1):
            return self.layer(x * scale)

    lens = EEGLens(Scaled().eval(), Adapter([ActivationSite("layer", "layer")]))
    batch = SignalBatch(torch.ones(1, 1, 1, 2), ("a",), ("C3",), 200, "test")
    donor = lens.run_with_cache(batch, scale=2).cache["layer"]
    with pytest.raises(ValidationError, match="configuration"):
        lens.run_with_interventions(batch, interventions=[Replacement("layer", donor)], scale=1)
    assert not lens.model.layer._forward_hooks
    with pytest.raises(ValidationError):
        replace(batch, sampling_rate=float("nan"))
    batch.data.fill_(float("nan"))
    with pytest.raises(ValidationError, match="Non-finite"):
        lens.run_with_cache(batch)
