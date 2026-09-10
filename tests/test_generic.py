from collections import namedtuple

import pytest
import torch
from torch import nn

from eeglens import Ablation, ActivationSite, EEGLens, GenericAdapter, SignalBatch, inspect_modules
from eeglens.errors import ValidationError


class Structured(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        if self.kind == "dict":
            return {"hidden": x * 2, "aux": "preserved"}
        return namedtuple("Result", "hidden aux")(x * 2, "preserved")


class ContinuousCNN(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.conv = nn.Conv1d(2, 3, 3, padding=1)
        self.container = Structured(kind)
        self.kind = kind

    def forward(self, x, scale=1):
        out = self.container(self.conv(x))
        assert (out["aux"] if self.kind == "dict" else out.aux) == "preserved"
        h = out["hidden"] if self.kind == "dict" else out.hidden
        return h.mean((1, 2)) * scale


@pytest.mark.parametrize("kind,key", [("dict", "hidden"), ("namedtuple", 0)])
def test_generic_continuous_native_observation_and_intervention(kind, key):
    model = ContinuousCNN(kind).eval()
    batch = SignalBatch(torch.randn(2, 2, 3, 4), ("a", "b"), ("C3", "C4"), 100, "continuous")
    adapter = GenericAdapter(
        [ActivationSite("features", "container", "batch", key)],
        forward=lambda model, batch, **kw: model(batch.data.flatten(2), **kw),
    )
    lens = EEGLens(model, adapter)
    baseline = lens.run_with_cache(batch, scale=2)
    torch.testing.assert_close(
        baseline.output, model(batch.data.flatten(2), scale=2), rtol=0, atol=0
    )
    result = lens.run_with_interventions(batch, interventions=[Ablation("features")], scale=2)
    torch.testing.assert_close(result.output, torch.zeros(2), rtol=0, atol=0)
    assert not model.container._forward_hooks
    torch.testing.assert_close(lens.run_with_cache(batch, scale=2).output, baseline.output)


def test_discovery_is_structural_and_validation_precedes_forward():
    model = ContinuousCNN("dict").eval()
    info = inspect_modules(model)
    assert next(r for r in info if r.path == "conv").direct_parameters == 21
    batch = SignalBatch(torch.ones(1, 2, 1, 4), ("a",), ("C3", "C4"), 100, "fixture")

    def reject(batch):
        raise ValidationError("Unsupported sampling rate")

    lens = EEGLens(
        model, GenericAdapter([ActivationSite("conv", "conv", "batch")], validate=reject)
    )
    with pytest.raises(ValidationError, match="sampling rate"):
        lens.run_with_cache(batch)
    assert not model.conv._forward_hooks
