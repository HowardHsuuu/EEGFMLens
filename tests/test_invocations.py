import pytest
import torch
from torch import nn

from eeglens import Ablation, ActivationSite, Adapter, EEGLens, SignalBatch
from eeglens.errors import HookExecutionError


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Identity()

    def forward(self, x):
        a = self.shared(x + 1)
        b = self.shared(x + 3)
        return a + 2 * b


def test_reused_module_calls_have_independent_effects():
    batch = SignalBatch(torch.ones(2, 1, 2, 3), ("a", "b"), ("C3",), 200, "fixture")
    model = Shared().eval()
    adapter = Adapter(
        [
            ActivationSite("first", "shared", call_index=0, expected_calls=2),
            ActivationSite("second", "shared", call_index=1, expected_calls=2),
        ]
    )
    lens = EEGLens(model, adapter)
    r = lens.run_with_cache(batch)
    torch.testing.assert_close(r.cache["first"].tensor, batch.data + 1)
    torch.testing.assert_close(r.cache["second"].tensor, batch.data + 3)
    for name, expected in [("first", 2 * (batch.data + 3)), ("second", batch.data + 1)]:
        out = lens.run_with_interventions(batch, interventions=[Ablation(name)])
        torch.testing.assert_close(out.output, expected)
    assert not model.shared._forward_hooks


def test_undeclared_reuse_still_fails_and_cleans_hooks():
    model = Shared().eval()
    lens = EEGLens(model, Adapter([ActivationSite("single", "shared")]))
    batch = SignalBatch(torch.ones(1, 1, 1, 2), ("a",), ("C3",), 200, "fixture")
    with pytest.raises(HookExecutionError):
        lens.run_with_cache(batch)
    assert not model.shared._forward_hooks
