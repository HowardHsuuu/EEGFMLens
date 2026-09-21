from dataclasses import dataclass, field

import pytest
import torch
from torch import nn

from eegfmlens import ActivationSite, Adapter, EEGLens, Selection, SignalBatch, SubspaceAblation
from eegfmlens.errors import ValidationError


@dataclass
class InvalidEdit:
    kind: str
    site: str = "hidden"
    selection: Selection = Selection()

    def apply(self, current, batch, layout, model_id):
        if self.kind == "shape":
            return current.transpose(1, 2)  # Same element count; restore must not silently reshape.
        if self.kind == "dtype":
            return current.double()
        if self.kind == "nonfinite":
            return torch.full_like(current, float("nan"))
        return None


@dataclass
class DiagnosticEdit:
    site: str = "hidden"
    selection: Selection = Selection()
    diagnostics: dict = field(default_factory=dict)

    def apply(self, current, batch, layout, model_id):
        self.diagnostics["changed_elements"] = current.numel()
        return current + 1


@pytest.mark.parametrize("warning", [None, "explicit warning"])
def test_custom_diagnostics_do_not_require_subspace_specific_metadata(warning):
    model = nn.Sequential(nn.Identity()).eval()
    lens = EEGLens(model, Adapter([ActivationSite("hidden", "0")]))
    batch = SignalBatch(torch.zeros(1, 1, 1, 4), ("a",), ("C3",), 200, "fixture")
    edit = DiagnosticEdit()
    if warning is not None:
        edit.multiplier_warning = warning
    result = lens.run_with_interventions(batch, interventions=[edit])
    torch.testing.assert_close(result.output, torch.ones_like(batch.data), rtol=0, atol=0)
    record = result.metadata["interventions"][0]
    assert record["diagnostics"] == {"changed_elements": 4}
    assert record["reference_selection"] is None
    assert record["multiplier_warning"] == warning
    edit.diagnostics["changed_elements"] = -1
    assert record["diagnostics"]["changed_elements"] == 4
    assert not model[0]._forward_hooks


@pytest.mark.parametrize("kind", ["shape", "dtype", "nonfinite", "nontensor"])
def test_invalid_intervention_is_rejected_and_hooks_recover(kind):
    model = nn.Sequential(nn.Identity()).eval()
    lens = EEGLens(model, Adapter([ActivationSite("hidden", "0")]))
    batch = SignalBatch(torch.randn(2, 2, 3, 4), ("a", "b"), ("C3", "C4"), 200, "fixture")
    existing_calls = []
    handle = model[0].register_forward_hook(lambda *args: existing_calls.append(1))
    try:
        with pytest.raises(ValidationError, match="Intervention at hidden"):
            lens.run_with_interventions(batch, interventions=[InvalidEdit(kind)])
        assert len(model[0]._forward_hooks) == 1
        torch.testing.assert_close(lens.run_with_cache(batch).output, batch.data, rtol=0, atol=0)
        assert len(existing_calls) == 2
    finally:
        handle.remove()


def test_bad_inputs_are_rejected_before_execution():
    model = nn.Sequential(nn.Identity()).eval()
    lens = EEGLens(model, Adapter([ActivationSite("hidden", "0")]))
    batch = SignalBatch(torch.randn(1, 1, 1, 4), ("a",), ("C3",), 200, "fixture")
    calls = []
    handle = model[0].register_forward_hook(lambda *args: calls.append(1))
    try:
        with pytest.raises(ValidationError, match="SignalBatch"):
            lens.run_with_cache(batch.data)
        with pytest.raises(ValidationError, match="not a string"):
            lens.run_with_cache(batch, sites="hidden")
        with pytest.raises(ValidationError, match="Interventions require"):
            lens.run_with_interventions(batch, interventions=[object()])
        assert calls == []
        with pytest.raises(ValidationError, match="must be tensors"):
            lens.run_with_interventions(batch, interventions=[SubspaceAblation("hidden", [], [])])
        assert len(model[0]._forward_hooks) == 1
    finally:
        handle.remove()
