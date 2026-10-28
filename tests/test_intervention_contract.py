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


@dataclass
class LocalityEdit:
    selection: Selection
    inplace: bool = False
    escape: bool = True
    mutate_declaration: bool = False
    site: str = "hidden"

    def apply(self, current, batch, layout, model_id):
        mask = self.selection.mask(current, batch, layout)
        if self.mutate_declaration:
            self.selection = Selection()
        if self.inplace:
            current.add_(1) if self.escape else current.masked_fill_(mask, 1)
            return current
        return current + (torch.ones_like(current) if self.escape else mask)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("mutate_declaration", [False, True])
def test_custom_selection_escape_rejected_without_native_or_input_damage(
    inplace, mutate_declaration
):
    model = nn.Sequential(nn.Identity()).eval()
    lens = EEGLens(model, Adapter([ActivationSite("hidden", "0")]))
    batch = SignalBatch(torch.zeros(1, 2, 2, 4), ("a",), ("C3", "C4"), 200, "fixture")
    selection = Selection(sensors=("C3",), patches=(0,))
    with pytest.raises(ValidationError, match="outside its selection"):
        lens.run_with_interventions(
            batch, interventions=[LocalityEdit(selection, inplace, True, mutate_declaration)]
        )
    assert torch.count_nonzero(batch.data) == 0
    assert not model[0]._forward_hooks
    assert torch.equal(lens.run_with_cache(batch).output, batch.data)


@pytest.mark.parametrize("inplace", [False, True])
def test_custom_selected_edit_accepted_and_metadata_binds_original_declaration(inplace):
    model = nn.Sequential(nn.Identity()).eval()
    lens = EEGLens(model, Adapter([ActivationSite("hidden", "0")]))
    batch = SignalBatch(torch.zeros(1, 2, 2, 4), ("a",), ("C3", "C4"), 200, "fixture")
    selection = Selection(sensors=("C3",), patches=(0,))
    result = lens.run_with_interventions(
        batch, interventions=[LocalityEdit(selection, inplace, False, True)]
    )
    expected = torch.zeros_like(batch.data)
    expected[:, 0, 0] = 1
    assert torch.equal(result.output, expected)
    assert torch.count_nonzero(batch.data) == 0
    record = result.metadata["interventions"][0]
    assert record["sensors"] == ("C3",) and record["patches"] == (0,)
    assert record["selection_support_verified"]


def test_custom_invalid_physical_selection_is_checked_before_apply():
    model = nn.Sequential(nn.Identity()).eval()
    lens = EEGLens(model, Adapter([ActivationSite("hidden", "0", layout="batch")]))
    batch = SignalBatch(torch.zeros(1, 2, 2, 4), ("a",), ("C3", "C4"), 200, "fixture")
    with pytest.raises(ValidationError, match="no sensor/patch selector"):
        lens.run_with_interventions(batch, interventions=[LocalityEdit(Selection(sensors=("C3",)))])
    assert not model[0]._forward_hooks


def test_axis_custom_escape_is_rejected():
    from eegfmlens import AxisSelection

    model = nn.Sequential(nn.Identity()).eval()
    lens = EEGLens(model, Adapter([ActivationSite("hidden", "0")]))
    batch = SignalBatch(torch.zeros(1, 2, 2, 4), ("a",), ("C3", "C4"), 200, "fixture")
    with pytest.raises(ValidationError, match="outside its selection"):
        lens.run_with_interventions(batch, interventions=[LocalityEdit(AxisSelection(2, (0,)))])
