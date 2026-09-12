from dataclasses import replace

import pytest
import torch
from torch import nn

from eeglens import (
    Ablation,
    ActivationSite,
    Adapter,
    AxisSelection,
    EEGLens,
    Replacement,
    Selection,
    SignalBatch,
    SweepTarget,
    load_run,
    patching_sweep,
    save_run,
)
from eeglens.errors import ValidationError


class Pair(nn.Module):
    def forward(self, x):
        hidden = x.flatten(1, 2)
        return hidden + 100, hidden


class Weighted(nn.Module):
    def __init__(self):
        super().__init__()
        self.pair = Pair()
        self.register_buffer("weights", torch.arange(1, 7).float()[None, :, None])

    def forward(self, x):
        first, second = self.pair(x)
        return first.sum(1) + (second * self.weights).sum(1)


def fixture():
    model = Weighted().eval()
    lens = EEGLens(model, Adapter([ActivationSite("summary", "pair", "batch", tensor_index=1)]))
    batch = SignalBatch(
        torch.arange(48).float().reshape(2, 2, 3, 4), ("a", "b"), ("C3", "C4"), 200, "synthetic"
    )
    return lens, batch


def test_axis_replacement_native_parity_trial_alignment_and_export(tmp_path):
    lens, source = fixture()
    reversed_source = replace(source, data=source.data.flip(0), trial_ids=("b", "a"))
    donor = lens.run_with_cache(reversed_source)
    recipient = replace(source, data=torch.zeros_like(source.data))
    selection = AxisSelection(axis=1, indices=(1, 4))
    result = lens.run_with_interventions(
        recipient,
        sites=("summary",),
        interventions=[Replacement("summary", donor.cache["summary"], selection)],
    )

    def native(module, inputs, output):
        second = output[1].clone()
        second[:, [1, 4]] = source.data.flatten(1, 2)[:, [1, 4]]
        return output[0], second

    handle = lens.model.pair.register_forward_hook(native)
    try:
        expected = lens.model(recipient.data)
    finally:
        handle.remove()
    torch.testing.assert_close(result.output, expected, rtol=0, atol=0)
    assert torch.count_nonzero(result.cache["summary"].tensor[:, [0, 2, 3, 5]]) == 0
    record = result.metadata["interventions"][0]
    assert record["axis"] == 1 and record["indices"] == (1, 4)
    assert "sensors" not in record and record["donor_trial_ids"] == ("b", "a")
    saved = load_run(save_run(result, tmp_path / "run"))
    assert saved.metadata["interventions"][0]["indices"] == [1, 4]
    torch.testing.assert_close(
        lens.run_with_cache(source).output, lens.model(source.data), rtol=0, atol=0
    )
    assert not lens.model.pair._forward_hooks


def test_identity_and_axis_ablation_preserve_tuple_sibling():
    lens, batch = fixture()
    run = lens.run_with_cache(batch)
    selection = AxisSelection(1, (2,))
    identity = lens.run_with_interventions(
        batch, interventions=[Replacement("summary", run.cache["summary"], selection)]
    )
    torch.testing.assert_close(identity.output, run.output, rtol=0, atol=0)
    zero = lens.run_with_interventions(batch, interventions=[Ablation("summary", selection)])
    expected = run.output - batch.data.flatten(1, 2)[:, 2] * 3
    torch.testing.assert_close(zero.output, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "axis,indices",
    [(0, (1,)), (-1, (1,)), (True, (1,)), (1, ()), (1, (1, 1)), (1, (-1,)), (1, (True,)), (1, [1])],
)
def test_invalid_axis_selections(axis, indices):
    with pytest.raises(ValidationError):
        AxisSelection(axis, indices)


@pytest.mark.parametrize("selection", [AxisSelection(3, (0,)), AxisSelection(1, (6,))])
def test_runtime_bounds_fail_and_hooks_recover(selection):
    lens, batch = fixture()
    baseline = lens.run_with_cache(batch).output
    with pytest.raises(ValidationError, match="out of range"):
        lens.run_with_interventions(batch, interventions=[Ablation("summary", selection)])
    torch.testing.assert_close(lens.run_with_cache(batch).output, baseline, rtol=0, atol=0)
    assert not lens.model.pair._forward_hooks


def test_physical_selection_is_not_inferred_for_batch_layout():
    lens, batch = fixture()
    with pytest.raises(ValidationError, match="no sensor/patch selector"):
        lens.run_with_interventions(
            batch, interventions=[Ablation("summary", Selection(sensors=("C3",)))]
        )


def test_physical_sweep_explicitly_rejects_axis_targets():
    original, batch = fixture()
    del original
    lens = EEGLens(
        Weighted().eval(),
        Adapter([ActivationSite("summary", "pair", "patch_tokens", tensor_index=1)]),
    )
    with pytest.raises(ValidationError, match="requires physical Selection"):
        patching_sweep(
            lens,
            batch,
            batch,
            lambda output, b: output.sum(),
            sites=("summary",),
            targets=(SweepTarget("index", AxisSelection(1, (1,))),),
        )
