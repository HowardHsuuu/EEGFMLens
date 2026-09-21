from dataclasses import replace

import pytest
import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    EEGLens,
    Selection,
    SignalBatch,
    SweepTarget,
    restoration_sweep,
)
from eegfmlens.errors import ValidationError


class RestorationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Identity()
        self.late = nn.Identity()

    def forward(self, data):
        hidden = self.early(data)
        hidden = self.late(hidden * 2)
        return hidden[:, 0, 1, 0] + hidden[:, 1, 2, 0]


def fixture():
    clean = SignalBatch(
        torch.tensor(
            [
                [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
                [[[7.0], [8.0], [9.0]], [[10.0], [11.0], [12.0]]],
            ]
        ),
        ("a", "b"),
        ("C3", "C4"),
        200,
        "restoration-test",
    )
    recipient = replace(clean, data=torch.zeros_like(clean.data))
    model = RestorationModel().eval()
    lens = EEGLens(
        model,
        Adapter([ActivationSite("early", "early"), ActivationSite("late", "late")]),
    )
    return lens, clean, recipient


def test_restoration_workflow_scores_known_causal_coordinates():
    lens, clean, recipient = fixture()
    result = restoration_sweep(
        lens,
        clean,
        recipient,
        sites=("early", "late"),
        random_controls=False,
    )
    assert result.metadata["workflow"]["name"] == "restoration_sweep"
    assert all(row["clean_score"] == pytest.approx(0) for row in result.rows)
    event = {
        (row["site"], row["target"], row["trial_id"]): row
        for row in result.rows
        if row["kind"] == "event"
    }
    assert event[("early", "C3:patch1", "a")]["clean_error_reduction"] > 0
    assert event[("late", "C4:patch2", "b")]["clean_error_reduction"] > 0
    assert event[("early", "C3:patch0", "a")]["clean_error_reduction"] == pytest.approx(0)


def test_restoration_workflow_preserves_controls_and_cleanup():
    lens, clean, recipient = fixture()
    target = SweepTarget(
        "causal",
        Selection(sensors=("C3",), patches=(1,)),
        off_event=Selection(sensors=("C4",), patches=(0,)),
    )
    result = restoration_sweep(lens, clean, recipient, sites=("early",), targets=(target,))
    assert {row["kind"] for row in result.rows} == {"identity", "event", "off_event", "random"}
    assert not lens.model.early._forward_hooks


def test_restoration_workflow_rejects_non_tensor_output():
    lens, clean, recipient = fixture()
    lens.model.forward = lambda data: (data, data)
    with pytest.raises(ValidationError, match="batch-first tensor"):
        restoration_sweep(lens, clean, recipient, sites=("early",))


def test_restoration_workflow_rejects_zero_clean_reference():
    lens, clean, recipient = fixture()

    def zero_forward(data):
        hidden = lens.model.early(data)
        hidden = lens.model.late(hidden * 2)
        return hidden[:, 0, 0, 0] * 0

    lens.model.forward = zero_forward
    with pytest.raises(ValidationError, match="near-zero clean reference"):
        restoration_sweep(lens, clean, recipient, sites=("early",))
