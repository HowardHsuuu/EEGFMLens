"""Reject ambiguous hook descriptors before they can select the wrong output."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from eeglens import Ablation, ActivationSite, Adapter, EEGLens, SignalBatch
from eeglens.errors import ValidationError


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("name", 0),
        ("module_path", None),
        ("layout", ""),
        ("writable", "false"),
        ("writable", 1),
        ("tensor_index", True),
        ("tensor_index", 1.0),
    ],
)
def test_ambiguous_site_descriptor_rejected(field, value):
    with pytest.raises(ValidationError):
        Adapter([replace(ActivationSite("signal", ""), **{field: value})])


def test_non_site_rejected():
    with pytest.raises(ValidationError, match="ActivationSite"):
        Adapter([object()])


def test_root_tuple_output_and_read_only_site():
    class Pair(nn.Module):
        def forward(self, x):
            return x * 2, x * 3

    batch = SignalBatch(torch.ones(1, 1, 1, 2), ("trial",), ("C3",), 200, "fixture")
    model = Pair().eval()
    lens = EEGLens(
        model,
        Adapter(
            [
                ActivationSite("first", "", tensor_index=0, writable=False),
                ActivationSite("last", "", tensor_index=-1),
            ]
        ),
    )
    with pytest.raises(ValidationError, match="not writable"):
        lens.run_with_interventions(batch, interventions=[Ablation("first")])
    assert not model._forward_hooks
    result = lens.run_with_interventions(batch, interventions=[Ablation("last")])
    torch.testing.assert_close(result.output[0], batch.data * 2, atol=0, rtol=0)
    torch.testing.assert_close(result.output[1], torch.zeros_like(batch.data), atol=0, rtol=0)
    assert not model._forward_hooks
