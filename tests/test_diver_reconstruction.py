"""Mask semantics and output ancestry, independent of external checkpoints."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eegfmlens import Ablation, DIVERAdapter, EEGLens, Replacement, Selection, SignalBatch
from eegfmlens.errors import ValidationError


class MaskGenerator(nn.Module):
    def forward(self, x):
        return torch.rand(x.shape[:-1], device=x.device) > 0.5


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Identity()])

    def forward(self, x):
        return self.encoder.layers[0](x)


class NativeFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.patcher = SimpleNamespace(patch_len=500, stride=500)
        self.token_manager = SimpleNamespace(RAN_PREPEND=False)
        self.mask_generator = MaskGenerator()
        self.embedding = nn.Linear(500, 4, bias=False)
        self.encoder = Encoder()
        self.head = nn.Linear(4, 4)
        org = nn.Module()
        org.heads = nn.ModuleDict({"time_head": nn.Linear(4, 500, bias=False)})
        self.heads = nn.ModuleDict({"org": org})

    def forward(self, x, *, data_info_list, use_mask):
        self.token_manager.RAN_PREPEND = True
        self.token_manager.PREPEND_PARAMS = (x.shape,)
        if use_mask:
            x = x.masked_fill(self.mask_generator(x).unsqueeze(-1), 0)
        features = self.encoder(self.embedding(x))
        result = {
            "y": self.head(features),
            "y_org": {"time_head_output": self.heads["org"].heads["time_head"](features)},
        }
        self.token_manager.RAN_PREPEND = False
        return result


def fixture():
    torch.manual_seed(71)
    model = NativeFixture().eval()
    adapter = DIVERAdapter(
        model, channels=("C3", "C4"), positions=torch.zeros(2, 3), output="reconstruction"
    )
    lens = EEGLens(model, adapter)
    batch = SignalBatch(torch.randn(2, 2, 3, 500), ("a", "b"), ("C3", "C4"), 500, "fixture")
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[:, 0, 1] = True
    return lens, batch, mask


def test_time_head_ancestry_mask_and_physical_selection():
    lens, batch, mask = fixture()
    assert "features.output" not in lens.adapter.sites
    run = lens.run_with_cache(batch, mask=mask.tolist())
    # Independent native input-masking reference; fixture embedding is deterministic.
    raw = batch.data.masked_fill(mask.unsqueeze(-1), 0)
    expected = lens.model(raw, data_info_list=[], use_mask=False)["y_org"]["time_head_output"]
    torch.testing.assert_close(run.output, expected, rtol=0, atol=0)
    own = lens.run_with_interventions(
        batch,
        mask=mask.tolist(),
        interventions=[Replacement("reconstruction.output", run.cache["reconstruction.output"])],
    )
    torch.testing.assert_close(own.output, expected, rtol=0, atol=0)
    edited = lens.run_with_interventions(
        batch,
        mask=mask.tolist(),
        interventions=[Ablation("reconstruction.output", Selection(sensors=("C4",), patches=(2,)))],
    )
    manual = expected.clone()
    manual[:, 1, 2] = 0
    assert expected[:, 1, 2].abs().sum() > 0
    torch.testing.assert_close(edited.output, manual, rtol=0, atol=0)
    # The unrelated feature head is actually executed but does not feed the time head.
    h = lens.model.head.register_forward_hook(lambda m, a, o: o * 0)
    try:
        unaffected = lens.adapter.forward(lens.model, batch, mask=mask.tolist())
    finally:
        h.remove()
    torch.testing.assert_close(unaffected, expected, rtol=0, atol=0)
    assert not lens.model.mask_generator._forward_hooks


def test_mask_failures_and_exception_restore_native_bookkeeping():
    lens, batch, mask = fixture()
    for kwargs in ({}, {"mask": mask.int().tolist()}, {"mask": [True]}, {"mask": None}):
        with pytest.raises(ValidationError):
            lens.run_with_cache(batch, **kwargs)
        assert not lens.model.mask_generator._forward_hooks

    def fail(module, inputs, output):
        raise RuntimeError("injected native failure")

    h = lens.model.embedding.register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            lens.run_with_cache(batch, mask=mask.tolist())
    finally:
        h.remove()
    assert not lens.model.mask_generator._forward_hooks
    assert lens.model.token_manager.RAN_PREPEND is False
    assert not hasattr(lens.model.token_manager, "PREPEND_PARAMS")
    assert torch.isfinite(lens.run_with_cache(batch, mask=mask.tolist()).output).all()
