"""Independent analytical attention oracle and scoped runtime fault checks."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from eegfmlens import EEGLens, ProjectedAttentionReplacement, Selection, SignalBatch
from eegfmlens.adapters.base import Adapter
from eegfmlens.adapters.steegformer import STEEGFormerMAEAdapter
from eegfmlens.errors import ValidationError
from eegfmlens.types import ActivationSite

INPUT = "decoder.0.attention_input"
SITE = "decoder.0.native_attention"


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)
        self.proj = nn.Linear(4, 4)
        self.attn_drop = nn.Dropout(0)
        self.proj_drop = nn.Dropout(0)
        self.num_heads = 2
        self.calls = 0
        self.fail_at = None

    def forward(self, x):
        self.calls += 1
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, 2, 2).permute(2, 0, 3, 1, 4).unbind(0)
        if self.calls == self.fail_at:
            raise RuntimeError("injected native failure")
        attn = ((q @ k.transpose(-2, -1)) / 2**0.5).softmax(-1)
        return self.proj_drop(
            self.proj((self.attn_drop(attn) @ v).transpose(1, 2).reshape(b, n, 4))
        )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.attn = Attention()

    def forward(self, x):
        return self.attn(self.norm(x))


class SmallAdapter(STEEGFormerMAEAdapter):
    """Uses the real ST physical expose/restore on a small attention fixture."""

    def __init__(self):
        self.channels = ("C4", "Cz", "C3")
        Adapter.__init__(
            self, [ActivationSite(INPUT, "norm", "tokens"), ActivationSite(SITE, "attn", "tokens")]
        )

    def validate(self, batch):
        assert batch.channels == self.channels and batch.data.shape[1:] == (3, 2, 4)

    def metadata(self):
        return {"scope": "small analytical attention fixture, not a native checkpoint"}

    def forward(self, model, batch, **kwargs):
        assert set(kwargs) == {"mask"}
        x = batch.data.masked_fill(torch.as_tensor(kwargs["mask"])[..., None], 0)
        native = x.transpose(1, 2).reshape(len(x), 6, 4)
        return model(torch.cat((torch.zeros(len(x), 1, 4), native), dim=1))


@pytest.fixture
def setup():
    with torch.random.fork_rng():
        torch.manual_seed(73)
        model = Model().eval()
        lens = EEGLens(model, SmallAdapter())
        batch = SignalBatch(torch.randn(2, 3, 2, 4), ("a", "b"), lens.adapter.channels, 128, "test")
        donor = replace(batch, data=torch.randn_like(batch.data))
    mask = torch.zeros(2, 3, 2, dtype=torch.bool)
    mask[:, 1, 0] = True
    kwargs = {"mask": mask.tolist()}
    live = lens.run_with_cache(batch, sites=(INPUT, SITE), **kwargs)
    other = lens.run_with_cache(donor, sites=(INPUT, SITE), **kwargs)
    return lens, batch, kwargs, live, other


def oracle(lens, live, donor, component, head, sensor):
    """Direct linear algebra, without the package's replay or projection hook."""
    module = lens.model.attn
    x = lens.adapter.restore(
        live.cache[INPUT].tensor, lens.adapter.require(INPUT), live.cache[INPUT].native_shape
    )
    y = lens.adapter.restore(
        donor.cache[INPUT].tensor, lens.adapter.require(INPUT), donor.cache[INPUT].native_shape
    )
    values = [
        torch.nn.functional.linear(z, module.qkv.weight, module.qkv.bias)
        .reshape(2, 7, 3, 2, 2)
        .permute(2, 0, 3, 1, 4)
        for z in (x, y)
    ]
    mixed = values[0].clone()
    indices = {"q": [0], "k": [1], "v": [2], "qk": [0, 1], "all": [0, 1, 2]}[component]
    for i in indices:
        if head is None:
            mixed[i] = values[1][i]
        else:
            mixed[i, :, head] = values[1][i, :, head]
    q, k, v = mixed.unbind(0)
    weights = torch.softmax(torch.einsum("bhid,bhjd->bhij", q, k) * (2**-0.5), -1)
    output = torch.einsum("bhij,bhjd->bhid", weights, v).transpose(1, 2).reshape(2, 7, 4)
    projected = torch.nn.functional.linear(output, module.proj.weight, module.proj.bias)
    expected = live.output.clone()
    token = 1 + 1 * 3 + lens.adapter.channels.index(sensor)
    expected[:, token] = projected[:, token]
    return expected


@pytest.mark.parametrize("component", ["q", "k", "v", "qk", "all"])
@pytest.mark.parametrize("head", [None, 1])
@pytest.mark.parametrize("sensor", ["C3", "C4"])
def test_projected_components_match_independent_physical_oracle(setup, component, head, sensor):
    lens, batch, kwargs, live, donor = setup
    edit = ProjectedAttentionReplacement(
        lens,
        SITE,
        live.cache[INPUT],
        donor.cache[INPUT],
        component=component,
        selection=Selection(sensors=(sensor,), patches=(1,)),
        head=head,
    )
    result = lens.run_with_interventions(batch, interventions=[edit], sites=(SITE,), **kwargs)
    torch.testing.assert_close(
        result.output, oracle(lens, live, donor, component, head, sensor), atol=1e-6, rtol=1e-6
    )
    selected = edit.selection.mask(result.cache[SITE].tensor, batch, "tokens")
    assert torch.equal(result.cache[SITE].tensor[~selected], live.cache[SITE].tensor[~selected])
    assert edit.diagnostics["selected_elements"] == 8
    assert len(edit.diagnostics["selected_projected_edit_rms"]) == 2
    assert result.metadata["interventions"][0]["parameters"]["component"] == component
    assert not lens.model.attn.qkv._forward_hooks
    lens.assert_model_unchanged()


def make_edit(setup, **overrides):
    lens, _, _, live, donor = setup
    args = dict(component="q", selection=Selection(sensors=("C4",), patches=(1,)), head=1)
    args.update(overrides)
    return ProjectedAttentionReplacement(lens, SITE, live.cache[INPUT], donor.cache[INPUT], **args)


def test_own_state_exact_and_cache_ownership(setup):
    lens, batch, kwargs, live, donor = setup
    edit = ProjectedAttentionReplacement(
        lens,
        SITE,
        live.cache[INPUT],
        live.cache[INPUT],
        component="all",
        selection=Selection(sensors=("C3",), patches=(1,)),
    )
    live.cache[INPUT].tensor.add_(900)
    result = lens.run_with_interventions(batch, interventions=[edit], **kwargs)
    assert torch.equal(result.output, live.output)
    assert edit.diagnostics["projected_edit_rms"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", "other"),
        ("trial_ids", ("b", "a")),
        ("execution_id", "different"),
        ("preprocessing_id", "other"),
        ("site", SITE),
    ],
)
def test_mismatched_cache_rejected(setup, field, value):
    lens, _, _, live, donor = setup
    with pytest.raises(ValidationError, match="identities"):
        ProjectedAttentionReplacement(
            lens,
            SITE,
            live.cache[INPUT],
            replace(donor.cache[INPUT], **{field: value}),
            component="q",
            selection=Selection(sensors=("C4",), patches=(1,)),
        )


@pytest.mark.parametrize("head", [True, -1, 2, 1.0])
def test_invalid_heads_rejected(setup, head):
    with pytest.raises(ValidationError, match="head"):
        make_edit(setup, head=head)


def test_changed_execution_rejected_with_cleanup(setup):
    lens, batch, kwargs, _, _ = setup
    edit = make_edit(setup)
    mask = torch.as_tensor(kwargs["mask"])
    mask[:, 0, 0] = True
    with pytest.raises(ValidationError, match="execution configuration"):
        lens.run_with_interventions(batch, interventions=[edit], mask=mask.tolist())
    assert all(not m._forward_hooks for m in lens.model.modules())
    lens.assert_model_unchanged()


def test_stale_live_cache_rejected(setup):
    lens, batch, kwargs, _, _ = setup
    edit = make_edit(setup)
    changed = replace(batch, data=batch.data.flip(-1))
    with pytest.raises(ValidationError, match="does not reproduce"):
        lens.run_with_interventions(changed, interventions=[edit], **kwargs)
    assert all(not m._forward_hooks for m in lens.model.modules())


def test_native_replay_failure_cleans_both_hook_scopes(setup):
    lens, batch, kwargs, _, _ = setup
    edit = make_edit(setup)
    lens.model.attn.calls = 0
    lens.model.attn.fail_at = 2
    with pytest.raises(RuntimeError, match="injected"):
        lens.run_with_interventions(batch, interventions=[edit], **kwargs)
    assert all(not m._forward_hooks for m in lens.model.modules())
    lens.model.attn.fail_at = None
    lens.assert_model_unchanged()
    lens.run_with_cache(batch, sites=(SITE,), **kwargs)


def test_existing_qkv_hook_not_removed_or_composed(setup):
    lens = setup[0]
    handle = lens.model.attn.qkv.register_forward_hook(lambda m, a, o: o)
    try:
        with pytest.raises(ValidationError, match="existing qkv"):
            make_edit(setup)
        assert len(lens.model.attn.qkv._forward_hooks) == 1
    finally:
        handle.remove()
