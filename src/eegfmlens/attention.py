"""Projected Q/K/V replacement in declared native ST decoder attention.

The native attention forward remains authoritative. A temporary projection hook
exchanges components during replay and is removed even when native execution
raises. Physical selection limits written attention outputs, not key/value input
tokens. This distinction matters for queries that attend to remote values.
"""

import re
from dataclasses import replace

import torch

from .adapters.steegformer import STEEGFormerMAEAdapter
from .errors import ValidationError
from .interventions import Selection
from .provenance import tensor_digest
from .types import Activation

_COMPONENTS = {"q": (0,), "k": (1,), "qk": (0, 1), "v": (2,), "all": (0, 1, 2)}


def _geometry(module, live, donor, component, head):
    if component not in _COMPONENTS:
        raise ValidationError("Unknown projected attention component")
    if (
        not isinstance(live, torch.Tensor)
        or not isinstance(donor, torch.Tensor)
        or live.ndim != 3
        or live.shape != donor.shape
        or live.dtype != donor.dtype
        or live.device != donor.device
        or not live.is_floating_point()
        or not torch.isfinite(live).all()
        or not torch.isfinite(donor).all()
    ):
        raise ValidationError("Projected attention input geometry/dtype/device differs")
    if module.training or module.attn_drop.p != 0 or module.proj_drop.p != 0:
        raise ValidationError("Projected replay requires deterministic eval attention")
    heads = module.num_heads
    if (
        type(heads) is not int
        or heads < 1
        or live.shape[-1] % heads
        or (head is not None and (type(head) is not int or not 0 <= head < heads))
    ):
        raise ValidationError("Invalid attention head geometry")
    if module.qkv._forward_hooks or module.qkv._forward_pre_hooks:
        raise ValidationError("Projected replay cannot compose with existing qkv hooks")
    return live.shape[0], live.shape[1], live.shape[2], heads


def _project(module, value, shape):
    b, n, width, heads = shape
    projected = module.qkv(value)
    if projected.shape != (b, n, 3 * width) or not torch.isfinite(projected).all():
        raise ValidationError("Native packed Q/K/V projection differs")
    return projected.reshape(b, n, 3, heads, width // heads)


def _replay(module, live, donor, component, head):
    shape = _geometry(module, live, donor, component, head)
    b, n, width, _ = shape
    before, after = _project(module, live, shape), _project(module, donor, shape)
    mixed = before.clone()
    terms = []
    for index in _COMPONENTS[component]:
        if head is None:
            mixed[:, :, index] = after[:, :, index]
            terms.append(after[:, :, index] - before[:, :, index])
        else:
            mixed[:, :, index, head] = after[:, :, index, head]
            terms.append(after[:, :, index, head] - before[:, :, index, head])
    delta = torch.cat([term.reshape(b, n, -1) for term in terms], dim=-1)
    calls = 0

    def substitute(_module, _args, output):
        nonlocal calls
        calls += 1
        if calls != 1 or output.shape != (b, n, 3 * width):
            raise ValidationError("Native attention must invoke packed qkv exactly once")
        return mixed.reshape(b, n, 3 * width)

    handle = module.qkv.register_forward_hook(substitute)
    try:
        # Bypass only the outer output hook to avoid recursively applying this
        # intervention; every native attention operation still runs unchanged.
        result = module.forward(live)
        if calls != 1 or result.shape != live.shape or not torch.isfinite(result).all():
            raise ValidationError("Native attention replay/output geometry differs")
    finally:
        handle.remove()
    return result, delta


class ProjectedAttentionReplacement:
    """Replace projected components using matched ST attention-input caches.

    ``site`` must be a declared ``decoder.N.native_attention`` output. ``live``
    and ``donor`` are Activation snapshots from its ``attention_input`` site,
    with identical trial order and execution configuration. ``selection``
    chooses physical output queries; selected Q/K/V components are borrowed at
    every token (optionally for one head). The native live replay must agree
    with the current site before an edit is allowed. Stale live caches fail.

    Use with ``EEGLens.run_with_interventions``. Do not concurrently execute the
    underlying native model or cache its internal qkv/projection submodules
    during replay. This API reports displacement, never a causal success gate.
    """

    def __init__(self, lens, site, live, donor, *, component, selection, head=None):
        if not isinstance(lens.adapter, STEEGFormerMAEAdapter):
            raise ValidationError("Projected attention requires the native ST MAE adapter")
        if not isinstance(site, str) or not re.fullmatch(r"decoder\.\d+\.native_attention", site):
            raise ValidationError("Expected a native decoder attention output site")
        if not isinstance(selection, Selection):
            raise ValidationError("Projected attention requires a physical Selection")
        input_site = site.replace(".native_attention", ".attention_input")
        declared, input_declared = lens.adapter.require(site), lens.adapter.require(input_site)
        if (
            declared.layout != "tokens"
            or input_declared.layout != "tokens"
            or not declared.writable
            or not input_declared.writable
            or declared.expected_calls != 1
            or input_declared.expected_calls != 1
            or declared.call_index != 0
            or input_declared.call_index != 0
        ):
            raise ValidationError("Projected attention needs single-call physical token sites")
        if not isinstance(live, Activation) or not isinstance(donor, Activation):
            raise ValidationError("Projected replay requires package Activation snapshots")
        fields = (
            "site",
            "model_id",
            "trial_ids",
            "channels",
            "preprocessing_id",
            "sampling_rate",
            "stride",
            "patch_samples",
            "unit",
            "layout",
            "native_shape",
            "execution_id",
        )
        if (
            live.site != input_site
            or live.model_id != lens.model_id
            or live.layout != "tokens"
            or not live.execution_id
            or any(getattr(live, key) != getattr(donor, key) for key in fields)
        ):
            raise ValidationError("Projected cache identities/coordinates/execution differ")
        if (
            live.tensor.shape != donor.tensor.shape
            or live.tensor.dtype != donor.tensor.dtype
            or live.tensor.device != donor.tensor.device
        ):
            raise ValidationError("Projected cache tensor geometry differs")
        lens.assert_model_unchanged()
        self.site, self.selection = site, selection
        self.component, self.head = component, head
        self.live = replace(live, tensor=live.tensor.detach().clone())
        self.donor = replace(donor, tensor=donor.tensor.detach().clone())
        self._adapter = lens.adapter
        self._module = lens.model.get_submodule(declared.module_path)
        self._model_id = lens.model_id
        self.diagnostics = {}
        self.multiplier_warning = None
        _geometry(self._module, self._native(self.live), self._native(self.donor), component, head)

    def _native(self, activation):
        return self._adapter.restore(
            activation.tensor, self._adapter.require(activation.site), activation.native_shape
        ).clone()

    def apply(self, current, batch, layout, model_id):
        live = self.live
        if (
            model_id != self._model_id
            or layout != "tokens"
            or live.trial_ids != batch.trial_ids
            or live.channels != batch.channels
            or live.preprocessing_id != batch.preprocessing_id
            or live.sampling_rate != batch.sampling_rate
            or live.stride != batch.stride
            or live.patch_samples != batch.data.shape[-1]
            or live.unit != batch.unit
            or current.dtype != live.tensor.dtype
            or current.device != live.tensor.device
        ):
            raise ValidationError("Projected recipient/cache provenance differs")
        declared = self._adapter.require(self.site)
        native_live, native_donor = self._native(live), self._native(self.donor)
        own_native, _ = _replay(self._module, native_live, native_live, "all", None)
        own = self._adapter.expose(own_native, declared, batch)
        if own.shape != current.shape or not torch.allclose(own, current, atol=5e-5, rtol=1e-5):
            raise ValidationError("Live attention cache does not reproduce current native output")
        changed_native, delta = _replay(
            self._module, native_live, native_donor, self.component, self.head
        )
        changed = self._adapter.expose(changed_native, declared, batch)
        selected = self.selection.mask(current, batch, layout)
        if not selected.any() or selected.all():
            raise ValidationError("Projected output selection must be nonempty and nonglobal")
        physical_delta = self._adapter.expose(delta, declared, batch)
        query_support = self.selection.mask(physical_delta, batch, layout)
        count = current.shape[0]
        counts = query_support.reshape(count, -1).sum(-1)
        if not torch.all(counts == counts[0]) or counts[0] == 0:
            raise ValidationError("Projected physical support differs across trials")
        rms = physical_delta[query_support].reshape(count, -1).double().square().mean(-1).sqrt()
        self.diagnostics = {
            "mode": self.component,
            "head": self.head,
            "native_replay_max_error": float((own - current).abs().max()),
            "projected_edit_rms": float(delta.float().square().mean().sqrt()),
            "selected_projected_edit_rms": rms.tolist(),
            "selected_elements": int(selected.sum()),
        }
        return torch.where(selected, changed, current)

    def provenance(self):
        return {
            "component": self.component,
            "head": self.head,
            "live_input_sha256": tensor_digest(self.live.tensor),
            "donor_input_sha256": tensor_digest(self.donor.tensor),
            "input_site": self.live.site,
            "output_selection_semantics": "physical queries; projected donor components at all tokens",
        }
