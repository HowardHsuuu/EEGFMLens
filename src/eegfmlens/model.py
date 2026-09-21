"""Scoped execution over a native model; never reconstructs its forward."""

import hashlib
import json
import threading
import uuid
import weakref
from dataclasses import dataclass
from typing import Any

import torch

from .errors import HookExecutionError, ValidationError
from .interventions import AxisSelection, Selection, _selection_metadata
from .provenance import tensor_digest
from .types import Activation, RunResult, SignalBatch


@dataclass(frozen=True)
class SiteCapability:
    """Operations that one declared activation site supports."""

    name: str
    module_path: str
    layout: str
    writable: bool
    selectors: tuple[str, ...]
    call_index: int
    expected_calls: int


class EEGLens:
    """Wrap an existing eval-mode model with declared adapter sites.

    The wrapper never changes training mode or parameters. This first runtime
    is inference-only. One run per wrapper may be active at a time. Do not run
    the underlying model concurrently or wrap it twice.
    """

    def __init__(self, model, adapter, *, model_id: str | None = None):
        self.model = model
        self.adapter = adapter
        self.model_id = model_id or f"instance:{uuid.uuid4().hex}"
        self.manifest: dict[str, Any] = {
            "identity": self.model_id,
            "checkpoint_sha256": None,
        }
        self._lock = threading.Lock()
        for site in adapter.sites.values():
            model.get_submodule(site.module_path)
        owner = getattr(model, "_eegfmlens_owner", None)
        if owner is not None and owner() is not None:
            raise ValidationError("Native model already has a live EEGLens wrapper")
        model._eegfmlens_owner = weakref.ref(self)
        self._initial_state = self._state()

    def _state(self):
        tensors = [*self.model.named_parameters(), *self.model.named_buffers()]
        return (
            tuple(
                (name, id(t), t.data_ptr(), t._version, str(t.dtype), str(t.device))
                for name, t in tensors
            ),
            tuple((name, id(m)) for name, m in self.model.named_modules()),
        )

    def sites(self):
        return tuple(self.adapter.sites.values())

    def capabilities(self) -> tuple[SiteCapability, ...]:
        """Describe usable operations without running the native model."""

        physical = {"bcpd", "spatial", "temporal", "tokens", "patch_tokens"}
        return tuple(
            SiteCapability(
                site.name,
                site.module_path,
                site.layout,
                site.writable,
                ("whole", "axis", "sensor", "patch")
                if site.layout in physical
                else ("whole", "axis"),
                site.call_index,
                site.expected_calls,
            )
            for site in self.sites()
        )

    def run_with_cache(self, batch: SignalBatch, *, sites=None, **kwargs):
        return self._run(batch, sites=sites, interventions=(), kwargs=kwargs)

    def run_with_interventions(self, batch: SignalBatch, *, interventions, sites=(), **kwargs):
        return self._run(batch, sites=sites, interventions=interventions, kwargs=kwargs)

    def _run(self, batch, *, sites, interventions, kwargs):
        if not self._lock.acquire(blocking=False):
            raise ValidationError("Concurrent or reentrant execution is unsupported")
        handles = []
        try:
            if self._state() != self._initial_state:
                raise ValidationError(
                    "Model state changed after wrapping; create a fresh model and wrapper"
                )
            if any(m.training for m in self.model.modules()):
                raise ValidationError("Set the native model to eval() before execution")
            if not isinstance(batch, SignalBatch):
                raise ValidationError("Expected a SignalBatch")
            batch.__post_init__()  # Tensors may have been mutated since construction.
            try:
                execution = json.dumps(kwargs, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "Execution kwargs must be JSON-serializable scalars/containers"
                ) from exc
            execution_id = hashlib.sha256(execution.encode()).hexdigest()
            input_digest = tensor_digest(batch.data)
            self.adapter.validate(batch)
            if isinstance(sites, (str, bytes)):
                raise ValidationError(
                    "Cache sites must be a collection of site names, not a string"
                )
            selected = tuple(self.adapter.sites if sites is None else sites)
            if any(not isinstance(name, str) or not name for name in selected):
                raise ValidationError("Cache sites must contain nonempty names")
            if len(set(selected)) != len(selected):
                raise ValidationError("Duplicate cache sites")
            patches = {}
            for intervention in interventions:
                if (
                    not isinstance(getattr(intervention, "site", None), str)
                    or not isinstance(
                        getattr(intervention, "selection", None), (Selection, AxisSelection)
                    )
                    or not callable(getattr(intervention, "apply", None))
                ):
                    raise ValidationError(
                        "Interventions require a site, Selection or AxisSelection and callable apply"
                    )
                site = self.adapter.require(intervention.site)
                if not site.writable or site.name in patches:
                    raise ValidationError("Site not writable or multiple interventions at one site")
                patches[site.name] = intervention
            targets = [
                (
                    self.adapter.require(n).module_path,
                    self.adapter.require(n).tensor_index,
                    self.adapter.require(n).call_index,
                )
                for n in patches
            ]
            if len(set(targets)) != len(targets):
                raise ValidationError("Multiple interventions target the same native invocation")
            names = list(dict.fromkeys([*selected, *patches]))
            calls = dict.fromkeys(names, 0)
            cache = {}

            def make_hook(site):
                def hook(module, args, output):
                    calls[site.name] += 1
                    if calls[site.name] > site.expected_calls:
                        raise HookExecutionError(
                            f"Site {site.name} executed more than "
                            + ("once" if site.expected_calls == 1 else "expected")
                        )
                    if calls[site.name] - 1 != site.call_index:
                        return None
                    tensor = output if site.tensor_index is None else output[site.tensor_index]
                    if not isinstance(tensor, torch.Tensor):
                        raise ValidationError(f"Non-tensor output at {site.name}")
                    semantic = self.adapter.expose(tensor, site, batch)
                    if site.name in patches:
                        intervention = patches[site.name]
                        donor = getattr(intervention, "donor", None)
                        if donor is not None and donor.execution_id != execution_id:
                            raise ValidationError("Donor execution configuration mismatch")
                        expected_shape, expected_dtype, expected_device = (
                            semantic.shape,
                            semantic.dtype,
                            semantic.device,
                        )
                        semantic = intervention.apply(semantic, batch, site.layout, self.model_id)
                        if (
                            not isinstance(semantic, torch.Tensor)
                            or semantic.shape != expected_shape
                            or semantic.dtype != expected_dtype
                            or semantic.device != expected_device
                        ):
                            raise ValidationError(
                                f"Intervention at {site.name} must preserve tensor shape, dtype and device"
                            )
                        if not torch.isfinite(semantic).all():
                            raise ValidationError(
                                f"Intervention at {site.name} produced non-finite values"
                            )
                    if site.name in selected:
                        cache[site.name] = Activation(
                            semantic.detach().clone(),
                            site.name,
                            self.model_id,
                            batch.trial_ids,
                            batch.channels,
                            batch.preprocessing_id,
                            batch.sampling_rate,
                            batch.stride,
                            batch.data.shape[-1],
                            batch.unit,
                            site.layout,
                            tuple(tensor.shape),
                            execution_id,
                        )
                    if site.name not in patches:
                        return None  # observer preserves the exact native output object
                    replacement = self.adapter.restore(semantic, site, tuple(tensor.shape))
                    if site.tensor_index is None:
                        return replacement
                    if isinstance(output, dict):
                        mapping = output.copy()
                        mapping[site.tensor_index] = replacement
                        return mapping
                    if isinstance(output, tuple) and hasattr(output, "_fields"):
                        sequence = list(output)
                        sequence[site.tensor_index] = replacement
                        return type(output)(*sequence)
                    sequence = list(output)
                    sequence[site.tensor_index] = replacement
                    return tuple(sequence) if isinstance(output, tuple) else sequence

                return hook

            for name in names:
                site = self.adapter.require(name)
                handles.append(
                    self.model.get_submodule(site.module_path).register_forward_hook(
                        make_hook(site)
                    )
                )
            with torch.no_grad():
                output = self.adapter.forward(self.model, batch, **kwargs)
            missing = [
                name
                for name, count in calls.items()
                if count != self.adapter.require(name).expected_calls
            ]
            if missing:
                raise HookExecutionError(f"Declared sites did not execute: {missing}")
            if self._state() != self._initial_state:
                raise ValidationError("Model state mutated during execution")
            records = []
            for intervention in patches.values():
                record = {
                    "type": type(intervention).__name__,
                    "site": intervention.site,
                    "call_index": self.adapter.require(intervention.site).call_index,
                    "expected_calls": self.adapter.require(intervention.site).expected_calls,
                    **_selection_metadata(intervention.selection),
                }
                for name in ("basis", "center"):
                    if hasattr(intervention, name):
                        record[name + "_sha256"] = tensor_digest(getattr(intervention, name))
                if hasattr(intervention, "donor"):
                    donor = intervention.donor
                    record.update(
                        donor_sha256=tensor_digest(donor.tensor),
                        donor_trial_ids=donor.trial_ids,
                        donor_model_id=donor.model_id,
                    )
                if hasattr(intervention, "diagnostics"):
                    record["diagnostics"] = json.loads(
                        json.dumps(intervention.diagnostics, allow_nan=False)
                    )
                    reference = getattr(intervention, "reference_selection", None)
                    record["reference_selection"] = (
                        None if reference is None else _selection_metadata(reference)
                    )
                    record["multiplier_warning"] = getattr(intervention, "multiplier_warning", None)
                records.append(record)
            return RunResult(
                output,
                cache,
                calls,
                self.model_id,
                uuid.uuid4().hex,
                {
                    "trial_ids": batch.trial_ids,
                    "preprocessing_id": batch.preprocessing_id,
                    "channels": batch.channels,
                    "sampling_rate": batch.sampling_rate,
                    "patch_stride_samples": batch.stride,
                    "unit": batch.unit,
                    "input_shape": tuple(batch.data.shape),
                    "torch_version": str(torch.__version__),
                    "model": dict(self.manifest),
                    "adapter": self.adapter.metadata(),
                    "interventions": records,
                    "input_sha256": input_digest,
                    "execution_kwargs": json.loads(execution),
                    "mode": "inference",
                    "cache_stage": "post_intervention",
                },
            )
        finally:
            for handle in handles:
                handle.remove()
            self._lock.release()
