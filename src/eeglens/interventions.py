"""Declarative same-model patching and controlled ablation."""

from dataclasses import dataclass

import torch

from .errors import ValidationError
from .types import Activation, SignalBatch


@dataclass(frozen=True)
class Selection:
    """Select whole patches by sensor and patch index; no sub-patch precision."""

    sensors: tuple[str, ...] | None = None
    patches: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.sensors is not None and (
            not isinstance(self.sensors, tuple) or any(not isinstance(s, str) for s in self.sensors)
        ):
            raise ValidationError("Sensors must be a tuple of names")
        if self.patches is not None and (
            not isinstance(self.patches, tuple) or any(type(p) is not int for p in self.patches)
        ):
            raise ValidationError("Patches must be a tuple of integer indices")

    def mask(self, tensor: torch.Tensor, batch: SignalBatch, layout: str):
        b, c, p, _ = batch.data.shape
        if layout not in {"bcpd", "spatial", "temporal", "tokens", "patch_tokens"}:
            if self.sensors is not None or self.patches is not None:
                raise ValidationError("This site has no sensor/patch selector")
            return torch.ones_like(tensor, dtype=torch.bool)
        sensors = self.sensors if self.sensors is not None else batch.channels
        patches = self.patches if self.patches is not None else tuple(range(p))
        if not sensors or not patches:
            raise ValidationError("Empty selection")
        if len(set(sensors)) != len(sensors) or len(set(patches)) != len(patches):
            raise ValidationError("Duplicate selectors")
        if any(s not in batch.channels for s in sensors) or any(k < 0 or k >= p for k in patches):
            raise ValidationError("Unknown sensor or out-of-range patch")
        grid = torch.zeros((c, p), device=tensor.device, dtype=torch.bool)
        for s in sensors:
            grid[batch.channels.index(s), list(patches)] = True
        if layout in {"tokens", "patch_tokens"}:
            if self.sensors is None and self.patches is None:
                return torch.ones_like(tensor, dtype=torch.bool)
            prefix = torch.zeros(int(layout == "tokens"), device=tensor.device, dtype=torch.bool)
            grid = torch.cat([prefix, grid.flatten()])
            return grid[None, :, None].expand_as(tensor)
        return grid[None, :, :, None].expand_as(tensor)


@dataclass(frozen=True)
class AxisSelection:
    """Select explicit indices of one exposed tensor axis, without physical labels.

    Axis zero is reserved for trial identity and cannot be selected. Axes and
    indices are nonnegative and checked against the activation at execution.
    The caller must verify the model-specific meaning of the selected indices.
    """

    axis: int
    indices: tuple[int, ...]

    def __post_init__(self):
        if type(self.axis) is not int or self.axis < 1:
            raise ValidationError("AxisSelection requires a positive non-batch axis")
        if (
            not isinstance(self.indices, tuple)
            or not self.indices
            or any(type(i) is not int or i < 0 for i in self.indices)
            or len(set(self.indices)) != len(self.indices)
        ):
            raise ValidationError("AxisSelection requires unique nonnegative integer indices")

    def mask(self, tensor: torch.Tensor, batch: SignalBatch, layout: str):
        if self.axis >= tensor.ndim or max(self.indices) >= tensor.shape[self.axis]:
            raise ValidationError("AxisSelection axis or index is out of range")
        axis_mask = torch.zeros(tensor.shape[self.axis], device=tensor.device, dtype=torch.bool)
        axis_mask[list(self.indices)] = True
        shape = [1] * tensor.ndim
        shape[self.axis] = tensor.shape[self.axis]
        return axis_mask.reshape(shape).expand_as(tensor)


def _selection_metadata(selection):
    if isinstance(selection, AxisSelection):
        return {"axis": selection.axis, "indices": selection.indices}
    return {"sensors": selection.sensors, "patches": selection.patches}


@dataclass(frozen=True)
class Replacement:
    """Replace selected recipient values, matching donor rows by unique trial IDs."""

    site: str
    donor: Activation
    selection: Selection | AxisSelection = Selection()

    def apply(self, current, batch, layout, model_id):
        d = self.donor
        if d.site != self.site or d.model_id != model_id or d.layout != layout:
            raise ValidationError("Donor site, model identity or layout mismatch")
        if (d.channels, d.preprocessing_id, d.sampling_rate, d.stride, d.patch_samples, d.unit) != (
            batch.channels,
            batch.preprocessing_id,
            batch.sampling_rate,
            batch.stride,
            batch.data.shape[-1],
            batch.unit,
        ):
            raise ValidationError("Donor preprocessing or coordinate metadata mismatch")
        if len(set(d.trial_ids)) != len(d.trial_ids) or d.tensor.shape[0] != len(d.trial_ids):
            raise ValidationError("Invalid donor trial metadata")
        if any(t not in d.trial_ids for t in batch.trial_ids):
            raise ValidationError("Recipient trial has no matching donor")
        if d.tensor.device != current.device or d.tensor.dtype != current.dtype:
            raise ValidationError(
                "Donor device/dtype mismatch; convert deliberately before caching"
            )
        values = d.tensor[[d.trial_ids.index(t) for t in batch.trial_ids]]
        if values.shape != current.shape or not torch.isfinite(values).all():
            raise ValidationError("Donor shape mismatch or non-finite values")
        return torch.where(self.selection.mask(current, batch, layout), values, current)


@dataclass(frozen=True)
class Ablation:
    """Zero a selected activation region; a perturbation, not a causal proof."""

    site: str
    selection: Selection | AxisSelection = Selection()

    def apply(self, current, batch, layout, model_id):
        return current.masked_fill(self.selection.mask(current, batch, layout), 0)


@dataclass(frozen=True)
class SubspaceAblation:
    """Remove a centered orthonormal feature subspace at an activation site.

    Basis columns are directions in the exposed tensor's final feature axis.
    Fit basis/center only on training data; this operation does not fit them.
    """

    site: str
    basis: torch.Tensor
    center: torch.Tensor
    selection: Selection | AxisSelection = Selection()

    def apply(self, current, batch, layout, model_id):
        q, center = self.basis, self.center
        if not isinstance(q, torch.Tensor) or not isinstance(center, torch.Tensor):
            raise ValidationError("Basis and center must be tensors")
        if q.ndim != 2 or q.shape[0] != current.shape[-1] or not 0 < q.shape[1] <= q.shape[0]:
            raise ValidationError(
                "Basis must have shape [features, rank], with 0 < rank <= features"
            )
        if center.shape != (current.shape[-1],):
            raise ValidationError("Center must have shape [features]")
        if any(t.device != current.device or t.dtype != current.dtype for t in (q, center)):
            raise ValidationError("Basis/center device or dtype mismatch")
        if not torch.isfinite(q).all() or not torch.isfinite(center).all():
            raise ValidationError("Non-finite basis or center")
        if not torch.allclose(
            q.T @ q, torch.eye(q.shape[1], device=q.device, dtype=q.dtype), atol=1e-5, rtol=1e-5
        ):
            raise ValidationError("Basis columns must be orthonormal")
        erased = current - ((current - center) @ q) @ q.T
        return torch.where(self.selection.mask(current, batch, layout), erased, current)
