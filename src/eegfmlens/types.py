"""Explicit identities and tensor metadata for native model execution."""

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .errors import ValidationError


@dataclass(frozen=True)
class SignalTransform:
    """One deliberate signal-space edit recorded on a :class:`SignalBatch`.

    Parameters are immutable JSON scalars so a transformed batch can be traced
    without retaining the original EEG.  These records describe experimental
    edits; they do not replace ``preprocessing_id``.
    """

    name: str
    parameters: tuple[tuple[str, str | int | float | bool], ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("Signal transform name must be nonempty")
        if not isinstance(self.parameters, tuple):
            raise ValidationError("Signal transform parameters must be a tuple")
        keys = []
        for item in self.parameters:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or type(item[1]) not in {str, int, float, bool}
            ):
                raise ValidationError("Signal transform parameters must be key/scalar pairs")
            if isinstance(item[1], float) and not math.isfinite(item[1]):
                raise ValidationError("Signal transform parameters must be finite")
            keys.append(item[0])
        if len(set(keys)) != len(keys):
            raise ValidationError("Signal transform parameter names must be unique")


@dataclass(frozen=True)
class SignalBatch:
    """Model-ready patches, with provenance; this class does not preprocess EEG.

    ``data`` is [batch, sensor, patch, sample]. Trial IDs must be unique within
    the batch. ``preprocessing_id`` identifies the complete processing recipe,
    excluding experimental corruption so clean/corrupt pairs remain compatible.
    """

    data: torch.Tensor
    trial_ids: tuple[str, ...]
    channels: tuple[str, ...]
    sampling_rate: float
    preprocessing_id: str
    unit: str = "model_scaled"
    patch_stride_samples: int | None = None
    transforms: tuple[SignalTransform, ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.data, torch.Tensor)
            or self.data.ndim != 4
            or not self.data.is_floating_point()
        ):
            raise ValidationError("Expected floating [batch, sensor, patch, sample] data")
        for name in ("trial_ids", "channels"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(v, str) or not v for v in values
            ):
                raise ValidationError(f"{name} must be a tuple of nonempty strings")
        b, c, p, s = self.data.shape
        if min(b, c, p, s) < 1:
            raise ValidationError("Empty input axes are not supported")
        if len(self.trial_ids) != b or len(set(self.trial_ids)) != b:
            raise ValidationError("trial_ids must be unique and match batch size")
        if len(self.channels) != c or len(set(self.channels)) != c:
            raise ValidationError("channels must be unique and match sensor axis")
        if (
            not isinstance(self.preprocessing_id, str)
            or not self.preprocessing_id
            or not isinstance(self.unit, str)
            or not self.unit
            or not isinstance(self.sampling_rate, (int, float))
            or not math.isfinite(self.sampling_rate)
            or self.sampling_rate <= 0
        ):
            raise ValidationError(
                "A preprocessing ID, unit and positive sampling rate are required"
            )
        if self.patch_stride_samples is not None and (
            type(self.patch_stride_samples) is not int or self.patch_stride_samples <= 0
        ):
            raise ValidationError("Patch stride must be positive")
        if not isinstance(self.transforms, tuple) or any(
            not isinstance(transform, SignalTransform) for transform in self.transforms
        ):
            raise ValidationError("transforms must be a tuple of SignalTransform records")
        for transform in self.transforms:
            transform.__post_init__()
        if not torch.isfinite(self.data).all():
            raise ValidationError("Non-finite input")

    @property
    def stride(self) -> int:
        return self.patch_stride_samples or self.data.shape[-1]


@dataclass(frozen=True)
class ActivationSite:
    """A native output site, exposed in an explicitly declared layout.

    Layout bcpd is batch/sensor/patch/feature; spatial and temporal restore
    CBraMod's folded batch axes to bcpd. Tokens retain LaBraM's CLS token.
    """

    name: str
    module_path: str
    layout: str = "bcpd"
    tensor_index: int | str | None = None
    writable: bool = True
    call_index: int = 0
    expected_calls: int = 1


@dataclass(frozen=True)
class Activation:
    """Detached owned snapshot, with native and exposed layout metadata."""

    tensor: torch.Tensor
    site: str
    model_id: str
    trial_ids: tuple[str, ...]
    channels: tuple[str, ...]
    preprocessing_id: str
    sampling_rate: float
    stride: int
    patch_samples: int
    unit: str
    layout: str
    native_shape: tuple[int, ...]
    execution_id: str = ""


@dataclass
class RunResult:
    """Native output and selected snapshots, plus actual execution evidence."""

    output: Any
    cache: dict[str, Activation]
    calls: dict[str, int]
    model_id: str
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
