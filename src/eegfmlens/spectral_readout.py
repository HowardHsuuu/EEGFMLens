"""Leakage-explicit activation readouts for EEG amplitude spectra."""

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .errors import ValidationError
from .probes import Pooling, ProbeScore, RidgeProbe, activation_matrix, fit_ridge_probe, r2_score
from .provenance import tensor_digest
from .spectral import FrequencyBand
from .types import Activation, SignalBatch

SpectralReadoutScope = Literal["patch", "trial"]
SpectralTargetTransform = Literal["amplitude", "log1p_amplitude"]


@dataclass(frozen=True)
class SpectralTargets:
    """Channel-mean one-sided amplitude targets with explicit row coordinates."""

    frequencies: torch.Tensor
    values: torch.Tensor
    row_coordinates: tuple[tuple[str, int | None], ...]
    trial_ids: tuple[str, ...]
    channels: tuple[str, ...]
    scope: SpectralReadoutScope
    transform: SpectralTargetTransform
    sampling_rate: float
    samples: int
    patches: int
    patch_samples: int
    stride: int
    preprocessing_id: str
    detrended: bool
    source_sha256: str

    def __post_init__(self):
        if (
            not isinstance(self.frequencies, torch.Tensor)
            or self.frequencies.ndim != 1
            or not self.frequencies.is_floating_point()
            or self.frequencies.numel() < 1
            or not torch.isfinite(self.frequencies).all()
            or bool((self.frequencies < 0).any())
            or (
                self.frequencies.numel() > 1
                and not bool((self.frequencies[1:] > self.frequencies[:-1]).all())
            )
        ):
            raise ValidationError("Spectral target frequencies must be finite and increasing")
        if (
            not isinstance(self.values, torch.Tensor)
            or self.values.ndim != 2
            or not self.values.is_floating_point()
            or self.values.shape != (len(self.row_coordinates), self.frequencies.numel())
            or self.values.device != self.frequencies.device
            or self.values.dtype != self.frequencies.dtype
            or not torch.isfinite(self.values).all()
            or bool((self.values < 0).any())
        ):
            raise ValidationError("Spectral target values must match rows and frequencies")
        if self.scope not in {"patch", "trial"}:
            raise ValidationError("Spectral target scope must be patch or trial")
        if self.transform not in {"amplitude", "log1p_amplitude"}:
            raise ValidationError("Unknown spectral target transform")
        if (
            not isinstance(self.trial_ids, tuple)
            or not self.trial_ids
            or len(set(self.trial_ids)) != len(self.trial_ids)
            or any(not isinstance(value, str) or not value for value in self.trial_ids)
            or not isinstance(self.channels, tuple)
            or not self.channels
            or len(set(self.channels)) != len(self.channels)
            or any(not isinstance(value, str) or not value for value in self.channels)
        ):
            raise ValidationError("Spectral target identities are invalid")
        expected = (
            tuple((trial, patch) for trial in self.trial_ids for patch in range(self.patches))
            if self.scope == "patch"
            else tuple((trial, None) for trial in self.trial_ids)
        )
        if self.row_coordinates != expected:
            raise ValidationError("Spectral target row coordinates do not match their scope")
        if (
            not isinstance(self.sampling_rate, (int, float))
            or not math.isfinite(self.sampling_rate)
            or self.sampling_rate <= 0
            or any(
                type(value) is not int or value < 1
                for value in (self.samples, self.patches, self.patch_samples, self.stride)
            )
            or not isinstance(self.preprocessing_id, str)
            or not self.preprocessing_id
            or type(self.detrended) is not bool
            or not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
        ):
            raise ValidationError("Spectral target provenance is invalid")
        expected_samples = (
            self.patch_samples if self.scope == "patch" else self.patches * self.patch_samples
        )
        if (
            self.samples != expected_samples
            or (self.scope == "trial" and self.stride != self.patch_samples)
            or float(self.frequencies[-1]) > self.sampling_rate / 2 + 1e-9
        ):
            raise ValidationError("Spectral target FFT geometry is inconsistent")


def amplitude_spectral_targets(
    batch: SignalBatch,
    *,
    scope: SpectralReadoutScope = "patch",
    transform: SpectralTargetTransform = "log1p_amplitude",
    f_min: float = 0.5,
    f_max: float | None = None,
    detrend: bool = True,
) -> SpectralTargets:
    """Create channel-mean amplitude targets from model-ready EEG.

    Patch rows are ordered trial-major, then patch. Trial scope concatenates
    contiguous nonoverlapping patches before the FFT. Amplitudes use one-sided
    ``2 / samples`` scaling, except at DC and Nyquist.
    """

    if not isinstance(batch, SignalBatch):
        raise ValidationError("Spectral targets require a SignalBatch")
    batch.__post_init__()
    if scope not in {"patch", "trial"}:
        raise ValidationError("Spectral target scope must be patch or trial")
    if transform not in {"amplitude", "log1p_amplitude"}:
        raise ValidationError("Unknown spectral target transform")
    nyquist = batch.sampling_rate / 2
    high = nyquist if f_max is None else f_max
    if type(f_min) not in {int, float} or type(high) not in {int, float}:
        raise ValidationError("Spectral target frequency bounds must be numeric")
    bounds = torch.tensor([f_min, high], dtype=torch.float64)
    if (
        not bool(torch.isfinite(bounds).all())
        or f_min < 0
        or high <= f_min
        or high > nyquist + 1e-9
    ):
        raise ValidationError("Spectral target bounds require 0 <= f_min < f_max <= Nyquist")
    if type(detrend) is not bool:
        raise ValidationError("Spectral target detrend must be a boolean")

    b, c, p, s = batch.data.shape
    if scope == "patch":
        series = batch.data.permute(0, 2, 1, 3).reshape(b * p, c, s)
        samples = s
        rows: tuple[tuple[str, int | None], ...] = tuple(
            (trial, patch) for trial in batch.trial_ids for patch in range(p)
        )
    else:
        if batch.stride != s:
            raise ValidationError(
                "Trial spectral targets require contiguous nonoverlapping patches"
            )
        series = batch.data.reshape(b, c, p * s)
        samples = p * s
        rows = tuple((trial, None) for trial in batch.trial_ids)
    working = series.float() if series.dtype in {torch.float16, torch.bfloat16} else series
    if detrend:
        working = working - working.mean(dim=-1, keepdim=True)
    coefficients = torch.fft.rfft(working, dim=-1)
    amplitude = coefficients.abs() / samples
    if samples > 1:
        stop = -1 if samples % 2 == 0 else None
        amplitude[..., 1:stop] *= 2
    amplitude = amplitude.mean(dim=1)
    frequencies = torch.fft.rfftfreq(
        samples,
        d=1.0 / batch.sampling_rate,
        device=working.device,
        dtype=working.dtype,
    )
    upper = frequencies <= high if abs(high - nyquist) < 1e-9 else frequencies < high
    keep = (frequencies >= f_min) & upper
    if not bool(keep.any()):
        raise ValidationError("Spectral target bounds contain no FFT bins")
    values = amplitude[:, keep]
    if transform == "log1p_amplitude":
        values = torch.log1p(values)
    return SpectralTargets(
        frequencies[keep],
        values,
        rows,
        batch.trial_ids,
        batch.channels,
        scope,
        transform,
        batch.sampling_rate,
        samples,
        p,
        s,
        batch.stride,
        batch.preprocessing_id,
        detrend,
        tensor_digest(batch.data),
    )


def _matching_provenance(activation: Activation, targets: SpectralTargets) -> bool:
    return (
        activation.trial_ids == targets.trial_ids
        and activation.channels == targets.channels
        and activation.preprocessing_id == targets.preprocessing_id
        and activation.sampling_rate == targets.sampling_rate
        and activation.patch_samples == targets.patch_samples
        and activation.stride == targets.stride
    )


def activation_spectral_matrix(
    activation: Activation,
    targets: SpectralTargets,
    *,
    trial_pooling: Pooling = "mean",
) -> torch.Tensor:
    """Align one cached activation with trial- or patch-scope spectral rows.

    Patch scope averages the activation's sensor positions to match the target's
    channel-mean amplitude. ``tokens`` and ``patch_tokens`` must use the declared
    channel-major token order. Trial scope delegates to ``activation_matrix``.
    """

    if not isinstance(activation, Activation) or not isinstance(targets, SpectralTargets):
        raise ValidationError("Spectral activation alignment requires Activation and targets")
    targets.__post_init__()
    if not _matching_provenance(activation, targets):
        raise ValidationError("Activation and spectral target provenance differ")
    if targets.scope == "trial":
        matrix = activation_matrix(activation, pooling=trial_pooling)
    else:
        if trial_pooling != "mean":
            raise ValidationError("trial_pooling applies only to trial-scope spectral targets")
        tensor = activation.tensor
        b, c, p = len(targets.trial_ids), len(targets.channels), targets.patches
        if activation.layout in {"bcpd", "spatial", "temporal"}:
            if tensor.ndim != 4 or tensor.shape[:3] != (b, c, p):
                raise ValidationError("Activation does not match patch spectral geometry")
            matrix = tensor.permute(0, 2, 1, 3).mean(dim=2).reshape(b * p, -1)
        elif activation.layout in {"tokens", "patch_tokens"}:
            offset = int(activation.layout == "tokens")
            if tensor.ndim != 3 or tensor.shape[:2] != (b, offset + c * p):
                raise ValidationError("Activation tokens do not match patch spectral geometry")
            matrix = tensor[:, offset:].reshape(b, c, p, -1).mean(dim=1).reshape(b * p, -1)
        else:
            raise ValidationError("Patch spectral alignment requires a physical token layout")
    if matrix.shape[0] != targets.values.shape[0] or not torch.isfinite(matrix).all():
        raise ValidationError("Activation rows do not match spectral target rows")
    return matrix


@dataclass(frozen=True)
class SpectralReadout:
    """A held-out linear map from activation coordinates to amplitude targets."""

    probe: RidgeProbe
    frequencies: torch.Tensor
    scope: SpectralReadoutScope
    transform: SpectralTargetTransform
    sampling_rate: float

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.probe.predict(features)

    def direction_signature(self, directions: torch.Tensor) -> torch.Tensor:
        """Predicted spectral change per activation-unit step along directions."""

        if (
            not isinstance(directions, torch.Tensor)
            or directions.ndim not in {1, 2}
            or not directions.is_floating_point()
            or directions.shape[-1] != self.probe.weight.shape[0]
            or directions.device != self.probe.weight.device
            or directions.dtype != self.probe.weight.dtype
            or not torch.isfinite(directions).all()
        ):
            raise ValidationError("Readout directions must match its activation feature axis")
        return directions @ self.probe.weight

    def band_signature(self, directions: torch.Tensor, band: FrequencyBand) -> torch.Tensor:
        """Mean signed direction signature inside a declared frequency band."""

        if not isinstance(band, FrequencyBand):
            raise ValidationError("band_signature requires a FrequencyBand")
        band.__post_init__()
        nyquist = self.sampling_rate / 2
        if band.high_hz > nyquist + 1e-9:
            raise ValidationError("Frequency band exceeds the spectral readout Nyquist")
        upper = (
            self.frequencies <= band.high_hz
            if abs(band.high_hz - nyquist) < 1e-9
            else self.frequencies < band.high_hz
        )
        mask = (self.frequencies >= band.low_hz) & upper
        if not bool(mask.any()):
            raise ValidationError("Frequency band contains no spectral readout bins")
        return self.direction_signature(directions)[..., mask].mean(dim=-1)


@dataclass(frozen=True)
class SpectralReadoutResult:
    readout: SpectralReadout
    train: ProbeScore
    test: ProbeScore
    train_rows: int
    test_rows: int
    metadata: dict[str, Any]


def _split(mask: torch.Tensor, rows: int, name: str) -> torch.Tensor:
    if (
        not isinstance(mask, torch.Tensor)
        or mask.ndim != 1
        or mask.shape[0] != rows
        or mask.dtype != torch.bool
    ):
        raise ValidationError(f"{name} must be a boolean mask with one value per spectral row")
    indices = mask.nonzero(as_tuple=False).flatten()
    if indices.numel() < 2:
        raise ValidationError(f"{name} must select at least two spectral rows")
    return indices


def fit_spectral_readout(
    features: torch.Tensor,
    targets: SpectralTargets,
    *,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    alpha: float = 1.0,
) -> SpectralReadoutResult:
    """Fit and score an amplitude readout on a caller-declared disjoint split."""

    if not isinstance(targets, SpectralTargets):
        raise ValidationError("Spectral readout requires SpectralTargets")
    targets.__post_init__()
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 2
        or not features.is_floating_point()
        or features.shape[0] != targets.values.shape[0]
        or features.shape[1] < 1
        or features.device != targets.values.device
        or features.dtype != targets.values.dtype
        or not torch.isfinite(features).all()
    ):
        raise ValidationError("Spectral readout features must match finite target rows")
    train = _split(train_mask, features.shape[0], "train_mask")
    test = _split(test_mask, features.shape[0], "test_mask")
    if train_mask.device != test_mask.device:
        raise ValidationError("Spectral readout masks must share a device")
    if bool((train_mask & test_mask).any()):
        raise ValidationError("Spectral readout train and test masks must be disjoint")
    train = train.to(features.device)
    test = test.to(features.device)
    probe = fit_ridge_probe(features[train], targets.values[train], alpha=alpha)
    readout = SpectralReadout(
        probe,
        targets.frequencies.detach().clone(),
        targets.scope,
        targets.transform,
        targets.sampling_rate,
    )
    return SpectralReadoutResult(
        readout,
        r2_score(targets.values[train], readout.predict(features[train])),
        r2_score(targets.values[test], readout.predict(features[test])),
        train.numel(),
        test.numel(),
        {
            "feature_sha256": tensor_digest(features),
            "target_sha256": tensor_digest(targets.values),
            "frequency_sha256": tensor_digest(targets.frequencies),
            "source_sha256": targets.source_sha256,
            "train_mask_sha256": tensor_digest(train_mask),
            "test_mask_sha256": tensor_digest(test_mask),
            "weight_sha256": tensor_digest(probe.weight),
            "intercept_sha256": tensor_digest(probe.intercept),
            "scope": targets.scope,
            "transform": targets.transform,
            "alpha": probe.alpha,
        },
    )
