"""Frequency-domain measurements and controlled edits for model-ready EEG.

The functions in this module preserve the :class:`SignalBatch` coordinate
contract.  They are intended to be composed with activation caching and
patching, so a signal-space perturbation can be traced through a native model.
"""

from dataclasses import dataclass, replace
from typing import Literal

import torch

from .errors import ValidationError
from .provenance import tensor_digest
from .types import SignalBatch, SignalTransform

SpectralScope = Literal["patch", "trial"]
SpectralComponent = Literal["complex", "amplitude", "phase"]


@dataclass(frozen=True)
class FrequencyBand:
    """Half-open frequency interval ``[low_hz, high_hz)``.

    A band whose upper edge equals Nyquist includes the Nyquist bin.
    """

    low_hz: float
    high_hz: float
    name: str | None = None

    def __post_init__(self):
        values = torch.tensor([self.low_hz, self.high_hz], dtype=torch.float64)
        if not bool(torch.isfinite(values).all()) or self.low_hz < 0 or self.high_hz <= self.low_hz:
            raise ValidationError("Frequency band requires finite 0 <= low_hz < high_hz")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValidationError("Frequency band name must be nonempty when supplied")


CANONICAL_BANDS: dict[str, FrequencyBand] = {
    "delta": FrequencyBand(1.0, 4.0, "delta"),
    "theta": FrequencyBand(4.0, 8.0, "theta"),
    "alpha": FrequencyBand(8.0, 13.0, "alpha"),
    "beta": FrequencyBand(13.0, 30.0, "beta"),
    "gamma": FrequencyBand(30.0, 45.0, "gamma"),
}


@dataclass(frozen=True)
class PowerSpectrum:
    """One-sided periodogram in signal-unit squared per Hz.

    ``power`` is ``[batch, sensor, patch, frequency]`` for patch scope and
    ``[batch, sensor, frequency]`` for trial scope.
    """

    frequencies: torch.Tensor
    power: torch.Tensor
    scope: SpectralScope
    sampling_rate: float
    samples: int
    detrended: bool


@dataclass(frozen=True)
class WelchPowerSpectrum:
    """One-sided Welch PSD with explicit window and overlap geometry."""

    frequencies: torch.Tensor
    power: torch.Tensor
    scope: SpectralScope
    sampling_rate: float
    segment_samples: int
    overlap_samples: int
    segments: int
    detrended: bool


def _validate_scope(batch: SignalBatch, scope: SpectralScope) -> tuple[torch.Tensor, int]:
    batch.__post_init__()
    if scope == "patch":
        return batch.data, batch.data.shape[-1]
    if scope != "trial":
        raise ValidationError("Spectral scope must be 'patch' or 'trial'")
    if batch.stride != batch.data.shape[-1]:
        raise ValidationError(
            "Trial-scope spectra require contiguous nonoverlapping patches; "
            "use patch scope for overlapping or gapped inputs"
        )
    b, c, p, s = batch.data.shape
    return batch.data.reshape(b, c, p * s), p * s


def _band_mask(frequencies: torch.Tensor, band: FrequencyBand) -> torch.Tensor:
    band.__post_init__()
    nyquist = float(frequencies[-1])
    if band.high_hz > nyquist + 1e-9:
        raise ValidationError(f"Band upper edge {band.high_hz:g} Hz exceeds Nyquist {nyquist:g} Hz")
    upper = (
        frequencies <= band.high_hz
        if abs(band.high_hz - nyquist) < 1e-9
        else frequencies < band.high_hz
    )
    mask = (frequencies >= band.low_hz) & upper
    if not bool(mask.any()):
        raise ValidationError("Frequency band contains no FFT bins at this sampling geometry")
    return mask


def power_spectrum(
    batch: SignalBatch,
    *,
    scope: SpectralScope = "trial",
    detrend: bool = True,
) -> PowerSpectrum:
    """Compute a one-sided FFT periodogram without changing the input batch.

    Detrending removes only the mean over the transformed time axis.  This is a
    periodogram, not Welch averaging or an aperiodic/oscillatory decomposition.
    """

    series, samples = _validate_scope(batch, scope)
    working = series.float() if series.dtype in {torch.float16, torch.bfloat16} else series
    if detrend:
        working = working - working.mean(dim=-1, keepdim=True)
    coefficients = torch.fft.rfft(working, dim=-1)
    power = coefficients.abs().square() / (batch.sampling_rate * samples)
    if samples > 1:
        stop = -1 if samples % 2 == 0 else None
        power[..., 1:stop] *= 2
    frequencies = torch.fft.rfftfreq(
        samples, d=1.0 / batch.sampling_rate, device=working.device, dtype=working.dtype
    )
    return PowerSpectrum(frequencies, power, scope, batch.sampling_rate, samples, detrend)


def welch_power_spectrum(
    batch: SignalBatch,
    *,
    scope: SpectralScope = "trial",
    segment_samples: int | None = None,
    overlap_samples: int | None = None,
    detrend: bool = True,
) -> WelchPowerSpectrum:
    """Estimate a one-sided PSD with Hann-windowed overlapping segments."""

    series, samples = _validate_scope(batch, scope)
    segment_samples = min(512, samples) if segment_samples is None else segment_samples
    if type(segment_samples) is not int or not 2 <= segment_samples <= samples:
        raise ValidationError("Welch segment_samples must be an integer in [2, signal samples]")
    overlap_samples = segment_samples // 2 if overlap_samples is None else overlap_samples
    if type(overlap_samples) is not int or not 0 <= overlap_samples < segment_samples:
        raise ValidationError("Welch overlap_samples must be an integer in [0, segment_samples)")
    step = segment_samples - overlap_samples
    working = series.float() if series.dtype in {torch.float16, torch.bfloat16} else series
    frames = working.unfold(-1, segment_samples, step)
    if detrend:
        frames = frames - frames.mean(dim=-1, keepdim=True)
    window = torch.hann_window(
        segment_samples,
        periodic=True,
        dtype=working.dtype,
        device=working.device,
    )
    coefficients = torch.fft.rfft(frames * window, dim=-1)
    power = coefficients.abs().square() / (batch.sampling_rate * window.square().sum())
    if segment_samples > 1:
        stop = -1 if segment_samples % 2 == 0 else None
        power[..., 1:stop] *= 2
    power = power.mean(dim=-2)
    frequencies = torch.fft.rfftfreq(
        segment_samples,
        d=1.0 / batch.sampling_rate,
        device=working.device,
        dtype=working.dtype,
    )
    return WelchPowerSpectrum(
        frequencies,
        power,
        scope,
        batch.sampling_rate,
        segment_samples,
        overlap_samples,
        frames.shape[-2],
        detrend,
    )


def band_power(
    spectrum: PowerSpectrum | WelchPowerSpectrum,
    band: FrequencyBand,
) -> torch.Tensor:
    """Integrate a periodogram over a declared frequency band."""

    if not isinstance(spectrum, (PowerSpectrum, WelchPowerSpectrum)):
        raise ValidationError("Expected a PowerSpectrum or WelchPowerSpectrum")
    mask = _band_mask(spectrum.frequencies, band)
    samples = spectrum.samples if isinstance(spectrum, PowerSpectrum) else spectrum.segment_samples
    bin_width = spectrum.sampling_rate / samples
    return spectrum.power[..., mask].sum(dim=-1) * bin_width


def _edited_batch(
    batch: SignalBatch, data: torch.Tensor, transform: SignalTransform
) -> SignalBatch:
    data = data.to(dtype=batch.data.dtype)
    if data.shape != batch.data.shape or not torch.isfinite(data).all():
        raise ValidationError("Spectral edit produced invalid model input")
    return replace(batch, data=data, transforms=(*batch.transforms, transform))


def scale_frequency_band(
    batch: SignalBatch,
    band: FrequencyBand,
    scale: float,
    *,
    scope: SpectralScope = "trial",
) -> SignalBatch:
    """Scale complex coefficients in a band while preserving their phase.

    ``scale=0`` removes the selected discrete bins and ``scale=1`` is an exact
    numerical identity.  The sharp band boundary can cause time-domain ringing;
    callers should use bands and controls appropriate to their hypothesis.
    """

    value = torch.tensor(scale, dtype=torch.float64)
    if not bool(torch.isfinite(value)) or scale < 0:
        raise ValidationError("Frequency scale must be finite and nonnegative")
    series, samples = _validate_scope(batch, scope)
    working = series.float() if series.dtype in {torch.float16, torch.bfloat16} else series
    coefficients = torch.fft.rfft(working, dim=-1)
    frequencies = torch.fft.rfftfreq(
        samples, d=1.0 / batch.sampling_rate, device=working.device, dtype=working.dtype
    )
    mask = _band_mask(frequencies, band)
    edited = coefficients.clone()
    edited[..., mask] *= scale
    restored = torch.fft.irfft(edited, n=samples, dim=-1)
    if scope == "trial":
        restored = restored.reshape_as(batch.data)
    transform = SignalTransform(
        "scale_frequency_band",
        (
            ("low_hz", float(band.low_hz)),
            ("high_hz", float(band.high_hz)),
            ("scale", float(scale)),
            ("scope", scope),
        ),
    )
    return _edited_batch(batch, restored, transform)


def patch_frequency_band(
    recipient: SignalBatch,
    donor: SignalBatch,
    band: FrequencyBand,
    *,
    component: SpectralComponent = "complex",
    scope: SpectralScope = "trial",
) -> SignalBatch:
    """Patch donor spectrum into recipient, aligning rows by trial identity.

    ``complex`` replaces amplitude and phase, ``amplitude`` retains recipient
    phase, and ``phase`` retains recipient amplitude.  No resampling, channel
    remapping, cross-trial mixing, or normalization is performed.
    """

    if component not in {"complex", "amplitude", "phase"}:
        raise ValidationError("Spectral component must be complex, amplitude or phase")
    recipient_series, samples = _validate_scope(recipient, scope)
    donor_series, donor_samples = _validate_scope(donor, scope)
    if donor_samples != samples or donor.data.shape[1:] != recipient.data.shape[1:]:
        raise ValidationError("Donor and recipient signal geometry differs")
    for attribute in ("channels", "sampling_rate", "preprocessing_id", "unit", "stride"):
        if getattr(donor, attribute) != getattr(recipient, attribute):
            raise ValidationError(f"Donor and recipient {attribute} differ")
    if any(trial not in donor.trial_ids for trial in recipient.trial_ids):
        raise ValidationError("Recipient trial has no matching donor")
    order = [donor.trial_ids.index(trial) for trial in recipient.trial_ids]
    donor_series = donor_series[order]
    if (
        donor_series.dtype != recipient_series.dtype
        or donor_series.device != recipient_series.device
    ):
        raise ValidationError("Donor and recipient device or dtype differs")
    working_recipient = (
        recipient_series.float()
        if recipient_series.dtype in {torch.float16, torch.bfloat16}
        else recipient_series
    )
    working_donor = donor_series.to(dtype=working_recipient.dtype)
    recipient_fft = torch.fft.rfft(working_recipient, dim=-1)
    donor_fft = torch.fft.rfft(working_donor, dim=-1)
    frequencies = torch.fft.rfftfreq(
        samples,
        d=1.0 / recipient.sampling_rate,
        device=working_recipient.device,
        dtype=working_recipient.dtype,
    )
    mask = _band_mask(frequencies, band)
    edited = recipient_fft.clone()
    if component == "complex":
        edited[..., mask] = donor_fft[..., mask]
    elif component == "amplitude":
        edited[..., mask] = torch.polar(
            donor_fft[..., mask].abs(), recipient_fft[..., mask].angle()
        )
    else:
        edited[..., mask] = torch.polar(
            recipient_fft[..., mask].abs(), donor_fft[..., mask].angle()
        )
    restored = torch.fft.irfft(edited, n=samples, dim=-1)
    if scope == "trial":
        restored = restored.reshape_as(recipient.data)
    transform = SignalTransform(
        "patch_frequency_band",
        (
            ("low_hz", float(band.low_hz)),
            ("high_hz", float(band.high_hz)),
            ("component", component),
            ("scope", scope),
            ("donor_sha256", tensor_digest(donor.data)),
        ),
    )
    return _edited_batch(recipient, restored, transform)
