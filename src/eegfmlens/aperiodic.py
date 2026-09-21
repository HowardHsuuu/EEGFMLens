"""Optional periodic/aperiodic spectral fits and controlled signal edits."""

import hashlib
import json
import math
from dataclasses import dataclass, replace
from importlib.metadata import version
from typing import Literal

import torch

from .errors import ValidationError
from .spectral import FrequencyBand, SpectralScope, welch_power_spectrum
from .types import SignalBatch, SignalTransform

SpectralRemoval = Literal["aperiodic", "periodic", "both"]


@dataclass(frozen=True)
class AperiodicChannelFit:
    """Fixed-mode spectral parameters for one named EEG channel.

    ``gaussians`` contains the exact FOOOF fit components as
    ``(center_hz, log10_height, standard_deviation_hz)`` tuples. These are
    reconstruction parameters, rather than the derived center/power/bandwidth
    summaries returned by ``peak_params_``.
    """

    channel: str
    offset: float
    exponent: float
    gaussians: tuple[tuple[float, float, float], ...]
    r_squared: float


@dataclass(frozen=True)
class AperiodicDecomposition:
    """A channel-wise fit reusable on batches with the same signal contract."""

    fits: tuple[AperiodicChannelFit, ...]
    channels: tuple[str, ...]
    sampling_rate: float
    preprocessing_id: str
    unit: str
    fit_range: FrequencyBand
    scope: SpectralScope
    segment_samples: int
    overlap_samples: int
    peak_width_limits: tuple[float, float]
    max_peaks: int
    min_peak_height: float
    estimator: str = "fooof-fixed"
    estimator_version: str = "unknown"

    def __post_init__(self):
        self.fit_range.__post_init__()
        if (
            not isinstance(self.channels, tuple)
            or not self.channels
            or any(not isinstance(channel, str) or not channel for channel in self.channels)
            or len(set(self.channels)) != len(self.channels)
        ):
            raise ValidationError("Aperiodic channels must be unique nonempty strings")
        if tuple(fit.channel for fit in self.fits) != self.channels:
            raise ValidationError("Aperiodic fits must match the declared channel order")
        if (
            not math.isfinite(self.sampling_rate)
            or self.sampling_rate <= 0
            or not self.preprocessing_id
            or not self.unit
            or not self.estimator
            or not self.estimator_version
            or self.scope not in {"patch", "trial"}
        ):
            raise ValidationError("Aperiodic decomposition has an invalid signal contract")
        if (
            type(self.segment_samples) is not int
            or self.segment_samples < 2
            or type(self.overlap_samples) is not int
            or not 0 <= self.overlap_samples < self.segment_samples
            or type(self.max_peaks) is not int
            or self.max_peaks < 0
            or not math.isfinite(self.min_peak_height)
            or self.min_peak_height < 0
        ):
            raise ValidationError("Aperiodic decomposition has invalid fit settings")
        if (
            len(self.peak_width_limits) != 2
            or not all(math.isfinite(value) for value in self.peak_width_limits)
            or not 0 < self.peak_width_limits[0] < self.peak_width_limits[1]
        ):
            raise ValidationError("Aperiodic peak-width limits are invalid")
        for fit in self.fits:
            values = [fit.offset, fit.exponent, fit.r_squared]
            for gaussian in fit.gaussians:
                if len(gaussian) != 3 or gaussian[2] <= 0:
                    raise ValidationError(
                        "Periodic Gaussian parameters require center, height and positive deviation"
                    )
                values.extend(gaussian)
            if not all(math.isfinite(value) for value in values):
                raise ValidationError("Aperiodic fit parameters must be finite")

    @property
    def digest(self) -> str:
        payload = {
            "fits": [
                {
                    "channel": fit.channel,
                    "offset": fit.offset,
                    "exponent": fit.exponent,
                    "gaussians": fit.gaussians,
                    "r_squared": fit.r_squared,
                }
                for fit in self.fits
            ],
            "channels": self.channels,
            "sampling_rate": self.sampling_rate,
            "preprocessing_id": self.preprocessing_id,
            "unit": self.unit,
            "fit_range": (self.fit_range.low_hz, self.fit_range.high_hz),
            "scope": self.scope,
            "segment_samples": self.segment_samples,
            "overlap_samples": self.overlap_samples,
            "peak_width_limits": self.peak_width_limits,
            "max_peaks": self.max_peaks,
            "min_peak_height": self.min_peak_height,
            "estimator": self.estimator,
            "estimator_version": self.estimator_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def fit_aperiodic_decomposition(
    batch: SignalBatch,
    *,
    fit_range: FrequencyBand = FrequencyBand(1, 45, "aperiodic-fit"),
    scope: SpectralScope = "trial",
    segment_samples: int | None = None,
    overlap_samples: int | None = None,
    peak_width_limits: tuple[float, float] = (1, 12),
    max_peaks: int = 6,
    min_peak_height: float = 0.1,
) -> AperiodicDecomposition:
    """Fit one fixed-mode FOOOF model per channel on an explicit reference batch.

    Install ``eegfmlens[aperiodic]`` to use this method. Fits average Welch PSDs
    over trials and, for patch scope, patches. Fit on a training/reference cohort
    and reuse the returned object on evaluation batches to keep the intervention
    fixed.
    """

    try:
        from fooof import FOOOF
    except ImportError as exc:
        raise ValidationError(
            "Aperiodic fitting requires the optional 'aperiodic' dependency"
        ) from exc
    batch.__post_init__()
    if batch.data.device.type != "cpu":
        raise ValidationError("Aperiodic fitting currently requires a CPU SignalBatch")
    fit_range.__post_init__()
    if fit_range.low_hz <= 0:
        raise ValidationError("Aperiodic fitting requires a strictly positive lower frequency")
    bounds = torch.tensor(peak_width_limits, dtype=torch.float64)
    if (
        len(peak_width_limits) != 2
        or not bool(torch.isfinite(bounds).all())
        or not 0 < peak_width_limits[0] < peak_width_limits[1]
    ):
        raise ValidationError("peak_width_limits must contain two increasing positive values")
    if type(max_peaks) is not int or max_peaks < 0:
        raise ValidationError("max_peaks must be a nonnegative integer")
    if (
        not isinstance(min_peak_height, (int, float))
        or not math.isfinite(min_peak_height)
        or min_peak_height < 0
    ):
        raise ValidationError("min_peak_height must be finite and nonnegative")
    spectrum = welch_power_spectrum(
        batch,
        scope=scope,
        segment_samples=segment_samples,
        overlap_samples=overlap_samples,
    )
    if fit_range.high_hz > float(spectrum.frequencies[-1]) + 1e-9:
        raise ValidationError("Aperiodic fit range exceeds the Welch Nyquist frequency")
    reduce_axes = (0,) if scope == "trial" else (0, 2)
    average_power = spectrum.power.mean(dim=reduce_axes)
    frequencies = spectrum.frequencies.detach().cpu().numpy()
    fits = []
    for channel_index, channel in enumerate(batch.channels):
        power = average_power[channel_index].detach().cpu().numpy()
        model = FOOOF(
            peak_width_limits=peak_width_limits,
            max_n_peaks=max_peaks,
            min_peak_height=min_peak_height,
            aperiodic_mode="fixed",
            verbose=False,
        )
        try:
            model.fit(frequencies, power, [fit_range.low_hz, fit_range.high_hz])
        except Exception as exc:
            raise ValidationError(f"Aperiodic fit failed for channel {channel!r}") from exc
        if not model.has_model:
            raise ValidationError(f"Aperiodic fit produced no model for channel {channel!r}")
        offset, exponent = (float(value) for value in model.aperiodic_params_)
        gaussians_list: list[tuple[float, float, float]] = []
        for row in model.gaussian_params_:
            if len(row) != 3:
                raise ValidationError(f"Periodic Gaussian fit is malformed for channel {channel!r}")
            center, height, standard_deviation = (float(value) for value in row)
            gaussians_list.append((center, height, standard_deviation))
        gaussians = tuple(gaussians_list)
        values = torch.tensor(
            [
                offset,
                exponent,
                float(model.r_squared_),
                *(value for row in gaussians for value in row),
            ],
            dtype=torch.float64,
        )
        if not bool(torch.isfinite(values).all()):
            raise ValidationError(f"Aperiodic fit is non-finite for channel {channel!r}")
        fits.append(
            AperiodicChannelFit(
                channel,
                offset,
                exponent,
                gaussians,
                float(model.r_squared_),
            )
        )
    return AperiodicDecomposition(
        tuple(fits),
        batch.channels,
        batch.sampling_rate,
        batch.preprocessing_id,
        batch.unit,
        fit_range,
        scope,
        spectrum.segment_samples,
        spectrum.overlap_samples,
        (float(peak_width_limits[0]), float(peak_width_limits[1])),
        max_peaks,
        float(min_peak_height),
        "fooof-fixed",
        version("fooof"),
    )


def remove_fitted_spectral_component(
    batch: SignalBatch,
    decomposition: AperiodicDecomposition,
    *,
    component: SpectralRemoval = "aperiodic",
) -> SignalBatch:
    """Divide out a fitted periodic and/or aperiodic amplitude component.

    Phase and frequencies outside the fitted range are retained. Removal is
    multiplicative and partial: it tests dependence on the fitted component; it
    does not claim that the reconstructed signal contains no such physiology.
    """

    batch.__post_init__()
    if not isinstance(decomposition, AperiodicDecomposition):
        raise ValidationError("Expected an AperiodicDecomposition")
    if component not in {"aperiodic", "periodic", "both"}:
        raise ValidationError("Spectral component must be aperiodic, periodic or both")
    for attribute in ("channels", "sampling_rate", "preprocessing_id", "unit"):
        if getattr(batch, attribute) != getattr(decomposition, attribute):
            raise ValidationError(f"Aperiodic decomposition {attribute} differs")
    if decomposition.scope == "trial":
        if batch.stride != batch.data.shape[-1]:
            raise ValidationError(
                "Trial-scope component removal requires contiguous nonoverlapping patches"
            )
        shape = batch.data.shape
        series = batch.data.reshape(shape[0], shape[1], shape[2] * shape[3])
    elif decomposition.scope == "patch":
        series = batch.data
    else:
        raise ValidationError("Aperiodic decomposition has an invalid scope")
    working = series.float() if series.dtype in {torch.float16, torch.bfloat16} else series
    samples = working.shape[-1]
    frequencies = torch.fft.rfftfreq(
        samples,
        d=1.0 / batch.sampling_rate,
        device=working.device,
        dtype=working.dtype,
    )
    fit_range = decomposition.fit_range
    in_range = (frequencies >= fit_range.low_hz) & (frequencies <= fit_range.high_hz)
    if not bool(in_range.any()):
        raise ValidationError("Aperiodic fit range contains no bins for this batch")
    coefficients = torch.fft.rfft(working, dim=-1)
    edited = coefficients.clone()
    for channel_index, fit in enumerate(decomposition.fits):
        divisor = torch.ones_like(frequencies)
        if component in {"periodic", "both"}:
            log_peaks = torch.zeros_like(frequencies)
            for center, height, standard_deviation in fit.gaussians:
                log_peaks += height * torch.exp(
                    -(frequencies - center).square() / (2 * standard_deviation**2)
                )
            divisor *= torch.sqrt(torch.pow(10.0, log_peaks))
        if component in {"aperiodic", "both"}:
            safe_frequency = frequencies.clamp_min(fit_range.low_hz)
            log_background = fit.offset - fit.exponent * torch.log10(safe_frequency)
            divisor *= torch.sqrt(torch.pow(10.0, log_background))
        divisor = torch.where(in_range, divisor.clamp_min(1e-8), 1)
        edited[:, channel_index] /= divisor
    restored = torch.fft.irfft(edited, n=samples, dim=-1)
    if decomposition.scope == "trial":
        restored = restored.reshape_as(batch.data)
    restored = restored.to(dtype=batch.data.dtype)
    if restored.shape != batch.data.shape or not torch.isfinite(restored).all():
        raise ValidationError("Spectral component removal produced invalid model input")
    return replace(
        batch,
        data=restored,
        transforms=(
            *batch.transforms,
            SignalTransform(
                "remove_fitted_spectral_component",
                (
                    ("component", component),
                    ("fit_sha256", decomposition.digest),
                    ("low_hz", float(fit_range.low_hz)),
                    ("high_hz", float(fit_range.high_hz)),
                    ("scope", decomposition.scope),
                ),
            ),
        ),
    )
