"""Neurophysiological signal descriptors for EEG-FM interpretation studies."""

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch

from .errors import ValidationError
from .spectral import CANONICAL_BANDS, FrequencyBand, SpectralScope, power_spectrum
from .types import SignalBatch

FeatureAggregation = Literal["flatten", "mean"]


@dataclass(frozen=True)
class EEGFeatureSet:
    """Named trial-aligned descriptors with retained EEG coordinates.

    Trial-scope values have shape ``[trial, channel, feature]``. Patch-scope
    values have shape ``[trial, channel, patch, feature]``.
    """

    values: torch.Tensor
    feature_names: tuple[str, ...]
    trial_ids: tuple[str, ...]
    channels: tuple[str, ...]
    scope: SpectralScope
    metadata: dict[str, Any]

    def matrix(self, *, aggregation: FeatureAggregation = "flatten") -> torch.Tensor:
        """Return a probe-ready ``[trial, feature]`` matrix."""

        if aggregation == "flatten":
            return self.values.reshape(self.values.shape[0], -1)
        if aggregation == "mean":
            axes = tuple(range(1, self.values.ndim - 1))
            return self.values.mean(dim=axes)
        raise ValidationError("Feature aggregation must be flatten or mean")

    def column_names(self, *, aggregation: FeatureAggregation = "flatten") -> tuple[str, ...]:
        """Names in the same order as :meth:`matrix` columns."""

        if aggregation == "mean":
            return self.feature_names
        if aggregation != "flatten":
            raise ValidationError("Feature aggregation must be flatten or mean")
        if self.scope == "trial":
            return tuple(
                f"{channel}:{feature}"
                for channel in self.channels
                for feature in self.feature_names
            )
        patches = self.values.shape[2]
        return tuple(
            f"{channel}:patch-{patch}:{feature}"
            for channel in self.channels
            for patch in range(patches)
            for feature in self.feature_names
        )


@dataclass(frozen=True)
class EEGConnectivity:
    """Trial-aligned channel-by-channel signal relationship matrices.

    Trial-scope values are ``[trial, channel, channel]``; patch-scope values
    are ``[trial, patch, channel, channel]``.
    """

    values: torch.Tensor
    measure: str
    trial_ids: tuple[str, ...]
    channels: tuple[str, ...]
    scope: SpectralScope
    metadata: dict[str, Any]


def _series(batch: SignalBatch, scope: SpectralScope) -> torch.Tensor:
    batch.__post_init__()
    if scope == "patch":
        return batch.data
    if scope != "trial":
        raise ValidationError("Feature scope must be patch or trial")
    if batch.stride != batch.data.shape[-1]:
        raise ValidationError("Trial-scope features require contiguous nonoverlapping patches")
    batch_size, channels, patches, samples = batch.data.shape
    return batch.data.reshape(batch_size, channels, patches * samples)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    threshold = torch.finfo(numerator.dtype).eps
    return torch.where(denominator > threshold, numerator / denominator, 0)


def time_domain_features(
    batch: SignalBatch,
    *,
    scope: SpectralScope = "trial",
) -> EEGFeatureSet:
    """Compute nine transparent time-domain descriptors per channel/patch."""

    signal = _series(batch, scope)
    if signal.shape[-1] < 3:
        raise ValidationError("Time-domain features require at least three samples")
    signal = signal.float() if signal.dtype in {torch.float16, torch.bfloat16} else signal
    difference = signal[..., 1:] - signal[..., :-1]
    second_difference = difference[..., 1:] - difference[..., :-1]
    activity = signal.var(dim=-1, unbiased=False)
    derivative_activity = difference.var(dim=-1, unbiased=False)
    mobility = torch.sqrt(_safe_ratio(derivative_activity, activity))
    derivative_mobility = torch.sqrt(
        _safe_ratio(second_difference.var(dim=-1, unbiased=False), derivative_activity)
    )
    complexity = _safe_ratio(derivative_mobility, mobility)
    values = torch.stack(
        (
            activity,
            mobility,
            complexity,
            torch.sqrt(activity),
            torch.sqrt(signal.square().mean(dim=-1)),
            ((signal[..., 1:] * signal[..., :-1]) < 0).to(signal.dtype).mean(dim=-1),
            difference.abs().sum(dim=-1),
            torch.sqrt(derivative_activity),
            signal.amax(dim=-1) - signal.amin(dim=-1),
        ),
        dim=-1,
    )
    if not torch.isfinite(values).all():
        raise ValidationError("Time-domain feature extraction produced non-finite values")
    return EEGFeatureSet(
        values,
        (
            "hjorth.activity",
            "hjorth.mobility",
            "hjorth.complexity",
            "standard_deviation",
            "root_mean_square",
            "zero_crossing_rate",
            "line_length",
            "derivative_standard_deviation",
            "peak_to_peak",
        ),
        batch.trial_ids,
        batch.channels,
        scope,
        {
            "samples": signal.shape[-1],
            "variance": "population",
            "line_length": "sum_absolute_first_difference",
        },
    )


def _frequency_mask(frequencies: torch.Tensor, band: FrequencyBand) -> torch.Tensor:
    band.__post_init__()
    nyquist = float(frequencies[-1])
    if band.high_hz > nyquist + 1e-9:
        raise ValidationError(f"Feature band {band.high_hz:g} Hz exceeds Nyquist {nyquist:g} Hz")
    upper = (
        frequencies <= band.high_hz
        if abs(band.high_hz - nyquist) < 1e-9
        else frequencies < band.high_hz
    )
    mask = (frequencies >= band.low_hz) & upper
    if not bool(mask.any()):
        raise ValidationError("Feature band contains no FFT bins")
    return mask


def spectral_features(
    batch: SignalBatch,
    *,
    scope: SpectralScope = "trial",
    bands: Mapping[str, FrequencyBand] = CANONICAL_BANDS,
    frequency_range: FrequencyBand | None = None,
    edge_fraction: float = 0.95,
) -> EEGFeatureSet:
    """Compute band power and broadband spectral descriptors.

    For each supplied band, absolute log10 power and power relative to the
    declared broadband range are returned. Broadband entropy is normalized to
    ``[0, 1]``; centroid and edge are reported in Hz. This is a periodogram
    summary, not an aperiodic/periodic parameterization.
    """

    if not isinstance(bands, Mapping) or not bands:
        raise ValidationError("At least one named frequency band is required")
    if any(not isinstance(name, str) or not name for name in bands):
        raise ValidationError("Frequency-band names must be nonempty strings")
    if not isinstance(edge_fraction, (int, float)) or not 0 < edge_fraction < 1:
        raise ValidationError("Spectral edge fraction must lie strictly between zero and one")
    spectrum = power_spectrum(batch, scope=scope)
    if frequency_range is None:
        low = min(band.low_hz for band in bands.values())
        high = max(band.high_hz for band in bands.values())
        frequency_range = FrequencyBand(low, high, "broadband")
    broadband_mask = _frequency_mask(spectrum.frequencies, frequency_range)
    if int(broadband_mask.sum()) < 2:
        raise ValidationError("Broadband spectral descriptors require at least two FFT bins")
    bin_width = spectrum.sampling_rate / spectrum.samples
    broadband = spectrum.power[..., broadband_mask]
    broadband_power = broadband.sum(dim=-1) * bin_width
    tiny = torch.finfo(broadband.dtype).tiny
    if bool((broadband_power <= torch.finfo(broadband.dtype).eps).any()):
        raise ValidationError("Spectral descriptors are undefined for zero broadband power")
    features: list[torch.Tensor] = []
    names: list[str] = []
    for name, band in bands.items():
        if not isinstance(band, FrequencyBand):
            raise ValidationError("Frequency bands must be FrequencyBand records")
        mask = _frequency_mask(spectrum.frequencies, band)
        power = spectrum.power[..., mask].sum(dim=-1) * bin_width
        features.extend((torch.log10(power.clamp_min(tiny)), _safe_ratio(power, broadband_power)))
        names.extend((f"{name}.log10_power", f"{name}.relative_power"))
    probability = broadband / broadband.sum(dim=-1, keepdim=True).clamp_min(tiny)
    entropy = -(probability * probability.clamp_min(tiny).log()).sum(dim=-1)
    entropy = entropy / torch.log(
        torch.tensor(probability.shape[-1], device=probability.device, dtype=probability.dtype)
    )
    frequencies = spectrum.frequencies[broadband_mask]
    centroid = (probability * frequencies).sum(dim=-1)
    cumulative = probability.cumsum(dim=-1)
    edge_index = (cumulative >= edge_fraction).to(torch.int64).argmax(dim=-1)
    edge = frequencies[edge_index]
    features.extend((entropy, centroid, edge))
    names.extend(
        ("spectral_entropy", "spectral_centroid_hz", f"spectral_edge_{edge_fraction:g}_hz")
    )
    values = torch.stack(features, dim=-1)
    if not torch.isfinite(values).all():
        raise ValidationError("Spectral feature extraction produced non-finite values")
    return EEGFeatureSet(
        values,
        tuple(names),
        batch.trial_ids,
        batch.channels,
        scope,
        {
            "estimator": "one_sided_periodogram",
            "detrended": spectrum.detrended,
            "broadband_hz": (frequency_range.low_hz, frequency_range.high_hz),
            "edge_fraction": float(edge_fraction),
        },
    )


def _channel_first(series: torch.Tensor, scope: SpectralScope) -> torch.Tensor:
    return series if scope == "trial" else series.permute(0, 2, 1, 3)


def channel_correlation(
    batch: SignalBatch,
    *,
    scope: SpectralScope = "trial",
) -> EEGConnectivity:
    """Pearson channel-correlation matrices for each trial or patch."""

    signal = _channel_first(_series(batch, scope), scope)
    signal = signal.float() if signal.dtype in {torch.float16, torch.bfloat16} else signal
    centered = signal - signal.mean(dim=-1, keepdim=True)
    norm = torch.linalg.vector_norm(centered, dim=-1)
    if bool((norm <= torch.finfo(norm.dtype).eps).any()):
        raise ValidationError("Channel correlation is undefined for a constant channel")
    normalized = centered / norm.unsqueeze(-1)
    values = normalized @ normalized.transpose(-1, -2)
    return EEGConnectivity(
        values.clamp(-1, 1),
        "pearson_correlation",
        batch.trial_ids,
        batch.channels,
        scope,
        {},
    )


def _band_analytic(
    batch: SignalBatch,
    band: FrequencyBand,
    scope: SpectralScope,
) -> torch.Tensor:
    signal = _series(batch, scope)
    working = signal.float() if signal.dtype in {torch.float16, torch.bfloat16} else signal
    samples = working.shape[-1]
    coefficients = torch.fft.rfft(working, dim=-1)
    frequencies = torch.fft.rfftfreq(
        samples,
        d=1.0 / batch.sampling_rate,
        device=working.device,
        dtype=working.dtype,
    )
    mask = _frequency_mask(frequencies, band)
    coefficients = torch.where(mask, coefficients, 0)
    filtered = torch.fft.irfft(coefficients, n=samples, dim=-1)
    full = torch.fft.fft(filtered, dim=-1)
    multiplier = working.new_zeros(samples)
    multiplier[0] = 1
    if samples % 2 == 0:
        multiplier[samples // 2] = 1
        multiplier[1 : samples // 2] = 2
    else:
        multiplier[1 : (samples + 1) // 2] = 2
    return torch.fft.ifft(full * multiplier, dim=-1)


def band_connectivity(
    batch: SignalBatch,
    band: FrequencyBand,
    *,
    measure: Literal[
        "phase_lag_index", "phase_locking_value", "magnitude_squared_coherence"
    ] = "phase_lag_index",
    scope: SpectralScope = "trial",
) -> EEGConnectivity:
    """Compute one phase/amplitude relationship matrix in a frequency band."""

    analytic = _channel_first(_band_analytic(batch, band, scope), scope)
    if measure == "phase_lag_index":
        phase = torch.angle(analytic)
        difference = phase.unsqueeze(-2) - phase.unsqueeze(-3)
        values = torch.sign(torch.sin(difference)).mean(dim=-1).abs()
    elif measure == "phase_locking_value":
        phase = torch.angle(analytic)
        difference = phase.unsqueeze(-2) - phase.unsqueeze(-3)
        values = torch.exp(1j * difference).mean(dim=-1).abs()
    elif measure == "magnitude_squared_coherence":
        cross = (analytic.unsqueeze(-2) * analytic.unsqueeze(-3).conj()).mean(dim=-1)
        auto = analytic.abs().square().mean(dim=-1)
        denominator = auto.unsqueeze(-2) * auto.unsqueeze(-1)
        if bool((denominator <= torch.finfo(denominator.dtype).eps).any()):
            raise ValidationError("Band coherence is undefined for a zero-power channel")
        values = cross.abs().square() / denominator
    else:
        raise ValidationError(
            "Connectivity measure must be phase_lag_index, phase_locking_value or "
            "magnitude_squared_coherence"
        )
    if not torch.isfinite(values).all():
        raise ValidationError("Band connectivity produced non-finite values")
    return EEGConnectivity(
        values.clamp(0, 1),
        measure,
        batch.trial_ids,
        batch.channels,
        scope,
        {"band_hz": (band.low_hz, band.high_hz), "band_name": band.name},
    )
