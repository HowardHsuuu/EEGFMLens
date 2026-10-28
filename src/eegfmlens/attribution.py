"""Per-trial gradient attribution in native EEG-FM execution paths."""

import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Sequence

import torch

from .errors import ValidationError
from .provenance import tensor_digest
from .spectral import FrequencyBand, SpectralScope
from .types import SignalBatch

AttributionMethod = Literal["gradient", "input_x_gradient", "integrated_gradients"]


@dataclass(frozen=True)
class AttributionResult:
    """Trial-aligned input and activation attribution tensors.

    ``input_multiplier`` is the raw gradient for gradient/input×gradient and
    the path-averaged gradient for integrated gradients.  It is retained so
    linear, EEG-specific coordinate transforms can propagate attribution
    without dividing by zero-valued input samples.
    """

    method: AttributionMethod
    objective: torch.Tensor
    baseline_objective: torch.Tensor | None
    input_attribution: torch.Tensor
    input_multiplier: torch.Tensor
    input_delta: torch.Tensor
    site_attributions: dict[str, torch.Tensor]
    site_multipliers: dict[str, torch.Tensor]
    site_deltas: dict[str, torch.Tensor]
    trial_ids: tuple[str, ...]
    channels: tuple[str, ...]
    sampling_rate: float
    patch_stride_samples: int
    preprocessing_id: str
    run_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SpectralAttribution:
    """Additive one-sided frequency attribution derived through the inverse DFT."""

    frequencies: torch.Tensor
    attribution: torch.Tensor
    scope: SpectralScope
    trial_ids: tuple[str, ...]
    channels: tuple[str, ...]
    conservation_error: torch.Tensor

    def mean_over_channels(self) -> torch.Tensor:
        return self.attribution.mean(dim=1)


@dataclass(frozen=True)
class SourceAttribution:
    """Additive attribution in a caller-supplied EEG source space.

    Tensors use ``[trial, source, patch, sample]``. The supplied forward model
    maps source deltas back to the channel order recorded by the attribution
    result; EEGFMLens does not estimate the inverse solution.
    """

    attribution: torch.Tensor
    source_delta: torch.Tensor
    source_multiplier: torch.Tensor
    source_names: tuple[str, ...]
    channels: tuple[str, ...]
    trial_ids: tuple[str, ...]
    reconstruction_rmse: torch.Tensor
    relative_reconstruction_error: torch.Tensor
    conservation_error: torch.Tensor
    forward_sha256: str
    source_delta_sha256: str

    def sum_over_time(self) -> torch.Tensor:
        """Return signed additive attribution with shape ``[trial, source]``."""

        return self.attribution.flatten(2).sum(dim=-1)

    def mean_over_time(self) -> torch.Tensor:
        """Return signed mean attribution with shape ``[trial, source]``."""

        return self.attribution.flatten(2).mean(dim=-1)


def _single(batch: SignalBatch, index: int) -> SignalBatch:
    return replace(batch, data=batch.data[index : index + 1], trial_ids=(batch.trial_ids[index],))


def _aligned_baseline(batch: SignalBatch, baseline: SignalBatch) -> SignalBatch:
    baseline.__post_init__()
    if set(batch.trial_ids) != set(baseline.trial_ids):
        raise ValidationError("Integrated-gradients baseline trial IDs must match exactly")
    for attribute in ("channels", "sampling_rate", "preprocessing_id", "unit", "stride"):
        if getattr(batch, attribute) != getattr(baseline, attribute):
            raise ValidationError(f"Integrated-gradients baseline {attribute} differs")
    if batch.data.shape != baseline.data.shape:
        raise ValidationError("Integrated-gradients baseline shape differs")
    if batch.data.device != baseline.data.device or batch.data.dtype != baseline.data.dtype:
        raise ValidationError("Integrated-gradients baseline device or dtype differs")
    order = [baseline.trial_ids.index(trial) for trial in batch.trial_ids]
    return replace(baseline, data=baseline.data[order], trial_ids=batch.trial_ids)


def attribute(
    lens,
    batch: SignalBatch,
    objective: Callable[[Any, SignalBatch], torch.Tensor],
    *,
    method: AttributionMethod = "integrated_gradients",
    baseline: SignalBatch | None = None,
    sites: Sequence[str] = (),
    steps: int = 32,
    execution_kwargs: dict[str, Any] | None = None,
) -> AttributionResult:
    """Attribute a scalar objective independently for every trial.

    Integrated gradients uses the trapezoid rule over ``steps`` intervals and
    requires an explicit baseline. Site attribution follows the same input path
    and integrates downstream gradient against each interval's activation
    change. This discrete path conductance remains valid when a site's activation
    path is nonlinear.
    """

    batch.__post_init__()
    if method not in {"gradient", "input_x_gradient", "integrated_gradients"}:
        raise ValidationError("Unknown attribution method")
    if isinstance(sites, (str, bytes)):
        raise ValidationError("Attribution sites must be a collection, not a string")
    sites = tuple(sites)
    if method == "integrated_gradients":
        if baseline is None:
            raise ValidationError("Integrated gradients requires an explicit SignalBatch baseline")
        if type(steps) is not int or steps < 1:
            raise ValidationError("Integrated-gradients steps must be a positive integer")
        baseline = _aligned_baseline(batch, baseline)
    elif baseline is not None:
        raise ValidationError("A baseline is used only for integrated gradients")
    kwargs = dict(execution_kwargs or {})
    objectives = []
    baseline_objectives = []
    input_attributions = []
    input_multipliers = []
    input_deltas = []
    site_attributions: dict[str, list[torch.Tensor]] = {name: [] for name in sites}
    site_multipliers: dict[str, list[torch.Tensor]] = {name: [] for name in sites}
    site_deltas: dict[str, list[torch.Tensor]] = {name: [] for name in sites}
    completeness_errors = []
    execution_record = None

    for index in range(len(batch.trial_ids)):
        observed = _single(batch, index)
        if method != "integrated_gradients":
            run = lens._differentiate(observed, objective, sites=sites, kwargs=kwargs)
            multiplier = run.input_gradient
            delta = torch.ones_like(run.input_value) if method == "gradient" else run.input_value
            input_attribution = multiplier if method == "gradient" else delta * multiplier
            objectives.append(run.objective)
            input_attributions.append(input_attribution)
            input_multipliers.append(multiplier)
            input_deltas.append(delta)
            for name in sites:
                site_multiplier = run.site_gradients[name]
                site_delta = (
                    torch.ones_like(run.site_values[name])
                    if method == "gradient"
                    else run.site_values[name]
                )
                site_multipliers[name].append(site_multiplier)
                site_deltas[name].append(site_delta)
                site_attributions[name].append(
                    site_multiplier if method == "gradient" else site_delta * site_multiplier
                )
            execution_record = run.execution_kwargs
            continue

        assert baseline is not None
        reference = _single(baseline, index)
        delta = observed.data - reference.data
        path_runs = []
        for alpha in torch.linspace(
            0,
            1,
            steps + 1,
            device=observed.data.device,
            dtype=observed.data.dtype,
        ):
            point = replace(reference, data=reference.data + alpha * delta)
            path_runs.append(lens._differentiate(point, objective, sites=sites, kwargs=kwargs))
        weights = observed.data.new_ones(steps + 1)
        weights[[0, -1]] = 0.5
        input_multiplier = (
            torch.stack(
                [weight * run.input_gradient for weight, run in zip(weights, path_runs)]
            ).sum(0)
            / steps
        )
        input_attribution = delta * input_multiplier
        objectives.append(path_runs[-1].objective)
        baseline_objectives.append(path_runs[0].objective)
        input_attributions.append(input_attribution)
        input_multipliers.append(input_multiplier)
        input_deltas.append(delta)
        completeness_errors.append(
            input_attribution.sum() - (path_runs[-1].objective - path_runs[0].objective).sum()
        )
        for name in sites:
            site_multiplier = (
                torch.stack(
                    [weight * run.site_gradients[name] for weight, run in zip(weights, path_runs)]
                ).sum(0)
                / steps
            )
            site_delta = path_runs[-1].site_values[name] - path_runs[0].site_values[name]
            conductance = torch.stack(
                [
                    0.5
                    * (
                        path_runs[path_index].site_gradients[name]
                        + path_runs[path_index + 1].site_gradients[name]
                    )
                    * (
                        path_runs[path_index + 1].site_values[name]
                        - path_runs[path_index].site_values[name]
                    )
                    for path_index in range(steps)
                ]
            ).sum(0)
            site_multipliers[name].append(site_multiplier)
            site_deltas[name].append(site_delta)
            site_attributions[name].append(conductance)
        execution_record = path_runs[-1].execution_kwargs

    result = AttributionResult(
        method,
        torch.cat(objectives),
        torch.cat(baseline_objectives) if baseline_objectives else None,
        torch.cat(input_attributions),
        torch.cat(input_multipliers),
        torch.cat(input_deltas),
        {name: torch.cat(values) for name, values in site_attributions.items()},
        {name: torch.cat(values) for name, values in site_multipliers.items()},
        {name: torch.cat(values) for name, values in site_deltas.items()},
        batch.trial_ids,
        batch.channels,
        batch.sampling_rate,
        batch.stride,
        batch.preprocessing_id,
        uuid.uuid4().hex,
        {
            "model_id": lens.model_id,
            "sites": sites,
            "steps": steps if method == "integrated_gradients" else None,
            "input_sha256": tensor_digest(batch.data),
            "baseline_sha256": tensor_digest(baseline.data) if baseline is not None else None,
            "execution_kwargs": execution_record,
            "trial_execution": "independent",
            "completeness_error": [float(value) for value in completeness_errors],
        },
    )
    tensors = [
        result.objective,
        result.input_attribution,
        result.input_multiplier,
        *result.site_attributions.values(),
        *result.site_multipliers.values(),
    ]
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValidationError("Attribution produced non-finite values")
    return result


def patch_attribution(result: AttributionResult, *, absolute: bool = False) -> torch.Tensor:
    """Aggregate input samples to ``[trial, channel, patch]`` attribution."""

    values = result.input_attribution.abs() if absolute else result.input_attribution
    return values.sum(dim=-1)


def channel_attribution(result: AttributionResult, *, absolute: bool = False) -> torch.Tensor:
    """Aggregate input attribution to ``[trial, channel]``."""

    return patch_attribution(result, absolute=absolute).sum(dim=-1)


def temporal_attribution(result: AttributionResult, *, absolute: bool = False) -> torch.Tensor:
    """Aggregate input attribution to ``[trial, patch]``."""

    return patch_attribution(result, absolute=absolute).sum(dim=1)


def spectral_attribution(
    result: AttributionResult,
    *,
    scope: SpectralScope = "trial",
) -> SpectralAttribution:
    """Map input×gradient or IG attribution through an inverse-DFT basis.

    This is a numerically stable form of the EEG-PRISM linear propagation rule:
    it uses the retained gradient multiplier directly instead of dividing the
    time-domain attribution by input samples that may be zero.
    """

    if result.method not in {"input_x_gradient", "integrated_gradients"}:
        raise ValidationError(
            "Spectral attribution requires input×gradient or integrated gradients"
        )
    delta, multiplier = result.input_delta, result.input_multiplier
    if scope == "trial":
        if result.patch_stride_samples != delta.shape[-1]:
            raise ValidationError(
                "Trial-scope attribution requires contiguous nonoverlapping patches"
            )
        b, c, p, s = delta.shape
        delta = delta.reshape(b, c, p * s)
        multiplier = multiplier.reshape(b, c, p * s)
    elif scope != "patch":
        raise ValidationError("Spectral attribution scope must be patch or trial")
    samples = delta.shape[-1]
    signal_fft = torch.fft.fft(delta, dim=-1)
    multiplier_fft = torch.fft.fft(multiplier, dim=-1)
    full = (signal_fft.real * multiplier_fft.real + signal_fft.imag * multiplier_fft.imag) / samples
    one_sided = full[..., : samples // 2 + 1].clone()
    stop = samples // 2 if samples % 2 == 0 else samples // 2 + 1
    for frequency_index in range(1, stop):
        one_sided[..., frequency_index] += full[..., samples - frequency_index]
    frequencies = torch.fft.rfftfreq(
        samples,
        d=1.0 / result.sampling_rate,
        device=delta.device,
        dtype=delta.dtype,
    )
    time_total = result.input_attribution.sum(dim=-1)
    if scope == "trial":
        time_total = time_total.sum(dim=-1)
    conservation_error = one_sided.sum(dim=-1) - time_total
    return SpectralAttribution(
        frequencies,
        one_sided,
        scope,
        result.trial_ids,
        result.channels,
        conservation_error,
    )


def source_attribution(
    result: AttributionResult,
    source_delta: torch.Tensor,
    forward_matrix: torch.Tensor,
    source_names: tuple[str, ...],
) -> SourceAttribution:
    """Propagate additive input attribution through an EEG forward model.

    ``forward_matrix`` has shape ``[channel, source]`` and must use
    ``result.channels`` order. ``source_delta`` is the source estimate relative
    to zero for input×gradient, or observed minus baseline source estimates for
    integrated gradients. Its forward projection need only approximate
    ``result.input_delta``; per-trial reconstruction and conservation errors
    make inverse-model limitations visible rather than silently accepting them.
    """

    if not isinstance(result, AttributionResult) or result.method not in {
        "input_x_gradient",
        "integrated_gradients",
    }:
        raise ValidationError("Source attribution requires input×gradient or integrated gradients")
    if (
        not isinstance(source_delta, torch.Tensor)
        or not source_delta.is_floating_point()
        or source_delta.ndim != 4
        or not torch.isfinite(source_delta).all()
    ):
        raise ValidationError(
            "Source delta must be a finite floating [trial, source, patch, sample] tensor"
        )
    if (
        not isinstance(forward_matrix, torch.Tensor)
        or not forward_matrix.is_floating_point()
        or forward_matrix.ndim != 2
        or not torch.isfinite(forward_matrix).all()
    ):
        raise ValidationError(
            "EEG forward matrix must be a finite floating [channel, source] tensor"
        )
    if (
        not isinstance(source_names, tuple)
        or not source_names
        or any(not isinstance(name, str) or not name for name in source_names)
        or len(set(source_names)) != len(source_names)
    ):
        raise ValidationError("Source names must be a nonempty tuple of unique nonempty strings")
    delta, multiplier = result.input_delta, result.input_multiplier
    if delta.ndim != 4 or multiplier.shape != delta.shape:
        raise ValidationError("Input attribution tensors must use [trial, channel, patch, sample]")
    if forward_matrix.shape != (delta.shape[1], source_delta.shape[1]):
        raise ValidationError("Forward matrix dimensions differ from channel and source axes")
    if len(result.channels) != delta.shape[1] or len(source_names) != source_delta.shape[1]:
        raise ValidationError("Channel or source names differ from their tensor axes")
    if source_delta.shape[0] != delta.shape[0] or source_delta.shape[2:] != delta.shape[2:]:
        raise ValidationError("Source delta differs from attribution trial or time geometry")
    if any(
        value.device != delta.device or value.dtype != delta.dtype
        for value in (source_delta, forward_matrix)
    ):
        raise ValidationError(
            "Source delta and forward matrix must match attribution device and dtype"
        )

    reconstructed = torch.einsum("cm,bmpt->bcpt", forward_matrix, source_delta)
    residual = reconstructed - delta
    reconstruction_rmse = residual.square().flatten(1).mean(dim=1).sqrt()
    input_rms = delta.square().flatten(1).mean(dim=1).sqrt()
    relative_error = reconstruction_rmse / input_rms.clamp_min(torch.finfo(delta.dtype).eps)
    source_multiplier = torch.einsum("cm,bcpt->bmpt", forward_matrix, multiplier)
    attribution = source_delta * source_multiplier
    conservation_error = attribution.flatten(1).sum(dim=1) - result.input_attribution.flatten(
        1
    ).sum(dim=1)
    return SourceAttribution(
        attribution.detach().clone(),
        source_delta.detach().clone(),
        source_multiplier.detach().clone(),
        source_names,
        result.channels,
        result.trial_ids,
        reconstruction_rmse.detach().clone(),
        relative_error.detach().clone(),
        conservation_error.detach().clone(),
        tensor_digest(forward_matrix),
        tensor_digest(source_delta),
    )


def spectral_band_attribution(result: SpectralAttribution, band: FrequencyBand) -> torch.Tensor:
    """Sum additive frequency attribution within a declared band."""

    band.__post_init__()
    nyquist = float(result.frequencies[-1])
    if band.high_hz > nyquist + 1e-9:
        raise ValidationError("Attribution band exceeds Nyquist")
    upper = (
        result.frequencies <= band.high_hz
        if abs(band.high_hz - nyquist) < 1e-9
        else result.frequencies < band.high_hz
    )
    mask = (result.frequencies >= band.low_hz) & upper
    if not bool(mask.any()):
        raise ValidationError("Attribution band contains no FFT bins")
    return result.attribution[..., mask].sum(dim=-1)
