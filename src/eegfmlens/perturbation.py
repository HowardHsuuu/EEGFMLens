"""Trial-independent perturbation curves for attribution faithfulness."""

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import torch

from .errors import ValidationError
from .interventions import Selection
from .provenance import tensor_digest
from .spectral import FrequencyBand, SpectralScope, scale_frequency_band
from .types import SignalBatch, SignalTransform


@dataclass(frozen=True)
class InputTarget:
    """Named channel/patch region for progressive input occlusion."""

    name: str
    selection: Selection


@dataclass(frozen=True)
class BandTarget:
    """Named frequency band for progressive spectral removal."""

    name: str
    band: FrequencyBand


@dataclass(frozen=True)
class PerturbationCurve:
    """Scores after zero or more cumulative perturbations.

    ``scores`` has shape ``[trial, 1 + target]``.  Column zero is unperturbed.
    AOPC is the mean unperturbed-score drop over all perturbed columns.  Scores
    must be higher-is-better; raw AOPC should not be compared across models with
    different score scales.
    """

    target_names: tuple[str, ...]
    scores: torch.Tensor
    aopc: torch.Tensor
    trial_ids: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def drops(self) -> torch.Tensor:
        return self.scores[:, :1] - self.scores[:, 1:]


def _score(score: Callable, output: Any, batch: SignalBatch) -> torch.Tensor:
    with torch.no_grad():
        value = score(output, batch)
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (1,)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValidationError("Score must return one finite floating value per trial, shape [B]")
    return value.detach().clone()


def _single(batch: SignalBatch, index: int) -> SignalBatch:
    return replace(batch, data=batch.data[index : index + 1], trial_ids=(batch.trial_ids[index],))


def _validate_names(targets: Sequence[InputTarget] | Sequence[BandTarget]) -> tuple[str, ...]:
    names = tuple(target.name for target in targets)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValidationError("Perturbation targets require nonempty names")
    if len(set(names)) != len(names):
        raise ValidationError("Perturbation target names must be unique")
    return names


def _curve(scores: list[list[torch.Tensor]], names, trials, metadata) -> PerturbationCurve:
    matrix = torch.stack([torch.cat(row) for row in scores])
    aopc = (matrix[:, :1] - matrix[:, 1:]).mean(dim=1)
    return PerturbationCurve(names, matrix, aopc, trials, metadata)


def occlusion_curve(
    lens,
    batch: SignalBatch,
    baseline: SignalBatch,
    score: Callable,
    targets: Sequence[InputTarget],
    *,
    execution_kwargs: dict[str, Any] | None = None,
) -> PerturbationCurve:
    """Progressively replace disjoint channel/patch regions with a baseline.

    Target order is caller supplied, typically from a held-out attribution map.
    Each trial is executed independently to prevent batch coupling.  A baseline
    must be explicit because zero, mean and physiological-reference occlusion
    answer different questions.
    """

    batch.__post_init__()
    baseline.__post_init__()
    names = _validate_names(targets)
    if set(batch.trial_ids) != set(baseline.trial_ids):
        raise ValidationError("Occlusion baseline trial IDs must match exactly")
    for attribute in ("channels", "sampling_rate", "preprocessing_id", "unit", "stride"):
        if getattr(batch, attribute) != getattr(baseline, attribute):
            raise ValidationError(f"Occlusion baseline {attribute} differs")
    if batch.data.shape != baseline.data.shape:
        raise ValidationError("Occlusion baseline shape differs")
    if batch.data.device != baseline.data.device or batch.data.dtype != baseline.data.dtype:
        raise ValidationError("Occlusion baseline device or dtype differs")
    order = [baseline.trial_ids.index(trial) for trial in batch.trial_ids]
    baseline = replace(baseline, data=baseline.data[order], trial_ids=batch.trial_ids)
    probe = torch.empty_like(batch.data, dtype=torch.bool)
    occupied = torch.zeros_like(probe)
    masks = []
    for target in targets:
        if not isinstance(target, InputTarget):
            raise ValidationError("occlusion_curve expects InputTarget records")
        mask = target.selection.mask(probe, batch, "bcpd")
        if bool((occupied & mask).any()):
            raise ValidationError("Progressive occlusion targets must be disjoint")
        occupied |= mask
        masks.append(mask)
    kwargs = dict(execution_kwargs or {})
    rows = []
    for index, trial_id in enumerate(batch.trial_ids):
        current = _single(batch, index)
        reference = _single(baseline, index)
        row = [_score(score, lens.run_with_cache(current, sites=[], **kwargs).output, current)]
        for target, mask in zip(targets, masks):
            single_mask = mask[index : index + 1]
            current = replace(
                current,
                data=torch.where(single_mask, reference.data, current.data),
                transforms=(
                    *current.transforms,
                    SignalTransform("occlude_input", (("target", target.name),)),
                ),
            )
            row.append(
                _score(score, lens.run_with_cache(current, sites=[], **kwargs).output, current)
            )
        rows.append(row)
    return _curve(
        rows,
        names,
        batch.trial_ids,
        {
            "method": "progressive channel/patch baseline replacement",
            "target_order": names,
            "score_direction": "higher_is_better",
            "trial_execution": "independent",
            "model_id": lens.model_id,
            "input_sha256": tensor_digest(batch.data),
            "baseline_sha256": tensor_digest(baseline.data),
            "execution_kwargs": kwargs,
        },
    )


def spectral_perturbation_curve(
    lens,
    batch: SignalBatch,
    score: Callable,
    targets: Sequence[BandTarget],
    *,
    scope: SpectralScope = "trial",
    execution_kwargs: dict[str, Any] | None = None,
) -> PerturbationCurve:
    """Progressively zero caller-ordered, nonoverlapping frequency bands."""

    batch.__post_init__()
    names = _validate_names(targets)
    intervals: list[tuple[float, float]] = []
    for target in targets:
        if not isinstance(target, BandTarget):
            raise ValidationError("spectral_perturbation_curve expects BandTarget records")
        target.band.__post_init__()
        for low, high in intervals:
            if max(low, target.band.low_hz) < min(high, target.band.high_hz):
                raise ValidationError("Progressive spectral targets must not overlap")
        intervals.append((target.band.low_hz, target.band.high_hz))
    kwargs = dict(execution_kwargs or {})
    rows = []
    for index in range(len(batch.trial_ids)):
        current = _single(batch, index)
        row = [_score(score, lens.run_with_cache(current, sites=[], **kwargs).output, current)]
        for target in targets:
            current = scale_frequency_band(current, target.band, 0, scope=scope)
            row.append(
                _score(score, lens.run_with_cache(current, sites=[], **kwargs).output, current)
            )
        rows.append(row)
    return _curve(
        rows,
        names,
        batch.trial_ids,
        {
            "method": "progressive phase-preserving frequency-band removal",
            "scope": scope,
            "target_order": names,
            "score_direction": "higher_is_better",
            "trial_execution": "independent",
            "model_id": lens.model_id,
            "input_sha256": tensor_digest(batch.data),
            "execution_kwargs": kwargs,
        },
    )


def attribution_cosine_consistency(
    attributions: Sequence[torch.Tensor], *, absolute: bool = True
) -> torch.Tensor:
    """Mean per-trial cosine similarity between attribution methods.

    Returns a symmetric ``[method, method]`` matrix.  Every tensor must use the
    same trial-first coordinates; the function performs no alignment by label.
    """

    if len(attributions) < 2:
        raise ValidationError("Consistency requires at least two attribution tensors")
    shape = attributions[0].shape
    if len(shape) < 2:
        raise ValidationError("Attributions require trial and feature axes")
    flattened = []
    for attribution in attributions:
        if (
            not isinstance(attribution, torch.Tensor)
            or attribution.shape != shape
            or not attribution.is_floating_point()
            or not torch.isfinite(attribution).all()
        ):
            raise ValidationError("Attribution tensors must be finite and identically shaped")
        values = attribution.abs() if absolute else attribution
        values = values.reshape(values.shape[0], -1)
        norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
        if bool((norms <= torch.finfo(norms.dtype).eps).any()):
            raise ValidationError("Cosine consistency is undefined for a zero attribution map")
        flattened.append(values / norms)
    count = len(flattened)
    result = attributions[0].new_empty((count, count))
    for left in range(count):
        for right in range(count):
            result[left, right] = (flattened[left] * flattened[right]).sum(1).mean()
    return result
