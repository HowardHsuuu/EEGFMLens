"""Controlled cumulative SAE-feature interventions on native model outputs."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch

from .errors import ValidationError
from .interventions import AxisSelection, Selection
from .model import EEGLens
from .provenance import tensor_digest
from .sae import (
    SAECodeReference,
    SAEFeatureAblation,
    SAEFeatureClamping,
    TopKSAE,
)
from .types import RunResult, SignalBatch

SAESweepMode = Literal["ablate", "clamp"]


@dataclass(frozen=True)
class SAEFeatureSweepResult:
    """Trial-level scores for observed and random cumulative feature rankings.

    Observed score tensors have shape ``[trial, step]``. Random controls have
    shape ``[draw, trial, step]``. Step zero is always the unmodified model.
    The package retains raw metric scales and directions.
    """

    mode: SAESweepMode
    site: str
    feature_ranking: tuple[int, ...]
    feature_counts: tuple[int, ...]
    feature_fractions: torch.Tensor
    scores: dict[str, torch.Tensor]
    random_scores: dict[str, torch.Tensor]
    random_rankings: tuple[tuple[int, ...], ...]
    trial_ids: tuple[str, ...]
    seed: int
    metadata: dict[str, Any]

    def _metric(self, metric: str) -> torch.Tensor:
        try:
            return self.scores[metric]
        except KeyError as error:
            raise ValidationError(f"Unknown SAE sweep metric: {metric}") from error

    def mean_delta(self, metric: str) -> torch.Tensor:
        """Mean score change from the unmodified model at every step."""

        values = self._metric(metric)
        return (values - values[:, :1]).mean(dim=0)

    def integrated_mean_delta(self, metric: str) -> float:
        """Trapezoidal area of the signed mean change over dictionary fraction."""

        return float(torch.trapezoid(self.mean_delta(metric), self.feature_fractions))

    def area_between(self, first_metric: str, second_metric: str) -> float:
        """Area of ``first - second`` mean score; caller assigns its meaning."""

        first, second = self._metric(first_metric), self._metric(second_metric)
        return float(torch.trapezoid((first - second).mean(dim=0), self.feature_fractions))

    def random_integrated_mean_delta(self, metric: str) -> torch.Tensor:
        """One signed integrated mean change per random ranking."""

        self._metric(metric)
        random = self.random_scores[metric]
        if random.shape[0] == 0:
            return random.new_empty((0,))
        delta = (random - random[:, :, :1]).mean(dim=1)
        return torch.trapezoid(delta, self.feature_fractions, dim=1)

    def random_area_between(self, first_metric: str, second_metric: str) -> torch.Tensor:
        """One ``first - second`` mean-score area per random ranking."""

        self._metric(first_metric)
        self._metric(second_metric)
        first = self.random_scores[first_metric]
        second = self.random_scores[second_metric]
        if first.shape[0] == 0:
            return first.new_empty((0,))
        return torch.trapezoid((first - second).mean(dim=1), self.feature_fractions, dim=1)


def _single(batch: SignalBatch, index: int) -> SignalBatch:
    return replace(batch, data=batch.data[index : index + 1], trial_ids=(batch.trial_ids[index],))


def _score_metrics(
    metrics: tuple[tuple[str, Callable], ...],
    run_metrics: tuple[tuple[str, Callable], ...],
    run: RunResult,
    batch: SignalBatch,
) -> dict[str, torch.Tensor]:
    values = {}
    for name, metric in metrics:
        with torch.no_grad():
            value = metric(run.output, batch)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (1,)
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValidationError(
                f"SAE sweep metric {name!r} must return one finite floating value per trial"
            )
        values[name] = value.detach().clone()
    for name, metric in run_metrics:
        with torch.no_grad():
            value = metric(run, batch)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (1,)
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValidationError(
                f"SAE sweep metric {name!r} must return one finite floating value per trial"
            )
        values[name] = value.detach().clone()
    return values


def _intervention(
    mode: SAESweepMode,
    site: str,
    sae: TopKSAE,
    features: tuple[int, ...],
    reference: SAECodeReference | None,
    selection: Selection | AxisSelection,
):
    if mode == "ablate":
        return SAEFeatureAblation(site, sae, features, selection)
    assert reference is not None
    return SAEFeatureClamping(site, sae, features, reference, selection)


def _run_ranking(
    lens: EEGLens,
    batch: SignalBatch,
    site: str,
    sae: TopKSAE,
    ranking: tuple[int, ...],
    counts: tuple[int, ...],
    metrics: tuple[tuple[str, Callable], ...],
    run_metrics: tuple[tuple[str, Callable], ...],
    cache_sites: tuple[str, ...],
    mode: SAESweepMode,
    reference: SAECodeReference | None,
    selection: Selection | AxisSelection,
    kwargs: dict[str, Any],
    baseline: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    all_metrics = (*metrics, *run_metrics)
    rows: dict[str, list[torch.Tensor]] = {name: [] for name, _ in all_metrics}
    for trial_index in range(len(batch.trial_ids)):
        current = _single(batch, trial_index)
        trial_values: dict[str, list[torch.Tensor]] = {name: [] for name, _ in all_metrics}
        if baseline is None:
            run = lens.run_with_cache(current, sites=cache_sites, **kwargs)
            values = _score_metrics(metrics, run_metrics, run, current)
        else:
            values = {name: baseline[name][trial_index, :1] for name, _ in all_metrics}
        for name, _ in all_metrics:
            trial_values[name].append(values[name])
        for count in counts[1:]:
            edit = _intervention(
                mode,
                site,
                sae,
                ranking[:count],
                reference,
                selection,
            )
            run = lens.run_with_interventions(
                current,
                interventions=(edit,),
                sites=cache_sites,
                **kwargs,
            )
            values = _score_metrics(metrics, run_metrics, run, current)
            for name, _ in all_metrics:
                trial_values[name].append(values[name])
        for name, _ in all_metrics:
            rows[name].append(torch.cat(trial_values[name]))
    return {name: torch.stack(values) for name, values in rows.items()}


def _selection_metadata(selection: Selection | AxisSelection) -> dict[str, Any]:
    if isinstance(selection, AxisSelection):
        return {"type": "axis", "axis": selection.axis, "indices": selection.indices}
    return {"type": "physical", "sensors": selection.sensors, "patches": selection.patches}


def sae_feature_sweep(
    lens: EEGLens,
    batch: SignalBatch,
    site: str,
    sae: TopKSAE,
    feature_ranking: tuple[int, ...],
    feature_counts: tuple[int, ...],
    metrics: Mapping[str, Callable],
    *,
    run_metrics: Mapping[str, Callable] | None = None,
    cache_sites: tuple[str, ...] = (),
    mode: SAESweepMode = "ablate",
    reference: SAECodeReference | None = None,
    selection: Selection | AxisSelection = Selection(),
    random_draws: int = 0,
    seed: int = 0,
    execution_kwargs: dict[str, Any] | None = None,
) -> SAEFeatureSweepResult:
    """Evaluate cumulative ranked SAE edits and same-count random rankings.

    Every trial executes independently and every step starts from the original
    activation. Output and run metric names and score directions are caller
    declarations. Run metrics can inspect explicitly requested post-intervention
    caches. Fit feature rankings, SAE parameters, code references and downstream
    readouts without using the evaluation trials.
    """

    if not isinstance(lens, EEGLens):
        raise ValidationError("SAE feature sweep requires an EEGLens")
    if not isinstance(batch, SignalBatch):
        raise ValidationError("SAE feature sweep requires a SignalBatch")
    batch.__post_init__()
    if not isinstance(site, str) or not site:
        raise ValidationError("SAE feature sweep requires a nonempty site name")
    declared = lens.adapter.require(site)
    if not declared.writable:
        raise ValidationError("SAE feature sweep requires a writable site")
    if not isinstance(sae, TopKSAE):
        raise ValidationError("SAE feature sweep requires a TopKSAE")
    if (
        not isinstance(feature_ranking, tuple)
        or not feature_ranking
        or any(
            type(feature) is not int or feature < 0 or feature >= sae.n_features
            for feature in feature_ranking
        )
        or len(set(feature_ranking)) != len(feature_ranking)
    ):
        raise ValidationError("SAE feature ranking must contain unique in-range indices")
    if (
        not isinstance(feature_counts, tuple)
        or len(feature_counts) < 2
        or feature_counts[0] != 0
        or any(type(count) is not int or count < 0 for count in feature_counts)
        or any(left >= right for left, right in zip(feature_counts, feature_counts[1:]))
        or feature_counts[-1] > len(feature_ranking)
    ):
        raise ValidationError(
            "SAE feature counts must start at zero, increase, and fit the ranking"
        )
    if not isinstance(metrics, Mapping) or any(
        not isinstance(name, str) or not name or not callable(metric)
        for name, metric in metrics.items()
    ):
        raise ValidationError("SAE feature sweep metrics require unique names and callables")
    run_metrics = {} if run_metrics is None else run_metrics
    if not isinstance(run_metrics, Mapping) or any(
        not isinstance(name, str) or not name or not callable(metric)
        for name, metric in run_metrics.items()
    ):
        raise ValidationError("SAE feature sweep run_metrics require unique names and callables")
    if not metrics and not run_metrics:
        raise ValidationError("SAE feature sweep requires at least one metric")
    if set(metrics) & set(run_metrics):
        raise ValidationError("SAE feature sweep metric names must be unique across metric kinds")
    if (
        not isinstance(cache_sites, tuple)
        or len(set(cache_sites)) != len(cache_sites)
        or any(not isinstance(name, str) or not name for name in cache_sites)
    ):
        raise ValidationError("SAE feature sweep cache_sites must be unique nonempty names")
    for cache_site in cache_sites:
        lens.adapter.require(cache_site)
    metric_items = tuple(metrics.items())
    run_metric_items = tuple(run_metrics.items())
    if mode not in {"ablate", "clamp"}:
        raise ValidationError("SAE feature sweep mode must be ablate or clamp")
    if mode == "clamp" and not isinstance(reference, SAECodeReference):
        raise ValidationError("Clamp sweeps require a fitted SAECodeReference")
    if mode == "ablate" and reference is not None:
        raise ValidationError("A code reference is used only for clamp sweeps")
    if not isinstance(selection, (Selection, AxisSelection)):
        raise ValidationError("SAE feature sweep selection is invalid")
    if type(random_draws) is not int or random_draws < 0:
        raise ValidationError("SAE feature sweep random_draws must be a nonnegative integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValidationError("SAE feature sweep seed must be an integer in [0, 2^63)")
    if execution_kwargs is not None and (
        not isinstance(execution_kwargs, dict)
        or any(not isinstance(name, str) for name in execution_kwargs)
    ):
        raise ValidationError("SAE feature sweep execution_kwargs must be a string-keyed dict")
    kwargs = dict(execution_kwargs or {})
    observed = _run_ranking(
        lens,
        batch,
        site,
        sae,
        feature_ranking,
        feature_counts,
        metric_items,
        run_metric_items,
        cache_sites,
        mode,
        reference,
        selection,
        kwargs,
    )
    generator = torch.Generator().manual_seed(seed)
    random_rankings = tuple(
        tuple(torch.randperm(sae.n_features, generator=generator).tolist())
        for _ in range(random_draws)
    )
    random_curves = [
        _run_ranking(
            lens,
            batch,
            site,
            sae,
            ranking,
            feature_counts,
            metric_items,
            run_metric_items,
            cache_sites,
            mode,
            reference,
            selection,
            kwargs,
            observed,
        )
        for ranking in random_rankings
    ]
    random_scores = {
        name: torch.stack([curve[name] for curve in random_curves])
        if random_curves
        else values.new_empty((0, *values.shape))
        for name, values in observed.items()
    }
    parameter_hashes = {
        f"{name}_sha256": tensor_digest(value) for name, value in sae.state_dict().items()
    }
    fractions = batch.data.new_tensor(feature_counts).div(sae.n_features)
    return SAEFeatureSweepResult(
        mode,
        site,
        feature_ranking,
        feature_counts,
        fractions,
        observed,
        random_scores,
        random_rankings,
        batch.trial_ids,
        seed,
        {
            "model_id": lens.model_id,
            "input_sha256": tensor_digest(batch.data),
            "sae": {
                "input_dim": sae.input_dim,
                "n_features": sae.n_features,
                "k": sae.k,
                **parameter_hashes,
            },
            "reference_sha256": tensor_digest(reference.values) if reference is not None else None,
            "output_metric_names": tuple(name for name, _ in metric_items),
            "run_metric_names": tuple(name for name, _ in run_metric_items),
            "metric_names": tuple(name for name, _ in (*metric_items, *run_metric_items)),
            "cache_sites": cache_sites,
            "selection": _selection_metadata(selection),
            "execution_kwargs": kwargs,
            "execution_batch_size": 1,
            "random_control": "seeded uniform permutations of the full SAE dictionary",
        },
    )
