"""Paired, trial-independent activation sweeps over native EEG models."""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

import torch

from .errors import ValidationError
from .interventions import Replacement, Selection
from .types import Activation, SignalBatch


@dataclass
class MatchedReplacement:
    """Match donor-delta norms separately for every trial, never across a batch.

    Invalid controls leave that row unchanged and report no valid intervention.
    Large multipliers are retained and flagged, never silently clipped.
    """

    site: str
    donor: Activation
    selection: Selection
    reference_selection: Selection | None = None
    multiplier_warning: float = 2.0
    diagnostics: list[dict] = field(default_factory=list, init=False)

    def apply(self, current, batch, layout, model_id):
        if not math.isfinite(self.multiplier_warning) or self.multiplier_warning <= 0:
            raise ValidationError("Multiplier warning threshold must be finite and positive")
        selected = Replacement(self.site, self.donor, self.selection).apply(
            current, batch, layout, model_id
        )
        delta = selected - current
        norm = torch.linalg.vector_norm(delta.flatten(1), dim=1)
        target = norm
        if self.reference_selection is not None:
            reference = Replacement(self.site, self.donor, self.reference_selection).apply(
                current, batch, layout, model_id
            )
            target = torch.linalg.vector_norm((reference - current).flatten(1), dim=1)
        unavailable = (norm < 1e-12) & (target > 1e-8)
        multiplier = (
            torch.ones_like(norm)
            if self.reference_selection is None
            else target / norm.clamp_min(1e-12)
        )
        multiplier = multiplier.masked_fill(unavailable, 0)
        changed = current + delta * multiplier.reshape((-1,) + (1,) * (current.ndim - 1))
        actual = torch.linalg.vector_norm((changed - current).flatten(1), dim=1)
        self.diagnostics = []
        for i, trial in enumerate(batch.trial_ids):
            missing = bool(unavailable[i])
            zero = float(target[i]) < 1e-8
            match = bool(torch.isclose(actual[i], target[i], rtol=1e-4, atol=1e-6))
            self.diagnostics.append(
                dict(
                    trial_id=trial,
                    status="unmatchable_zero_control"
                    if missing
                    else "zero_target"
                    if zero
                    else "matched"
                    if match
                    else "numerical_mismatch",
                    valid=not missing and not zero and match,
                    donor_delta_norm=float(norm[i]),
                    target_norm=float(target[i]),
                    actual_delta_norm=float(actual[i]),
                    multiplier=None if missing else float(multiplier[i]),
                    excessive_multiplier=not missing
                    and float(multiplier[i]) > self.multiplier_warning,
                )
            )
        if not torch.isfinite(changed).all():
            raise ValidationError("Non-finite matched replacement")
        return changed


@dataclass(frozen=True)
class SweepTarget:
    """An event/region and optional caller-defined location controls.

    Off-event semantics are supplied by the caller's annotations. Random controls
    may be explicitly fixed; otherwise the sweep samples a disjoint translation
    along time on the same sensors, with identical patch count and spacing.
    """

    name: str
    selection: Selection
    off_event: Selection | None = None
    random_position: Selection | None = None


def patch_grid(batch: SignalBatch) -> tuple[SweepTarget, ...]:
    """One target per channel/time patch, with no implicit CLS selection."""
    return tuple(
        SweepTarget(f"{channel}:patch{p}", Selection(sensors=(channel,), patches=(p,)))
        for channel in batch.channels
        for p in range(batch.data.shape[2])
    )


def _random_location(batch, target, trial_id, seed):
    patches = target.selection.patches or tuple(range(batch.data.shape[2]))
    offsets = sorted(p - min(patches) for p in patches)
    options = [
        tuple(start + p for p in offsets)
        for start in range(batch.data.shape[2] - max(offsets))
        if not set(start + p for p in offsets).intersection(patches)
    ]
    if not options:
        return None
    payload = json.dumps([seed, trial_id, target.name, target.selection.sensors, patches])
    rng = random.Random(int.from_bytes(hashlib.sha256(payload.encode()).digest(), "big"))
    return Selection(sensors=target.selection.sensors, patches=rng.choice(options))


def _single(batch, index):
    return replace(batch, data=batch.data[index : index + 1], trial_ids=(batch.trial_ids[index],))


def _score(score, output, batch):
    with torch.no_grad():
        value = score(output, batch)
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (len(batch.trial_ids),)
        or not torch.isfinite(value).all()
    ):
        raise ValidationError("Score must return one finite torch scalar per trial, shape [B]")
    return float(value[0])


def _patch_change(clean, recipient, batch):
    a, b = clean.tensor, recipient.tensor
    if clean.layout in {"tokens", "patch_tokens"}:
        prefix = int(clean.layout == "tokens")
        a, b = (
            a[:, prefix:].reshape(1, len(batch.channels), batch.data.shape[2], -1),
            b[:, prefix:].reshape(1, len(batch.channels), batch.data.shape[2], -1),
        )
    if a.ndim != 4:
        raise ValidationError("Patching sweep requires channel/time/feature sites")
    return (
        torch.linalg.vector_norm(b - a, dim=-1)
        / torch.linalg.vector_norm(a, dim=-1).clamp_min(1e-8)
    )[0].tolist()


@dataclass
class SweepResult:
    """JSON-safe per-trial records; no implicit cross-subject aggregation."""

    rows: list[dict]
    baselines: list[dict]
    metadata: dict

    def summary(self, *, groups: dict[str, str] | None = None):
        """Equal-group means of valid effects; report every excluded/flagged row.

        Supply subject IDs to avoid treating correlated trials as independent.
        With no mapping each trial is its own group. No significance inference.
        """
        grouped = {}
        for r in self.rows:
            if r["kind"] == "identity":
                continue
            key = (r["site"], r["target"], r["kind"])
            grouped.setdefault(key, []).append(r)
        result = []
        for (site, target, kind), rows in grouped.items():
            values = {}
            for r in rows:
                if not r["valid"]:
                    continue
                group = r["trial_id"] if groups is None else groups[r["trial_id"]]
                values.setdefault(group, []).append(r["clean_error_reduction"])
            means = [sum(v) / len(v) for v in values.values()]
            result.append(
                dict(
                    site=site,
                    target=target,
                    kind=kind,
                    rows=len(rows),
                    valid_rows=sum(r["valid"] for r in rows),
                    groups=len(means),
                    flagged_rows=sum(r.get("excessive_multiplier", False) for r in rows),
                    mean_clean_error_reduction=sum(means) / len(means) if means else None,
                )
            )
        return result

    def save(self, path):
        path = Path(path)
        with path.open("x") as f:
            json.dump(
                dict(
                    schema="eeglens.sweep.v1",
                    rows=self.rows,
                    baselines=self.baselines,
                    metadata=self.metadata,
                ),
                f,
                indent=2,
                allow_nan=False,
            )
            f.write("\n")


def patching_sweep(
    lens,
    clean: SignalBatch,
    recipient: SignalBatch,
    score: Callable,
    *,
    sites: Sequence[str] | None = None,
    targets: Sequence[SweepTarget] | None = None,
    random_controls: bool = True,
    seed: int = 0,
    multiplier_warning: float = 2.0,
    recovery_site: str | None = None,
    execution_kwargs: dict | None = None,
) -> SweepResult:
    """Run layer × target paired replacements with audited location controls.

    Each trial executes independently (B=1), bounding cache memory and avoiding
    model batch coupling. Pairing and random choices use trial IDs, not row order.
    Scores receive native output and the corresponding single-trial SignalBatch.
    Caches retain only the current pair. No task fitting or data preprocessing.
    """
    clean.__post_init__()
    recipient.__post_init__()
    if set(clean.trial_ids) != set(recipient.trial_ids):
        raise ValidationError("Clean and recipient trial IDs must match exactly")
    for attr in ["channels", "sampling_rate", "preprocessing_id", "unit", "stride"]:
        if getattr(clean, attr) != getattr(recipient, attr):
            raise ValidationError(f"Paired input {attr} mismatch")
    if clean.data.shape[1:] != recipient.data.shape[1:]:
        raise ValidationError("Paired input shapes differ")
    sites = (
        tuple(
            s.name
            for s in lens.sites()
            if s.writable and s.layout in {"bcpd", "spatial", "temporal", "tokens", "patch_tokens"}
        )
        if sites is None
        else tuple(sites)
    )
    targets = tuple(patch_grid(recipient) if targets is None else targets)
    if not sites or len(set(sites)) != len(sites):
        raise ValidationError("Nonempty unique sites required")
    if any(not t.name for t in targets) or len({t.name for t in targets}) != len(targets):
        raise ValidationError("Target names must be nonempty and unique")
    for site in sites:
        declared = lens.adapter.require(site)
        if not declared.writable or declared.layout not in {
            "bcpd",
            "spatial",
            "temporal",
            "tokens",
            "patch_tokens",
        }:
            raise ValidationError("Sweep requires writable channel/time/feature sites")
    # Check selector validity and equal, disjoint control supports before execution.
    grid = torch.empty(
        (1, len(recipient.channels), recipient.data.shape[2], 1), device=recipient.data.device
    )
    for target in targets:
        if any(
            selection is not None and not isinstance(selection, Selection)
            for selection in (target.selection, target.off_event, target.random_position)
        ):
            raise ValidationError("patching_sweep requires physical Selection, not raw tensor axes")
        mask = target.selection.mask(grid, recipient, "bcpd")
        for control in [target.off_event, target.random_position]:
            if control is not None:
                other = control.mask(grid, recipient, "bcpd")
                if int(mask.sum()) != int(other.sum()) or bool((mask & other).any()):
                    raise ValidationError(
                        "Location controls must be disjoint and have equal patch count"
                    )
    kwargs = dict(execution_kwargs or {})
    cache_sites = list(dict.fromkeys([*sites, *([recovery_site] if recovery_site else [])]))
    rows, baselines = [], []
    for index, trial in enumerate(recipient.trial_ids):
        donor_batch = _single(clean, clean.trial_ids.index(trial))
        batch = _single(recipient, index)
        donor = lens.run_with_cache(donor_batch, sites=cache_sites, **kwargs)
        corrupt = lens.run_with_cache(batch, sites=cache_sites, **kwargs)
        clean_score, corrupt_score = (
            _score(score, donor.output, donor_batch),
            _score(score, corrupt.output, batch),
        )
        baseline = dict(
            trial_id=trial,
            clean_score=clean_score,
            recipient_score=corrupt_score,
            clean_input_sha256=donor.metadata["input_sha256"],
            recipient_input_sha256=corrupt.metadata["input_sha256"],
            patch_changes={
                site: _patch_change(donor.cache[site], corrupt.cache[site], batch) for site in sites
            },
        )
        baselines.append(baseline)
        for site in sites:
            identity = lens.run_with_interventions(
                batch, interventions=[Replacement(site, corrupt.cache[site])], **kwargs
            )
            torch.testing.assert_close(identity.output, corrupt.output, rtol=1e-5, atol=1e-6)
            rows.append(
                dict(
                    trial_id=trial,
                    site=site,
                    target=None,
                    kind="identity",
                    status="passed",
                    valid=True,
                    clean_score=clean_score,
                    recipient_score=corrupt_score,
                    patched_score=_score(score, identity.output, batch),
                )
            )
            for target in targets:
                controls = [("event", target.selection, None)]
                if target.off_event is not None:
                    controls.append(("off_event", target.off_event, target.selection))
                if random_controls:
                    controls.append(
                        (
                            "random",
                            target.random_position or _random_location(batch, target, trial, seed),
                            target.selection,
                        )
                    )
                for kind, selection, reference in controls:
                    common = dict(
                        trial_id=trial,
                        site=site,
                        target=target.name,
                        kind=kind,
                        clean_score=clean_score,
                        recipient_score=corrupt_score,
                    )
                    if selection is None:
                        rows.append(
                            dict(
                                **common,
                                status="no_disjoint_random_location",
                                valid=False,
                                patched_score=None,
                            )
                        )
                        continue
                    edit = MatchedReplacement(
                        site, donor.cache[site], selection, reference, multiplier_warning
                    )
                    patched = lens.run_with_interventions(batch, interventions=[edit], **kwargs)
                    diagnostics = edit.diagnostics[0]
                    value = _score(score, patched.output, batch)
                    valid = diagnostics["valid"]
                    rows.append(
                        dict(
                            **common,
                            **{k: v for k, v in diagnostics.items() if k != "trial_id"},
                            sensors=selection.sensors,
                            patches=selection.patches,
                            reference_sensors=reference.sensors if reference else None,
                            reference_patches=reference.patches if reference else None,
                            patched_score=value if valid else None,
                            delta=value - corrupt_score if valid else None,
                            clean_error_reduction=abs(corrupt_score - clean_score)
                            - abs(value - clean_score)
                            if valid
                            else None,
                        )
                    )
        if recovery_site:
            full = lens.run_with_interventions(
                batch,
                interventions=[Replacement(recovery_site, donor.cache[recovery_site])],
                **kwargs,
            )
            torch.testing.assert_close(full.output, donor.output, rtol=1e-5, atol=1e-6)
        after = lens.run_with_cache(batch, sites=[], **kwargs)
        torch.testing.assert_close(after.output, corrupt.output, rtol=1e-5, atol=1e-6)
    return SweepResult(
        rows,
        baselines,
        dict(
            model=dict(lens.manifest),
            sites=sites,
            seed=seed,
            multiplier_warning=multiplier_warning,
            recovery_site=recovery_site,
            execution_kwargs=kwargs,
            execution_batch_size=1,
            identity_and_cleanup="passed",
        ),
    )
