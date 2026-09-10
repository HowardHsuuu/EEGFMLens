# Paired activation-patching sweeps

`patching_sweep` runs native donor replacement, location controls and identity checks through the public EEGLens interface. Supply two model-ready `SignalBatch` objects, an `EEGLens` model, and a pure score function. The score receives `(native_output, single_trial_batch)` and returns a finite Torch tensor of shape `[1]`. It can select a class logit, a frozen readout margin, or another prespecified scalar. No head fitting, preprocessing or data download is performed.

```python
from eeglens import patching_sweep

result = patching_sweep(lens, clean, recipient, score)
result.save("new-sweep.json")
```

By default it scans every writable channel/time site and every channel × time patch. Specify a smaller set of sites/targets for larger inputs. Trials are paired by unique IDs, and matching preprocessing, channel order, sample rate and input geometry are required. Deliberately relabel donor trial IDs yourself if your scientific pairing uses different source examples; the API never guesses correspondence.

## Annotated regions and controls

```python
from eeglens import Selection, SweepTarget, patching_sweep

region = SweepTarget(
    "event",
    Selection(sensors=("C3",), patches=(5, 6, 7)),
    off_event=Selection(sensors=("C3",), patches=(1, 2, 3)),
)
result = patching_sweep(
    lens,
    clean,
    recipient,
    score,
    sites=("blocks.2.output", "blocks.5.output"),
    targets=(region,),
    seed=91,
)
```

The target's donor values are restored without scaling. Off-event and random location controls have the same selected patch count and no overlap with the target. The caller is responsible for validating that an `off_event` selection is physiologically off-event using annotations. Automatic random controls translate the target's time-patch pattern, on the same sensors, to a disjoint location. Their seed depends on trial ID, target definition/name and the explicit seed, so row order and partitioning do not change the selected location. `random_position=Selection(...)` supplies a fixed control instead. With no disjoint location, the row is recorded as unavailable; no fallback to an overlapping control occurs.

Selections apply to every trial passed to one call. Group examples with common coordinates or make separate calls for different annotation-derived regions. All selected token operations exclude LaBraM CLS unless an unrestricted full-site plumbing replacement is explicitly requested through `recovery_site`.

`targets=()` is observation-only: it still records paired per-patch changes and identity checks but performs no event/location replacements. `random_controls=False` disables random controls. `recovery_site="blocks.11.output"` asserts that full-site donor replacement reproduces the clean native output; set this only for a known sufficient site. It is a plumbing assertion, not a physiological conclusion.

## Numerical contracts and records

Each trial is executed independently with batch size one, regardless of the input container's batch size. This bounds activation-cache memory and prevents batch-coupled models or scores from changing the comparison when input batches are partitioned. It is intentionally not accelerated batched inference. The lower-level public `MatchedReplacement` also computes each trial's norm independently when used in a genuinely batched native run.

For a selected control, scale the clean-minus-recipient activation delta to the target region's clean-minus-recipient Frobenius norm. Record the donor norm, target norm, actual applied norm, multiplier and status separately for every trial. The norm tolerance is `rtol=1e-4, atol=1e-6`. A zero control delta with nonzero target is unavailable. Near-zero targets (`<1e-8`) are uninformative. Numerical mismatches are invalid. These rows have null effect/score fields, rather than a fabricated zero effect. The low-level intervention leaves an unavailable row unchanged and exposes its diagnostics.

Multipliers above `multiplier_warning` (default 2) are retained and flagged, not clipped or automatically removed. A numerically valid match can still be scientifically implausible. Check these flags before interpreting location specificity. A warning threshold is an inspection aid, not a universal physiological validity criterion.

`SweepResult` contains:

- `baselines`: clean/recipient scores and input hashes, plus each site's `[channel][patch]` relative activation distances. Patch-only distances exclude CLS.
- `rows`: trial, site, target, control type, actual selections, reference selection, validity diagnostics, patched score, raw score delta and reduction in absolute clean-score error.
- `metadata`: model/checkpoint identity, execution kwargs, sites, seed, threshold and independent-trial execution policy.

Identity replacement is checked at every site; cleanup is checked by another recipient forward after the interventions. Failure raises instead of producing a completed sweep. `save(path)` writes schema `eeglens.sweep.v1` and refuses to overwrite an existing file. Lower-level `save_run` records matched-intervention diagnostics and reference selections too.

`result.summary(groups={trial_id: subject_id, ...})` reports equal-group means of valid clean-error reductions, the contributing counts and multiplier flags. Without `groups`, every trial is one group. This descriptive summary does not compute significance, turn correlated trials into independent samples, or automatically exclude large multipliers. For repeated trials, supply scientifically appropriate grouping and a prespecified statistical analysis.

## Evidence and limits

The offline [example](../examples/patching_sweep.py) has a known causal answer, including irrelevant coordinates. Tests verify both causal coordinates, exact intervention effects, donor permutation, per-trial norm matching, random-location partition invariance, CLS preservation, invalid controls, score shape, serialization and hook cleanup. The full phase experiment uses these public functions; see [sweep validation](sweep-validation.md) for the exact comparison against its preserved original results.

This interface does not infer that a region represents a physiological concept. Attention/convolution/normalization can distribute information across positions. Score recovery and patch distances remain distinct observations. Current real-model validation is local CPU float32 inference with CBraMod and LaBraM; hardware, gradient and compiled-mode support are unchanged.
