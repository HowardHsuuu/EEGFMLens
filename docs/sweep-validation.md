# Public sweep validation — 2026-09-09

The public API is validated against a known causal answer and a complete reproduction of the existing CBraMod/LaBraM phase experiment. Its purpose is reliable, reusable intervention execution; the earlier inconclusive spindle-specific interpretation is unchanged.

## Goal-by-goal audit

| Requirement | Concrete evidence | Result |
|---|---|---|
| Layer × channel × time sweeps with paired EEG and caller score | `patching_sweep`, default `patch_grid`, explicit `SweepTarget`, paired scores and per-patch baseline changes | Public API implemented; no task-specific fitting or preprocessing in the runtime |
| Identity, off-event/random controls and matching strength | Identity at every site; disjoint equal-size controls; stable trial-ID random translations; public `MatchedReplacement` | Per-trial donor, target and actual delta norms, multipliers and validity/large-multiplier flags retained |
| Known causal sites and irrelevant sites | Analytic two-layer/two-channel grid with exact positive, negative and zero score effects; CLS-prefixed multi-channel token model | Every expected effect and patch coordinate passes |
| Batch/order invariance | Whole input vs split/reversed trials; reversed donor rows; direct batched norm matching with different norms per trial | Exactly equal sweep records; matching never uses a batch-wide norm |
| Invalid controls do not become fabricated results | Zero controls, zero targets, non-disjoint controls, score-shape errors, null invalid effects, grouped summaries and serialization | Tested; invalid records excluded from descriptive effects and large multipliers explicitly flagged |
| Full real-model reproduction through public interface | Fresh `research/phase/run.py` run with migrated `evaluate.py`, both pretrained and both random encoders | Every original response/intervention record and aggregate estimate exactly reproduced |
| User-facing entry point works outside editable sources | Wheel build; wheel-import assertion and executable known-answer example; wheel Python sources compared byte-for-byte | Passed using existing local cached build dependencies; no network or publication |

## Complete real-data comparison

The new run regenerated all 423 paired cases (141 windows, three replicates) and signal/QC records exactly. The reference artifacts were checked against their original SHA256 manifest before comparison. This was a full rerun, not an eight-case-only comparison.

| Model | Response records exactly equal | Intervention records exactly equal | Additional identity checks |
|---|---:|---:|---:|
| CBraMod pretrained | 10,575 | 25,380 | 10,575 |
| LaBraM pretrained | 10,575 | 25,380 | 10,575 |
| CBraMod random | 10,575 | 0 | 10,575 |
| LaBraM random | 10,575 | 0 | 10,575 |
| Total | 42,300 | 50,760 | 42,300 |

All subject-level summary values and intervals are exactly equal to the preserved pre-refactor results. All 50,760 intervention records have valid numerical norm matches. The new audit records flag 4,465 CBraMod and 4,301 LaBraM interventions with multipliers above two; these are not silently discarded or declared physiologically valid. The original phase result remains beyond-whole-window-spectrum sensitivity without sufficient evidence for spindle-specific morphology.

The preserved reference is the external workspace directory `research/eeglens_phase/run-v1`. The full public-API run is `research/eeglens_sweep/full-v1`, and the full comparison record is `research/eeglens_sweep/full-comparison.json`. Runtime source hashes captured during execution match the final runtime sources; all output artifact hashes were rechecked. The checked-in [comparison record](../research/phase/results/sweep-comparison.json) contains counts and verification outcomes, not EEG or weights.

## Tests and executable example

33 tests passed, including official checkpoint loading/identity, core runtime contracts, phase signal/statistics checks and eight sweep tests. An upstream LaBraM TorchScript deprecation warning remains; it does not prevent these CPU executions. The runnable `examples/patching_sweep.py` independently checks all 24 analytic grid effects and was also executed with imports forced to the built wheel.

Reproduce the complete analysis using the commands in [the phase README](../research/phase/README.md), then compare:

```bash
python research/phase/compare_sweep.py \
  --reference /absolute/path/to/preserved-original-phase-run \
  --rerun /absolute/path/to/new-public-sweep-run \
  --output /absolute/path/to/new-comparison.json
```

Test commands from the repository root:

```bash
EEGLENS_CBRAMOD_CHECKPOINT=/absolute/path/to/cbramod.pth \
EEGLENS_LABRAM_CHECKPOINT=/absolute/path/to/labram-base.pth \
python -m pytest -q tests research/phase/test_phase.py
python examples/patching_sweep.py
```

The currently tested real EEG is the existing single-central-channel DREAMS experiment; multi-channel grid/CLS coordinates are tested with analytic models and the native adapter contracts, not claimed as a new multi-channel physiological study. Runtime execution is local CPU float32; gradients, compiled models, GPU behavior and additional foundation models are outside this milestone. Independent-trial execution prioritizes pairing and comparison consistency over batching throughput. Statistical inference and off-event annotation validity remain caller responsibilities.
