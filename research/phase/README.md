# EEGLens: phase and spindle morphology

A fixed-head follow-up to `research/spindle`, using the corrected DREAMS inputs and official CBraMod/LaBraM checkpoints. See [PROTOCOL.md](PROTOCOL.md) for the distinction between exact whole-window spectrum preservation and local segment preservation. No readouts are trained in this experiment.

From the repository root, in the same environment used for the spindle experiment:

```bash
python research/phase/run.py \
  --previous /absolute/path/to/corrected-spindle-run \
  --events /absolute/path/to/DREAMS/spindles \
  --cbramod /absolute/path/to/cbramod.pth \
  --labram /absolute/path/to/labram-base.pth \
  --work /absolute/path/to/new-phase-run
```

The destination must not already exist. Runs sequentially on CPU with two Torch threads. Dependencies are the spindle experiment dependencies plus matplotlib. Checkpoints and source EEG stay outside this repository. The parent run must include its artifact hash manifest, corrected inputs, saved raw-coordinate held-out heads, and all four encoder feature files.

Outputs:

- `prepared/signals.npz`: deterministic float32 paired inputs; `cases.json`: selection, seeds, exclusions; `quality.json`: signal constraints and morphology/distribution diagnostics.
- `results/*-responses.json`: each case, variant, layer and all 15 patch distances, plus fixed-head margins.
- `results/*-restorations.json`: event, off-event and random controls, actual activation norms, multipliers, margins, and unavailable-control status.
- `results/summary.json`: subject-level estimates and descriptive bootstrap intervals for every layer, including paired restoration comparisons.
- `results/overview.png`, `results/patches.png`: descriptive figures. Event-relative edge columns can contain fewer subjects because some patch coordinates fall outside the 15-second window.
- `provenance.json`, `artifact_hashes.json`: code, environment, checkpoint and artifact fingerprints.

The evaluation now uses public `SweepTarget` and `patching_sweep`; no private runtime access or local intervention implementation is needed. Compatibility tables retain the original fields, and `*-sweep-audit.json` additionally records identity checks, input-specific selections and richer per-trial diagnostics. Inspect `status`: unavailable, zero-target and numerical-mismatch interventions have null effect fields. Controls with excessive multipliers remain flagged even when their norm matches numerically. See [the sweep guide](../../docs/sweeps.md).

Interpretation concerns: local joins, changes in the sample-value distribution, time relocation, global attention receptive fields, only one random-weight seed, eight excerpts and no independent confirmation set. N2 prediction among selected N2 windows measures sensitivity, not classification accuracy or clinical utility. Any mechanism conclusion concerns the frozen encoder plus the saved staging head.

Tests:

```bash
python -m pytest tests research/phase/test_phase.py
```

The completed [results](RESULTS.md) support beyond-whole-window-spectrum sensitivity, while spindle-specific interpretation remains unsupported. See the [completion audit](AUDIT.md). Checked-in `results/summary.json`, `overview.png` and `patches.png` are aggregate outputs; the waveform example stays in the external work directory.

For an independent full signal regeneration and one-case-per-subject repeat of both pretrained models:

```bash
python research/phase/verify.py \
  --work /absolute/path/to/completed-phase-run \
  --previous /absolute/path/to/corrected-spindle-run \
  --events /absolute/path/to/DREAMS/spindles \
  --cbramod /absolute/path/to/cbramod.pth \
  --labram /absolute/path/to/labram-base.pth \
  --output /absolute/path/to/new-verification-directory
```
