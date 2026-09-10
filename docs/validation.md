# Validation and reproduction

## What was exercised locally

The environment, 22-test count and S001R01 table below describe the initial validation snapshot. Current package gates are tracked in [public readiness](public-readiness.md); the current real-EEG local-intervention check is described next.

## Current real-EEG local-intervention check

A subsequent a8 validation used a newly created local Python 3.12 venv installed from the wheel with `[eeg,labram,dev]` through normal dependency resolution. Both real-EEG examples pass outside the checkout with the same exact local-intervention checks. See `research/model_validation/results/real-eeg-fresh-a8.json` and `fresh-environment-a8.json` for results, versions and provenance. This is a fresh environment on the same macOS ARM machine, not another platform or an independent native-study replication. The a7 records below remain historical.

`examples/real_eeg.py` now compares selected C3 / second-patch ablation with an independent raw-coordinate native hook. Both the downstream output and edited first-block cache must match exactly (`atol=rtol=0`); every unselected activation, including LaBraM CLS, must remain unchanged. The direct native forward bypasses the adapter's forward method. This checks intervention semantics on the loaded encoder, separately from the upstream-source comparison below.

Both official checkpoints pass on two four-second windows from local EEGMMIDB S001R04. The EDF SHA256 is `3d161f88e1c00632585287d2ce584c2bc0f08862438eb255ea8723e00fac693d`. The example filters 0.5–75 Hz, resamples to 200 Hz and uses microvolts divided by 100 without rereferencing. This differs from the unfiltered shared-input MI study; these windows are instrumentation examples, not labeled motor-imagery trial evaluations.

The same script also passed outside the checkout using Python isolated mode and the installed 0.1.0a7 wheel (macOS ARM, Python 3.12, torch 2.14.0, NumPy 2.5.3, MNE 1.13.0, scipy 1.18.1, timm 1.0.29). Dependency consistency passed. Import origin, example and result hashes, and versions are retained in [installed-example evidence](../research/model_validation/results/real-eeg-installed-a7.json). This environment was previously used for wheel validation and then received MNE; it is not a fresh-machine installation. The updated example postdates the a7 sdist.

Use a non-editable installation of the local wheel and its EEG/LaBraM extras in your environment, then run from any directory with absolute paths:

```bash
python -m pip install '/absolute/path/eeglens-0.1.0a7-py3-none-any.whl[eeg,labram]'
python -I /absolute/path/eeglens/examples/real_eeg.py \
  --model cbramod --checkpoint /absolute/path/cbramod.pth \
  --edf /absolute/path/S001R04.edf --output /absolute/path/new-result.json
```

For LaBraM, change the model to `labram` and supply its checkpoint. The output parent directory must exist, and an existing output file is rejected before model execution. No data or weights are downloaded by the example. This provides a bounded real-data instrumentation recipe; the full scientific study still needs its separately documented artifacts and environment.

## Initial validation snapshot

macOS arm64, CPU float32, Python 3.12.14, PyTorch 2.14.0, timm 0.4.12, NumPy 2.5.3, MNE 1.12.1. The default suite uses synthetic fixtures; two additional tests require official checkpoint paths. GitHub CI is configured separately, not reported as having run.

Local full suite: **22 tests passed**, including identity and selected ablation at every declared site in both official checkpoints. Ruff lint/format checks passed. Wheel and source distribution built successfully; retained vendor licenses are included. The wheel was also installed into a separate core-only environment.

The real-data recipe uses [EEGMMIDB 1.0.0 S001R01](https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf), available under the dataset's Open Data Commons Attribution license. EDF SHA256: `4743b736131a7e147c150e8b37711029b6cda5e356c4b3e8261a03cdcaaf8b0c`. Model URLs and hashes are in [models.md](models.md). Download those three public files separately into a local data directory, following their terms; the package performs no downloads.

From the repository, substituting your local paths:

```bash
python -m pip install -e '.[dev,labram,eeg]'
pytest -m 'not integration'
EEGLENS_CBRAMOD_CHECKPOINT=/path/cbramod.pth \
EEGLENS_LABRAM_CHECKPOINT=/path/labram-base.pth pytest
python examples/real_eeg.py --model cbramod --checkpoint /path/cbramod.pth --edf /path/S001R01.edf
python examples/real_eeg.py --model labram --checkpoint /path/labram-base.pth --edf /path/S001R01.edf
```

The original script asserted native/cache parity, identity replacement, final-block recovery, effective selected C3/patch-1 ablation, input corruption effect and cleanup. These checks use `atol=1e-6, rtol=1e-5`; observed differences are reported below. Current local-ablation oracle checks additionally require exact equality as described above.

| Official checkpoint, two EEG windows | CBraMod features | LaBraM patch tokens |
|---|---:|---:|
| Output shape | `[2,19,4,200]` | `[2,76,200]` |
| Native/cache maximum difference | 0 | 0 |
| Identity replacement difference | 0 | 0 |
| Full final-block recovery difference | 0 | 0 |
| Cleanup difference | 0 | 0 |
| Selected ablation maximum difference | 1.2155663 | 19.1415 |

Magnitude is model-specific and is not a comparative quality score. These are feature-level instrumentation results on two windows, not accuracy, robustness or physiological-mechanism findings. Broader subject/task and device coverage remains future work.

## Independent upstream forward

To test that vendored computation agrees with the original source, check out the exact revisions in [models.md](models.md) and run:

```bash
python examples/upstream_parity.py --model cbramod --upstream /path/CBraMod --checkpoint /path/cbramod.pth --edf /path/S001R01.edf
python examples/upstream_parity.py --model labram --upstream /path/LaBraM --checkpoint /path/labram-base.pth --edf /path/S001R01.edf
```

This imports the supplied local source and compares equal encoder weights and output configuration. LaBraM's weights are mapped by the audited loader before loading the independently imported encoder; this checks forward equivalence, not an independent checkpoint converter.

Both independent upstream comparisons produced maximum absolute difference **0** on the same two EEG windows.
