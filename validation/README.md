# Native integration validation

Core tests require only EEGLens and pytest. Native tests execute explicit upstream
source checkouts; checkpoint tests also require weights. No test downloads those
resources implicitly. Follow [model setup](../docs/models.md) to obtain pinned sources.

```bash
python -m pip install -e '.[dev,labram]'
EEGFMLENS_CBRAMOD_SOURCE=/path/to/CBraMod \
EEGFMLENS_LABRAM_SOURCE=/path/to/LaBraM \
pytest -m native
```

To add official checkpoint tests:

```bash
EEGFMLENS_CBRAMOD_SOURCE=/path/to/CBraMod \
EEGFMLENS_LABRAM_SOURCE=/path/to/LaBraM \
EEGFMLENS_CBRAMOD_CHECKPOINT=/path/to/cbramod.pth \
EEGFMLENS_LABRAM_CHECKPOINT=/path/to/labram-base.pth \
pytest -m 'native or integration'
```

The source fixtures check Git revisions and tracked Python changes before import.
Tests selected without the required source paths fail explicitly; unavailable
checkpoint paths are skipped. CPU float32 is the validated target.

The a10 migration was checked against the previous bundled implementations on
both official checkpoints: all 49 CBraMod and 37 LaBraM cached sites and selected
C3/patch interventions matched exactly. This was a two-model migration check,
not a fresh validation of all eleven models. The historical component reports
are now supplied separately; they do not turn that migration into an eleven-model rerun.

## Pinned-source catalog matrix

CI also checks EEGPT, BIOT, CSBrain, DIVER-1 and ST-EEGFormer one checkout per
job. To reproduce one job locally:

```bash
EEGFMLENS_NATIVE_MODEL=eegpt \
EEGFMLENS_NATIVE_SOURCE=/path/to/EEGPT \
pytest -q tests/test_native_catalog.py
```

Use the exact reference returned by `model_info(...)` and install that upstream
project's runtime dependencies. `tools/native_sources.py` rejects a different Git
HEAD or modified tracked Python before import.

## Current checkpoint suite for all eleven families

[Current executable checks](current_checkpoints/README.md) cover 14 native
component views and 638 site/batch records. They bind the current runtime,
runner and source/weight bytes, and include direct native baselines, identity,
independent intervention and cleanup controls. Follow that document for local
asset layout, dependencies, compatibility adjustments and replay commands.

```bash
python validation/verify_current_checkpoints.py
```

This command verifies retained evidence; the documented runners repeat model
execution. The [coverage reference](../docs/model-coverage.md) and
[input contracts](../docs/input-contracts.md) identify component and geometry
limits. Historical reports in [checkpoint_reports](checkpoint_reports/README.md)
remain separate and unchanged. Participant arrays are not distributed.

The public model connection path is the package catalog and `connect(...)`,
documented in [model setup](../docs/models.md). Installed-wheel tests cover all
eleven adapter contracts; seven families additionally use pinned upstream source
in CI. A checkpoint replay is a separate local check requiring supplied assets.
