# Native integration validation

Core tests require only EEGLens and pytest. Native tests execute explicit upstream
source checkouts; checkpoint tests also require weights. No test downloads those
resources implicitly. Follow [model setup](../docs/models.md) to obtain pinned sources.

```bash
python -m pip install -e '.[dev,labram]'
EEGLENS_CBRAMOD_SOURCE=/path/to/CBraMod \
EEGLENS_LABRAM_SOURCE=/path/to/LaBraM \
pytest -m native
```

To add official checkpoint tests:

```bash
EEGLENS_CBRAMOD_SOURCE=/path/to/CBraMod \
EEGLENS_LABRAM_SOURCE=/path/to/LaBraM \
EEGLENS_CBRAMOD_CHECKPOINT=/path/to/cbramod.pth \
EEGLENS_LABRAM_CHECKPOINT=/path/to/labram-base.pth \
pytest -m 'native or integration'
```

The source fixtures check Git revisions and tracked Python changes before import.
Tests selected without the required source paths fail explicitly; unavailable
checkpoint paths are skipped. CPU float32 is the validated target.

The a10 migration was checked against the previous bundled implementations on
both official checkpoints: all 49 CBraMod and 37 LaBraM cached sites and selected
C3/patch interventions matched exactly. This was a two-model migration check,
not a fresh validation of all eleven models. Generated reports are not shipped
with the package repository.

## Other models

The [coverage table](../docs/model-coverage.md) and [input contracts](../docs/input-contracts.md)
identify supported native components and geometry. Generated checkpoint reports
and machine-specific experiment runners are intentionally excluded from this
tool repository. The pytest commands above are the maintained portable
source/checkpoint checks.

The normal public integration path is the package catalog and `connect(...)`,
documented in [model setup](../docs/models.md). The installed-wheel suite checks
that catalog and the adapter contracts for all eleven families on every supported
OS/Python combination. Only CBraMod and LaBraM currently run against pinned live
upstream source in CI; this distinction is reported explicitly.
