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
C3/patch interventions matched exactly. [Migration result](migration/external-parity-a10.json).
This is a two-model migration check, not a fresh validation of all eleven models.

## Other models

The [coverage table](../docs/model-coverage.md) and [input contracts](../docs/input-contracts.md)
identify supported native components and geometry. Standalone `validate_*.py`
runners preserve the original per-model assays. Several legacy runners still
expect the external workspace layout described in their archived execution notes;
they are developer integration tools, not the installation path for the package.
The modern two-model source/checkpoint commands above are portable.

Older results and milestone write-ups live in [archive/](https://github.com/HowardHsuuu/EEGFMLens/blob/main/validation/archive/README.md).
JSON/XML records are retained byte-for-byte; paths and hashes describe the original
execution. They do not claim that old code was rerun after relocation.
