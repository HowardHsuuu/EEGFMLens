# EEGFMLens

<p align="center">
  <img src="docs/assets/eegfmlens-hero.png" alt="EEG signals passing through a foundation model while a lens reveals a sparse internal causal circuit." width="100%">
</p>

[![Tests](https://github.com/HowardHsuuu/EEGFMLens/actions/workflows/tests.yml/badge.svg)](https://github.com/HowardHsuuu/EEGFMLens/actions/workflows/tests.yml)

Inspect and intervene on EEG foundation models with explicit sensor and time axes.

The Python package is **`eegfmlens`**. It runs native PyTorch models, caches module
outputs, and applies trial-paired replacements, ablations and patching sweeps.
Adapters cover eleven EEG model families within the [documented scope](docs/model-coverage.md).
Model implementations, checkpoints and data are supplied separately.

**Status: 0.1.0a11, public source alpha.** CPU/float32 is the validated execution target.
This repository does not claim a PyPI release. See [what changed in a11](docs/migration-a11.md).

## Install

Python 3.10+:

```bash
git clone https://github.com/HowardHsuuu/EEGFMLens.git
cd EEGFMLens
python -m pip install .
python examples/quickstart.py
python examples/model_catalog.py
```

Core dependencies are PyTorch and NumPy. The quickstart is offline and uses a
small synthetic model; no model code, weights or data are downloaded implicitly.

## Flagship workflow

Run a complete layer × sensor/time activation-restoration study with a known-answer
offline model:

```bash
python -m pip install '.[visualization]'
python examples/restoration_workflow.py --demo --output /tmp/eegfmlens-restoration
```

The output directory contains the full controlled sweep, an auditable manifest,
and a heatmap. The same workflow accepts pinned CBraMod or LaBraM source, a local
checkpoint, and public EEGMMIDB data. See the
[activation-restoration workflow](docs/restoration-workflow.md) for the research
question, controls, interpretation, and real-model command.

## Cache and patch

```python
from dataclasses import replace
import torch
from torch import nn
from eegfmlens import ActivationSite, Adapter, EEGLens, Replacement, SignalBatch

model = nn.Sequential(nn.Linear(200, 32), nn.GELU(), nn.Linear(32, 16)).eval()
lens = EEGLens(model, Adapter([ActivationSite("features", "2")]))
batch = SignalBatch(
    torch.randn(2, 3, 4, 200),
    ("trial-1", "trial-2"),
    ("C3", "CZ", "C4"),
    200,
    "synthetic-v1",
)
clean = lens.run_with_cache(batch, sites=["features"])
recipient = replace(batch, data=torch.zeros_like(batch.data))
patched = lens.run_with_interventions(
    recipient,
    interventions=[Replacement("features", clean.cache["features"])],
)
torch.testing.assert_close(patched.output, clean.output)
```

For real models, load the native model and checkpoint from its upstream project,
then use the common connection API:

```python
from eegfmlens import connect, model_info

print(model_info("biot"))
lens = connect(native_model.eval(), "biot", channels=checkpoint_channels)
for site in lens.capabilities():
    print(site.name, site.layout, site.selectors)
```

The catalog covers all eleven verified families and reports required adapter
options, input contracts, references and selection semantics. CBraMod and LaBraM
also have strict checkpoint helpers accepting external constructors. See
[model setup and checkpoint loading](docs/models.md).

## What you can do

- Cache independent activation snapshots at declared module outputs.
- Replace donors by trial ID; select verified sensor/patch coordinates or explicit raw axes.
- Zero activations or erase a supplied feature subspace.
- Run paired patching sweeps with identity, location and norm-matched controls.
- Run an end-to-end activation-restoration workflow with reports and figures.
- Save outputs, activations and provenance in versioned local bundles.
- Connect another PyTorch model using `GenericAdapter` and a native forward callback.

Supported families: **CBraMod, LaBraM, EEGPT, BIOT, BENDR, BrainOmni, CSBrain,
NeuroRVQ, SignalJEPA, DIVER-1 and ST-EEGFormer**. Coverage is component-specific;
some internal axes have no validated electrode/time mapping. BrainOmni and DIVER
require paired RNG in the tested paths. [Coverage and limitations](docs/model-coverage.md).

Individual heads, QKV editing, gradients, GPU/mixed precision, compiled execution
and cross-model activation transport are not validated public capabilities.

## Learn more

- [Supported models and setup](docs/models.md) · [Input contracts](docs/input-contracts.md)
- [API](docs/api.md) · [Custom models](docs/custom-models.md)
- [Flagship restoration workflow](docs/restoration-workflow.md) · [Patching sweeps](docs/sweeps.md)
- [Real EEG example](docs/validation.md) · [Integration testing](validation/README.md)
- [Contributing](CONTRIBUTING.md) · [Release process](docs/releasing.md) · [Changelog](CHANGELOG.md)

Please use [GitHub issues](https://github.com/HowardHsuuu/EEGFMLens/issues) for bugs
and questions. Include the package version, native source/checkpoint versions and
an example that reproduces the issue.

Inspired by [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
and [WorldModelLens](https://github.com/Bhavith-Chandra/WorldModelLens).
The tool and native conformance tests live here; dataset-specific scientific
studies are maintained separately.

EEGFMLens code is MIT licensed. Attribution for the small retained LaBraM channel
mapping is kept with its source and in [LICENSES](LICENSES/README.md). External
models, weights and datasets retain their respective licenses.
