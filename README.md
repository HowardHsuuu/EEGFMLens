# EEGFMLens

[![Tests](https://github.com/HowardHsuuu/EEGFMLens/actions/workflows/tests.yml/badge.svg)](https://github.com/HowardHsuuu/EEGFMLens/actions/workflows/tests.yml)

Inspect and intervene on EEG foundation models with explicit sensor and time axes.

The Python package is **`eeglens`**. It runs native PyTorch models, caches module
outputs, and applies trial-paired replacements, ablations and patching sweeps.
Adapters cover eleven EEG model families within the [documented scope](docs/model-coverage.md).
Model implementations, checkpoints and data are supplied separately.

**Status: 0.1.0a10, source alpha.** CPU/float32 is the validated execution target.
This repository does not claim a PyPI release. See [migration from a9](docs/migration-a10.md).

## Install

Python 3.10+:

```bash
git clone https://github.com/HowardHsuuu/EEGFMLens.git
cd EEGFMLens
python -m pip install .
python examples/quickstart.py
```

Core dependencies are PyTorch and NumPy. The quickstart is offline and uses a
small synthetic model; no model code, weights or data are downloaded implicitly.

## Cache and patch

```python
from dataclasses import replace
import torch
from torch import nn
from eeglens import ActivationSite, Adapter, EEGLens, Replacement, SignalBatch

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

For real models, follow [model setup and checkpoint loading](docs/models.md).
CBraMod and LaBraM have strict checkpoint helpers accepting external constructors.
For the other families, wrap an already loaded native model with its adapter.
All adapters declare supported output paths; they do not resample or normalize EEG.

## What you can do

- Cache independent activation snapshots at declared module outputs.
- Replace donors by trial ID; select verified sensor/patch coordinates or explicit raw axes.
- Zero activations or erase a supplied feature subspace.
- Run paired patching sweeps with identity, location and norm-matched controls.
- Save outputs, activations and provenance in versioned local bundles.
- Connect another PyTorch model using `GenericAdapter` and a native forward callback.

Supported families: **CBraMod, LaBraM, EEGPT, BIOT, BENDR, BrainOmni, CSBrain,
NeuroRVQ, SignalJEPA, DIVER-1 and ST-EEGFormer**. Coverage is component-specific;
some internal axes have no validated electrode/time mapping. BrainOmni and DIVER
require paired RNG in the tested paths. [Coverage and limitations](docs/model-coverage.md).

Individual heads, QKV editing, gradients, GPU/mixed precision, compiled execution
and cross-model activation transport are not validated public capabilities.

## Learn more

- [Model setup](docs/models.md) · [Input contracts](docs/input-contracts.md)
- [API](docs/api.md) · [Custom models](docs/custom-models.md)
- [Patching sweeps](docs/sweeps.md) · [Known-answer example](examples/patching_sweep.py)
- [Real EEG example](docs/validation.md) · [Integration testing](validation/README.md)
- [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

Please use [GitHub issues](https://github.com/HowardHsuuu/EEGFMLens/issues) for bugs
and questions. Include the package version, native source/checkpoint versions and
an example that reproduces the issue.

Inspired by [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
and [WorldModelLens](https://github.com/Bhavith-Chandra/WorldModelLens).
The tool and native conformance tests live here; dataset-specific scientific
studies are maintained separately.

EEGFMLens code is MIT licensed. The retained LaBraM channel metadata has its
[own notice](THIRD_PARTY_NOTICES.md). External models, weights and datasets retain
their respective licenses.
