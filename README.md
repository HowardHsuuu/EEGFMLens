# EEGLens

**Inspect and intervene on EEG foundation models, with explicit sensor and time axes.**

EEGLens runs native PyTorch models, caches intermediate activations, and applies paired replacements or ablations. It provides a common interface for eleven EEG model families: CBraMod, LaBraM, EEGPT, BIOT, BENDR (encoder and contextualizer), BrainOmni (tiny encode), CSBrain, NeuroRVQ, SignalJEPA, DIVER-1 (EEG) and ST-EEGFormer (small). Each adapter declares its native tensor layouts and supported output paths.

**Status: 0.1.0a9, public source alpha.** Installable from this source tree; no PyPI release is claimed. Official checkpoints and a small real EEG example have been exercised on CPU in float32. Inspired by TransformerLens; public readiness remains under [explicit audit](docs/public-readiness.md).

## Install

From this repository, with Python 3.10 or newer:

```bash
python -m pip install .
# Optional: LaBraM checkpoint loading and EDF examples
python -m pip install '.[labram,eeg]'
python examples/quickstart.py
```

Core requires PyTorch and NumPy. LaBraM adds timm/einops; EDF preprocessing adds MNE. Nothing downloads on import or during execution. The quickstart uses random weights and synthetic input to demonstrate execution, not pretrained behavior.

## Patch a native model

```python
from dataclasses import replace
import torch
from eeglens import CBraModAdapter, EEGLens, Replacement, SignalBatch
from eeglens.models import CBraMod

model = CBraMod(n_layer=2).eval()  # Offline example, random weights
lens = EEGLens(model, CBraModAdapter(model))
batch = SignalBatch(
    data=torch.randn(2, 3, 4, 200),  # trial, sensor, patch, sample
    trial_ids=("trial-1", "trial-2"),
    channels=("C3", "CZ", "C4"),
    sampling_rate=200,
    preprocessing_id="synthetic-v1",
)
site = "blocks.1.output"
clean = lens.run_with_cache(batch, sites=[site])
recipient = replace(batch, data=torch.zeros_like(batch.data))
patched = lens.run_with_interventions(
    recipient, interventions=[Replacement(site, clean.cache[site])]
)
torch.testing.assert_close(patched.output, clean.output)
```

For pretrained execution, use `load_cbramod("checkpoint.pth")` or `load_labram("checkpoint.pth")`, imported from `eeglens`. Loaders use local files, record SHA256, and strictly check encoder keys. See [model loading](docs/models.md) and the [real EEG recipe](docs/validation.md).

## Paired patching analysis

Use `patching_sweep(lens, clean, recipient, score)` to scan channel/time positions across native layers. The public API pairs donors by trial ID, executes trials independently, checks identity/cleanup, and records per-trial norm-matched location controls with explicit invalid/large-multiplier diagnostics. See the [sweep guide](docs/sweeps.md) and runnable [known-answer example](examples/patching_sweep.py).

## Available operations

- Inspect supported sites; cache selected activations as independent detached snapshots.
- Replace activations with donor rows matched by trial ID, validating model identity and preprocessing coordinates.
- Zero selected sensors/patches or erase a supplied orthonormal feature subspace.
- Preserve native computation and remove owned hooks after success or failure.
- Save outputs, activations and provenance in a versioned JSON/tensor bundle.
- Report raw paired effects; leave recovery undefined without meaningful baseline degradation.

| Model | Writable and observable sites | Exposed shape |
|---|---|---|
| CBraMod | Embedding; block, spatial, temporal and MLP outputs | `[B,C,P,D]`; branches restore folded axes |
| LaBraM base | Patch embedding; block, attention and MLP outputs | `[B,C*P,D]` at embedding; `[B,1+C*P,D]` after CLS insertion |

The table above describes the original two adapters with sensor/patch semantics. The [model coverage table](docs/model-coverage.md) records all eleven verified families, exact checkpoint/component scope and limitations. Other models expose conservative batch layouts where token-to-electrode/time mapping is not validated. BrainOmni and DIVER-1 comparisons require paired RNG because native attention applies dropout in eval mode.

All supported sites are module **outputs**. QKV, attention probabilities, individual heads, gradients, compiled execution, mixed precision, GPU execution, learned steering and cross-model transport are not validated capabilities. Unsupported sites raise errors. See [API contracts](docs/api.md).

## Repository scope

This repository contains the reusable tool, adapters, examples, tests and native
integration validation. Dataset-specific studies, scientific results and response
replay archives are maintained separately and are not release requirements.

Use [validation/](validation/README.md) for native-hook conformance checks and
[the acceptance ledger](docs/public-readiness.md) for verified platform and model scope.

## Documentation

- [API and runtime guarantees](docs/api.md)
- [Models and preprocessing](docs/models.md)
- [Reproduce validation](docs/validation.md)
- [Architecture](docs/architecture.md) · [Roadmap](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

Inspired by [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [WorldModelLens](https://github.com/Bhavith-Chandra/WorldModelLens). Pinned EEG model sources and attribution are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Code is MIT licensed. Weights and datasets retain their own terms; neither is included.

## Bring your own model

Use `GenericAdapter` with explicit hook sites and a native forward callback to connect other PyTorch models without editing the runtime. `inspect_modules` lists candidate paths. See [custom model integration](docs/custom-models.md) for input conversion, tensor layouts, structured outputs and validation. Checkpoint-backed checks cover eleven families, including the complete BENDR encoder/contextualizer composition. See [verified model coverage](docs/model-coverage.md) for exact weights, tested sites and limitations. Generic connectivity is not automatic architecture validation.
