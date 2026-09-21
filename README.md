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

**Status: 0.1.0a19, public source alpha.** CPU/float32 is the validated native-model
execution target. This repository does not claim a PyPI release. See
[what changed in a19](docs/migration-a19.md).

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
Install `.[aperiodic]` only when FOOOF-based periodic/aperiodic fitting is needed.

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
- Measure and patch EEG frequency bands in amplitude, phase or full complex spectrum.
- Compute Welch PSDs, named time/frequency descriptors, channel correlation and
  band-limited PLI, PLV and magnitude-squared coherence.
- Fit held-out activation-to-amplitude-spectrum readouts with exact trial/patch row
  alignment, then decode activation or SAE directions into signed spectral changes.
- Fit a periodic/aperiodic decomposition on a reference cohort and reuse it as a
  phase-preserving signal intervention through the optional dependency group.
- Attribute objectives to input samples and declared internal sites with gradients,
  input × gradient, integrated gradients and path conductance.
- Propagate additive attribution into EEG frequency coordinates and evaluate it with
  progressive channel/patch or frequency-band perturbation curves.
- Propagate additive attribution into a caller-supplied EEG source solution while
  reporting forward reconstruction and attribution-conservation error.
- Fit held-out layer-wise ridge probes, Euclidean concept subspaces and
  covariance-aware LEACE erasers with same-rank random controls.
- Fit held-out ridge concept directions and measure objective sensitivity with
  native-site TCAV plus a random-label permutation null.
- Audit group variance and cross-group condition-direction consistency.
- Train Top-K sparse autoencoders, profile concept-related features, and ablate,
  steer or target-centroid clamp selected SAE codes.
- Evaluate ranked SAE features with cumulative ablation or clamping curves, multiple
  named output metrics and seeded random-feature rankings.
- Test a hypothesized source-to-mediator path with identity-controlled path patching.
- Compare trial-matched layers within or across models using linear CKA or RSA,
  with optional within-group centering to expose subject-identity similarity.
- Run paired patching sweeps with identity, location and norm-matched controls.
- Run an end-to-end activation-restoration workflow with reports and figures.
- Save outputs, activations and provenance in versioned local bundles.
- Connect another PyTorch model using `GenericAdapter` and a native forward callback.

These functions use the same `SignalBatch`, `Activation` and intervention contracts.
EEG-aware analyses are composed into a mechanistic study when its hypothesis needs
signal frequency, sensor or subject structure; they are not a separate workflow.

Supported families: **CBraMod, LaBraM, EEGPT, BIOT, BENDR, BrainOmni, CSBrain,
NeuroRVQ, SignalJEPA, DIVER-1 and ST-EEGFormer**. Coverage is component-specific;
some internal axes have no validated electrode/time mapping. BrainOmni and DIVER
require paired RNG in the tested paths. [Coverage and limitations](docs/model-coverage.md).

Individual heads, QKV editing, GPU/mixed precision, compiled execution and
cross-model activation transport are not validated public capabilities. Gradient
methods have analytic synthetic evidence; model-family-specific gradient support
must be established on each exact native component and checkpoint.

## Interpretability methods

EEGFMLens treats EEG-aware operations and general mechanistic methods as one
composable experiment API. For example, a frequency-band or phase edit produces a
traceable `SignalBatch`; the same batch can then be followed through activation
caches, probes, SAE features, restoration sweeps or a hypothesized circuit path.

| Question | Public methods |
|---|---|
| Which signal property changes the model? | periodograms, band power, phase-preserving band scaling, trial-matched amplitude/phase/complex spectral patching |
| Which named EEG properties are present? | Welch PSD, Hjorth/time descriptors, band power/entropy/centroid/edge, correlation, PLI, PLV, magnitude-squared coherence |
| What amplitude structure can an activation or feature direction predict? | aligned trial/patch spectral targets, held-out ridge readout, per-frequency R² and direction/band signatures |
| Does periodic or aperiodic structure carry an effect? | reference-fitted FOOOF decomposition and reusable component removal |
| Which input coordinates support an objective? | gradient, input × gradient, integrated gradients, channel/time aggregation, additive frequency attribution |
| Which estimated neural sources support an objective? | source-space propagation through a supplied source delta and EEG forward matrix, with inverse diagnostics |
| Is an attribution map behaviorally faithful? | progressive channel/patch occlusion, spectral-band removal, AOPC, cross-method cosine consistency |
| Where is information represented? | layer-wise ridge probes, group-variance decomposition, condition-direction consistency |
| Does the model use that representation? | Euclidean subspace removal, covariance-aware LEACE, same-rank controls, activation restoration |
| Is an objective locally sensitive to a named concept? | held-out ridge CAV, native-site TCAV, raw directional sensitivity, random-label null |
| Can a sparse feature mediate behavior selectively? | Top-K SAE training/metrics, concept profiles, feature ablation/steering/clamping, cumulative target/off-target curves and random-feature controls |
| Does an effect travel through a proposed path? | source-to-mediator path patching with identity controls |
| Do different models share trial geometry? | trial-ID-aligned linear CKA and RSA, optionally after within-group centering |

The implementations draw method definitions and controls from
[FMScope / *The Identity Trap*](https://arxiv.org/abs/2606.06647),
[BrainPEC / *What Do EEG Foundation Models Capture?*](https://arxiv.org/abs/2605.11410),
[*Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders*](https://arxiv.org/abs/2605.13930),
[*Beyond Accuracy*](https://arxiv.org/abs/2605.17562),
[EEG Foundation Models for BCI Learn Diverse Features of Electrophysiology](https://arxiv.org/abs/2506.01867),
[From Clever Hans to Scientific Discovery](https://arxiv.org/abs/2605.11885),
[EEG-PRISM](https://arxiv.org/abs/2608.13676), and
[EEG-Xplain](https://arxiv.org/abs/2609.15687). Circuit design also follows the
replacement-fidelity boundary made explicit by
[CLT-Forge](https://arxiv.org/abs/2603.21014). Cross-model comparison follows
[linear CKA](https://arxiv.org/abs/1905.00414); aperiodic intervention follows the
fit/remove design used by FMScope. Concept erasure follows
[LEACE](https://arxiv.org/abs/2306.03819) with an independently implemented,
reference-fitted low-rank affine map. Concept sensitivity follows the original
[TCAV formulation](https://proceedings.mlr.press/v80/kim18d.html). See the
[method guide](docs/interpretability.md) for exact semantics, controls, omissions
and clean-room implementation provenance.

An offline known-answer walkthrough connects a spectral intervention, a held-out
probe and a path test:

```bash
python examples/interpretability_methods.py
python examples/cross_model_analysis.py
python examples/concept_erasure.py
python examples/concept_attribution.py
python examples/feature_intervention_sweep.py
python examples/spectral_readout.py
python examples/source_attribution.py
```

## Learn more

- [Supported models and setup](docs/models.md) · [Input contracts](docs/input-contracts.md)
- [API](docs/api.md) · [Custom models](docs/custom-models.md)
- [Interpretability methods](docs/interpretability.md)
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
