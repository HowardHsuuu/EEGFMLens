# Model coverage and evidence

EEGFMLens has public adapters for eleven EEG model families. “Supported” means
that the named native component has an explicit input contract, declared hook
sites, native-output parity checks, intervention controls and failure tests. It
does not mean that every checkpoint, classifier, device or preprocessing pipeline
from that family is covered.

The exact connection options and current source references are available through
`supported_models()` and summarized in [model setup](models.md).

## What current CI proves

Every commit builds and installs the wheel on Linux, macOS and Windows with
Python 3.10 and 3.12. The installed suite exercises the runtime plus all eleven
adapter contracts with synthetic native-shaped fixtures. It checks site routing,
tensor layouts, input rejection, intervention locality, cleanup and the public
model catalog.

A separate CI job checks CBraMod and LaBraM against pinned, unmodified upstream
source checkouts. It does not download checkpoints. Official-checkpoint parity
for those two loaders is maintained as a local integration test requiring
explicit weight paths.

The other nine families have checkpoint-backed integration evidence summarized
below. Those integrations are not rerun by current public CI, so their scope is
reported separately from the installed-wheel and pinned-source jobs. Generated
reports and machine-specific runners are outside this package repository.

## Checkpoint-backed scope

| Family | Verified native component | Physical selection | Material limitation |
| --- | --- | --- | --- |
| CBraMod | official pretrained 12-block model | sensor/patch, including folded branches | no task classifier |
| LaBraM | official pretrained student encoder and learned final norm | channel-major sensor/patch tokens | no functional QKV hook; at most 16 native patches |
| EEGPT | EEGTransformer encoder with converted pretrained weights | none on summary/window token sites | 256 Hz configured geometry only |
| BIOT | PREST 16-channel and six-dataset 18-channel encoders | none on STFT sequence | checkpoint channel vocabulary must be declared |
| BENDR | convolutional encoder and encoder/contextualizer composition | none on downsampled/context sequence | no DN3 preprocessing or classifier |
| BrainOmni | tiny checkpoint native `encode` path | none on latent sensor units | synthetic geometry; pair RNG for native eval dropout |
| CSBrain | official pretrained encoder | sensor/patch after reversing native region sort | declared channel order only |
| NeuroRVQ | raw four-branch pretrained backbone features | sensor/patch on all branch invocations | pretraining heads and absent downstream norms excluded explicitly |
| SignalJEPA | local and contextual encoder with full channel embeddings | none on local/context features | decoder is loaded but not executed |
| DIVER-1 | official EEG features and native time reconstruction | embedding and final physical outputs | synthetic xyz geometry; pair RNG for native eval dropout |
| ST-EEGFormer | official small `forward_features` encoder | none on time-major token sites | no classifier or base/large variants |

All checkpoint-backed integration runs used eager CPU float32. The recorded
checks cover native observation, identity replacement, a separately implemented
zero-ablation oracle, finite outputs, hook cleanup and recovery after reordered
trial donors. Current fixture tests exercise dense feature-subspace semantics and
input-boundary failures for every exported adapter; see
[validation instructions](../validation/README.md) for executable commands.

## What is not claimed

The evidence does not establish downstream accuracy, clinical validity,
physiological source localization, universal token semantics or equivalence for
arbitrary model variants. Sites with `batch` layout permit whole-activation and
explicit `AxisSelection` edits only. Check `lens.capabilities()` on the actual
model instance before choosing a scientific intervention.

REVE and EEGMamba adapters remain experimental and are not part of the eleven:
REVE has architecture-only checks without accessible official weights, while the
inspected EEGMamba configuration requires an unavailable native CUDA/Triton path.
Neither is exported from the package root or returned by `supported_models()`.
