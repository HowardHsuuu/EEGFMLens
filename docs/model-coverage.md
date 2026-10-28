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

A separate CI matrix checks CBraMod, LaBraM, EEGPT, BIOT, CSBrain, DIVER-1 and
ST-EEGFormer against pinned, unmodified upstream source checkouts. Every job
compares direct native output with EEGLens observation, checks every declared
site executes, verifies identity replacement, matches zero ablation to an
independent native hook, checks a nonzero effect, and verifies hook cleanup. It
does not download checkpoints. Official-checkpoint parity for the two packaged
loaders remains a local integration test requiring explicit weight paths.

All eleven families now have current-release checkpoint executions: 14 declared
component views and 638 site/batch records. Native observation, identity,
independent intervention and cleanup checks passed at batch sizes 1 and 2 in
CPU float32. See [runners and retained reports](../validation/current_checkpoints/README.md)
for component scope, exact assets, dependencies, compatibility adjustments and
replay commands. `python validation/verify_current_checkpoints.py` verifies the
retained reports against this runtime and its runners without loading models.

This local checkpoint suite is distinct from public source CI: BENDR, BrainOmni,
NeuroRVQ and SignalJEPA are not added to the seven-job upstream CI matrix.
The 13 historical views and 488 records in `validation/checkpoint_reports` remain
unchanged, with a separate verifier and six-study native summary.

## Checkpoint-backed scope

| Family | Verified native component | Physical selection | Public CI evidence | Material limitation |
| --- | --- | --- | --- | --- |
| CBraMod | official pretrained 12-block model, including explicit-mask waveform reconstruction | sensor/patch, including folded branches | contract + pinned source; masked-call native test | no task classifier |
| LaBraM | official pretrained student encoder and learned final norm | channel-major sensor/patch tokens | contract + pinned source | no functional QKV hook; at most 16 native patches |
| EEGPT | EEGTransformer encoder; separately, a strictly loaded native masked encoder/predictor/reconstructor checkpoint | time patches at `predictor.combined` in the masked variant; no electrode identity for summary tokens | encoder contract + pinned source; full-checkpoint local admission | 256 Hz configured geometry; the masked endpoint and time-only selection are separate from the encoder-only contract |
| BIOT | PREST 16-channel and six-dataset 18-channel encoders | none on STFT sequence | contract + pinned source | checkpoint channel vocabulary must be declared |
| BENDR | convolutional encoder and encoder/contextualizer composition | none on downsampled/context sequence | contract | no DN3 preprocessing or classifier |
| BrainOmni | tiny checkpoint native `encode` path | none on latent sensor units | contract | synthetic geometry; pair RNG for native eval dropout |
| CSBrain | official pretrained encoder | sensor/patch after reversing native region sort | contract + pinned source | declared channel order only |
| NeuroRVQ | raw four-branch pretrained backbone features | sensor/patch on all branch invocations | contract | pretraining heads and absent downstream norms excluded explicitly |
| SignalJEPA | local and contextual encoder with full channel embeddings | none on local/context features | contract | decoder is loaded but not executed |
| DIVER-1 | official EEG features and native time reconstruction | embedding, augmented block/encoder sites, and final physical outputs | contract + pinned source | special row/column excluded by named selections; synthetic xyz geometry; pair RNG for native eval dropout |
| ST-EEGFormer | official small `forward_features` encoder; separately, caller-supplied native MAE reconstruction model through `STEEGFormerMAEAdapter` | decoder sites reorder time-major tokens into physical sensor/patch coordinates; visible-only encoder sites remain whole-activation | encoder contract + pinned source; MAE native-source test | no classifier or bundled MAE checkpoint; study-checkpoint replay must be verified separately |

All checkpoint-backed integration runs used eager CPU float32. The report-specific
checks include native observation, feature/coordinate edits, identity replacement,
zero-ablation oracles, finite outputs, cleanup and reordered donor recovery.
Not every report asserts every check; consult its oracle and recorded fields.
In the historical reports for the four shared encoder runs (EEGPT, BIOT variants and BENDR encoder),
baseline comparison shares adapter input preparation. BrainOmni records its CPU
communication-import adaptation. Neither is labelled independent preprocessing
validation or unmodified upstream execution. Current fixture tests exercise dense feature-subspace semantics and
input-boundary failures for every exported adapter; see
[validation instructions](../validation/README.md) for executable commands.

## What is not claimed

The evidence does not establish downstream accuracy, clinical validity,
physiological source localization, universal token semantics or equivalence for
arbitrary model variants. Sites with `batch` layout permit whole-activation and
explicit `AxisSelection` edits only. Check `lens.capabilities()` on the actual
model instance before choosing a scientific intervention.

The distributed package contains adapters only for the eleven families in the
catalog. Architecture sketches without full validation are kept out of the
runtime and wheel.
