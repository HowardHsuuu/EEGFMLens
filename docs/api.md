# API and runtime contracts

## Model and input

`supported_models()` returns eleven immutable `ModelSpec` records. Each contains
the upstream reference, verified component, input summary, variants and required
adapter options. `model_info(family)` resolves documented aliases.

`connect(model, family, variant=None, model_id=None, **adapter_options)` is the
recommended entry point for an already constructed and loaded supported model.
It checks model type, family, variant and option names, constructs the verified
adapter and records the integration identity in the run manifest. It does not
load weights, execute upstream code, preprocess EEG, call `.eval()` or move the
model. CBraMod and LaBraM additionally provide strict local checkpoint helpers.

Every `ModelSpec` also exposes `evidence`, a tuple of maintained validation tiers:
`contract-ci` for installed synthetic contract tests, `pinned-source-ci` for
direct tests against an exact clean upstream commit, and `checkpoint-evaluated`
for the documented local checkpoint campaign.

`EEGLens(model, adapter, model_id=None)` wraps an existing eval-mode PyTorch model. It does not call `.eval()`, move the model or change parameters. Official loaders do these before wrapping. Configure device/dtype first. Only eager CPU float32 has integration evidence in this alpha.

`SignalBatch(data, trial_ids, channels, sampling_rate, preprocessing_id, unit="model_scaled", patch_stride_samples=None, transforms=())` requires finite floating `[batch,sensor,patch,sample]` data. IDs/channels are tuples of nonempty unique strings. Omitted stride means nonoverlapping patches. This type does not filter, scale, rereference or reorder EEG. The preprocessing ID must identify the complete recipe, excluding experimental corruption so clean/recipient pairing remains possible. Trial IDs identify the same source epoch. Library signal edits append immutable `SignalTransform` records without changing the preprocessing identity.

`lens.sites()` returns declarations with name, native module path, layout,
tuple-output index and writability. `lens.capabilities()` returns a public,
immutable summary for each site, including valid selector kinds and invocation
counts. Custom adapters use `Adapter([ActivationSite(...)])`; override `validate`
and `forward` as needed. The generic `batch` layout does not infer sensor or patch
coordinates. Use whole-activation selection or `AxisSelection` for raw axes whose
meaning you have verified. Custom adapters are not automatically certified model support.

## Execution

```python
clean = lens.run_with_cache(batch, sites=["blocks.0.output"])
patched = lens.run_with_interventions(batch, interventions=[patch], sites=[])
```

`sites=None` caches all declared sites; `[]` caches none. Intervened sites execute even if uncached. By default each selected/intervened site must fire exactly once. Reused modules can declare `expected_calls` and a zero-based `call_index` on `ActivationSite`; only that invocation is cached or edited, and the total invocation count is checked. Results contain native `output`, `cache`, actual `calls`, `model_id`, unique `run_id` and `metadata`. Cache tensors are detached clones on the original device, **after intervention**. They are owned snapshots, not read-only tensors: modifying one intentionally changes later donors.

`run_with_cache` defaults to caching every declared site; `run_with_interventions`
defaults to caching none. Request downstream sites explicitly when an intervened
run will supply a donor for a later intervention:

```python
from eegfmlens import Replacement

induced = lens.run_with_interventions(batch, interventions=[patch], sites=["blocks.5.output"])
receiver_patch = Replacement("blocks.5.output", induced.cache["blocks.5.output"])
```

Executing an intervention at a site does not automatically put that site, or its
downstream states, in the returned cache. The cached donor above includes the
first intervention's effects.

Observation preserves the native output object. Interventions replace only the declared tensor of a tuple output. Owned hooks are removed in `finally`; existing hooks survive and remain the caller's responsibility. Do not overlap direct native calls with wrapped execution. One live wrapper per native model and one active run per wrapper are enforced.

Changes to parameter/buffer version, storage, device or module identity since wrapping cause errors. Create a fresh model/wrapper after training or conversion. This does not detect every `.data` mutation or arbitrary Python attribute change; do not mutate model configuration after wrapping. Custom model IDs are caller assertions; official loaders derive identity from checkpoint bytes and output configuration.

Generic kwargs must be JSON-serializable and match exactly between donor and recipient. Built-in adapters reject runtime kwargs and use declared unmasked paths. Cache and intervention calls use `torch.no_grad()`; `attribute` uses a separate scoped gradient path and does not populate parameter `.grad` fields. Training, persistent gradient retention, stochastic replay and chunked caches are not supplied. Select sites/batch sizes to fit memory.

## Interventions

`AxisSelection(axis=1, indices=(2, 5))` selects explicit indices of an exposed
activation tensor axis. Use it for native summary tokens or other internal
coordinates whose meaning you have verified in the model. Axis zero is reserved
for trial identity; negative axes, duplicate indices and out-of-range indices
are rejected. It does not infer electrode or time labels. With `Replacement`,
unselected recipient values remain intact and donor rows still align by trial ID.
The run records the axis, indices and real donor provenance. `Ablation`,
`SubspaceAblation` and `LEACEAblation` also accept this selector;
`patching_sweep` requires physical `Selection` targets and does not accept raw
axis coordinates.

```python
from eegfmlens import AxisSelection, Replacement

# For a verified [batch, native_token, feature] exposed site:
patch = Replacement(site, donor.cache[site], AxisSelection(axis=1, indices=(2, 5)))
result = lens.run_with_interventions(recipient, interventions=[patch])
```

```python
from eegfmlens import Ablation, Replacement, Selection, SubspaceAblation

selection = Selection(sensors=("C3",), patches=(1,))
patch = Replacement("blocks.0.output", clean.cache["blocks.0.output"], selection)
zero = Ablation("blocks.0.output", selection)
# basis: [D,rank], orthonormal columns; center: [D], fitted separately
erase = SubspaceAblation("blocks.0.output", basis, center, selection)
```

Fit covariance-aware concept erasure on reference rows, then reuse the fixed
operator at a declared activation site:

```python
from eegfmlens import LEACEAblation, fit_leace_eraser, fit_random_subspace_control

eraser = fit_leace_eraser(reference_features, reference_concepts)
leace = LEACEAblation("blocks.0.output", eraser, selection)
control = fit_random_subspace_control(
    reference_features,
    rank=eraser.rank,
    seed=17,
)
random = SubspaceAblation("blocks.0.output", control.basis, control.center, selection)
```

One intervention per site is allowed. Sensor/patch selectors take their Cartesian product across all trials/features. They select whole patches, not sample windows or exclusive receptive fields. LaBraM order is sensor-major, patch-minor. An unrestricted selection includes CLS; explicit sensor or patch selections exclude CLS, even if they name all sensors/patches.

Replacement matches donor rows by trial ID, allowing reordering or donor supersets. It requires identical site/model/layout, channel order, preprocessing ID, sampling rate, stride, patch length, units, remaining shape, dtype and device. No implicit casting, resampling or cross-model patching occurs. Subspace ablation computes `x - ((x-center) @ basis) @ basis.T`. `LEACEAblation` applies the fitted affine map along the final feature axis and records fit dimensions, numerical settings and parameter hashes. It rejects an `AxisSelection` over that feature axis because the resulting partial edit would not erase the fitted concept. Fit every basis, center or eraser on reference/training data separately.

## Metrics and storage

`restoration_sweep(lens, clean, recipient, ...)` is the higher-level default for
tensor-valued model outputs. It computes a separately executed clean reference
for every trial and scores negative normalized L2 output error before delegating
the controlled intervention grid to `patching_sweep`. See the
[flagship workflow](restoration-workflow.md).

`paired_effect(clean, recipient, patched, epsilon=1e-8)` expects scalar **higher-is-better** metrics; negate losses. It reports `delta=patched-recipient` and `recovery=delta/(clean-recipient)` only when the denominator exceeds epsilon. Recovery is not clipped; otherwise it is `None` with a reason. Callers own readouts, aggregation and statistical inference.

`save_run(run, "new-directory")` writes JSON and tensor files, refusing existing destinations. `load_run(...)` checks schema/SHA256, uses weights-only loading and returns CPU tensors. Supported outputs: tensors, JSON scalars, string-keyed dictionaries, lists and tuples. Starting with a7, list/tuple subclasses (including namedtuples) serialize as plain lists/tuples: values and order survive, but class identity and field names do not. This fixes an a6-and-earlier defect that could write an unreadable bundle for these subclasses. Other arbitrary Python classes are rejected. Failed saves clean temporary files.

Records contain checkpoint/source metadata where available, input hash, execution kwargs, coordinates, PyTorch version, selectors and donor/subspace hashes. They do **not** include raw inputs, fitted basis/center or preprocessing code; donor tensors must be saved separately unless included in the cache. Keep these artifacts for replay. Checksums detect changes, not authenticity.

## Attribution, perturbation, EEG concepts, alignment, probe, SAE and path APIs

`attribute(lens, batch, objective, ...)` computes raw gradient, input × gradient or
integrated gradients one trial at a time. Integrated gradients requires an explicit,
trial-aligned baseline. Requested site results use declared semantic layouts; the
integrated form is path conductance rather than total activation change times one
averaged gradient. `patch_attribution`, `channel_attribution` and
`temporal_attribution` aggregate input coordinates.

`spectral_attribution` maps additive input × gradient or integrated-gradient results
through an inverse-DFT basis and reports conservation error. `source_attribution`
uses a caller-supplied source delta and `[channel, source]` EEG forward matrix and
reports forward reconstruction plus attribution-conservation error; it does not solve
the inverse problem. `occlusion_curve` and
`spectral_perturbation_curve` evaluate ordered spatial/temporal or frequency targets
with full per-trial score curves and AOPC. `attribution_cosine_consistency` compares
aligned maps across methods. Definitions, leakage boundaries and native-evidence
limits are in the [interpretability method guide](interpretability.md).

`power_spectrum` and `band_power` measure patch- or trial-scope spectra.
`welch_power_spectrum` adds explicit Hann segment and overlap geometry.
`scale_frequency_band` and `patch_frequency_band` return new `SignalBatch` objects
that can be passed directly to `EEGLens`. Spectral donor rows align by trial ID.

`time_domain_features` and `spectral_features` return named `EEGFeatureSet`
records; `.matrix()` makes their aggregation explicit for probing. Sensor-space
relationships are available through `channel_correlation` and `band_connectivity`.
The optional `fit_aperiodic_decomposition` and
`remove_fitted_spectral_component` functions separate reference fitting from a
reusable periodic/aperiodic intervention.

`linear_cka`, `rsa_correlation` and `cross_model_similarity` compare paired trial
geometry. The high-level API requires `Activation` records, aligns exact trial-ID
sets and reports the model/site identities. A `groups` argument removes group means
before comparison; it does not estimate uncertainty or choose exchangeability blocks.

`fit_concept_direction` fits a binary ridge CAV using explicit disjoint train and
test masks and returns held-out balanced accuracy. `tcav_score` evaluates native-site
objective gradients along that direction with an explicit trial or position sampling
unit. `tcav_permutation_test` adds same-split random-label CAVs without rerunning the
model. It returns raw sensitivities and null scores; it does not infer subject-level
exchangeability or multiple-comparison correction.

`activation_matrix`, `fit_ridge_probe`, `layerwise_ridge_probe`,
`fit_cross_covariance_subspace`, `fit_leace_eraser`,
`fit_random_subspace_control`, `group_variance_decomposition` and
`within_group_contrast_consistency` provide representation diagnostics and
controlled erasure without implicit splits. `TopKSAE`, `train_sae`,
`sae_concept_profile`, `fit_sae_code_reference`, `SAEFeatureAblation`,
`SAEFeatureSteering` and `SAEFeatureClamping` expose sparse features, descriptive
concept rankings and residual-preserving interventions. `sae_feature_sweep` executes
a prespecified cumulative feature ranking with ablation or reference-code clamping,
retains `[trial, step]` scores for every named metric and evaluates seeded random
feature rankings at the same counts. Metric scale, direction and scientific meaning
remain caller declarations. `path_patch` composes
two activation replacements to test a declared source-to-mediator route. Exact
definitions and interpretation boundaries are in the
[interpretability method guide](interpretability.md).
