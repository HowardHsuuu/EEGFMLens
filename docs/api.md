# API and runtime contracts

## Model and input

`EEGLens(model, adapter, model_id=None)` wraps an existing eval-mode PyTorch model. It does not call `.eval()`, move the model or change parameters. Official loaders do these before wrapping. Configure device/dtype first. Only eager CPU float32 has integration evidence in this alpha.

`SignalBatch(data, trial_ids, channels, sampling_rate, preprocessing_id, unit="model_scaled", patch_stride_samples=None)` requires finite floating `[batch,sensor,patch,sample]` data. IDs/channels are tuples of nonempty unique strings. Omitted stride means nonoverlapping patches. This type does not filter, scale, rereference or reorder EEG. The preprocessing ID must identify the complete recipe, excluding experimental corruption so clean/recipient pairing remains possible. Trial IDs identify the same source epoch.

`lens.sites()` returns declarations with name, native module path, layout, tuple-output index and writability. Custom adapters use `Adapter([ActivationSite(...)])`; override `validate` and `forward` as needed. The generic `batch` layout supports only whole-activation selection. Custom adapters are not automatically certified model support.

## Execution

```python
clean = lens.run_with_cache(batch, sites=["blocks.0.output"])
patched = lens.run_with_interventions(batch, interventions=[patch], sites=[])
```

`sites=None` caches all declared sites; `[]` caches none. Intervened sites execute even if uncached. By default each selected/intervened site must fire exactly once. Reused modules can declare `expected_calls` and a zero-based `call_index` on `ActivationSite`; only that invocation is cached or edited, and the total invocation count is checked. Results contain native `output`, `cache`, actual `calls`, `model_id`, unique `run_id` and `metadata`. Cache tensors are detached clones on the original device, **after intervention**. They are owned snapshots, not read-only tensors: modifying one intentionally changes later donors.

Observation preserves the native output object. Interventions replace only the declared tensor of a tuple output. Owned hooks are removed in `finally`; existing hooks survive and remain the caller's responsibility. Do not overlap direct native calls with wrapped execution. One live wrapper per native model and one active run per wrapper are enforced.

Changes to parameter/buffer version, storage, device or module identity since wrapping cause errors. Create a fresh model/wrapper after training or conversion. This does not detect every `.data` mutation or arbitrary Python attribute change; do not mutate model configuration after wrapping. Custom model IDs are caller assertions; official loaders derive identity from checkpoint bytes and output configuration.

Generic kwargs must be JSON-serializable and match exactly between donor and recipient. Built-in adapters reject runtime kwargs and use declared unmasked paths. Calls use `torch.no_grad()`. Training, gradient retention, stochastic replay and chunked caches are not supplied. Select sites/batch sizes to fit memory.

## Interventions

`AxisSelection(axis=1, indices=(2, 5))` selects explicit indices of an exposed
activation tensor axis. Use it for native summary tokens or other internal
coordinates whose meaning you have verified in the model. Axis zero is reserved
for trial identity; negative axes, duplicate indices and out-of-range indices
are rejected. It does not infer electrode or time labels. With `Replacement`,
unselected recipient values remain intact and donor rows still align by trial ID.
The run records the axis, indices and real donor provenance. `Ablation` and
`SubspaceAblation` also accept this selector; `patching_sweep` requires physical
`Selection` targets and does not accept raw axis coordinates.

```python
from eeglens import AxisSelection, Replacement

# For a verified [batch, native_token, feature] exposed site:
patch = Replacement(site, donor.cache[site], AxisSelection(axis=1, indices=(2, 5)))
result = lens.run_with_interventions(recipient, interventions=[patch])
```

```python
from eeglens import Ablation, Replacement, Selection, SubspaceAblation

selection = Selection(sensors=("C3",), patches=(1,))
patch = Replacement("blocks.0.output", clean.cache["blocks.0.output"], selection)
zero = Ablation("blocks.0.output", selection)
# basis: [D,rank], orthonormal columns; center: [D], fitted separately
erase = SubspaceAblation("blocks.0.output", basis, center, selection)
```

One intervention per site is allowed. Sensor/patch selectors take their Cartesian product across all trials/features. They select whole patches, not sample windows or exclusive receptive fields. LaBraM order is sensor-major, patch-minor. An unrestricted selection includes CLS; explicit sensor or patch selections exclude CLS, even if they name all sensors/patches.

Replacement matches donor rows by trial ID, allowing reordering or donor supersets. It requires identical site/model/layout, channel order, preprocessing ID, sampling rate, stride, patch length, units, remaining shape, dtype and device. No implicit casting, resampling or cross-model patching occurs. Subspace ablation computes `x - ((x-center) @ basis) @ basis.T`. Fit basis/center on training data separately.

## Metrics and storage

`paired_effect(clean, recipient, patched, epsilon=1e-8)` expects scalar **higher-is-better** metrics; negate losses. It reports `delta=patched-recipient` and `recovery=delta/(clean-recipient)` only when the denominator exceeds epsilon. Recovery is not clipped; otherwise it is `None` with a reason. Callers own readouts, aggregation and statistical inference.

`save_run(run, "new-directory")` writes JSON and tensor files, refusing existing destinations. `load_run(...)` checks schema/SHA256, uses weights-only loading and returns CPU tensors. Supported outputs: tensors, JSON scalars, string-keyed dictionaries, lists and tuples. Starting with a7, list/tuple subclasses (including namedtuples) serialize as plain lists/tuples: values and order survive, but class identity and field names do not. This fixes an a6-and-earlier defect that could write an unreadable bundle for these subclasses. Other arbitrary Python classes are rejected. Failed saves clean temporary files.

Records contain checkpoint/source metadata where available, input hash, execution kwargs, coordinates, PyTorch version, selectors and donor/subspace hashes. They do **not** include raw inputs, fitted basis/center or preprocessing code; donor tensors must be saved separately unless included in the cache. Keep these artifacts for replay. Checksums detect changes, not authenticity.
