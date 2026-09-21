# Input and coordinate contracts

`SignalBatch.data` contains already prepared floating-point tensors shaped
`[trial, sensor, patch, sample]`. EEGLens does not infer units, filters, channel
aliases, references or resampling. Record the actual recipe in
`preprocessing_id` and `unit`; matching strings are provenance checks, not
verification that preprocessing was performed correctly. Trial IDs and channel
names must be unique, nonempty strings matching their axes; data must be finite
and all axes nonempty.

`patch_stride_samples` describes the distance between successive patch starts;
the default equals the sample axis length. `Selection(patches=(k,))` selects a
whole patch by index, not a time in seconds. With original sampling rate `f`, its
nominal sample interval starts at `k * stride` and spans `patch_samples`;
absolute recording/event time must be retained by the caller. No sample-level
selector is implied.

| Adapter | Rate | Input patch/sample requirement | Sensor and extent contract |
| --- | ---: | --- | --- |
| CBraMod | 200 Hz | Exactly 200 samples per patch | Uses caller-provided sensor order; no vocabulary remapping. Native checks sample 1, 4 and 5 patches. |
| LaBraM | 200 Hz | Exactly 200 samples per patch | Exact uppercase native channel vocabulary; indices must fit checkpoint embeddings. Time count cannot exceed `max_patches` (16 in tested base checkpoint). |
| CSBrain | 200 Hz | Exactly 200 samples per patch | Exact configured sensor order, restored from native region sorting. Native checks sample 1, 4 and 5 patches. |
| EEGPT | 256 Hz | Patch size and sensor/window counts must equal native configuration; contiguous patches | Native channel lookup is used; no alias inference. Tested checkpoint has 64-sample windows. |
| BIOT | 200 Hz | Container patches concatenate into a continuous signal; stride must equal sample length; total samples at least native `n_fft` | Exact configured complete channel vocabulary/order. Native STFT tokens are not input patches. |
| BENDR encoder/context | 256 Hz | Container patches concatenate, with stride equal to sample length | Exact configured input channels, including any auxiliary scale input. No automatic DN3 preprocessing. Downsampled time is not input patch index. |
| BrainOmni | 256 Hz | Container patches concatenate, with stride equal to sample length | Exact configured channels, explicit native position/direction vectors and sensor-type IDs. Latent sensor units are not electrodes. |
| SignalJEPA | Native `sfreq` (128 Hz tested) | Container patches concatenate; contiguous; total length must survive every native convolution width/stride | Exact configured channel order and unchanged native channel-index mapping. |
| DIVER EEG | 500 Hz | Contiguous 500-sample patches | Exact configured channel order and unchanged explicit xyz geometry. Augmented tokens have no physical selector. |
| ST-EEGFormer | 128 Hz | Contiguous container patches; total length must be divisible by native patch size and fit native time vocabulary | Exact configured channels mapped through supplied native lookup. Tested patch size is 16. |
| NeuroRVQ | 200 Hz | Contiguous 200-sample patches; time count fits native vocabulary | Case-normalized channel lookup; no general alias mapping. Native time indices align to the end of its vocabulary. |

For CBraMod, LaBraM and CSBrain, the adapter currently validates patch sample
length but does not prohibit a nondefault stride. Such input is caller-defined;
the validated MI recipe uses contiguous patches. Accepted input metadata is not
evidence that an overlapping recipe matches a checkpoint's pretraining.

## Feature and physical selections

The final exposed activation axis is the feature axis used by
`SubspaceAblation`; its orthonormal basis has shape `[features, rank]`. Changing
a dense direction is not the same as deleting an input sensor. Sensor/patch
selectors are supported only on verified physical layouts. EEGPT, BIOT, BENDR,
SignalJEPA, BrainOmni and ST-EEGFormer conservatively reject physical selectors.
DIVER permits them at physical embedding/head sites only. Use
`lens.capabilities()` on the connected model to inspect each site's layout,
writability and permitted selectors.

## Current contract tests

The installed test suite exercises all eleven adapters with native-shaped
fixtures. It covers valid input routing, sampling rates, channel order and
geometry checks, selection boundaries, identity replacement, intervention
locality, hook cleanup and recovery after rejected inputs. It also checks
continuous-input repartitioning for BIOT, BENDR and SignalJEPA, geometry mutation
for BrainOmni, DIVER and ST-EEGFormer, and channel-alias collisions for EEGPT and
NeuroRVQ.

These fixture tests establish the package contract; they do not establish
downstream accuracy, clinical validity or universal checkpoint compatibility.
Checkpoint-backed component scope and current CI boundaries are stated in
[model coverage](model-coverage.md) and [validation instructions](../validation/README.md).
