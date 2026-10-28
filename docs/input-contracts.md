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

## Electrode coordinate sources

The development `load_elc_geometry` API binds native coordinates to an explicit
ASA ELC file and SHA-256. The caller supplies the exact required unit and names
the source coordinate frame. Labels match without case sensitivity; no channel
aliases, head-coordinate transformation, centering, normalization or unit
conversion is performed. The complete coordinate/label inventory is validated
before selecting electrodes in the requested order.

```python
from eegfmlens import DIVERAdapter, load_elc_geometry

geometry = load_elc_geometry(
    coordinate_file,
    channels=channels,
    expected_sha256=trusted_geometry_sha256,
    unit="mm",
    coordinate_frame="standard_1005_native",
)
adapter = DIVERAdapter(model, channels=channels, positions=geometry.as_tensor())
geometry_provenance = geometry.metadata()
```

This example assumes the checkpoint requires the declared native millimeter
coordinates. A file hash verifies identity, not whether that geometry is
scientifically appropriate. The caller must verify the model's source contract.
The coordinate frame label is a declaration; an ELC unit/label parser cannot
establish anatomical alignment. `as_tensor()` returns a fresh float32 tensor;
float64 may be requested explicitly. The geometry object stores immutable
coordinate tuples, while the adapter independently checks its cloned tensor
against mutation.

The loader does not import MNE or silently locate its bundled montage. An
explicitly selected MNE-distributed ELC file can be supplied with a trusted
hash. Raw EDF/BrainVision decoding and independent reader checks may still use
MNE outside EEGFMLens. Recording readers, cohort enrollment, event/window
choices and filtering recipes remain explicit upstream preparation; the
intervention tool consumes the resulting `SignalBatch` and model geometry.

## Adapter input grids

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
| DIVER EEG | 500 Hz | Contiguous 500-sample patches | Exact configured channel order and unchanged explicit xyz geometry. Block and encoder sites retain the special row/column but map named selections only to physical positions. |
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
selectors are supported only on verified physical layouts. The EEGPT
encoder-only adapter, BIOT, BENDR, SignalJEPA and BrainOmni conservatively
reject physical selectors. The separate EEGPT masked-reconstruction adapter
accepts time-patch selection at `predictor.combined`, but its learned summary
tokens have no electrode identity and reject sensor selection. ST-EEGFormer
encoder sites reject physical selectors; its separate native MAE decoder
adapter supports checked sensor/patch selection at named decoder sites.
DIVER permits them at physical embedding/head sites and at its augmented
block/encoder sites after the one-row/one-column native mapping is verified. Use
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


## Native target semantics and intervention evidence

For CSBrain waveform projection, `projection_order="input"` expresses the published
training loss's input-order target indexing. A physical C3 example can occupy
input/output index4 while its sorted encoder index is16. This option is separate
from encoder token ordering. Hidden-input and direct-native-output checks validate
the evaluation implementation; they cannot reconstruct the upstream training mask
or prove its training-log fidelity. Keep that limitation explicit in waveform studies.

A valid native-state replacement need not improve the scientific endpoint. Report
uncertain or oppositely signed effects together with state activity, matched active
controls, graph-null checks and uncertainty. A technical adapter pass is not a
learned-mechanism claim. Fresh native random models use the same adapter and physical
selection APIs; complete-seed/state provenance and study-level inference belong to
the consuming findings study, not to a second package intervention implementation.


When a study reuses measurements from an earlier package version, retain that
wheel identity and the original execution metadata. A later package's added
adapter or selection geometry does not relabel earlier measurements as having
used it. Compare the actually invoked adapter, coordinate mapping and intervention
code, and validate every newly executed native model against independent physical
support checks. A static comparison limited to unchanged code paths is useful
supporting evidence, not a replacement for native admission or scientific controls.


For long studies, package admission and successful process exit are separate
from complete scientific evidence. The consuming study should bind every
recorded output to its actual wheel and input contract, preserve interrupted
attempts, and verify model-state identity before comparing cohorts. Do not
change a running collector's installed dependencies during documentation or
release maintenance.


When moving a study's physical target, keep the input channel identities and
native ordering intact. Move target masking, readout and intervention/control
selections together. Any normalization fitted only on visible target samples
must use the new target's visible samples; relabeling an existing normalized
array is insufficient. Such study preprocessing checks do not substitute for
native checkpoint admission at the new physical support.


Native evaluation checks do not establish the training-mask provenance of a
released checkpoint. Report that provenance separately when it affects the
interpretation of masked reconstruction.
