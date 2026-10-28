# Models and preprocessing

`SignalBatch` uses `[batch, sensor, patch, sample]` containers, but the required sampling rate, sample count and channel convention depend on the adapter and checkpoint. The two checkpoint helpers documented here use float32 `[B,C,P,200]` patches at 200 Hz. Other integrations include 128, 256 and 500 Hz models and continuous-input models; use the [per-model input contracts](input-contracts.md), not the 200 Hz recipe, for those models. The caller owns filtering, referencing, scaling, channel order and epoch definitions. Matching tensor shapes do not establish matching preprocessing.

EEGLens supplies local checkpoint loaders for CBraMod and LaBraM only. For the other nine verified families, construct and strictly load the native upstream model, then use `connect(...)`. The adapters do not download or bundle upstream implementations or checkpoints. [Model coverage](model-coverage.md) identifies the evidence behind each component; [validation](../validation/README.md) explains the current test layers. The generic adapter interface is not a universal checkpoint loader.

## One connection API

`supported_models()` is the source of truth for the eleven public families.
`model_info(name)` accepts documented aliases, and `connect(...)` validates the
variant and option names before constructing the adapter.

```python
from eegfmlens import connect, model_info, supported_models

for spec in supported_models():
    print(spec.family, spec.variants)

spec = model_info("ST-EEGFormer")
lens = connect(
    native_model.eval(),
    spec.family,
    channels=channels,
    channel_mapping=official_channel_mapping,
)
```

Model construction, checkpoint loading and EEG preprocessing remain explicit
because their upstream APIs and licenses differ. The connection surface is
consistent:

| Family | Native component | `connect` options | Input contract | Physical sensor/patch selection |
| --- | --- | --- | --- | --- |
| `cbramod` | pretrained encoder/reconstruction model | none | 200 Hz, 200 samples/patch | all declared sites |
| `labram` | pretrained student encoder | optional `output` | 200 Hz, 200 samples/patch, native channel names | all declared sites |
| `eegpt` | EEGTransformer encoder; optional native encoder/predictor/reconstructor | `variant="masked_reconstruction"` for the full pipeline | 256 Hz, contiguous native patches; explicit `mask_x` and `mask_y` for reconstruction | physical time patches at `predictor.combined`; summary tokens have no sensor identity |
| `biot` | BIOTEncoder | `channels` | 200 Hz, checkpoint channel vocabulary | no; raw axis only |
| `bendr` | encoder + contextualizer | `channels`; optional `variant="encoder"` | 256 Hz, ordered inputs | no; raw axis only |
| `brainomni` | tiny checkpoint `encode` path | `channels`, `positions`, `sensor_types` | 256 Hz, native geometry | no; raw axis only |
| `csbrain` | pretrained encoder | `channels` | 200 Hz, 200 samples/patch | all declared sites |
| `neurorvq` | raw four-branch backbone | `channel_vocabulary` | 200 Hz, 200 samples/patch | all branch sites |
| `signaljepa` | local + contextual encoder | none | model rate/channel order | no; raw axis only |
| `diver` | EEG features/reconstruction | `channels`, `positions`; optional `output` | 500 Hz, 500 samples/patch | embedding/final physical sites |
| `steegformer` | small `forward_features` path | `channels`, `channel_mapping` | 128 Hz, official channel IDs | no; raw axis only |

After connection, `lens.capabilities()` lists the exact sites, layouts,
writability, invocation counts and allowed selectors for that model instance.
This is more precise than assuming a family-wide token meaning.

For EEGPT masked reconstruction, pass the strictly loaded upstream encoder,
target encoder, predictor and reconstructor as one model. The full-pipeline
variant requires ordered, disjoint physical masks for every run and exposes
`predictor.combined` as batch/time-patch/learned-summary/feature. Use
`Selection(patches=(...))` for physical time-window edits there; sensor
selection is rejected because the learned summary tokens are not electrodes.
Its native target is a per-patch normalized waveform, so error estimates must
be reported separately from other models' raw or standardized waveform errors.

```python
for site in lens.capabilities():
    print(site.name, site.selectors)
```

The upstream repository and tested reference for every family are available in
`model_info(...)`. A Git revision identifies the source used for validation; it
does not cause EEGLens to fetch or certify the caller's checkout.

## External source setup

Install dependencies and obtain the tested revisions explicitly:

```bash
python -m pip install '.[labram]'
git clone https://github.com/wjq-learning/CBraMod.git /path/to/CBraMod
git -C /path/to/CBraMod checkout b9e961003214326972c567eff390e75b0287e32a
git clone https://github.com/935963004/LaBraM.git /path/to/LaBraM
git -C /path/to/LaBraM checkout c431221e6cfd23dbfa9950e0180682fb322b0548
```

Replace `/path/to/...` with directories you choose. Run your script with the
relevant checkout on the Python import path, for example:

```bash
PYTHONPATH=/path/to/CBraMod python my_cbramod_analysis.py
PYTHONPATH=/path/to/LaBraM python my_labram_analysis.py
```

On Windows PowerShell, set `$env:PYTHONPATH` to that directory before invoking
Python. Use trusted checkouts: importing a constructor executes upstream code.
EEGLens never clones, installs or imports those implementations automatically.
The repository's real-EEG example accepts `--upstream` directly and verifies the
pinned revision. The package helpers accept named constructors from any explicit
installation, recording their source hash when available. A supplied
`source_revision` is caller-declared provenance, not automatic revision validation.

## CBraMod

Pinned [upstream](https://github.com/wjq-learning/CBraMod) source: `b9e961003214326972c567eff390e75b0287e32a`.

```python
from models.cbramod import CBraMod  # External CBraMod checkout on PYTHONPATH
from eegfmlens import load_cbramod

lens = load_cbramod("cbramod.pth", model_factory=CBraMod, output="features", device="cpu")
```

The official 12-block state dictionary is loaded strictly. `features` removes only the reconstruction projection after strict loading; output is `[B,C,P,200]`. `reconstruction` retains the projection. Neither is a task classifier. Site names are `waveform.code`, `embedding.output` and `blocks.i.{output,spatial.output,temporal.output,mlp.output}`. `waveform.code` exposes the native waveform convolution's channel-major positions as `[B,C,P,features]`, so `Selection(sensors=..., patches=...)` can target it without indexing native convolution axes. Spatial/temporal features have width 100 in this checkpoint; block features have width 200.

Weights: [pinned official Hugging Face file](https://huggingface.co/weighting666/CBraMod/resolve/500543c7e30bda1b22bfd51a49301b238dee21fd/pretrained_weights.pth), Apache-2.0 according to its repository metadata.

SHA256: `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`.

### CSBrain explicit physical masks

`CSBrainAdapter` also accepts `mask=` as a boolean `[batch, sensor, patch]`
array in `SignalBatch.channels` order. It permutes that mask into the native
region-sorted sensor order before calling the upstream model, which sorts
the input but does not sort its mask. Unmasked calls are unchanged. Returned
predictions remain exactly native; interpreting reconstruction target indices
requires checking the supplied checkpoint's training target contract separately
from the encoder cache layout. The adapter does not certify that scientific
contract or rescale microvolts automatically.

## LaBraM base

### Native pretraining prediction adapter (development)

`LaBraMPretrainingAdapter` accepts a caller-constructed, fully loaded native
`NeuralTransformerForMEM`, separately from the encoder-only `load_labram` helper.
It executes one explicit physical boolean mask `[batch, sensor, patch]` and returns
all patch logits `[batch, sensor*patch, vocabulary]` through the outer `lm_head`.
Call separately with a complementary mask for the native symmetric arm. Pass the
mask as a JSON-serializable list to `EEGLens.run_with_cache(..., mask=mask.tolist())`.

The native student block sites retain physical coordinates and gain
`normalization.output` (including CLS) and `prediction.output` (excluding CLS).
The student’s separately stored prediction head is not used by this native wrapper.
Tokenizer loading, target-code choice and task labels remain caller responsibilities;
this adapter neither fits a decoder nor certifies a checkpoint's training history.

Native masking occurs after TemporalConv. GroupNorm within that embedding can mix
information from raw target samples into other tokens before masking. Token masking
therefore does not guarantee raw-waveform hiding. Explicit raw occlusion requires
a separately identified input batch and changes the evaluation condition. Small
pinned-source tests cover both mask arms, support, replay and cleanup; full-checkpoint
admission remains separate. Previously released encoder behavior is unchanged.

### Encoder-only checkpoint helper

Pinned [upstream](https://github.com/935963004/LaBraM) source: `c431221e6cfd23dbfa9950e0180682fb322b0548`.

```python
from modeling_finetune import labram_base_patch200_200  # External LaBraM checkout
from eegfmlens import load_labram

lens = load_labram("labram-base.pth", model_factory=labram_base_patch200_200, output="patch_tokens")
```

Requires the `labram` extra. The loader extracts the pretrained student encoder, explicitly discards recognized pretraining heads and strictly loads every remaining key. The manifest lists discarded keys. It retains the learned `student.norm` using native `use_mean_pooling=False`; it does not silently initialize a new downstream `fc_norm` or classifier.

Outputs: `patch_tokens` → `[B,C*P,200]`; `all_tokens` → `[B,1+C*P,200]`; `pooled` → `[B,200]`, specifically **CLS**, not a mean, under this inference configuration. At most 16 patches are supported by the pretrained temporal positions. Channel labels must exactly match supported upstream channel names and fit its position table; unknown names raise errors.

Sites: `embedding.output` and `blocks.i.{output,attention.output,mlp.output}`. Blocks include CLS; patch embedding does not. Functional QKV is not a hookable native module output and is deliberately unavailable.

Weights: [official repository checkpoint](https://raw.githubusercontent.com/935963004/LaBraM/c431221e6cfd23dbfa9950e0180682fb322b0548/checkpoints/labram-base.pth). Upstream repository is MIT licensed; attribution for the retained channel mapping is in [LICENSES](../LICENSES/README.md).

SHA256: `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`.

This official checkpoint includes training metadata. Only for this exact hash,
loading allowlists the necessary NumPy scalar/dtype and argparse Namespace
types; it still uses `weights_only=True`. Unknown checkpoint files do not
receive this checkpoint-specific allowlist. Both loaders accept
`expected_sha256=` for explicit integrity enforcement.

## EEG demonstration

The real EEG example selects 19 EEGMMIDB sensors, filters 0.5–75 Hz, resamples to 200 Hz, scales microvolts by 1/100, and makes two four-second windows with one-second patches. It uses no rereferencing. This is an explicitly recorded demonstration recipe, not a claim to reproduce every upstream downstream-data pipeline. Sensor/time selectors describe activation coordinates, not exclusive physiological sources or receptive fields.

## DIVER native reconstruction

The same fully loaded native DIVER EEG model supports two distinct outputs.
`DIVERAdapter(..., output="features")` is the existing default. Select
`output="reconstruction"` for `y_org.time_head_output` with shape
`[batch, sensor, patch, 500]`. Load the complete native checkpoint including its
heads; an encoder-only checkpoint is insufficient.

```python
lens = connect(
    model.eval(), "diver", channels=channels, positions=positions, output="reconstruction"
)
mask = torch.zeros(batch.data.shape[:-1], dtype=torch.bool)
mask[:, channels.index("C3"), 1] = True
with torch.inference_mode(), torch.random.fork_rng(devices=[]):
    torch.manual_seed(9417)
    result = lens.run_with_cache(batch, mask=mask.tolist())
```

`True` masks a complete native input patch before embedding. Numeric masks,
wrong shapes and missing masks are rejected. Pair RNG when comparing runs:
upstream attention dropout remains active in evaluation, and the native mask
generator still consumes its ordinary RNG draw before the explicit override.
The temporary override is removed even if a downstream hook fails.

`blocks.*.attention.output` captures the projected native attention output
before the layer's dropout and residual addition;
`blocks.*.ffn.output` captures the feed-forward output before its residual
addition. These and `blocks.*.output` retain the native flattened
`(C+1)×(P+1)` grid, while `encoder.output` retains the unflattened grid.
Their `augmented_tokens` and
`augmented_grid` layouts map `Selection(sensors=("C3",), patches=(1,))` to the
physical C3/patch-1 position and exclude the prepended row and column.
`Selection()` without coordinates selects the entire native activation,
including the special positions. The adapter rejects an unexpected shape
instead of silently interpreting a different token geometry as EEG channels.

In reconstruction mode, `reconstruction.output` exposes the native time head
and supports sensor/patch selection. Its last axis is waveform samples, not a
latent feature basis. `features.output` is intentionally absent: the original
encoder `head` output does not feed the native time head. Consult
`lens.capabilities()` for each internal site's declared geometry. Output
replacement at a final head is a tool control, not evidence of a mechanistic
explanation.

Checkpoint-backed validation used official weights, synthetic three-sensor
inputs, batches 1/2 and paired CPU float32 runs. Thirty site/batch identity and
independent native-zero checks passed, including physical time-head selection
and cleanup. A separate pinned-upstream, random-weight test checks named
sensor/patch edits at attention, feed-forward, block and encoder outputs
against direct native hooks, including cleanup. The random-weight test is a
tool-contract check, not evidence of a scientific mechanism. Generated reports
and environment-specific runners are not shipped with the reusable package.


### Native LaBraM task loading and contextual codes

`load_labram_pretraining` and `load_labram_tokenizer` take an explicit local path,
a zero-argument native `model_factory`, and required `expected_sha256`. They load
the full state strictly; the pretraining loader retains both stored heads and
uses the outer head through `LaBraMPretrainingAdapter`. Only the byte-identical
official pretraining checkpoint may use `acknowledge_unused_logit_scale=True`;
the manifest records the omitted scalar and does not apply it to logits. Source
revision strings and factory hashes identify supplied code, not proof of native
equivalence or training provenance. Perform an independent native admission.

`LaBraMTokenizerAdapter` calls native `get_codebook_indices` and exposes encoder
sites. It requires an already initialized codebook and eval mode. The published
quantizer updates a usage statistic even in eval; this adapter lets that update
occur on a temporary clone and restores the original buffer object in `finally`.
Persistent codebook, counters and tensor versions remain unchanged; generic state
guards remain active. Do not call the native model concurrently. Codes depend on
context: the caller must declare which input generates labels and preserve that
choice across comparisons. Token masks do not guarantee raw-sample hiding.
