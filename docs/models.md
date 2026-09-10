# Models and preprocessing

`SignalBatch` uses `[batch, sensor, patch, sample]` containers, but the required sampling rate, sample count and channel convention depend on the adapter and checkpoint. The two bundled loaders documented here use float32 `[B,C,P,200]` patches at 200 Hz. Other integrations include 128, 256 and 500 Hz models and continuous-input models; use the [per-model input contracts](input-contracts.md), not the bundled-loader recipe, for those models. The caller owns filtering, referencing, scaling, channel order and epoch definitions. Matching tensor shapes do not establish matching preprocessing.

EEGLens supplies local checkpoint loaders for CBraMod and LaBraM only. For the other nine verified families, construct and strictly load the native upstream model, then pass it to its exported adapter and `EEGLens`. The adapters do not download or bundle those upstream implementations or checkpoints. [Model coverage](model-coverage.md) identifies each verified component; [validation runners](../validation/README.md) document the tested construction and loading paths. The generic adapter interface is not a universal checkpoint loader.

## CBraMod

Pinned [upstream](https://github.com/wjq-learning/CBraMod) source: `b9e961003214326972c567eff390e75b0287e32a`.

```python
from eeglens import load_cbramod

lens = load_cbramod("cbramod.pth", output="features", device="cpu")
```

The official 12-block state dictionary is loaded strictly. `features` removes only the reconstruction projection after strict loading; output is `[B,C,P,200]`. `reconstruction` retains the projection. Neither is a task classifier. Site names are `embedding.output` and `blocks.i.{output,spatial.output,temporal.output,mlp.output}`. Spatial/temporal features have width 100 in this checkpoint; block features have width 200.

Weights: [pinned official Hugging Face file](https://huggingface.co/weighting666/CBraMod/resolve/500543c7e30bda1b22bfd51a49301b238dee21fd/pretrained_weights.pth), Apache-2.0 according to its repository metadata.

SHA256: `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`.

## LaBraM base

Pinned [upstream](https://github.com/935963004/LaBraM) source: `c431221e6cfd23dbfa9950e0180682fb322b0548`.

```python
from eeglens import load_labram

lens = load_labram("labram-base.pth", output="patch_tokens")
```

Requires the `labram` extra. The loader extracts the pretrained student encoder, explicitly discards recognized pretraining heads and strictly loads every remaining key. The manifest lists discarded keys. It retains the learned `student.norm` using native `use_mean_pooling=False`; it does not silently initialize a new downstream `fc_norm` or classifier.

Outputs: `patch_tokens` → `[B,C*P,200]`; `all_tokens` → `[B,1+C*P,200]`; `pooled` → `[B,200]`, specifically **CLS**, not a mean, under this inference configuration. At most 16 patches are supported by the pretrained temporal positions. Channel labels must exactly match supported upstream channel names and fit its position table; unknown names raise errors.

Sites: `embedding.output` and `blocks.i.{output,attention.output,mlp.output}`. Blocks include CLS; patch embedding does not. Functional QKV is not a hookable native module output and is deliberately unavailable.

Weights: [official repository checkpoint](https://raw.githubusercontent.com/935963004/LaBraM/c431221e6cfd23dbfa9950e0180682fb322b0548/checkpoints/labram-base.pth). Upstream repository is MIT licensed; see its retained notice.

SHA256: `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`.

This legacy checkpoint includes training metadata. Only for this exact hash, loading allowlists the necessary NumPy scalar/dtype and argparse Namespace types; it still uses `weights_only=True`. Unknown checkpoint files do not receive this legacy allowlist. Both loaders accept `expected_sha256=` for explicit integrity enforcement.

## EEG demonstration

The real EEG example selects 19 EEGMMIDB sensors, filters 0.5–75 Hz, resamples to 200 Hz, scales microvolts by 1/100, and makes two four-second windows with one-second patches. It uses no rereferencing. This is an explicitly recorded demonstration recipe, not a claim to reproduce every upstream downstream-data pipeline. Sensor/time selectors describe activation coordinates, not exclusive physiological sources or receptive fields.
