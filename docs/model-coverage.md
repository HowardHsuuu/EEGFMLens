# Verified model coverage

The runtime has checkpoint-backed checks for eleven EEG model families. Scope matters: these are native encoder observation/intervention tests, not downstream accuracy validation or proof of EEG token semantics for every site.

| Family | Tested component / weights | Sites | Geometry |
|---|---|---:|---|
| CBraMod | Official pretrained model | Existing checkpoint and runtime suite | Sensor/patch and folded spatial/temporal branches |
| LaBraM | Official pretrained student encoder | Existing checkpoint and runtime suite | Channel-major patches with leading CLS |
| EEGPT | Official EEGTransformer code; Braindecode converted pretrained encoder weights | 10 | Window-folded attention, four trailing summary tokens |
| BIOT | Official encoder code; PREST 16-channel official weights and six-dataset 18-channel mirrored weights | 9 each | STFT sequence; no input-patch semantic mapping |
| BENDR | Official encoder + contextualizer; mirrored encoder weights and official contextualizer release | 6 encoder-only; 11 composed | Downsampled features; sequence-first context caches exposed batch-first |
| BrainOmni | Official tiny checkpoint, native encode | 12 | Latent sensor units; paired RNG required for native eval dropout |
| CSBrain | Official pretrained model | 14 | Original sensor ordering restored from region-sorted native activations |
| NeuroRVQ | Official pretrained backbone; raw four-branch features | 52 | Invocation-specific sites on shared blocks; channel-major patches with leading CLS |
| SignalJEPA | Full maintainer checkpoint including channel embeddings; native encoder forward | 10 | Local/contextual tokens; conservative batch layout |
| DIVER-1 | Official EEG checkpoint, full native state, muP configuration | 15 | Special channel/time tokens use batch layout; physical embedding/output axes retained |
| ST-EEGFormer | Official small checkpoint; strict pretrained encoder, native forward_features | 10 | Time-major tokens and CLS; conservative batch layout |

For the initial EEGPT, two BIOT checkpoints and BENDR encoder expansion, we tested batch sizes 1 and 2, exact equality with native observation, identity replacement, zero ablation against an independent manual native hook, finite outputs, hook cleanup and final-site clean recovery after trial reordering. All 68 site/batch conditions passed across the four new checkpoint configurations. Tests use deterministic synthetic model-ready inputs on CPU; no task labels or clinical claims are involved.

The initial expansion suite passed 45 tests, including CBraMod and LaBraM checkpoint checks and invocation-specific hook tests. That count is historical: the a6 installed-package gate passes 74 non-integration tests and the known-answer sweep in two local dependency environments; checkpoint/native runs remain separate evidence. The 68 conditions above describe the initial EEGPT/BIOT/BENDR encoder expansion; subsequent additions are reported separately below. See [current acceptance evidence](public-readiness.md) and [reproducible validation sources and results](../research/model_validation/README.md).

## Dense intervention evidence

Model-ready input constraints and independently checked boundaries are listed in [input contracts](input-contracts.md). Accepted metadata does not imply automatic preprocessing or universal checkpoint compatibility.

All eleven families / thirteen component-checkpoint views now have fixed-geometry dense rank-three native intervention evidence, totaling 796 site/batch/global-or-local conditions. See [dense validation scope](../research/model_validation/DENSE_VALIDATION.md). Three study models use real development EEG; other views use synthetic model-ready input. CPU float32 and the listed configurations are the verified scope, not all devices or model variants. Physical selectors are checked only where the adapter exposes that mapping; all other sites explicitly reject them.

## Expansion validation

BrainOmni tiny and CSBrain have additionally passed 24 and 28 site/batch conditions respectively. CSBrain also passed a sensor/patch-selection check that accounts for its native sensor permutation. SignalJEPA passed 20 conditions and reordered donor recovery both through unchanged direct-source modules (Torch 2.14) and standard Braindecode import (Torch 2.8). Complete BENDR passed 22 conditions and reordered donor recovery; REVE official weights currently return gated-access 403 with the available account.

BrainOmni uses the official native `encode` method, which excludes the final block. Its attention implementation applies dropout even in eval mode; comparisons were paired under the same RNG seed, not asserted to be deterministic across uncontrolled runs. The CPU-only source copy changes the unused distributed backend import from `deepspeed.comm` to `torch.distributed`; encoder arithmetic was not rewritten. Input positions are synthetic in this runtime test, not a real EEG/MEG preprocessing validation.

CSBrain uses the author's pretrained checkpoint after stripping the DDP `module.` prefix and strict loading. Cache tensors are mapped back from the native region-sorted sensor axis to the declared `SignalBatch.channels` order. Only single-execution sites are listed; shared repeatedly called embedding modules are excluded.

## Use a verified adapter

```python
from eeglens import EEGLens, EEGPTAdapter, BIOTAdapter, BENDREncoderAdapter

# Pass an already loaded native encoder, not an arbitrary downstream wrapper.
lens = EEGLens(eegpt_encoder.eval(), EEGPTAdapter(eegpt_encoder))

# Ordered channels must match the checkpoint's actual input convention.
lens = EEGLens(biot_encoder.eval(), BIOTAdapter(biot_encoder, channels=checkpoint_channels))
lens = EEGLens(bendr_encoder.eval(), BENDREncoderAdapter(bendr_encoder, channels=model_inputs))
```

Loading examples with strict state-dict checks are in `research/model_validation/validate.py`. Additional upstream code/weights stay external to the package and are not silently downloaded by these adapters. EEGPT requires 256 Hz and matching contiguous 64-sample patches in the tested configuration; BIOT uses 200 Hz; BENDR uses 256 Hz. Adapters validate declared input contracts but do not perform channel mapping, normalization or resampling.

EEGPT exposes `[B,window,token,feature]` caches after undoing the native window-folded batch axis. This uses the conservative `batch` layout, because four summary tokens are not electrodes. BIOT and BENDR likewise use `batch` layout. Whole-activation and feature-subspace interventions work; sensor/patch selections and sensor/time sweeps are not enabled for these layouts. That functionality needs an additional validated geometry mapping.

Shared BIOT per-channel embedding modules execute repeatedly; they are not listed as single-execution hook sites. Attention and feedforward branch outputs within the transformer execute once and are supported. BENDR now covers the native contextualizer composition, but not DN3 real EEG channel preparation or downstream classifiers.

NeuroRVQ passed 104 site/batch conditions, including independent native-hook ablation and unchanged outputs in the other three branches. Its upstream inference class declares downstream affine norms absent from the checkpoint; the validation replaces those with parameter-free identity and strictly loads the backbone after explicitly excluding unused pretraining heads, mask token and pretraining norm. This is raw backbone feature extraction, not validation of a pretrained downstream classifier. See `research/model_validation/validate_neurorvq.py`.

## Experimental integrations, excluded from the verified count

- `eeglens.adapters.reve.REVEAdapter`: 92 conditions passed on the official base architecture with random initialization. This checks native hooks/overlapping window execution only; official gated weights remain unavailable (403). It does not establish pretrained behavior. Not exported at package root.
- `eeglens.adapters.eegmamba.EEGMambaAdapter`: source-derived hook declarations for hidden/residual outputs; native Mamba2/Triton execution is unavailable on this Mac. No checkpoint validation. Not exported at package root. See [execution evidence](../research/model_validation/EXPANSION.md).

SignalJEPA loads every tensor of the full maintainer checkpoint strictly, including the pretrained 62-channel embedding table. Validation used unchanged Braindecode 1.8.1 source modules under Torch 2.14; package `__init__` imports were bypassed to avoid unrelated models requiring Torchaudio. This direct-source path is explicit and reproducible in the runner. Standard Braindecode 1.8.1 package import was also verified under Torch/Torchaudio 2.8.0 with the same checkpoint and all 20 conditions passing. The native forward uses the local encoder and transformer encoder; its decoder is loaded but not executed. No time-to-input-patch mapping is claimed.

`BENDRAdapter` accepts `torch.nn.Sequential(native_encoder, native_contextualizer)` after strict loading of both checkpoints. It preserves sequence-first strides on restoration: native conditioning uses a permuted convolution output, whereas transformer blocks emit contiguous sequence-first tensors. This avoids floating-point differences caused by changed kernel memory layouts; all comparisons still use zero tolerance. The full-composition runner is `validate_bendr_context.py`.

## DIVER-1 and ST-EEGFormer follow-up

DIVER-1 passed 30 site/batch conditions, reordered donor recovery, physical sensor/patch ablation, and recovery after an injected hook failure. Its native SDPA uses dropout in eval; all causal comparisons pair RNG seed 9417. It is the 500 Hz EEG checkpoint, not an iEEG preprocessing or checkpoint validation. Both special channel and special time positions remain in internal caches. The adapter restores upstream token-manager bookkeeping after exceptions, so a failed intervention does not poison the next run.

ST-EEGFormer-small passed 20 conditions and reordered donor recovery on six-second 128 Hz input. The official downstream `forward_features` path is retained with pretrained norm/CLS output and no classifier. Decoder-only checkpoint keys are explicitly excluded; the parent timm image `pos_embed` parameter is removed because the native EEG forward never uses it and it is absent from pretraining. All remaining model keys load strictly. Its time-major tokens are not given LaBraM's channel-major selectors.

[Reproduction, versions and checkpoint provenance](../research/model_validation/DIVER_STEEGFORMER.md). These add two verified families to the previous nine; no iEEG cross-modal result, downstream accuracy or biological interpretation is claimed.

## Feature-axis correction (2026-09-10)

BENDR convolution sites previously exposed native `[B,D,T]` tensors, while `SubspaceAblation` operates on the final feature axis. Whole-activation zero/identity checks did not detect that mismatch. Both BENDR adapters now expose convolution caches as `[B,T,D]`, and restore the original native layout for intervention. Context transformer sites already expose `[B,T,D]`. All 34 BENDR site/batch conditions were rerun with an additional independent manual erasure of two native feature channels; exact agreement passed. Regenerate old BENDR caches before feature-subspace analysis. This fixes one identified semantic defect; it is not a complete semantic certification of every site in all eleven models.
