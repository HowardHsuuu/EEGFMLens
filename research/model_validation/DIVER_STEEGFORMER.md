# DIVER-1 and ST-EEGFormer integration

This follow-up adds DIVER-1's EEG checkpoint and ST-EEGFormer-small to EEGLens. The original nine-family completion audit remains a historical record. Current coverage is eleven families (twelve public model-specific adapters because BENDR has two component scopes).

## Official sources and weights

- DIVER-1 source: https://github.com/DIVER-Project/DIVER-1 at `fae4d7c5a58f2f795ce767939ad191d9c7ba49b8`. Git LFS media returned 404. The author's documented Drive folder provided file `1xuQMEEmuLY47bKA1zgMYwtUCl0SwdKEr`, saved externally as `expansion/diver1-eeg.pt`. Its 102,787,420 bytes match LFS SHA256 `dbfa48289989475a52719b1bcb868e62a82877120ac8272ca7dab772e407b891`. The `module` state in this DeepSpeed-format checkpoint loads strictly. The original `apply_mup` helper attaches native base-shape metadata before loading, and the native encoder uses `mup=True`. Optimizer states are not used.
- ST-EEGFormer source: https://github.com/LiuyinYang1101/STEEGFormer at `542ee17918c3c2c36ba1d4ea02bedff5eb149370`. Official small release: https://github.com/LiuyinYang1101/STEEGFormer/releases/download/ST-EEGFormer-small/checkpoint-300.pth (393,972,208 bytes), saved as `expansion/steegformer-small.pth`. The official channel mapping is `pretrain/senloc_file/sen_chan_idx.pkl`. Its hash and the exact excluded decoder keys are recorded in results. Native inference source is `easy_start/models_vit_eeg.py`, with `num_classes=0, global_pool=False`. The unused parent timm image-position parameter is removed; no encoder forward is rewritten.

Upstream source and weights remain outside the EEGLens package under `research/eeglens_model_validation/repos` and `expansion`. The package only supplies adapters and validation code. No checkpoint downloads occur on import or ordinary adapter execution.

## Reproduce

From the parent `bcilab` workspace, with EEGLens on PYTHONPATH and external files located as above:

```bash
PYTHONPATH=eeglens/src MPLCONFIGDIR=research/eeglens_build_evidence/mpl-cache research/eeglens_model_validation/.venv-braindecode/bin/python eeglens/research/model_validation/validate_diver_steegformer.py --model diver
PYTHONPATH=eeglens/src MPLCONFIGDIR=research/eeglens_build_evidence/mpl-cache research/eeglens_model_validation/.venv-braindecode/bin/python eeglens/research/model_validation/validate_diver_steegformer.py --model steegformer
```

The isolated environment uses Torch 2.8.0, Torchvision 0.23.0, timm 1.0.10, jaxtyping 0.3.11 and mup 1.0.0. The previously validated LaBraM environment is unchanged. Each model runs in a separate process to avoid collisions between upstream packages named `models`.

## Evidence and scope

Results `diver.json` and `steegformer.json` record exact observation, identity replacement and independent manual-zero agreement at every declared site for batches 1 and 2, plus finite outputs, scoped cleanup and donor recovery after trial reordering. DIVER has 15 sites (30 conditions); ST-EEGFormer has 10 (20 conditions). Source, adapter, runner, runtime and checkpoint hashes plus git revisions are recorded. Inputs are synthetic, using C3/C4/F3; DIVER uses five seconds at 500 Hz with synthetic coordinates, ST-EEGFormer six seconds at 128 Hz and official channel IDs.

DIVER additionally verifies C4/time-patch-1 ablation and cleanup of native token-manager state after a deliberately failing encoder hook. Its original attention passes nonzero SDPA dropout even under eval. Validation pairs RNG seed 9417 using `torch.random.fork_rng`, preserving native arithmetic and restoring the caller's RNG afterward. Uncontrolled repeated calls are not claimed deterministic. The adapter itself does not reset RNG for users. The original reconstruction head still executes as part of native forward, while the adapter exposes `y`, the original-position backbone features.

ST-EEGFormer exposes native `forward_features`, including its pretrained final norm and CLS selection. Token order is time-major, so internal caches deliberately use batch layout; physical sensor/time selection is not claimed. Classification heads, base/large variants and fine-tuning are not validated here. DIVER iEEG weights and cross-modal preprocessing are also not validated. The intent is trustworthy native observation/intervention integration, not task performance or evidence of a physiological mechanism.
