# Reasonable-scope expansion

User scope: connect as many reasonable EEG foundation encoders as practical; exclude language-model pipelines such as NeuroLM. Existing five families remain supported. Priority additions are BrainOmni, REVE, CSBrain, SignalJEPA, and the missing BENDR contextualizer. Other public EEG-only encoders will be assessed for checkpoint access and local feasibility; this is not an assertion that every published model has been covered.

## Current evidence

- BrainOmni tiny: strict official checkpoint loading (332 keys), 12 native encode sites × two batch sizes, identity and manual ablation agreement, reordered final-site donor recovery. Native eval has unconditional SDPA dropout; comparisons explicitly share RNG seed 9137. A CPU-only copy replaces the distributed communication import with torch.distributed, using the uninitialized single-process branch. Source is external, not vendored into EEGLens. Position/sensor-type metadata is cloned, mutation-checked and hashed in run metadata. Test positions are synthetic.
- CSBrain: official Google Drive pretrained file `1Z67je1HQhClyG9XER_zJzFMAKSVNmnF0`, strict loading after DDP prefix removal. Fourteen sites × two batch sizes passed. An actual C3-REF/patch-1 zero edit matched the expected location in the native region-sorted output. The adapter restores original sensor order in caches and restores native order on writes. Repeated shared embedding modules are not single-execution sites.
- REVE: official `brain-bzh/reve-base` revision `fa9a2163a4b7c0a42c8e28b56077ef9c368944dc` returned 401 anonymously and gated-repository 403 with the existing account. Public source is available in the official GitHub checkout; pretrained weights were not obtained. No random-weight result is counted as pretrained validation.
- SignalJEPA: official-maintainer full channel-embedding checkpoint downloaded at `braindecode/signal-jepa`, revision `51232ee0795a60e4378c17befe1e2ea5e94450c4`. Strict full checkpoint loading and 20 native conditions plus reordered donor recovery passed under Torch 2.14 using unchanged source modules, bypassing package-level imports of unrelated models. SignalJEPAAdapter is exported. Standard import also passed under a separate Torch/Torchaudio 2.8.0 and Braindecode 1.8.1 environment.
- BENDR contextualizer: official release fully downloaded (612,693,181 bytes), both native components strictly loaded, 22 site/batch conditions and reordered donor recovery passed. BENDRAdapter is exported alongside BENDREncoderAdapter.

Synthetic runtime checks do not validate downstream task accuracy or biological interpretations. Nine families have checkpoint-backed support; REVE is architecture-only and EEGMamba has experimental hook declarations but lacks a supported local runtime. These exceptions remain explicit in model-coverage.md.

## Additional coverage gate

The primary-source check also identified [NeuroRVQ](https://github.com/KonstantinosBarmpas/NeuroRVQ), which releases a standalone EEG foundation model, and [EEGMamba](https://github.com/wjq-learning/EEGMamba), whose official implementation requires Mamba. Both are being inspected rather than silently counted as unsupported or excluded. NeuroLM remains explicitly outside the user's requested scope. Models requiring unavailable CUDA kernels will be documented separately from missing weights; a rewritten substitute is not native validation.

## EEGMamba native execution gate (2026-09-09)

The inspected official `models/eegmamba.py` selects Mamba2 with `headdim=50` and `d_state=64`. Its `modules/config_mamba.py` enables `fused_add_norm=True`; `modules/mixer_seq_simple.py` requires Triton layer/RMS normalization and calls the fused kernel. [Mamba's installation documentation](https://github.com/state-spaces/mamba#installation) lists Linux for the core runtime and NVIDIA/CUDA for CUDA execution (AMD has a separate path). This Mac has no supported native execution environment for the inspected configuration. No lab connection or rented GPU is authorized.

An experimental `eeglens.adapters.eegmamba.EEGMambaAdapter` declares native output hooks for future validation on a supported machine, but is not exported or counted as pretrained support. Blocks return hidden and residual tensors and reverse token order after each block; block sites deliberately use batch layout. A CPU rewrite or disabling fused kernels is not claimed equivalent to native pretrained validation.

NeuroRVQ is now publicly exported and has passed 104 native conditions plus reordered donor recovery. Full-suite tests pass (39), and BrainOmni/CSBrain validation was rerun after the runtime invocation change.
