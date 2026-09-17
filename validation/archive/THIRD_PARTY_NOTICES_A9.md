# Third-party code

`src/eeglens/_vendor/cbramod` contains the native CBraMod model and criss-cross transformer from https://github.com/wjq-learning/CBraMod at revision `b9e961003214326972c567eff390e75b0287e32a`, copyright 2025 Jiquan Wang, MIT licensed. The original license is included. The model import was made package-relative and its command-line demonstration removed. Model computation is unchanged.

CBraMod pretrained weights are separate artifacts; the official Hugging Face model card declares Apache-2.0. We do not bundle weights. Source-code licensing does not supersede checkpoint or dataset terms.

The bundled CBraMod transformer also contains helpers matching PyTorch's transformer implementation. `_get_activation_fn`, `_get_seq_len`, and `_detect_is_causal_mask` have exact AST matches against PyTorch 2.6.0 `torch/nn/modules/transformer.py`. `src/eeglens/_vendor/PYTORCH_LICENSE` retains the complete upstream [PyTorch v2.6.0 license](https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE), including its copyright notices and redistribution conditions. This reference documents verified code correspondence, not the original revision from which CBraMod was derived. CBraMod's native source remains unchanged; its separate MIT notice is retained.

`src/eeglens/_vendor/labram.py` and `labram_channels.py` are from LaBraM's `modeling_finetune.py` and channel order in `utils.py`, https://github.com/935963004/LaBraM at revision `c431221e6cfd23dbfa9950e0180682fb322b0548`, copyright 2024 Weibang Jiang, MIT licensed. `LABRAM_LICENSE` preserves the original terms. The model source is unmodified; the channel list is extracted without importing unrelated training utilities. The upstream model acknowledges BEiT-v2, timm, DeiT and DINO origins in its header.

For these acknowledged origins, the package also retains their license texts:

- BEiT v2 / Microsoft UniLM: `BEIT2_LICENSE` (MIT), reference revision `ca43e4cd19445a536f133bf2bc25b573b2f0c7c5`. The referenced `beit2/modeling_finetune.py` credits Copyright (c) 2022 Microsoft, by Zhiliang Peng. LaBraM has exact AST matches for `_cfg`, `DropPath`, `Mlp`, `Block.forward` and `PatchEmbed.forward` against this reference. LaBraM modifies the model for EEG; EEGLens retains LaBraM's model file unchanged.
- timm: `TIMM_LICENSE` (Apache-2.0), v0.4.12 reference revision `7096b52a613eefb4f6d8107366611c8983478b19`; vision-transformer source credits Copyright 2021 Ross Wightman. timm itself is an external dependency, not bundled here.
- DeiT: `DEIT_LICENSE` (Apache-2.0), reference revision `7e160fe43f0252d17191b71cbb5826254114ea5b`; model source credits Copyright (c) 2015-present, Facebook, Inc., all rights reserved.
- DINO: `DINO_LICENSE` (Apache-2.0), reference revision `7c446df5b9f45747937fb0d72314eb9f7b66930a`; vision-transformer source credits Copyright (c) Facebook, Inc. and its affiliates.

These reference revisions identify inspected snapshots, not proven original derivation commits. Retaining the acknowledged projects' licenses does not imply that every file from those projects is bundled. The package-level MIT license does not replace third-party notices or terms. Source correspondence and retrieved-file hashes are recorded in `research/model_validation/results/labram-upstream-provenance.json` in the source repository.
