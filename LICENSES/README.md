# Retained third-party material

EEGFMLens does not include upstream model implementations or pretrained weights.
One small piece of adapter metadata is retained:

| Package file | Source | Terms |
| --- | --- | --- |
| `src/eeglens/adapters/labram_channels.py` | LaBraM `utils.py`, `standard_1020` channel order at revision `c431221e6cfd23dbfa9950e0180682fb322b0548` | [LaBraM MIT license](LaBraM-MIT.txt), Copyright (c) 2024 Weibang Jiang |

Source: <https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/utils.py>

The ordered labels are stored locally so the adapter can map EEG channel names to
the checkpoint's learned embedding indices without importing upstream training
utilities. External source checkouts, dependencies, checkpoints and datasets
retain their own terms.
