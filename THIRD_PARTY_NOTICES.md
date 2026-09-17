# Third-party notices

EEGFMLens does not bundle upstream model implementations or pretrained weights.
It retains the following adapter metadata:

| File | Source | License |
| --- | --- | --- |
| `src/eeglens/adapters/labram_channels.py` | LaBraM `utils.py`, `standard_1020` channel order, revision `c431221e6cfd23dbfa9950e0180682fb322b0548` | MIT, Copyright (c) 2024 Weibang Jiang |

Source: https://github.com/935963004/LaBraM/tree/c431221e6cfd23dbfa9950e0180682fb322b0548

The ordered labels were extracted without importing the upstream training utilities.
The complete copyright and license text is in [LICENSES/LaBraM-MIT.txt](LICENSES/LaBraM-MIT.txt),
included in both the source distribution and wheel.

The package-level MIT license does not replace third-party terms. External source
checkouts, dependencies, checkpoints and datasets retain their own licenses.
