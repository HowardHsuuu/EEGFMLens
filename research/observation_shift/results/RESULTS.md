# Gain and readout pilot: preliminary results

These are exploratory results on the previously used eight-subject DREAMS cohort. They do not establish a foundation-model mechanism, genuine device generalization, or information loss. Internal correction and multichannel/independent-cohort experiments remain outstanding.

## Fixed historical head

| Model | Gain 0.5 | Gain 0.75 | Clean | Gain 1.5 | Gain 2 |
|---|---:|---:|---:|---:|---:|

| cbramod | 0.481 | 0.637 | 0.687 | 0.618 | 0.543 |

| labram | 0.593 | 0.592 | 0.600 | 0.597 | 0.616 |

| cbramod-random | 0.522 | 0.598 | 0.624 | 0.596 | 0.571 |

| labram-random | 0.582 | 0.601 | 0.610 | 0.600 | 0.598 |


## Paired effect of training-domain readout adaptation

Balanced-accuracy differences: augmented-head minus clean-trained head, with matched model/pooling/split/solver. Intervals are descriptive paired subject bootstraps (10,000 draws); no multiple-comparison claim. Augmented training sees gains 0.5 and 2 only.

| Model / pooling | Unseen gain 0.75 | Clean off-target effect | Unseen gain 1.5 |
|---|---:|---:|---:|

| cbramod / mean | +0.025 [-0.087, +0.129] | -0.053 [-0.122, +0.016] | +0.045 [-0.005, +0.104] |

| cbramod / flatten | -0.023 [-0.118, +0.068] | -0.060 [-0.134, +0.005] | -0.017 [-0.056, +0.026] |

| labram / mean | +0.022 [-0.013, +0.056] | -0.001 [-0.030, +0.024] | -0.008 [-0.049, +0.022] |

| labram / flatten | -0.004 [-0.032, +0.024] | -0.020 [-0.051, +0.012] | -0.021 [-0.041, -0.000] |

| cbramod-random / mean | -0.026 [-0.087, +0.026] | -0.062 [-0.120, -0.018] | -0.024 [-0.089, +0.038] |

| cbramod-random / flatten | -0.005 [-0.038, +0.030] | -0.018 [-0.043, +0.011] | -0.012 [-0.033, +0.008] |

| labram-random / mean | -0.001 [-0.032, +0.031] | -0.018 [-0.057, +0.023] | -0.005 [-0.039, +0.036] |

| labram-random / flatten | -0.014 [-0.028, -0.001] | -0.018 [-0.039, +0.005] | -0.017 [-0.045, +0.011] |


## Interpretation

CBraMod's historical fixed-head balanced accuracy falls from 0.687 at gain 1 to 0.481 at gain 0.5. A frozen-encoder mean-pooled probe trained using shifted training subjects reaches 0.696 at gain 0.5 on held-out subjects. This shows that the collapse of this particular fixed head is not sufficient evidence that the representation has lost all N2-discriminative information. It does not show complete information preservation.

The more demanding unseen-strength comparison is less decisive, and augmented training can harm clean-condition performance. Flattening all patch tokens does not automatically help on this small cohort. LaBraM's balanced accuracy stays near 0.60 while many individual decisions change: aggregate performance stability is not prediction invariance. Random-encoder controls also change with gain, so sensitivity alone cannot be attributed to pretraining.

Positive global calibration gain is invertible and is canceled numerically by per-window standardization (maximum input relative error below 8e-8). This is an input-equivalence check, not a measured improvement in normalized-model task accuracy; normalization may itself remove useful amplitude information. Known gain inversion is a trivial calibration baseline.

Next: test a training-derived correction at prespecified intermediate sites, with clean-condition harm, mismatched-pair controls, and unseen subjects/strengths. Oracle donor restoration by itself will not resolve whether the effect is transportable. No new physiology or publication-level novelty is claimed by this gate.
