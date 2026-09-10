# Calibration study: completed pilot and decision

Global gain repair is not supported as the main scientific contribution. Input normalization is a sufficient decision-invariance baseline here; internal correction effects are inconsistent and can harm clean predictions. This is a negative feasibility decision on a reused eight-subject cohort, not evidence that EEG FM mechanisms are solved.

## Simple comparators

Balanced accuracy is averaged equally across subjects. Scratch seeds are averaged within each subject first (eight independent subject units, not 24). All heads/checkpoints use subject-disjoint training and validation.

| Model / training | Gain 0.5 | 0.75 | 1 | 1.5 | 2 |
|---|---:|---:|---:|---:|---:|
| Small CNN / clean | 0.5249 | 0.5673 | 0.5907 | 0.5004 | 0.4481 |
| Small CNN / augmented | 0.5145 | 0.5346 | 0.5404 | 0.5318 | 0.5408 |
| Small CNN / normalized | 0.5573 | 0.5573 | 0.5573 | 0.5573 | 0.5573 |
| cbramod / normalized | — | 0.6869 | 0.6869 | 0.6869 | — |
| labram / normalized | — | 0.6099 | 0.6099 | 0.6099 | — |
| cbramod-random / normalized | — | 0.5837 | 0.5837 | 0.5837 | — |
| labram-random / normalized | — | 0.5824 | 0.5824 | 0.5824 | — |

All four normalized encoders have zero decision flips on all 808 windows across the three evaluated gains. Features are numerically close, not bitwise equal. This preprocessing removes absolute amplitude and is not a general clinical preprocessing recommendation.

## Functional interventions

The completed native evaluation contains 96,960 intervention records across two pretrained models, five sites, four methods and three gains. Every intervention was valid. Maps use only training paired activations; gain 0.75 and 1.5 are unseen strengths. Clean donor activations are used only for oracle diagnostics. Identity and full-final-site checks ran inside the native experiment. See CORRECTIONS.md for all sites; FINAL_SUMMARY.json additionally contains all 60 paired true-versus-null contrasts. Intervals are exploratory subject bootstraps without multiplicity correction. No best layer is selected.

## What the evidence resolves

- A failing historical head does not establish absent task information: training-domain augmentation can improve a new held-out readout. Gains vary by condition and do not prove a universal correction.

- Tested intermediate mean/affine maps do not provide a stable repair advantage with clean-condition preservation. Their failure cannot establish information destruction.

- Normalization makes this particular positive global scaling nuisance trivial while retaining useful task accuracy. It weakens the motivation for expanding gain repair.

- The small supervised CNN is a practical comparator only. Its three seeds do not replace multiple random-encoder seeds or an architecture-matched pretraining ablation.

## Limits and next decision

The cohort has already informed earlier analyses; the gain change is synthetic and single-channel. No real-device, montage, independent-cohort, cross-task, or causal-pretraining claim is justified. The broader shared-computation question requires a separately fixed protocol and data/novelty gate. Keep this pilot as a reproducible negative result and EEGLens validation; do not scale the same calibration experiment merely to seek significance.
