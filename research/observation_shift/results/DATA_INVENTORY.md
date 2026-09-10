# Existing dual-task data gate

This audit reuses existing annotations and baseline outcomes; no new head is fitted or selected. Unknown/boundary spindle labels remain excluded.

| Subject | All windows | Reviewed labeled | N2 negative | N2 positive |
|---|---:|---:|---:|---:|
| 1 | 102 | 60 | 14 | 28 |
| 2 | 103 | 56 | 10 | 33 |
| 3 | 93 | 49 | 30 | 4 |
| 4 | 104 | 59 | 13 | 5 |
| 5 | 102 | 60 | 14 | 29 |
| 6 | 90 | 55 | 17 | 27 |
| 7 | 101 | 57 | 36 | 12 |
| 8 | 113 | 60 | 9 | 27 |

Every subject has both spindle labels within N2, but subjects 3 and 4 have only four and five positive windows. The effective independent sample is eight subjects. The complete stage×spindle counts and original subject-disjoint split assignments are in DATA_INVENTORY.json.

Existing conditional spindle probes already ran in the earlier spindle study. Their equal-subject AUROC was spectral 0.8581, CBraMod 0.6750, LaBraM 0.7728. These are intermediate-feature probes selected using training/validation data, not independently validated final task-B heads. Repeating them as a new experiment would add no evidence.

Decision: retain DREAMS as a feasibility example, but do not use it as independent confirmation or infer shared computations from stage/event co-degradation. A prospective cross-task experiment still needs a frozen intervention, target-side selection exclusion, conditional controls and independent data.

Source: [DREAMS depositor record](https://doi.org/10.5281/ZENODO.2650141). The accompanying source PDF identifies CC BY-NC-ND 3.0; this inventory does not redistribute signals. Window filtering and annotation boundaries are specified in the prepared manifest.
