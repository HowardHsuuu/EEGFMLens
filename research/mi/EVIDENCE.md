# Reading the MI evidence

Start with `REPORT.md` for the fixed main study. The subsequent recoverability, iterative erasure, native contrast mapping and contrast-task analyses are explicitly exploratory; they do not replace the original test protocol.

`results/evidence-catalog.json` inventories the result manifests and hashes the current reports. It distinguishes inventories whose referenced files still match from historical snapshots whose documents or helpers have since changed. Historical hashes are retained; a later document edit is not retroactively claimed to have been present during a native run.

The catalog reports file consistency only. Native parity, fitting-role separation and numerical controls are established by their specific runners and retained results, within their stated configurations. A matching hash neither certifies scientific validity nor proves execution on an untested platform.

For the contrast-task study, all three models have complete three-fold summaries. The three-model report includes the clean-readout audit that limits physiological interpretation, and `contrast-task-three-model-evidence.json` records the completed artifact set. Earlier CBraMod-only and two-model inventories remain historical report snapshots. `contrast-native-v1` predates a removed unused import; `contrast-native-v2` records the rerun, while its explanatory document was later expanded. The catalog identifies those changes explicitly.

Raw EEG, weights and large local feature/response arrays are external workspace artifacts. Their paths and hashes are in the study manifests, but they are not included in the wheel. Use the documented data/checkpoint preparation before attempting full reproduction. Portable reports and native-validation snapshots are available in this repository; absence of local raw data is not evidence that a reproduced run has passed.
