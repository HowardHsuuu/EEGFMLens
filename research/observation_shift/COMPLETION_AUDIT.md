# Research-start completion audit — 2026-09-09

## Scope being closed

The active instruction “請開始” initiated the observation-shift investigation recorded in PROTOCOL.md. Its explicit initial deliverables are a checked related-work map, a reproducible gain gate and an evidence-based decision on the next causal experiment. The protocol additionally keeps the scientific objective open until information/readout/correction alternatives are tested and generalization limits are clear. Those tests now have results; the decision is to stop expanding global gain repair as the main contribution.

This closes the initiated pilot and its decision, not the larger ambition of explaining foundation-model transfer. The cross-task candidate suggested during the discussion is documented with its unresolved novelty/data gates; it is not presented as an executed experiment or a finished paper. No paper-level, clinical, real-device or causal-pretraining claim is made.

## Requirements and authoritative evidence

| Requirement | Evidence inspected | Assessment |
|---|---|---|
| Related work and distinguishability | RELATED_WORK.md; primary-source methods of 2605.11410 and 2606.06647; NEXT_QUESTION.md | Completed targeted check; broad novelty rejected, narrower novelty unverified |
| Reproducible calibration gate on all windows and controls | External gain-v1, model feature files and source/protocol hashes; report-v1/summary.json | Completed; 808 windows, five gains, pretrained/random CBraMod and LaBraM |
| Fixed-head failure vs tested readout recoverability | probes-v1/results.json and readouts.npz; summary tables | Completed mean/token-preserving, clean/augmented train-only comparisons; failure is not information absence |
| Training-derived internal corrections and controls | maps-v1; corrections-v1/{cbramod,labram}.json; report-v1/corrections.json | Completed 96,960 native records, five sites, four methods, three held-out/clean conditions |
| Native identity/oracle/full-final checks | Assertions in correct.py exercised during completed runs; no invalid result rows; test_corrections.py | Completed within the tested models/data; not physiological validation |
| Matched null strength and specificity reporting | Independent per-record norm comparison; FINAL_SUMMARY.json with 60 paired true-minus-null contrasts | Completed; no test-selected best site or multiplicity-corrected discovery claim |
| Simple input normalization with actual task performance | normalized-v1 native results for four encoders, all 24 subject/gain outcomes per encoder | Completed; zero decision flips, useful CBraMod task score; numerical features not identical |
| Practical supervised comparator | scratch-v1: 72 checkpoints, 360 outcomes; verification.json reloads every reported condition | Completed, seeds averaged within subject; not architecture-matched pretraining ablation |
| Provenance and reproducible report | report-v1/artifact_manifest.json verified against files; finalize.py; code lint | Completed for the saved pilot artifacts |
| Decide whether to scale this mechanism hypothesis | FINAL_REPORT.md and README.md | Stop expanding gain repair; simple normalization suffices for this nuisance |
| Explain broader generalization limits | PROTOCOL.md Phase 2; FINAL_REPORT.md; results/DATA_INVENTORY.md | Explicitly unresolved independent cohorts, real montages, multiple random encoders and architecture-matched training |

## Follow-up candidate, not completed results

NEXT_QUESTION.md asks whether an A-derived intervention transfers unchanged to B in one frozen encoder. The data inventory identifies existing within-N2 event labels, sparse positive counts and cohort reuse. Existing conditional probes were inspected instead of redundantly refitted. Independent data and a prospective target-excluded selection protocol remain required. The current evidence does not justify announcing shared circuits or claiming no prior work covers the question.
