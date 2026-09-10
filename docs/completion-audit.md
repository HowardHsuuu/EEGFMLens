# Goal completion audit — local a8

Follow-up: local a9 now retains the four LaBraM-acknowledged upstream licenses and source credits. Its eleven archive/source checks and two installed 87-test gates pass; see [current ledger](public-readiness.md). The a8 inventory below remains a historical audit snapshot.

Overall status: **incomplete**. The science deliverable is complete within its declared design; the broader public-readiness claim remains unproven. This audit does not equate TransformerLens-level quality with a test count or claim every configuration of eleven models is perfect.

| Original requirement | Current evidence | Assessment |
| --- | --- | --- |
| Eleven-model intervention semantics | 11 families / 13 views; coordinate and dense native-oracle checks, local-selection preservation and unsupported-selector rejection; input-boundary evidence | Satisfied for the recorded CPU float32 configurations and operations. Not exhaustive over architectures, hardware or all options |
| Three-model real EEG experiment with subject separation and controls | Fixed main study plus completed contrast-task follow-up: CBraMod, LaBraM, CSBrain, each 18 subjects / 810 trials / 71 conditions in the follow-up; rank/random/norm controls and off-target readouts | Completed. The follow-up reuses original training subjects and is exploratory; it is not new independent confirmation |
| Insightful result that determines the next scientific action | Recovery survives original-probe erasure; pooled scores obscure individual failures; all three final physiological readouts fail pooled generalization; no established selective shared dependence | Completed as a limited negative/measuring-method result. It does not establish distinct model mechanisms or absent physiological information. Current evidence does not justify cross-model steering |
| A reliable tool other researchers can use | a8 wheel and sdist; three installed environments including one new venv; 87 non-integration tests each; native real-EEG examples; portable response replay | Strong local evidence, incomplete broader public-readiness verification |

The current-state inventory rechecked all 26 referenced dense/input-boundary result digests, all three complete contrast summaries and all three a8 installed-artifact reports. `research/model_validation/results/goal-audit-a8.json` retains the inspected hashes. Current Ruff checks pass over the repository; 184 files are formatted. Hash consistency supports artifact identity, not scientific validity or a new native run.

## Remaining work toward public readiness

1. **Configured platform jobs executed.** Source commit `76f52c4` is published at `https://github.com/HowardHsuuu/EEGFMLens`. [GitHub run 34435514982](https://github.com/HowardHsuuu/EEGFMLens/actions/runs/34435514982) passed lint and all six Ubuntu/macOS/Windows × Python 3.10/3.12 installed-wheel jobs. All six downloaded JSON/JUnit pairs were verified; see [current evidence](public-readiness.md#executed-github-actions-evidence). This closes the previously missing platform gate for the non-integration suite. It does not rerun native checkpoint experiments on those platforms.
2. **Retain the limits of the source-attribution review.** The a9 follow-up records LaBraM/BEiT v2 AST correspondence, reference snapshots and the four acknowledged projects' license texts, verifies their archive inclusion, and preserves source credits. This completes the identified missing-notice work. Original historical derivation revisions remain unproven; do not claim complete reconstruction of every upstream origin. Direct-source byte parity and derivation provenance remain separate claims.
3. **Keep published artifacts consistent with their tested snapshot.** a8 itself was built and audited; no PyPI artifact release is claimed. Later audit-document edits are working-tree records; they are not silently treated as bytes in the already-tested sdist. Any replacement release artifact needs its own digest and archive checks. No upload is authorized merely by the existence of a local package.

## Optional follow-ups, not missing positive findings

A full nine-fold native rerun in another environment would strengthen reproduction evidence; current fresh-environment verification includes two-checkpoint native examples and complete retained-response reanalysis, not this expensive rerun. GPU support and additional model configurations likewise need their own tests before being advertised.

A new causal study should first establish a physiological readout that generalizes across evaluation subjects, then freeze a selective intervention and independent evaluation cohort. Finding a positive shared mechanism is not required to complete the present study. Do not keep changing ranks, readouts or subject selection in the completed experiment to manufacture that outcome.

Results: [main study](../research/mi/REPORT.md), [completed native comparison](../research/mi/CONTRAST_TASK_RESULTS.md), [recovery](../research/mi/RECOVERABILITY.md), [pooling diagnostic](../research/mi/FINAL_POOLING_DIAGNOSTIC.md). Tool/reproduction evidence: [public-readiness ledger](public-readiness.md), [real EEG example](validation.md), [portable replay](../research/mi/REPLAY.md).
