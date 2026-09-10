# Public readiness audit

Status: active, incomplete. Current local package: **0.1.0a9**, unpublished. TransformerLens-level quality is a quality ambition, not verified feature parity. This ledger states tested scope rather than declaring integrations perfect.

a9 retains the BEiT v2/Microsoft, timm, DeiT and DINO license texts acknowledged by LaBraM. Recorded reference snapshots show exact AST correspondence for several LaBraM/BEiT v2 components; they are not asserted to be original derivation commits. Eleven source/license checks pass for both archives. Two installed environments pass all 87 non-integration tests, 33-file byte verification, quickstart and the 24-effect analytic sweep. The modern environment was freshly created for a8 and upgraded to a9; it is not a fresh a9 dependency resolution. Production model Python code is unchanged. Evidence: `vendor-audit-a9.json`, `installed-modern-a9.json`, `installed-minimum-a9.json`, and `labram-upstream-provenance.json` under `research/model_validation/results/`.

The [goal completion audit](completion-audit.md) separates completed scientific deliverables, bounded integration evidence, remaining release gates and optional future studies.

a8 adds the PyTorch source notice after verifying three exact helper AST matches inside CBraMod's bundled transformer. The reference source matches the installed torch 2.6 RECORD hash, and the complete upstream v2.6.0 license is retained in both archives. Model Python code is unchanged. Both installed dependency environments pass 87 non-integration tests (two checkpoint tests excluded), isolated import, byte comparison of all 33 package Python files, dependency checks, quickstart and the 24-effect analytic sweep. Seven vendor source/license checks pass against the a8 archives. Evidence: `installed-clean-a8.json`, `installed-minimum-a8.json`, `vendor-audit-a8.json` and `pytorch-helper-provenance.json` under `research/model_validation/results/`. This is local upgrade coverage, not a fresh-machine or cross-platform run. Earlier a6/a7 paragraphs below retain historical scope.

a6 includes descriptor guards against boolean tuple indices and truthy string write flags. Both installed dependency environments pass 74 non-integration tests, isolated import, dependency consistency, quickstart and the analytic 24-effect patching sweep. All 33 package Python files match current source and installed wheel bytes. This is local macOS ARM CPU upgrade coverage; it does not imply that native checkpoint experiments were rerun for a6.

a7 includes the tuple/list-subclass export fix. Both installed dependency environments pass 75 non-integration tests, the quickstart and all 24 analytic sweep effects. All 33 package Python files match current source and installed wheel bytes; the six vendor source/license checks and both archive inventories pass. The model arithmetic and running contrast-task protocol are unchanged.

## Acceptance ledger

| Requirement | Current evidence | Remaining limit |
| --- | --- | --- |
| Eleven-family intervention semantics | 11 families / 13 views; 488 coordinate-subspace and 796 dense projection conditions; independent native output and locality comparisons | Sampled configurations and sites; not every architecture option, device or dtype |
| Input contracts | All 13 views have native boundary evidence; invalid inputs/configuration changes rejected before computation with recovery checks | Sampled lengths and malformed cases, not exhaustive public API validation |
| Real EEG science | Three-model subject-disjoint MI study, rank/energy controls, off-target readouts, fixed inference and random-initialization comparison | Six evaluation subjects; no established common mechanism; LaBraM pretraining includes EEGMMIDB |
| Insightful follow-up | Training-subject cross-fitting shows linear information can remain after original-probe erasure; pooled scores conceal subject heterogeneity | Exploratory, not independent confirmation or downstream causal evidence |
| Installed artifact | a9 wheel bytes checked against installed files; isolated import, dependency check, 87 non-integration tests, quickstart and analytic sweep | macOS ARM Python 3.12 CPU; a9 upgrades, with fresh dependency-resolution evidence from a8 |
| Release provenance | a9 direct source and acknowledged upstream notice retention, with recorded hashes and archive inclusion | Original derivation commits and every transitive upstream file are not established; checkpoints/datasets have separate terms |
| Platform coverage | Six OS/Python CI jobs configured | Remote CI unexecuted; Linux, Windows, Python 3.10 and GPU not validated |

## Authoritative evidence

Fresh-install follow-up: a newly created Python 3.12 venv installed the local a8 wheel with `[eeg,labram,dev]` through normal dependency resolution. All 87 non-integration tests, isolated import, 33-file byte checks, dependency consistency, quickstart and the 24-effect sweep pass. Both official-checkpoint real-EEG examples pass from outside the checkout, including exact native local-ablation and cache parity. The three-model response replay also reproduces every compared numerical result in this environment. Evidence is indexed by `research/model_validation/results/fresh-environment-a8.json`. This removes the reliance on previously installed research dependencies for these commands; it remains the same macOS ARM host and does not rerun the full nine-fold native study or test another platform.

Paths below are relative to the repository root.

- `research/model_validation/results/semantic-v1/summary.json`: coordinate-subspace inventory, historical run versions retained.
- `research/model_validation/results/dense-summary.json`: 796 conditions. All 13 referenced result digests rechecked against current files. BrainOmni's a5 rerun is in `adapter-window-v1/brainomni.json`; NeuroRVQ's rerun is retained in `dense-neurorvq-v1/`.
- `research/model_validation/results/input-boundaries-summary.json`: 11 families / 13 views; all referenced result digests rechecked. Counts have different case definitions and should not be combined into an inflated sample size.
- `research/model_validation/results/installed-minimum-a7.json`: older dependencies, current installed-artifact verifier.
- `research/model_validation/results/installed-clean-a7.json`: modern dependencies, current verifier. Historical a5-v2 retains the separate coverage run.
- `research/model_validation/results/vendor-audit-a7.json`: retained a7 wheel/sdist vendor inventory. Later documentation/tooling edits do not retroactively become part of that sdist.

Native comparisons use exact downstream output agreement, with paired CPU RNG where upstream evaluation retains dropout. Numerical residual checks have separately stated tolerances; they do not relax downstream parity. See [input contracts](input-contracts.md) and [dense validation](../research/model_validation/DENSE_VALIDATION.md).

## Replacement boundary coverage

`tests/test_replacement_boundaries.py` adds 12 donor-incompatibility cases: reordered channel coordinates, sampling rate, patch stride, patch sample count, unit, layout, site identity, duplicate trial IDs, wrong trial count, feature shape, NaN and infinity. Each rejected replacement is followed by a valid selected replacement with reversed recipient row order. Explicit expected values verify trial-ID alignment, local C3/patch selection and unchanged other coordinates; donor/input tensors and an existing user hook are preserved, and a subsequent clean run recovers exactly.

All 12 cases pass against source and, from outside the checkout with isolated imports, against installed a7 in both the torch 2.14 and torch 2.6 environments. Evidence: `research/model_validation/results/replacement-boundaries-a7.json` and its two JUnit files. This is a focused follow-up suite, not a new full-suite count or new native checkpoint run. Production package code is unchanged; these tests postdate the a7 sdist. Malformed custom intervention objects and untested device/configuration combinations are not certified by these cases.

## Installed-artifact verification

The package includes NeuroRVQ channel-alias collision rejection and BrainOmni window-configuration validation. BrainOmni native checks cover lengths 1, 512, 513 and 1536; padding matches native behavior. Accepting one sample describes computational compatibility, not useful EEG evidence.

The installed verifier originally imported EEGLens/torch in its coordinator. With local torch 2.6, spawning another torch process then failed with OpenMP SHM error 179. A minimal parent-import/child-import reproducer isolated this from the package tests. The coordinator now resolves the module without loading native runtimes and verifies actual import in a separate isolated subprocess before tests. No duplicate-runtime bypass or model change is used. The older environment passes after this change. This is a local process-lifecycle fix, not a claim about every OpenMP failure.

Historical full a5 suite JUnit records report 66 passing tests per dependency environment, including two checkpoint integration cases. The a5 installed-artifact gate excluded those two cases and recorded 64 passing tests. The current a7 gate records 75 passing non-integration tests, dependency consistency and quickstart separately. Evidence from different gates is not added together.

## Scientific conclusion and next work

The [main report](../research/mi/REPORT.md) does not establish that the models rely on a common physiological mechanism. Its fixed six-endpoint Holm correction cannot reach 0.05 with only six subjects under the chosen exact sign-flip test. This constrains inference rather than justifying changing the analysis after seeing results.

The [recoverability report](../research/mi/RECOVERABILITY.md) identifies a concrete interpretive problem: erasing a probe's coefficient span does not establish information removal. LaBraM's recovered beta pooled R² is 0.453, but only 8/18 individual subjects have positive recovered R². Follow-up should distinguish surviving linear information, subject heterogeneity and downstream dependence with controlled interventions; see [next goal](next-goal.md).

Remaining release work includes systematic API error coverage, an externally reproducible scientific example, executed platform coverage and complete release/provenance review. No public upload has been performed. Earlier milestones and superseded pending lists are preserved in [history](public-readiness-history.md).

A [portable three-model response replay](../research/mi/REPLAY.md) now reconstructs the completed contrast-task summaries, clean-readout evaluations and plots without checkpoints. The actual archive passed after extraction outside the checkout; corrupt files and rehashed incorrect expected numbers were rejected without successful output. This improves analysis portability but does not satisfy fresh-machine native-inference or cross-platform validation.

The real-EEG instrumentation example now has an independent native C3/patch ablation oracle, exact edited-cache/downstream parity and exact unselected-activation preservation. Both bundled checkpoint loaders pass from outside the checkout against the installed a7 wheel with the EEG dependencies present; see [reproduction recipe](validation.md) and `research/model_validation/results/real-eeg-installed-a7.json`. This is two windows on the existing macOS ARM CPU environment, not full scientific fresh-machine reproduction. The example source postdates the a7 sdist; package Python sources were not changed.

The current installed gate additionally executes `examples/patching_sweep.py` from outside the checkout. Both dependency environments reproduce all 24 analytically specified effects, including zero-effect coordinates; the example now requires the expected record count. Example source digests are retained in the `installed-*-a6-sweep.json` reports. These source tooling/example changes postdate the a6 sdist; package Python bytes are unchanged. This is intervention calibration with synthetic data, not new physiological evidence.

Science update: the [contrast-task report](../research/mi/CONTRAST_TASK_RESULTS.md) contains completed CBraMod/LaBraM/CSBrain results (each 18 subjects, 810 trials, 71 conditions). Clean final physiological readouts fail subject-wise generalization; loss increases alone therefore cannot establish information removal. The [pooling diagnostic](../research/mi/FINAL_POOLING_DIAGNOSTIC.md) shows that this failure also does not establish absence of final-layer information. Both analyses remain exploratory. The completed comparison does not establish selective shared μ/β dependence and does not support advancing to cross-model steering on that premise.
