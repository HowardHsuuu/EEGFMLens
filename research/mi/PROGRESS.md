# MI study progress and evidence

## Exploratory follow-up completed

Training-only nested subject cross-fitting and linear recoverability analysis are complete; see [RECOVERABILITY.md](RECOVERABILITY.md). Removing the original coefficient span does not remove all linearly decodable beta information: LaBraM pooled out-of-fold R² is 0.511 clean and 0.453 after refitting on the remaining features. However, recovered individual-subject R² is positive for only 8/18 subjects. Both information recoverability and subject heterogeneity qualify the main causal interpretation. An outer-label perturbation regression test verifies that evaluation labels cannot change that fold's predictions. The original test protocol and results remain unchanged.

## Latest checkpoint: three-model held-out study complete

All three pretrained test intervention runs and constructor-random test extractions are complete. The fixed six-endpoint inference and spectral test baseline have been evaluated. See [REPORT.md](REPORT.md) for the consolidated findings, limitations and next-experiment decision. CSBrain's task contrast is −0.01983 BA (descriptive interval −0.04149 to −0.00463; Holm p 0.375), while its target-loss contrast is 0.00344 (Holm p 0.625). No corrected endpoint meets 0.05. All three clean-output and per-trial matched-norm checks passed.

All exposed sites of the three study models now additionally pass 400 independent dense rank-three native intervention conditions on real development EEG, including local C4/patch-1 edits at batch sizes 1 and 2. These are separate from the eleven-family coordinate-subspace checks and do not establish dense support for the other eight models.

## Earlier partial held-out checkpoint

The chronological notes below include earlier states. All three pretrained models now have clean test features and frozen-probe evaluations (270 trials from six reserved subjects). CBraMod and LaBraM test interventions are complete; CSBrain test interventions remain running. Their validation interventions are all complete. No final three-model inference has been issued.

`evaluate_probes.py` now reproduces the previous independent validation summaries and evaluates frozen probes without refitting. It checks feature/probe hashes, training provenance, model identity, input contracts, electrode order and train/evaluation subject separation. Artifacts are local `probes-v1/{model}/{validation,test}-evaluation.json` and `random-probes-v1/{model}/{validation,test}-evaluation.json` where available.

CSBrain's constructor-random validation baseline is now complete: subject-mean BA **0.65563**, middle mu/beta pooled R² **−0.01360/0.05245**, versus pretrained BA **0.66634** and R² **0.05457/0.25322**. Like the other models, this does not yet establish a reliable task advantage from pretraining. Random test extraction uses the same previously selected seed 812 and frozen probes; no seed search or test fitting is performed.

Pretrained clean test mean subject BA is **0.56014 / 0.58424 / 0.66841** for CBraMod / LaBraM / CSBrain. These are descriptive scores, not evidence of significant between-model differences.

Full-dose concept-minus-norm-matched-random test contrasts:

| Model | Target normalized MSE change (descriptive 95% subject bootstrap interval) | Task BA change (same interval) |
| --- | --- | --- |
| CBraMod | −0.00491 (−0.06540, 0.04531) | −0.00246 (−0.01153, 0.00529) |
| LaBraM | 0.01995 (0.00068, 0.04135) | −0.00268 (−0.01120, 0.00789) |

CBraMod's validation target-loss effect has not replicated on test. LaBraM retains a small descriptive effect, but no multiple-testing conclusion is available. Neither currently resolves a task accuracy loss. The independent clean feature predictions match intervention-run clean outputs exactly; all six per-trial matched-norm conditions pass for both completed test models and CSBrain validation.

**Inference resolution limitation:** with six subjects, a one-sided exact sign-flip p-value cannot be smaller than 1/64. Under the frozen Holm family of six endpoints, the smallest adjusted p-value is at least 6/64 = **0.09375**. Consequently, this protocol cannot reject any endpoint at familywise 0.05 even with maximally consistent effects. The implementation verifies this floor. Retain the locked protocol and report effect sizes and this limitation; do not switch tests, reduce the family or add subjects after seeing outcomes to manufacture significance. A future confirmatory study needs an independently planned larger subject sample and effect-size-based power assessment.

The current evidence supports studying the gap between physiological decodability and task dependence, but does not establish shared mechanisms or explain a pretraining benefit. LaBraM's pretraining corpus overlap and the shared preprocessing limitations below continue to apply.

## Earlier validation pilot record

The following records describe earlier checkpoints, not current job status. These validation numbers are not held-out test results. All probe regularization was selected on this validation set, so these numbers are optimistic for final evaluation.

Training uses 810 trials from 18 subjects; validation uses 270 trials from 6 disjoint subjects. No test responses have been evaluated.

Exposure audit update: LaBraM's reported pretraining includes EEGMMIDB. Its downstream-held-out results must not be described as pretraining-unseen generalization. CBraMod and CSBrain report TUEG pretraining. See [primary-source audit](PRETRAINING_AUDIT.md); architecture/pretraining comparisons remain confounded by corpus exposure.

| Representation | Mean subject task balanced accuracy | Middle contrast mu pooled R² | Middle contrast beta pooled R² |
| --- | ---: | ---: | ---: |
| 95 spectral features | 0.52635 | — | — |
| CBraMod | 0.57987 | 0.21135 | 0.55671 |
| LaBraM | 0.56176 | 0.22468 | 0.47281 |
| CSBrain | 0.66634 | 0.05457 | 0.25322 |

The concept probes predict concurrent C4-C3 log-power asymmetries from middle-layer C4-C3 feature contrasts. Task accuracy is an average of subject-level balanced accuracies; the R² columns are pooled across validation trials and must not be described as subject-mean scores. Beta is more linearly decodable than mu under this particular setup, but decodability does not establish model dependence or superiority. The task differences have not been established as statistically reliable. These shared-input results are not replications of original model benchmark pipelines.

Evidence resides locally outside the repository in `research/eeglens_mi/spectral-v1/summary.json` and `research/eeglens_mi/probes-v1/{cbramod,labram,csbrain}/validation-probe-summary.json`. All three models have completed train/validation extraction and probe fitting. Validation interventions with matched random controls have started for all three; no intervention finding is available yet. Random-initialization baselines, pretraining-exposure audit, protocol lock and test evaluation remain required.

A useful descriptive contrast is that CSBrain has the highest task validation accuracy while the specified middle-layer asymmetry probes have lower R². This motivates testing whether task performance depends on these directions, but it does not by itself identify different mechanisms: layer choice, linear probe capacity, preprocessing and other EEG features remain possible explanations.

## Random-initialization controls

With one constructor seed (812), the random CBraMod encoder achieved validation task BA **0.61561**, with middle mu/beta R² **0.01237/0.03976**. Random LaBraM achieved task BA **0.56621**, with R² **0.03000/0.23356**. Thus the current pretrained versions improve these linear physiological probes but show no established task advantage over these random-feature controls. Do not characterize the current task result as explaining a pretraining benefit. One seed and validation selection limit this comparison; CSBrain's control remains underway. Artifacts are in local `random-probes-v1/{cbramod,labram}/validation-probe-summary.json`.

The fixed test inference implementation enumerates all 64 sign assignments for six subject effects and applies Holm correction across the six predefined model/endpoint tests. Constant-effect, null-effect and known Holm examples passed. This has not yet produced test p-values; inference requires all model responses. Sign exchangeability is an assumption, and nonsignificance is not an equivalence result.

## First validation intervention responses

Held-out protocol `test-protocol-v1.json` is now fixed before test preparation/evaluation. It records all three probe/checkpoint hashes, the fixed input recipe, middle/final sites, rank, doses, random seeds, subject-level endpoints and interpretation limits. Test preparation then produced 270 trials from the six reserved subjects; feature extraction has started. Test metrics have not yet been evaluated. Existing validation analyses remain labeled exploratory.

CBraMod and LaBraM completed all 270 trials across clean, concept and random-control conditions. At full dose, concept-minus-norm-matched-random target descriptor loss increases were respectively **0.04522** (subject-bootstrap interval 0.00400–0.09363) and **0.04574** (0.02963–0.06752), in training-variance-normalized MSE units averaged over mu/beta. Their task balanced-accuracy contrasts were **−0.00368** (−0.01636–0.00615) and **+0.00280** (−0.01087–0.02168). Off-target descriptor loss intervals include zero for both.

This is preliminary evidence that these interventions selectively impair the downstream physiological readouts without a clearly resolved task-accuracy decrease. It does not show that all mu/beta information was removed, that those signals are irrelevant to the model, or that the two models share a mechanism. Intervals are exploratory, from six validation subjects, without multiplicity correction. Redundancy, probe error and limited power remain explanations.

Clean response predictions exactly match the independently saved clean feature/readout predictions. Every random matched-control delta norm matches its paired concept norm within float32 tolerance. Maximum applied multipliers were 3.93 (CBraMod) and 3.25 (LaBraM); values were retained rather than silently capped. Evidence: local `responses-v1/{cbramod,labram}/summary.json` and `responses.npz`. CSBrain interventions are still running. All 90 recordings have now downloaded and passed EDF-size validation. Constructor-random model extraction has started with seed 812, separate from pretrained artifacts.
