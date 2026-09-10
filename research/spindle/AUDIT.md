# Completion and verification audit

Completed local experiment, 2026-09-09. Scientific positive claims are evaluated separately from execution completion.

| Requirement | Inspected evidence | Result |
|---|---|---|
| Public EEG + human events + staging | DREAMS archive MD5 matched official `2f8a101194e133dd4324f21047dce579`; eight EDFs, 360 stage entries each, expert event files | Available and temporally aligned |
| Signal units and provenance | EDF/text sample comparison, source SHA256 per subject, explicit uppercase-UV conversion; physical median absolute amplitude 9.98 uV, maximum 399.04 uV after preprocessing | Verified; initial incorrect run invalidated |
| Annotation coverage | 808 stable-stage windows; 308 reviewed N2 concept windows; end<=990 seconds, partial overlaps excluded | Unknown labels kept separate from negatives |
| Subject-held-out prediction | Eight outer folds, five train/two validation/one test subjects; test covariate/label perturbation regression test | Disjoint fitting; baselines completed for both pretrained and random encoders plus spectral features |
| Interpretable candidate and nuisance checks | Conditional N2 spindle probes at two fixed internal sites; validation-only selection; spectral/amplitude residual probes; second-rater sensitivity on six subjects | Decodability measured; no robust beyond-spectral claim established |
| Native interventions | 12,120 per-window/control records per model; identity and extracted-feature parity asserted in every held-out batch | Completed for both models |
| Perturbation controls | Ten random directions, shuffled labels, sigma direction, adjusted spindle direction and signal-defined slow-wave morphology; rank one, per-window matched norm | Maximum relative norm discrepancy < 1e-6 for both models |
| Paired specificity and uncertainty | 76 matched N2 positive/negative pairs across eight subjects; full matching balance diagnostics; 10,000 subject-bootstrap draws | Completed; imbalance and small-sample limitations explicitly reported |
| Reproduction | Two corrected full runs yield exactly equal input arrays, all pretrained/random features, baseline records, and all 24,240 intervention records | Verified; output artifact SHA256 entries checked |
| Tests | 22 core/checkpoint tests plus three experiment tests (split leakage, norm/CLS semantics, voltage conversion) | 25 passed; one upstream timm/PyTorch deprecation warning |
| Human-readable results | Generated RESULTS.md, aggregate summary JSON and inspected figure | Completed |

The valid final run used Python 3.12.14, torch 2.14.0 and MNE 1.12.1 on local macOS CPU. Runner output includes exact dependency versions, analysis-source hashes, checkpoint hashes, fitted parameters and artifact hashes. No lab machine, paid instance or external publication was used.

The primary per-window intervention records are identical between corrected run-v2 and run-v3. Run-v3 additionally retains concept intercepts and corrects random-control accuracy reporting to average the ten separate accuracies rather than score an averaged-logit ensemble. This aggregation correction does not change the primary linear margin contrast. Second-rater sensitivity and paired pretrained/random summaries are diagnostic additions with no model or layer reselection.

The local protocol is an internal decision record, not an externally registered study. The independent text export is used to verify numerical units; no raw waveform is redistributed here. K-complex records lacked a verified shared subject/time mapping, so their labels were not imported. The other-event control is explicitly operational slow-wave morphology, and expert K-complex specificity remains untested.

**Scientific outcome:** the experiment does not establish a shared spindle-specific mechanism or a reliable overall accuracy contribution from the erased spindle directions. It does establish that EEGLens can distinguish a decodable correlate from a stronger, unsupported mechanism claim through native, controlled interventions. These negative/ambiguous findings fulfill the experiment's falsifiable objective; they do not answer every question about EEG foundation-model mechanisms.

Subject interpretation follows the published DREAMS description of eight recordings from eight patients; the anonymized EDF headers alone cannot establish identity. See [the dataset-methods description](https://pmc.ncbi.nlm.nih.gov/articles/PMC11219251/). Expert-2 intervals are fixed-duration annotations, so that sensitivity analysis is not independent ground truth for exact spindle duration.
