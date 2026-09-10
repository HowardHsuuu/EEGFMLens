# Research plan

Status: historical cross-model transport proposal. The active work first tests within-model information dependence; see the [current study index](../research/mi/README.md), [results](../research/mi/REPORT.md) and [next milestone](next-goal.md). The transport experiment below has not been executed and remains contingent on reliable, selective within-model effects. Completed studies have not established those prerequisites.

## Main question

Can an independently fitted correspondence between EEG model representations predict their intervention responses on held-out subjects?

This separates three claims: models encode common physiological information; their activations can be aligned; corresponding interventions have selective functional effects. None alone establishes identical algorithms or causal mechanisms in the human brain.

Existing work already covers cross-model physiological probing and erasure, including universal candidate features, and SAE-based steering/selectivity. These are baselines, not missing capabilities we claim to invent. See [What Do EEG FMs Capture?](https://arxiv.org/abs/2605.11410) and [EEG SAE](https://arxiv.org/abs/2605.13930). A focused novelty audit of cross-model intervention transfer remains a pre-study gate.

## Initial protocol

1. Validate official CBraMod and LaBraM checkpoints and frozen-encoder readouts on a small public MI dataset. Readouts are trained separately for each encoder with comparable budgets and held fixed during interventions.
2. Run the same source trials through each native preprocessing pipeline. Use explicit trial/time anchors and pooling to obtain paired activation observations. Fit concepts, normalization and a capacity-limited mapping on training subjects; select layer pairs/rank on validation subjects.
3. Lock the mapping and candidate source directions before inspecting test responses. Apply within-model interventions in A and corresponding mapped interventions in B on held-out subjects. Never fit the mapping to test behavior.
4. Compare a prespecified response vector: target margin/loss, concept readout changes and off-target outcomes. Raw logits are not comparable across models. Fix concept sign on training data and calibrate intervention norms to each model's training activation scale.
5. Include independent per-model concept erasure, shuffled trial-pair mappings, rank/energy-matched random directions, identity and matched/wrong donor controls. Test dose-response and report global degradation separately.

For a linear mapping, transport a change rather than a raw state: if `h_B ≈ A h_A + b`, propose `delta_B = A delta_A`, apply it to B's own recipient state, and compare with B's own matched-donor delta. Means, whitening, scaling and rank are part of the saved transformation; inverse transformations must be included. This is a candidate experimental construction, not a claim of causal equivalence. Small-dose and matched-donor tests help diagnose out-of-distribution interventions.

The primary outcome is held-out response agreement and selectivity relative to controls, not merely a high CKA or successful feature probe. Pre-register the exact agreement statistic, dose aggregation and smallest meaningful effect after a training/validation pilot and before the test set is opened. Bootstrap/permutation units are subjects or independent sessions, not adjacent windows. Report failures and undefined recovery denominators.

## Scope and extensions

Start with [EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) MI and use [BNCI 2014-001](https://bnci-horizon-2020.eu/database/data-sets) for related-task cross-dataset validation. Mapping transport requires compatible, predefined activation views; dataset-specific sensor changes can invalidate it. Report this incompatibility rather than refitting on test outcomes. Audit pretraining overlap and artifact/subject confounds before generalization claims.

After the primary experiment, lock candidate correspondences and test another task with its own training-only readout. Same-task cross-dataset evaluation is not cross-task transfer. Add BrainOmni and paired [Wakeman–Henson EEG–MEG](https://www.nature.com/articles/sdata20151) later. EEG–fMRI requires HRF and scanner-artifact controls; ECoG is outside the initial study.

A negative result may indicate different functional organization, an inadequate mapping family, poor temporal pooling or unreliable measurements. Distinguish these explanations. If pretrained encoders have no measurable advantage over random-feature/simple EEG baselines, do not characterize the study as explaining a pretraining advantage.
