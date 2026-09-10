# Candidate scientific question: transferable internal dependence

Status: candidate protocol, not an established novelty claim or a completed cross-task experiment. Written after the calibration pilot. The gain experiment remains separately reported; its negative decision does not answer this question.

## Question

Within one unchanged pretrained EEG encoder, does an intervention learned on task A predict a selective behavioral effect on task B, without refitting its subspace, site, rank, or strength on B?

The distinction is between shared EEG feature names and shared internal dependence. Two task-specific analyses can both identify band power while relying on different activation directions or different model weights. Transfer of one fixed intervention is a stricter observable prediction, though it still does not establish a full computational circuit or a physiological brain mechanism.

## Closest prior evidence and remaining uncertainty

[What Do EEG Foundation Models Capture from Human Brain Signals?](https://arxiv.org/html/2605.11410v1), Sections 3.1–3.3, 4.1 and 4.3, already examines feature encoding, internal erasure and shared usage across tasks. Its stated setup loads task-fine-tuned checkpoints and estimates each dataset/model/feature eraser separately. Therefore, broad cross-task feature-use maps are already covered. A frozen, A-derived intervention transferred unchanged to B is a candidate distinction from that protocol, not proof of novelty across all research.

[Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders](https://arxiv.org/html/2605.13930v1) already motivates concept intervention and off-target/entanglement analysis. Using an SAE or showing selective edits alone is insufficient differentiation.

A targeted search on 2026-09-09 used “EEG foundation model shared circuits cross task mechanistic interpretability sparse autoencoder” and “EEG foundation model cross task causal intervention shared representations interpretability”. Before treating this candidate as a paper contribution, check intervention transport, cross-task erasure, and frozen-versus-fine-tuned comparisons in the closest papers and their code. Absence from this short search is not evidence of absence.

## Design requirements

1. Use exactly the same frozen backbone weights, preprocessing and hook semantics for both tasks. Fit distinct readouts on training subjects only. Record checkpoint and data hashes. Task A and task B are prediction targets, not claims that the person's mental state is experimentally manipulated.
2. Prefer a cohort with two independently meaningful annotations on the same recordings. This reduces dataset identity as a confound but does not remove label correlation. A sleep-stage/event pair needs within-stage event evaluation, usable positive/negative examples per held subject, and an additional target with a predicted null effect where feasible.
3. Fit candidate subspaces or interventions using A training subjects only; select site, rank and strength on A validation subjects. Freeze those choices before B outcomes. B training labels may fit its readout but cannot select the transported intervention. Keep B test subjects unavailable to both stages.
4. Measure the same intervention on A and B, with rank-matched and actual-edit-norm-matched controls, shuffled A target labels, clean prediction preservation, per-subject effects and intervention validity. A broad collapse is not shared computation.
5. Report an independently B-fitted intervention as a clearly labeled within-task comparison. Do not use its outcomes to redefine the A-to-B success criterion. Test both directions only if fixed beforehand; do not retain only the direction that works.
6. Compare pretrained with several random encoders and architecture-matched supervised training before making causal pretraining claims. A small CNN and a single random encoder are feasibility baselines only.

## First deliverables and gates

- An annotation inventory: source licenses, subjects, recording IDs, label definitions, class counts jointly by task and subject, and nonoverlapping train/validation/test assignment. Current single-channel DREAMS windows are only a feasibility option, not an independent confirmation cohort.
- A held-out baseline for both tasks. If B is near chance or its labels almost deterministically follow A, the proposed selective transfer measurement is not interpretable; change the data/target before running expensive interventions.
- A frozen A-derived intervention artifact carrying its training provenance and selection rule. EEGLens applies that artifact through the native encoder to both task readouts.
- A full effect table including nulls, off-target effects, unavailable conditions and uncertainty across subjects. No test-selected “best shared circuit”.

Positive evidence would be selective, held-subject intervention transport beyond matched controls with a prespecified sign or behavioral prediction. A null result would bound transport for the tested intervention family; it would not establish that the model has no shared computations. A global performance drop would fail the specificity gate.

## EEGLens scope

Reuse the existing native cache, scoped intervention and sweep APIs. Add a reusable artifact/provenance boundary only after the experiment requires it. Avoid building a circuit-discovery interface before deciding which claim it must test. Readout changes, concept decodability, and native behavioral interventions remain distinct recorded quantities.

## Existing-data gate outcome

The [annotation inventory](results/DATA_INVENTORY.md) found both within-N2 event labels in every subject, with only four and five positive windows in two subjects. Existing intermediate conditional probes already provide a feasibility baseline. Because the cohort and question have informed prior experiments, this is not prospective independent evidence. The calibration pilot is closed by [its audit](COMPLETION_AUDIT.md); this candidate remains a future research design with unresolved gates.
