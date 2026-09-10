# Scientific study and semantic validation proposal

Status: historical proposal, retained to explain the initial design and BENDR defect discovery. The three-model study and eleven-family semantic audit have since completed within their recorded scope. Start with the [current scientific reports](../research/mi/README.md) and [public-readiness ledger](public-readiness.md). Future-tense statements below describe the proposal at the time, not outstanding work or current findings. The checks establish bounded native execution fidelity, not perfect integration, clinical validity, or universal sensor/time/feature semantics.

## Current goal and ordering

The user-approved primary question is now: **when several EEG FMs encode the same physiological feature, does removing its internal representation affect their downstream task performance similarly on held-out subjects?** First establish within-model selective effects, then consider cross-model intervention transfer below as a follow-up. The transport proposal is not the immediate primary endpoint.

Completion requires all eleven families to pass an explicit intervention semantics audit, a reproducible three-model real-EEG study, and a documented interpretation of results and remaining identification limits. Public readiness also requires installation, public API documentation, reproducible examples, checkpoint/source provenance, failure behavior, cache compatibility, and supported-environment checks. Passing local synthetic tests alone does not establish TransformerLens-level quality. Publication or uploading artifacts is a separate action, not needed to prepare the release locally.

Initial study design: subject-disjoint motor imagery; training-only physiological concept fits and frozen task/off-target readouts; validation-only layer/rank/dose selection; middle-layer intervention followed by native downstream computation. Include dimension-matched random subspaces, energy-matched perturbations, and pretrained/random-feature/spectral baselines. Define the exact off-target variables and selection rule before examining test outcomes. Bootstrap subjects, not windows. Report each model's effect and uncertainty before asserting agreement. Shared sensitivity to a broad damaging intervention is not evidence of shared selective dependence.

The current candidate dataset is EEGMMIDB 1.0.0, whose source documentation distinguishes unilateral from bilateral task labels and records at 160 Hz. Use only explicitly identified unilateral imagery runs for the left/right study. Pretraining exposure remains an audit requirement; a downstream subject split cannot by itself prove that test subjects were unseen during foundation-model training. Source: https://physionet.org/content/eegmmidb/1.0.0/ (checked 2026-09-10).

## Recommended question

**Does representational alignment predict transferable intervention effects across EEG foundation models?**

The distinction is between shared decodable information and shared functional organization inside the models. On held-out subjects, can a correspondence fitted only on clean training activations transport a change from a source model's middle layer into another model and predict a selective effect on its task output? Existing physiological probing/erasure and SAE steering are relevant baselines, not novelty claims. A focused full-text novelty audit remains required before claiming this particular transport protocol is new.

Primary sources checked:
- https://arxiv.org/abs/2605.11410 — physiological probes, subspace erasure, and universal candidate features across models/tasks.
- https://arxiv.org/abs/2605.13930 — SAE concepts, steering selectivity and entanglement.
- https://arxiv.org/abs/2605.17562 — robustness, pooling and relevance analysis.
- https://arxiv.org/abs/2608.13676 — physiological-domain interpretation of model predictions.

## Minimal scientific protocol

1. Use one public motor-imagery task with subject-disjoint training/validation/test sets. Each model receives its documented montage, reference, units, sampling rate and normalization derived from the same raw trial. Record incompatible models rather than inventing channels or silently changing preprocessing. Audit pretraining overlap before generalization claims.
2. Train comparable frozen-encoder readouts and simple spectral/spatial baselines. Measure whether pretrained features actually outperform random initialization before interpreting a pretraining advantage. Random controls are explicit controls, not pretrained integration evidence.
3. On training subjects, fit bounded-rank clean-activation correspondences between selected middle layers. Choose layers/ranks/doses on validation subjects. Transport a delta into the recipient's own activation state; do not transplant an unscaled raw state or fit mappings to held-out intervention outcomes.
4. On held-out subjects, compare predicted and observed signed task-margin changes and off-target effects over multiple small doses. Normalize margins and doses within each model using training data; raw logits are not directly comparable. Report intervention-response agreement beyond representational similarity.
5. Compare against shuffled trial-pair mappings, rank/norm-matched random directions, independently fitted per-model concept erasure, and matched/wrong donors. Include a final-layer linear-readout control but do not use its algebraically predictable response as evidence of a shared computational mechanism. The primary test must include nonlinear downstream computation from middle layers.
6. Initially pilot a few architecturally different models, then extend the locked protocol to all compatible members of the eleven-family cohort. Repeat on a second task with separate training-only readouts if the initial study warrants it. Failure of a restricted mapping is not proof that models share no mechanisms.

## Other worthwhile questions

- **Temporal evidence integration:** do pretrained models use the same event duration/order or mainly local spectra? Match amplitudes and spectra as far as the manipulation permits, vary event duration/gap/order, and use internal interventions to distinguish context dependence from preprocessing/windowing effects. This is sensitive to time alignment, receptive fields and hidden patch boundaries. Our prior phase/spindle pilots should be treated as existing evidence and controls, not ignored or rerun without a new hypothesis.
- **Transferable physiology versus recording shortcuts:** when subject/session/montage information is attenuated, does physiological/task information survive and transfer? Keep training/test identities separate, use independent nuisance and physiological readouts, and report collateral task damage. This extends existing nuisance/robustness research only if it establishes a new, held-out intervention result; merely decoding subject identity is not enough.

## Engineering checks that accompany the science

Run these for every declared supported view, with expectations derived independently of the adapter:

- Native-vs-wrapped outputs and intervention effects on real model-ready EEG and controlled perturbations, across valid lengths/channel configurations. A physiologically plausible-looking result is not a correctness oracle.
- Explicit batch, feature, token and physical-time axes; test a partial feature erasure and a single physical sensor/time edit where supported. Unsupported physical mappings must reject selectors. Validate feature pooling separately from token pooling.
- Native preprocessing/units/reference and trial alignment; use known-time signal markers and explicit channel IDs. A changed montage is not universally an invariance, and native sensitivity must be distinguished from wrapper errors.
- Identity, matched/wrong donors, failure cleanup and cache reload. Keep stochastic comparison rules explicit: BrainOmni and DIVER native eval dropout requires controlled paired RNG, and batch/ordering tests must account for random-mask assignment rather than assuming a fixed global seed guarantees per-trial equivalence.
- Repeat effects over subjects and seeds. Lack of an expected physiological effect can be a model result; disagreement with independently executed native code is an integration failure.

## Defect found during this review

BENDR convolution caches exposed `[B,D,T]`, but feature-subspace erasure used the last axis. This could erase temporal directions while being described as feature erasure. Full zeroing and identity replacement both passed, illustrating their limited coverage. The adapters were corrected to `[B,T,D]`; a unit regression and independent native feature-channel erasure checks now cover both encoder-only and composed BENDR. All 34 native BENDR site/batch conditions passed including the added feature checks. Old BENDR caches must be regenerated. A full eleven-model semantic/preprocessing audit remains future work.

The next proposed milestone is a reproducible semantic-conformance matrix for all eleven families plus a small real-data intervention-transfer pilot. Neither milestone is claimed complete here.
