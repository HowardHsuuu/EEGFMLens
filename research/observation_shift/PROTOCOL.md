# Observation-shift mechanisms in EEG foundation models

Decision record: 2026-09-09, before new model outcomes. This starts an exploratory study; it is not an established novelty claim or independent replication of the previous DREAMS analyses.

## Scientific question

When an observation change alters an EEG FM's prediction, is task information still recoverable from its frozen representations, and can a training-derived internal correction transfer across subjects and shift strengths without a clean test-time donor?

The original question about generic robustness and layer-wise representation changes overlaps strongly with existing research. The distinguishing candidate is functional localization plus out-of-sample correction: oracle activation patching alone cannot distinguish damaged information from a readout mismatch, and a new probe failing does not prove information absence.

Competing, nonexclusive hypotheses:

- H1: information remains accessible under a different fixed-capacity readout; adapting a head on training-subject shifted features restores held-out performance.
- H2: a low-complexity transformation fitted only to paired training representations restores the original fixed head on unseen subjects/strengths. This supports transportable observation-dependent representation change, not physiological invariance by itself.
- H3: the tested heads/corrections fail. This is an upper bound on these methods' recoverability, not proof of information destruction.
- H4: apparent robustness is largely preprocessing/architecture. Normalization or random-weight controls explain it without invoking pretrained physiological representations.

## Phase 0: calibration-gain feasibility gate

Reuse all 808 corrected DREAMS 15-second windows and the existing held-out N2 heads, not only spindle-positive examples. Models: CBraMod and LaBraM, pretrained and the existing seed-4311 random encoders. Sampling, channel, units and all previous split assignments stay fixed. No causal claim about pretraining is possible from a single random initialization alone.

Counterfactual gains are fixed at 0.5, 0.75, 1, 1.5, 2. Apply positive global scaling to the model-ready waveform as a simulated calibration error. This is not a change in neuronal amplitude and is not true device or cross-subject generalization. The gain is reversible; verify that division by the known gain recovers inputs within float32 tolerance. A per-window normalization control must cancel this transformation up to numerical precision. Calibration correction is a trivial input-space baseline, not a discovery.

Save final per-patch features and pooled activations at embedding, blocks 2/5/8/11; record patch-resolved relative changes from the same clean input. Compare fixed-head balanced accuracy, AUROC, decision flips, signed margin shifts, and absolute margin differences normalized by each head's training-fold clean margin SD. Report all subjects and all strengths. Do not select examples by test-time degradation.

First verify clean final features against the previous artifacts. Measure whether the effect exists at all and whether the normalized input control explains it. If effects are numerical only, stop treating global gain as a useful learned-mechanism stressor. GroupNorm, affine biases and CBraMod's amplitude-spectrum branch must be inspected before interpreting layer patterns.

## Phase 1: readable information and functional correction

Keep the five-train/two-validation/one-test subject folds. Fit all scalers, heads, nuisance bases and correction maps exclusively on training subjects; choose regularization on validation subjects. No gain-specific refitting using held-out data. Compare clean-trained and shifted-training readouts using both mean-pooled and token-preserving features, controlling regularization and evaluation splits. Existing pooled heads remain the reference condition. Token-preserving readouts are a necessary confound control, not the intended novelty.

Fit candidate paired corrections on training gains 0.5 and 2; use validation subjects at those gains for selection, then test gains 0.75 and 1.5 on the held-out subject. A separate prespecified clean-condition check measures off-target harm. Start with a mean paired delta and regularized affine maps; add complexity only if their held-out diagnostics justify it. Controls include input gain inversion, normalization, identity maps and mismatched/permuted training pairs. A map that needs a test example's clean donor is an oracle, not a transferable correction.

EEGLens interventions then test candidate intermediate sites with held-out predictions, not only final-feature geometry. Clean-donor patching supplies an oracle diagnostic; identity and full-final-site replacement verify plumbing. Match locations/strengths where meaningful and disclose large multipliers. A shared mean correction and its controls must be fitted before held-out intervention analysis. Report effects on both originally correct and incorrect trials without choosing cases to inflate recovery.

## Phase 2: actual observation and generalization evidence

A paper-level claim additionally needs multichannel data and an independent task/cohort, real reference transformations, and a distinction between channel removal and zero-padding. The existing central-channel prepared array cannot support meaningful average rereferencing or missing-electrode claims. Do not synthesize these from a single channel. Montage changes alter information and geometry; they are not guaranteed label-preserving invertible nuisances.

Include multiple random seeds and a supervised-from-scratch comparison before attributing findings to foundation pretraining. A compact supervised model is a practical comparator but not an architecture-matched pretraining ablation. Compute and data feasibility must be checked locally before expanding training; no lab or paid resources are currently authorized.

## Reporting and stop conditions

Use equal-subject estimates, preserve per-window pairing, and label bootstrap intervals exploratory. This cohort has already informed earlier experiments; independent confirmation is required. The initial deliverables are a checked related-work map, reproducible gain gate, and an explicit decision about whether the next causal experiment is scientifically interpretable. The broader scientific objective remains open until the information/readout/correction alternatives are tested and the limits on generalization are clear. Do not declare foundation mechanisms solved from these calibration experiments.

### Phase 1 readout parameters, locked before readout fitting

The gain gate has completed feature extraction; this paragraph is fixed before examining its aggregate scores or fitting new probes. Fit logistic readouts with train-only StandardScaler, class-balanced LBFGS, C in {0.1, 1, 10}, max_iter=3000, seed 4311. Select C by equal-validation-subject balanced accuracy, averaging both training-domain gains for augmented heads. Compare mean pooling and flattened 15×200 patch tokens. Clean training uses gain 1; augmented training uses gains 0.5 and 2 only, with each replica weighted 0.5 so total sample weight equals the number of original training windows. Test all gains; 0.75 and 1.5 are unseen strengths. No preprocessing/head parameters use test subjects. Clean-trained vs augmented-trained probe comparisons use the same solver and selection rule; historical fixed heads are separately reported references. Save chosen parameters and all out-of-fold margins. Probe recovery indicates accessibility to this probe family, not a proof of information preservation or unique circuit localization.

### Phase 1 internal-correction parameters, before correction fitting/evaluation

Use the existing five sites and pretrained CBraMod/LaBraM. Each held-out subject receives maps fitted on its five training subjects at gains 0.5 and 2 only. One shared map per site must handle both gains without knowing the test gain. Standardize input and target feature coordinates using training tokens only. Ridge penalties are {0.01, 0.1, 1} times training-token count; choose using mean standardized reconstruction MSE, equally averaging the two validation subjects and two training-domain gains. Do not select by test task score.

Compare an additive mean paired correction, an equal-norm random-direction mean correction, a regularized affine map, and a permuted-pair affine control. The permuted control reassigns whole training windows while preserving patch order and uses the same permutation for both gains; its penalty selection uses the same validation criterion. At evaluation, match its actual edit norm per trial to the learned affine map's edit norm using only the current corrupted activation (no clean test donor). Report multipliers and null zero-delta failures. Keep CLS unchanged for partial/internal corrections. Evaluate all windows at gains 0.75, 1 and 1.5; report clean-condition harm and both seen/unseen subjects explicitly. Training and validation subjects are used for fitting/selection only; reported task predictions are always out of fold.

Maps act on all EEG patches at a site, not a selected physiological event. Their coefficients and parameter/source hashes are saved. Include native identity checks, paired clean donor oracle checks, and full-final-site restoration as plumbing. The oracle is diagnostic and is never used by the learned correction. Input gain inversion remains an explicit upper-bound calibration control, not a deployable learned mechanism. Intermediate maps are a restricted function class; failure does not prove lost information.

### Input-normalization comparator

Before normalized-input model outcomes, fix per-window zero-mean/unit-SD normalization over the single EEG channel's 15-second waveform, with epsilon 1e-8. This is a mathematical calibration control, not a recommended physical-unit preprocessing recipe. Extract mean-pooled native features at gains 0.75, 1 and 1.5 for all four encoders. Fit normalized-input heads only at gain 1 using the same subject splits, C grid, balanced LBFGS and equal-validation-subject selection as the clean-trained probes. Report out-of-fold accuracy and actual native-feature/decision invariance across all windows, rather than inferring task utility from input equality. Such normalization may remove useful amplitude information; compare its clean accuracy explicitly.

### Supervised-from-scratch practical comparator

After normalization/correction outcomes, add a clearly labeled practical comparator before drawing any comparison to task-specific training. Its architecture is fixed before training: Conv1d 1→8 (kernel 25, stride 4, padding 12), GELU, Conv1d 8→16 (kernel 15, stride 4, padding 7), GELU, global temporal average, linear logit. No BatchNorm or pretraining. Use three initialization seeds (9137/9138/9139, plus held-subject offset), the same subject folds, balanced BCE, AdamW lr 1e-3/weight_decay 1e-4, batch 32, at most 40 epochs, early stop after six epochs without validation improvement. Select checkpoints by equal-validation-subject BA (loss breaks ties). Compare clean-only training, training gains 0.5/2 sampled per example, and unit-SD-normalized training. Evaluate all five gains; average seeds within subject before uncertainty. This is not an architecture-matched pretraining ablation and does not prove a foundation-pretraining effect. It was added after the gain findings and is exploratory.
