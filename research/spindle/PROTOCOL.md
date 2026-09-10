# Spindle representation and held-out N2 decisions

Status: data feasibility audit; no model outcome inspected. 2026-09-09.

## Objective and interpretation

Test whether frozen CBraMod and LaBraM encode expert-annotated spindle events and whether intervening on candidate directions changes N2 predictions on unseen subjects. Negative results remain valid. This is a reusable EEGLens experiment, not a claim to invent EEG probing, erasure or spindle interpretation.

Prior work already probes and erases physiological features in these models, including sleep tasks: https://arxiv.org/abs/2605.11410. The useful distinction here is event-level annotation, held-out subjects, controls for spectral/amplitude information, and matched perturbation specificity. Any novelty assessment remains provisional until the full closest papers are inspected.

## Data gates before locking analysis

- EEG, staging and expert events must have verified temporal correspondence.
- Explicit annotation coverage: unreviewed time is not a negative event label.
- Entire subjects remain in one outer split. Related recording subsets require verified subject identity, not filename assumptions.
- Train/validation must contain N2 and non-N2; report subject and class counts. Never infer N2 labels from spindle presence.
- Verify sensor/reference and sample rates from EDF headers; map only genuine electrodes to model channel IDs.
- Use human spindle labels as the concept target; detector outputs may only be secondary analyses.
- DREAMS official warning: expert-1 event counts cut off after 1000 s. Audit files and use conservatively verified review coverage.
- MODA reviewed segments are N2-only; cannot alone supply a staging classifier. Underlying full PSG access has application requirements.

Sources: https://zenodo.org/records/2650142 ; https://github.com/klacourse/MODA_GC . Raw data stays outside the package; retain dataset licensing and provenance separately.

## Analysis to lock after label/compute audit

Use outer subject-held-out folds with inner subject-held-out validation for regularization and layer selection. All normalization, candidate directions, spectral residualization, readout fitting and intervention calibration use training/validation only. No test-set selection of favorable layers, concepts, thresholds or seeds.

Frozen FM features feed a simple linear N2-vs-other readout. Report balanced accuracy, AUROC and class/subject counts, compared with majority, spectral/amplitude features and a random-weight encoder. Distinguish supervised readout dependence from claims about pretraining itself. Keep model-native temporal limits explicit: LaBraM supports at most 16 one-second patches; 30-second epochs require a fixed chunking/aggregation policy chosen before outcomes.

Probe event presence or reviewed event burden in the same temporal windows. Compare raw activation probes, spectral/amplitude probes, and train-fitted residual controls. Spindle-relatedness beyond sigma power is a separate claim requiring evidence.

Primary intervention outcome: held-out N2 margin change on spindle-positive vs matched spindle-negative windows. Also report performance/calibration changes on all stages. Fit low-rank candidate subspaces on training subjects; apply at an internal layer and run the remaining native network. Avoid using only final-layer linear algebra as circuit evidence.

Controls: identity; random direction matched in rank and per-example perturbation norm; shuffled concept labels fitted only on training data; spectral/amplitude direction; another expert event if aligned annotations are available. Matching variables and calibration use training data. Preserve and report raw perturbation norms so direction effects are not confused with strength. Check whether unrelated classes/readouts also deteriorate.

Aggregate at subject level; use subject bootstrap or paired subject effects with intervals, not epochs as independent replicates. Small datasets require explicitly exploratory conclusions. Two architectures support a cross-model comparison, not a universal foundation-model claim.

## Completion evidence

1. Reproducible data manifests, coverage audit, disjoint split assertions and fixed protocol/config.
2. Useful held-out prediction baselines for both models, or a documented failed baseline with justified next action.
3. Held-out concept decodability and spectral/amplitude comparisons.
4. Native interventions plus matched controls, per-subject outcomes and uncertainty.
5. A report separating encoded information, readout dependence and nonspecific damage; no positive finding required.

## Pilot configuration locked before fitting any readout (2026-09-09)

DREAMS spindle archive matches official MD5 `2f8a101194e133dd4324f21047dce579`. Eight subjects, 808 stable-stage 15-second windows. Use native central electrode referenced to A1; common 0.5–20 Hz bandpass, 200 Hz resampling, microvolts/100. Stage 2 is positive; all other scored stages negative. All three five-second labels must agree. Primary concept is expert-1 spindle overlap >=0.5 seconds; zero overlap is negative, partial overlap below threshold is excluded. Only windows ending <=990 seconds are concept-reviewed. Expert-2 agreement is a secondary sensitivity analysis, never a selection criterion.

Outer folds: each subject 1..8 held out once. Validation subjects are the next two cyclic IDs; the other five are training. Fit all transforms on training. Linear logistic readouts with balanced classes; C in [0.1,1,10] selected by validation balanced accuracy for staging and AUROC for spindle probing. Fixed encoder output pooling is the mean over sensors and patch tokens (exclude LaBraM CLS). Candidate internal sites are blocks.5.output and blocks.8.output, selected using validation concept discrimination, not intervention outcomes.

The pilot reports 8 subject effects, not 808 independent subjects. Outcome-based changes to this configuration must be versioned as exploratory follow-up. No staging or concept predictions had been examined at this lock point. The dataset contains sleep-disorder patients; do not generalize to healthy populations without replication.

## Intervention configuration (locked after baseline fitting, before interventions)

Baseline results are now available; no intervention outcome has been inspected. Keep the baseline-selected internal site fixed. Compare rank-one raw spindle direction, spectrally adjusted spindle direction, sigma-power direction, shuffled-spindle direction and 10 Gaussian random directions. All non-identity control perturbations match the raw-spindle perturbation's per-window Frobenius norm, after excluding LaBraM CLS. Report actual norm error. The adjusted spindle direction removes the training-fitted spectral-to-activation linear span; report its held-out discrimination separately.

K-complex files have no identical central EEG signals and have anonymized identifiers insufficient to establish subject overlap. Do not merge them by filename. Use a separately named **signal-defined slow-wave morphology control**, not an expert K-complex label: count negative half-waves after 0.5–2 Hz filtering, lasting 0.25–1.0 seconds, with trough below -40 microvolts, excluding the outer one second of each 15-second window. This operational control is exploratory and does not replace future expert-annotated K-complex validation. Its direction is estimated from training N2 only.

For covariance adjustment, standardize the seven spectral/amplitude features on training N2 and ridge-regress activations on them with alpha=10. Fit a logistic spindle probe on residual activations using the preselected C; orthogonally remove the fitted nuisance coefficient span from its raw-coordinate direction. Fixed random seeds are 4311 + 31*held_subject + control_index. Match direction-control strength per example to isolate orientation rather than perturbation magnitude. Subject bootstrap uses 10,000 samples, seed 4311; all intervals remain exploratory given eight subjects.

## Summary rule locked before reading intervention outputs

Primary selectivity contrast uses spindle-positive vs spindle-negative reviewed N2 windows within each held subject. Match without replacement by minimum total Euclidean distance on the seven training-standardized spectral/amplitude covariates; discard pairs whose distance exceeds sqrt(7). Report pair counts and covariate balance rather than assume matching succeeded. Effects are changes in N2 margin divided by the training-stage-margin standard deviation; report per-subject paired differences (positive minus negative), and random-control means. Negative contrast indicates preferential N2-margin suppression on spindle-positive windows. Do not label a mechanism established solely from this contrast; require direction specificity beyond random, sigma and slow-wave controls and inspect the adjusted-direction evidence.

## Mandatory numerical correction and invalidated first run

Before accepting any result, amplitude QC revealed a reader unit bug: these EDFs use uppercase `UV`. MNE 1.12.1 normalizes its displayed unit to microvolts but its EDF scaling branch only recognizes `uV` and mu-V spellings, so the numerical values remained in microvolts while `get_data(units='uV')` multiplied by another million. The independent text signal exports match the unscaled EDF values to their 4-decimal precision at zero lag. The initial prepared/features/results directories were archived under `invalid-unit-v0` and are invalid for scientific interpretation.

Preparation v2 compares each EDF sample with the independent text microvolt export, selects the necessary conversion to volts, verifies the converted signal, and checks plausible amplitude before filtering/scaling for the models. This accommodates both affected and corrected reader versions. Subject 6 text has 140 extra trailing samples; all EDF-length samples align, with no time shift. The full pipeline is rerun with identical split/model/analysis rules after this engineering correction. No outcome-driven parameter changes are allowed.

## Final reporting detail

Random-control accuracy is the mean of the ten separately scored control accuracies, not the accuracy of an averaged-logit ensemble. Linear margin contrasts commute with averaging, so their primary definition is unchanged. Independent second-rater sensitivity uses the fixed first-rater-trained probes on the six available subjects, without layer/parameter reselection. Neither model shows a reliable overall balanced-accuracy drop from the primary spindle erasure; margin selectivity must not be presented as an accuracy contribution.
