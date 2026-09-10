# Within-model EEG information dependence

Status: the fixed three-model study is complete; exploratory native contrast-task follow-up is also complete. This directory contains research scripts and retained reports, not a self-contained dataset/checkpoint distribution.

## Start with the current evidence

| Analysis | Report | What it establishes |
| --- | --- | --- |
| Fixed subject-disjoint study | [Main results](REPORT.md) | Three models, random-initialization/spectral baselines and controlled interventions; no established shared physiological mechanism |
| Probe erasure and refitting | [Recoverability](RECOVERABILITY.md), [iterative recovery](ITERATIVE_RECOVERY.md) | Original-probe failure does not establish information removal; pooled and individual-subject scores differ |
| Native contrast versus sensor erasure | [Task comparison](CONTRAST_TASK_RESULTS.md) | All three models complete. Failed clean physiological readouts limit interpretation of intervention loss |
| Alternative final-layer readouts | [Pooling diagnostic](FINAL_POOLING_DIAGNOSTIC.md) | A failed readout does not establish absent information; this post-results diagnostic does not replace the fixed intervention readouts |

See [evidence navigation](EVIDENCE.md) for retained hashes and historical snapshots, and [pretraining exposure](PRETRAINING_AUDIT.md) before interpreting subject-disjoint results. The preparation and pilot sections below describe the original workflow; their historical pending statements are not the current study status.

## Reproduction boundaries

For a checkpoint-free replay of the completed contrast-task summaries, clean-readout evaluations and figures, use the [portable reanalysis bundle](REPLAY.md). Its archive was extracted outside the checkout and validated locally; this replays retained responses, not native model inference.

Commands through the frozen-test section assume the `eeglens/` repository directory. Sections explicitly marked “parent workspace” instead use its parent. Install EEGLens and the research dependencies in the Python environment used for the commands; the research scripts are not installed CLI entry points. Model execution additionally requires local checkpoint files, and CSBrain requires its upstream source checkout.

`extract.py --root /path/to/external/research` currently expects this external layout:

```text
/path/to/external/research/
  eeglens_build_evidence/cbramod.pth
  eeglens_build_evidence/labram-base.pth
  eeglens_model_validation/expansion/CSBrain.pth
  eeglens_model_validation/repos/CSBrain/models/CSBrain.py
```

The frozen protocol also pins original artifact hashes. Recomputing artifacts with a different environment may produce different bytes; a rejected hash must not be bypassed and described as reproducing the frozen run. A new run needs separately recorded provenance. Downloaded EEG, weights and large intermediate arrays are outside this repository, so the commands below alone do not constitute a verified fresh-machine reproduction. For a smaller instrumentation example accepting explicit EDF/checkpoint paths, see [real_eeg.py](../../examples/real_eeg.py); its preprocessing differs from this study and it makes no task or mechanism claim.

The primary question is whether decodable physiological information has comparable functional importance across EEG foundation models on subject-disjoint evaluation. Cross-model transport is a later experiment, contingent on selective within-model effects.

## Shared data preparation

Use local EEGMMIDB 1.0.0 unilateral motor-imagery runs 04, 08 and 12. Labels from other run types must not be mixed in. Dataset: https://physionet.org/content/eegmmidb/1.0.0/; cite Schalk (2009), DOI 10.13026/C28G6P, and retain Open Data Commons Attribution License v1.0 attribution.

```bash
python research/mi/inspect_recording.py --edf /data/S001R04.edf --output /work/inventory.json
python research/mi/prepare.py --edf /data/S001R04.edf --output /work/prepared
```

Preparation saves unfiltered, unresampled, original-reference EEG in SI volts, with 640 samples per four-second trial. It retains canonical channel names, subject/run/trial identifiers, source hashes and event sample anchors. Short events are explicitly excluded with reasons; duplicated trial IDs and misaligned onsets fail. Model-specific conversion must consume these same trials and record its own reference, units, filtering and resampling. Do not feed these volts directly to a model expecting scaled microvolts.

Shared descriptors use a separate common-average-reference copy: C4 minus C3 natural-log mean Welch PSD in [8,13) and [13,30) Hz. Non-target descriptors are mean O1/O2 log alpha PSD and global log RMS. Welch uses a Hann two-second window and one-second overlap. These are concurrent asymmetries, not baseline-relative ERD. Non-target descriptors are controls for collateral changes, not independent physiological ground truth.

Development evidence: S001R04 produced 15 complete trials (8 left, 7 right). Saved epochs were independently compared exactly with MNE native sample slices. Swapping C3/C4 reverses asymmetry signs while preserving the non-target descriptors. Multiplying EEG by three preserves asymmetries and shifts log PSD/log RMS by 2 log(3)/log(3). Local evidence is in `research/eeglens_mi/development-v1/verification.json` outside the package repository. This is single-subject pipeline QA, not generalization evidence.

## Required before scientific evaluation

`split-v1.json` now fixes 30 subjects without inspecting their recordings/labels (except prior S001 development): 18 training, 6 validation, 6 test. S001 is forced into training. Selection of the first 30 subject identifiers is a bounded study cohort, not a representative population sample. `fetch.py --split research/mi/split-v1.json --output /data/eegmmidb` explicitly downloads its 90 recordings with four workers, validates EDF record sizes and retains source hashes. Existing validated files are reused. A completed download report is written only when all jobs finish; a started process is not evidence that the dataset is complete.

- Fix subject-disjoint train/validation/test membership. S001 is development data and must not become held-out test evidence.
- Audit each model's pretraining exposure; downstream subject separation alone does not establish unseen-pretraining subjects.
- Audit model preprocessing and choose three compatible pretrained models. Preserve identical trial identity and physical time extent.
- Establish spectral and random-feature baselines, then fit task and descriptor readouts on training subjects only.
- Select middle-layer sites, ranks and intervention doses using validation subjects only. Lock the protocol before test evaluation.
- Compare concept erasure with rank-matched random subspaces and energy-matched controls; measure target and non-target response, collateral task damage and uncertainty over subjects.
- Interpret visual cue direction as a task confound, and distinguish power asymmetry from causal motor physiology. No mechanistic claim follows from a successful descriptor probe alone.

## Partition preparation

```bash
python research/mi/prepare_split.py --split research/mi/split-v1.json \
  --partition train --data /data/eegmmidb --output /work/train
```

Use separate output directories for each partition. The runner refuses an incomplete partition, an empty split group or repeated subjects across any groups; it does not silently shrink the cohort to available downloads. It checks that prepared subjects exactly match the requested partition, and records the frozen split hash in the manifest. Development checks verified that missing recordings and deliberately overlapping subject membership are rejected. This enforces membership only: all later fitted preprocessing and readouts must still restrict fitting to training data.

## Three-model extraction

`extract.py` loads official CBraMod, LaBraM and CSBrain weights from supplied local paths and consumes a prepared artifact only after checking its hash. The current experimental recipe is shared 19-channel, original-reference EEG, polyphase 160-to-200 Hz resampling, four one-second patches, microvolts divided by 100, and no additional bandpass. This explicitly controls the input across models; it is **not** presented as reproducing all original training pipelines. Preprocessing sensitivity and pretraining-exposure audits remain necessary for interpretation.

Layers 5 and 11 (zero-indexed) are retained as initial middle/final candidates. Pooling averages time within each electrode, preserving the physical electrode axis; LaBraM CLS is excluded and CSBrain's internal sorting is undone by its adapter. The saved final feature is the last encoder-block cache, not CSBrain's reconstruction projection. Files include trial IDs, labels, subject IDs, descriptor targets, channel order and source/artifact hashes. No probe fitting occurs during extraction.

```bash
PYTHONPATH=src python research/mi/extract.py --prepared /work/train \
  --model cbramod --root /path/to/external/research --output /work/train-features
```

The initial S001 development run is only an execution/alignment check. Do not use it as evidence for cross-subject decoding or selective physiological dependence.

## Probe fitting

`probes.py` accepts only train and validation feature artifacts with matching model, checkpoint, preprocessing, sites and pooling. Subject overlap is rejected. Feature mean/scale and ridge coefficients use training data only; validation MSE selects alpha independently for each target from 0.001, 0.01, 0.1, 1, 10, with penalty `n_train * alpha`. Task labels are fitted as -1/+1 regression margins, not calibrated probabilities. No test artifact is accepted by this fitting CLI.

Final-layer electrode-preserving features predict the task margin and four descriptors. Middle-layer C4-C3 feature contrasts predict mu/beta asymmetry. Their two native-coordinate coefficient vectors define the rank-two intervention span; numerically rank-deficient fits fail. The erasure center is the training mean over C3/C4 and trials. Planned intervention selects C3 and C4; it does not erase all electrodes. Random rank-matched and energy-matched interventions remain required before drawing a selective-dependence conclusion.

Synthetic implementation QA compared ridge coefficients against independently solved normal equations, checked the native-coordinate intercept, and verified that fixed-alpha fitted parameters are unchanged when validation data are perturbed. This is not evidence of successful EEG probing; full-partition fits have not yet run.

## Validation intervention pilot

`intervene.py` currently accepts validation artifacts only, pending a locked test protocol. It applies the fitted rank-two span at middle block 5 on C3/C4, with doses 0.5 and 1.0, and evaluates fixed final-layer readouts. Three seeded random rank-two spans (31, 71, 113) provide both unscaled-dose and per-trial norm-matched controls. It records actual delta norms and multipliers, retaining large gains for diagnosis rather than silently clipping them. Norm matching uses activations, not validation labels. The operation is constructed through public SubspaceAblation and same-trial Replacement APIs.

Implementation checks on a random CBraMod fixture verified exact native identity at dose zero, matched delta norms and exact preservation of unselected electrodes. This establishes control construction, not selective biological effects. Subject-level uncertainty, baseline models, multiplier diagnostics and a locked test evaluation remain required.

## Spectral task baseline

`spectral_baseline.py` uses the same 19 electrode locations and recorded reference, with 95 log-power features: five fixed bands [1,4), [4,8), [8,13), [13,30), [30,45) Hz per electrode. Welch runs at the native 160 Hz with two-second Hann windows and one-second overlap. Training-only ridge fitting and validation alpha selection match the model task-readout procedure. Report mean subject balanced accuracy as well as each subject's result. This baseline predicts task labels, not the descriptors from which some features can be reconstructed algebraically.

Implementation QA checked the expected 2 log(gain) feature shift under amplitude scaling, feature count, and balanced-accuracy behavior on imbalanced labels. The first full fit used 810 training trials from 18 subjects and 270 validation trials from 6 subjects; mean subject validation balanced accuracy was **0.52635**. This is validation-selected, not held-out test performance. Local evidence: `research/eeglens_mi/spectral-v1/summary.json`. Random-initialization model baselines remain additional required evidence before interpreting any pretraining advantage.

`summarize.py` reports intervention effects per subject, normalizes descriptor loss by training target variance, and uses subject bootstrap intervals. Concept-versus-matched-random differences are paired within subject before resampling. Its constant/symmetric-input bootstrap checks passed. These are exploratory intervals without multiplicity correction; an analysis script alone does not establish any intervention finding.

## Frozen test runner

`test_intervene.py` is a frozen held-out variant of the validation runner. It requires `--protocol test-protocol-v1.json` and `--split split-v1.json`, verifies all pinned source/probe/split hashes and test-subject membership before inference, and rejects any configuration differing from the implemented recipe. Its donor function and every intervention loop were compared by Python AST against the validation runner and matched exactly. The validation source is left intact for ongoing runs. A deliberately altered split hash was rejected before inference. The test runner reports its own hash and the protocol hash in each output manifest.

The three-model test and random-initialization comparisons are complete; current results and limitations are in `REPORT.md`. Native-preprocessing sensitivity and public-readiness work remain outstanding. Earlier pilot descriptions above are historical.
# Frozen-probe evaluation

Use `evaluate_probes.py` to evaluate existing coefficients without fitting on evaluation data:

```bash
PYTHONPATH=eeglens/src python eeglens/research/mi/evaluate_probes.py \
  --features research/eeglens_mi/features-v1/test/cbramod.npz \
  --training-features research/eeglens_mi/features-v1/train/cbramod.npz \
  --probes research/eeglens_mi/probes-v1/cbramod \
  --partition test \
  --output research/eeglens_mi/probes-v1/cbramod/test-evaluation.json
```

Run from the parent workspace in the research environment documented below. For constructor-random controls, use the corresponding `random-features-v1` and `random-probes-v1` paths. This script evaluates task balanced accuracy per subject and physiological concept R² pooled across trials; these aggregation levels are intentionally distinct. Consult `PROGRESS.md` for current results and the six-subject inference resolution limitation. Test evaluation does not imply that a subject was absent from pretraining (see `PRETRAINING_AUDIT.md`).

## Exploratory recovery visualization

See `RECOVERABILITY.md` for the completed training-subject cross-fitting analysis, reconstruction commands and score-audited PNG/PDF/CSV outputs. `plot_recoverability.py` needs only NumPy and Matplotlib and does not load model checkpoints. Its subject-wise view prevents pooled R² from being mistaken for consistent individual performance.

The [iterative recovery follow-up](ITERATIVE_RECOVERY.md) measures ranks 0/2/4/8/16/32 using fitting-only directions and rank-matched random controls. It reports remaining linear information, perturbation energy and non-target readouts, without claiming downstream causal effects.

The [mean-contrast intervention](CONTRAST_INTERVENTION.md) preserves C3/C4 common components and within-trial patch deviations. It has analytic and three-model native checks. The [task comparison](CONTRAST_TASK_RESULTS.md) contains completed full-subject results for all three models.

A [final pooling diagnostic](FINAL_POOLING_DIAGNOSTIC.md) shows that failed whole-electrode readouts do not establish absence of final-layer information; spatially constrained readouts improve LaBraM pooled beta prediction but remain heterogeneous across subjects.

See [evidence navigation](EVIDENCE.md) for current reports versus historical snapshots and the limits of hash-based verification.
