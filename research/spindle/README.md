# Spindle/N2 experiment

A subject-held-out experiment using frozen CBraMod and LaBraM, human spindle labels, internal interventions and strength-matched controls. Read [PROTOCOL.md](PROTOCOL.md) before interpreting outputs. This is an exploratory eight-subject study, not a clinical validation or a claim to introduce concept erasure.

## Data

Obtain `DatabaseSpindles.rar` from the [official DREAMS archive](https://zenodo.org/records/2650142). The [pinned public mirror](https://huggingface.co/datasets/lemon2004/DREAMS/resolve/b131c6d7380181a91047b573900af5186871c4bd/DatabaseSpindles.rar) was used after official gateway errors. Verify MD5 `2f8a101194e133dd4324f21047dce579` against the official record before extracting. Retain the included dataset license. Data and activation artifacts are not licensed under EEGLens's MIT license and must not be committed to the source repository.

Extract into a directory containing `excerpt1.edf` through `excerpt8.edf`, their text exports, hypnograms and visual-scoring files. The text exports are required for voltage-scale verification: these EDF headers use uppercase `UV`, which affected the numerical scaling in the tested MNE reader. Subject 6's text export has extra trailing samples; the EDF-length prefix aligns.

Obtain local official checkpoints using [the model documentation](../../docs/models.md). The pipeline does not download files implicitly.

## Run

From the EEGLens repository:

```bash
python -m pip install -e '.[labram,eeg,dev]'
python -m pip install -r research/spindle/requirements.txt
pytest research/spindle/test_experiment.py
python research/spindle/run.py \
  --data /path/to/extracted/DatabaseSpindles \
  --work /path/to/new-experiment-directory \
  --cbramod /path/to/cbramod.pth \
  --labram /path/to/labram-base.pth
python research/spindle/report.py \
  --work /path/to/new-experiment-directory \
  --destination research/spindle
```

The work directory must not already exist. All subjects stay disjoint across train/validation/test; eight folds train on five subjects, validate on two and test on one. A single CPU run fits local memory; no GPU is required for this dataset. The runner records source hashes, dependency versions and output hashes.

Outputs:

- `prepared/`: verified model inputs and source/annotation manifest.
- `features/`: pretrained and random-weight pooled features for both models.
- `results/baselines.json`: per-subject staging and spindle-probe metrics.
- `results/readouts.npz`: fitted fold-specific linear parameters.
- `results/*-directions.npz`: directions, centers and nuisance transformations.
- `results/*-interventions.json`: per-window original/changed margins and perturbation norms.
- `results/summary.json`: subject bootstrap intervals, matching diagnostics and control contrasts.

Individual steps are executable as `prepare.py`, `extract.py`, `baselines.py`, `intervene.py` and `summarize.py`. Their command-line help describes paths. No layer or hyperparameter is chosen from test/intervention outcomes.

## Interpretation limits

The completed local run and its limitations are in [RESULTS.md](RESULTS.md). Verification is recorded in [AUDIT.md](AUDIT.md). Two corrected full runs produced identical inputs, features, baselines and per-window intervention records.

N2 is recognized in stable 15-second windows, not a full clinical 30-second staging pipeline. Concept labels after 990 seconds are excluded due to the documented annotation cutoff. Probe fitting uses reviewed training N2, so it does not simply learn N2 vs other stages. Eight subjects offer limited precision; bootstrap intervals are descriptive and exploratory, with overlapping training sets across folds.

The spectral/amplitude adjustment is linear and may remove true spindle-related information as well as confounding. Failure after adjustment does not prove that the model lacks spindle morphology. Random-weight encoders are a single-seed diagnostic, not a fully characterized null distribution.

Other-event control is an explicitly signal-defined slow-wave morphology score. It is not a human K-complex annotation: the separate K-complex subset cannot be joined using excerpt numbers or its uninformative subject headers. Matching quality must be inspected; a caliper does not guarantee perfect covariate balance.

A full pipeline run demonstrates a falsifiable use of EEGLens. Encoding, decision dependence, perturbation specificity and common mechanisms are separate claims; do not equate any changed prediction with a physiological circuit.
