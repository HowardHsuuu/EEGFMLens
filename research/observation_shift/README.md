# Observation-shift mechanism study: pilot completed

The scientific question is whether observation-shift failure reflects a mismatch of the existing task readout, versus information that is no longer accessible to tested readout/correction families. This work does not claim a new robustness benchmark, a solved EEG foundation mechanism, or independent confirmation on DREAMS.

Read [the related-work map](RELATED_WORK.md), [the prospective protocol](PROTOCOL.md), and [the preliminary results](results/RESULTS.md). Existing robustness, layer probing, pooling and invariance-learning studies overlap substantially with the original broad proposal. The training-derived corrections and held-out functional tests are now complete. The calibration pilot does not support gain repair as the main scientific contribution. Read [the final report](results/FINAL_REPORT.md) and [the next candidate question](NEXT_QUESTION.md).

## Completed

- Gain gate: all 808 corrected windows, gains 0.5/0.75/1/1.5/2, two pretrained encoders and their existing random controls, fixed historical readouts, five-layer per-patch representations.
- Positive-gain inversion and normalization input-equivalence checks, and clean-feature reproduction against the previous study.
- Readability comparison: mean versus token-preserving logistic probes, clean versus shifted training, all subject-disjoint folds and unseen gain strengths. All scalers and heads are train-only; C is selected on the two validation subjects.
- Paired subject-level descriptive summaries. Source and protocol hashes are retained in the external feature/readout artifacts.

## Completed follow-up and decision

- Native mean/affine corrections, matched random/permuted controls and clean-condition harm: 96,960 valid intervention records, all five sites reported.
- Normalized-input readouts on all four encoders: zero decision flips across the three evaluated gains; CBraMod balanced accuracy 0.6869.
- Small supervised CNN: three seeds, eight held-subject folds, three training modes; 72 fitted checkpoints. Seeds are averaged within subjects.
- [All correction effects](results/CORRECTIONS.md), paired true-versus-null contrasts and final artifact hashes are available in the report artifacts.

The gain pilot is a reproducible negative feasibility result. Multiple random encoder seeds, architecture-matched training and independent multichannel/task data remain necessary for broader pretraining or real-device claims. They were not completed and no such claim is made. The next scientific candidate is separately specified in NEXT_QUESTION.md; cross-task experiments have not run.

## Reproduce

Use the existing environment for the spindle/phase experiments. All inputs and weights remain local and external to the package. Run each encoder sequentially with two CPU threads:

```bash
python research/observation_shift/gain_gate.py \
  --model cbramod --checkpoint /absolute/path/to/cbramod.pth \
  --previous /absolute/path/to/corrected-spindle-run \
  --output /absolute/path/to/gain-output
```

Repeat with `--random`, and for `--model labram` with its checkpoint, retaining the same output directory (each label writes a distinct file). It refuses to overwrite that label's feature file. Then:

```bash
python research/observation_shift/probes.py \
  --features /absolute/path/to/gain-output \
  --previous /absolute/path/to/corrected-spindle-run \
  --output /absolute/path/to/new-probe-output
python research/observation_shift/summarize.py \
  --gate /absolute/path/to/gain-output \
  --probes /absolute/path/to/new-probe-output \
  --output /absolute/path/to/report-output
```

`gain_gate.py` reuses the previous research model-construction helper, while all observations use the public EEGLens runtime. The gate caches activations; the subsequent `correct.py` stage performs native interventions. The existing prepared data are central-channel only; raw EDF channel inventories show additional but inconsistent EEG channels, which require a separate validated preprocessing/geometry design.

Follow-up commands (each fitting/evaluation output must be a fresh directory):

```bash
python research/observation_shift/fit_corrections.py --help
python research/observation_shift/correct.py --help
python research/observation_shift/normalize.py --help
python research/observation_shift/scratch.py --previous /absolute/path/to/corrected-spindle-run --output /absolute/path/to/study/scratch-v1
python research/observation_shift/summarize_corrections.py --source /absolute/path/to/study/corrections-v1 --output /absolute/path/to/study/report-v1
python research/observation_shift/finalize.py --root /absolute/path/to/study --output /absolute/path/to/study/report-v1
```

`finalize.py` expects the versioned subdirectories used in this study; it verifies expected result counts, averages scratch seeds within subjects, computes paired correction-minus-null contrasts and hashes the external artifacts. It does not substitute for native correctness checks or independent scientific replication.
