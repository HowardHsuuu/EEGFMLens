# Checkpoint-backed evidence reports

These are **recorded historical checks**, not a new evaluation of the current
package. They expose the evidence behind the eleven-family coverage statement:
13 component views and 488 site/batch records, with batch sizes 1 and 2.
All inputs were synthetic; no participant-level EEG or outputs are included.
Historical code used the `eeglens` namespace. The distributed package uses
`eegfmlens`; source digests identify the actual tested implementations.

## Inspect and verify

From the repository root, with Python 3.10 or later and no third-party libraries:

```sh
python validation/verify_checkpoint_reports.py
```

`manifest.json` binds every report to its SHA-256. Each report supplies the
checkpoint identity (digest, plus an informative label), native component,
implementation digests, input axes/shape/channels/rate, seed, every retained
site/batch result, oracle definition, numerical comparison rules, compatibility
changes and explicit limits. The digest identifies exact bytes; a label alone
is not sufficient to select weights. Checkpoint/source acquisition is described
in [model setup](../../docs/models.md). Some historical files were converted
weights or mirrors, as their component/checkpoint descriptions state.

The verifier checks integrity, schema, inventory, geometry and recorded outcomes.
It does **not** authenticate a downloaded checkpoint, execute a native model or
independently prove that a historical assertion was true. The per-site JSON
records are retained measurements/assertions, not regenerated model outputs.
Separate raw oracle tensors and machine-specific historical runners are not
included. The maintained native tests in [validation instructions](../README.md)
provide execution checks when the required sources/weights are supplied. The
[current checkpoint suite](../current_checkpoints/README.md) separately runs
14 component views on the current runtime; it does not relabel these historical
measurements as new executions.

## Historical component inventory

Input dimensions are batch, channel, patch, sample; native folding/reordering is
reported through cache/native shapes and oracle descriptions where retained.
The count is site/batch records, not participants or independent replications.

| Report | Exact component scope | Input geometry | Records |
| --- | --- | --- | --- |
| [cbramod](cbramod.json) | pretrained 12-block waveform model; unmasked synthetic forward | B x 3 x 2 x 200; 200 Hz | 98 |
| [labram](labram.json) | pretrained student forward_features; patch tokens and learned final norm | B x 3 x 2 x 200; 200 Hz | 74 |
| [eegpt](eegpt.json) | EEGTransformer target encoder; converted checkpoint, not masked reconstruction | B x 4 x 16 x 64; 256 Hz | 20 |
| [biot](biot.json) | BIOTEncoder, six-dataset 18-channel checkpoint from third-party mirror | B x 18 x 10 x 200; 200 Hz | 18 |
| [biot-prest](biot-prest.json) | BIOTEncoder, PREST 16-channel checkpoint | B x 16 x 10 x 200; 200 Hz | 18 |
| [bendr](bendr.json) | ConvEncoderBENDR, pretrained convolutional encoder | B x 20 x 4 x 256; 256 Hz | 12 |
| [bendr-context](bendr-context.json) | pretrained ConvEncoderBENDR and BENDRContextualizer composition | B x 20 x 4 x 256; 256 Hz | 22 |
| [brainomni-tiny](brainomni-tiny.json) | BrainOmni tiny encode; synthetic sensor geometry | B x 4 x 3 x 512; 256 Hz | 24 |
| [csbrain](csbrain.json) | CSBrain pretrained forward including projection; original region-sort mapping | B x 19 x 4 x 200; 200 Hz | 28 |
| [neurorvq](neurorvq.json) | NeuroRVQFM four raw backbone branches; pretraining heads and absent downstream norms excluded | B x 2 x 2 x 200; 200 Hz | 104 |
| [signaljepa](signaljepa.json) | SignalJEPA full-channel local/contextual encoder; decoder loaded but not executed | B x 62 x 3 x 128; 128 Hz | 20 |
| [diver](diver.json) | DIVER-1 EEG backbone y output; not the study iEEG time-reconstruction endpoint | B x 3 x 5 x 500; 500 Hz | 30 |
| [steegformer](steegformer.json) | ST-EEGFormer small forward_features; reconstruction decoder excluded | B x 3 x 48 x 16; 128 Hz | 20 |

## Oracle and compatibility qualifications

- EEGPT encoder, both BIOT variants and BENDR encoder compare cached execution
  against `adapter.forward`; this shares input preparation. Their native edit
  hooks are independent of the intervention selector. Neither comparison is an
  independent preprocessing oracle.
- CBraMod and LaBraM semantic reports cover direct baseline, physical-coordinate
  and partial/centered feature edits. They do not separately record the whole-zero
  and identity checks in that report; these must not be inferred from other rows.
- BrainOmni tiny records a CPU communication-import adaptation and paired RNG
  for native eval dropout. DIVER also pairs native dropout RNG. These historical
  checks are distinct from current CI against unmodified pinned source.
- NeuroRVQ excludes pretraining heads and newly initialized downstream norms,
  retaining raw pretrained branch features. ST-small excludes decoder keys and
  an unused image-position parameter. Exclusions are preserved in the JSON.
- A checkpoint identifier shared between two tests does not make their native
  components, masks or intervention paths interchangeable.

## Separate scientific execution evidence

`six_study_native_summary.json` is the manuscript's six-model native admission
summary, copied without changing its numerical fields. It distinguishes recorded
exact assertions from retained-array rechecks and states where raw oracle arrays
were not retained. The paper's native-validation appendix gives masks, sites,
counts and tolerances, and the companion study repository supplies the scientific
protocols. In particular, the historical EEGPT target encoder, DIVER EEG feature
path and ST-small encoder do **not** validate the masked reconstruction paths
used in the study. Those paths have their own admissions.

This evidence supports specified executions and interventions. It establishes
neither all-input equivalence, a model-wide mechanism, task accuracy nor
physiological validity. Runtime changes require tests of the affected paths.
