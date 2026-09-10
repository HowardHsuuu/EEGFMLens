# Native integration validation

These scripts validate EEGLens adapters against independent native hooks. They do
not fit scientific readouts or publish study results. The current scope is eleven
families / thirteen views; historical milestone sections below retain their
original counts. See [dense evidence](DENSE_VALIDATION.md) and the
[tool acceptance ledger](../docs/public-readiness.md).

Runners require explicit local upstream checkouts/checkpoints and the external
workspace layout recorded below; they never download those resources silently.
`checkpoint_fixtures.py` supplies model loading and input conversion for the three
spatial-model validators without importing any scientific analysis. Its recipe
identifier is retained for compatibility with historical evidence. JSON/XML
snapshots retain their original paths and hashes; moving scripts does not turn
those snapshots into fresh validation results.

# Additional pretrained encoder validation

Completed 2026-09-09 on local CPU (24 GiB host memory, torch 2.14.0, two threads). New families: EEGPT, BIOT and BENDR's convolutional encoder. BIOT additionally covers two checkpoint variants. No lab/paid compute or downstream training was used.

## Sources and weight provenance

- EEGPT native source: https://github.com/BINE022/EEGPT at `a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705`. Weight source: https://huggingface.co/braindecode/eegpt-pretrained/tree/e41cb3ae2ce4fd9eb736862292c91f8128d15618 . The official Figshare download returned 403. This is explicitly a converted checkpoint, not the original Figshare file. All `target_encoder.*` tensors load strictly after stripping the prefix; only the wrapper's `chans_id` buffer is excluded and replaced by the native channel-ID preparation from SignalBatch names. No missing encoder parameters are allowed.
- BIOT native source: https://raw.githubusercontent.com/ycq091044/BIOT/main/model/biot.py (exact downloaded source SHA256 in results). PREST weights: https://raw.githubusercontent.com/ycq091044/BIOT/main/pretrained-models/EEG-PREST-16-channels.ckpt . Six-dataset weights: https://huggingface.co/Tianyi1229/MindCine/blob/a0c5e4361a2f77b451d8845a51fb58596d08b67e/LLM_pretrained/BIOT/EEG-six-datasets-18-channels.ckpt . This second file is a third-party mirror, not an independent official download verification. Both full native encoder state dictionaries load strictly.
- BENDR native source: https://github.com/SPOClab-ca/BENDR at `ac918abaec111d15fcaa2a8fcd2bd3d8b0d81a10`. Official release https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha identifies the encoder and contextualizer assets; the direct encoder transfer reset partway through. Weight source used: https://huggingface.co/Tianyi1229/MindCine/blob/a0c5e4361a2f77b451d8845a51fb58596d08b67e/LLM_pretrained/BENDR/encoder.pt . The mirror SHA256 is recorded, but a complete original-release download was not available for byte comparison. The official ConvEncoderBENDR loads the full encoder state strictly.
- BENDR dependency DN3: https://github.com/SPOClab-ca/dn3 at `dc54fc61895fa82ea73ad74c6bba10963cdadab2`. Exactly one Python compatibility change was required: in `dn3/utils.py`, `from collections import Iterable` becomes `from collections.abc import Iterable`. Native encoder computation was not edited. The compatibility file's hash and diff are saved in external evidence.

Upstream code remains external. This package adds adapters and validation logic; it does not redistribute these newly downloaded model implementations or weights. The upstream BENDR repository did not include a top-level license file in this checkout; no BENDR source was copied into the package.

## Reproduce

The external root used here is `/Users/howardhsu/Desktop/bcilab/research/eeglens_model_validation`. It contains `repos/EEGPT`, `repos/BENDR`, `repos/dn3`, the downloaded `biot.py`, and `checkpoints/` using the filenames in `validate.py`. Use the pinned source revisions and weight URLs above, verify hashes against `results/`, and apply the one DN3 compatibility change.

The existing validation environment is `/Users/howardhsu/Desktop/bcilab/research/eeglens_survey_2026-09-09/.venv`. Additional installed dependencies include `parse==1.22.1`, `pyyaml-include==1.3.2`, `linear-attention-transformer==0.19.1` and its dependencies; `environment.json` records versions. Install EEGLens in the environment before running.

```bash
python validation/validate.py --root /absolute/path/to/external-root --model eegpt --output /absolute/path/to/results
python validation/validate.py --root /absolute/path/to/external-root --model biot --output /absolute/path/to/results
python validation/validate.py --root /absolute/path/to/external-root --model biot-prest --output /absolute/path/to/results
python validation/validate.py --root /absolute/path/to/external-root --model bendr --output /absolute/path/to/results
```

Each invocation overwrites its own small result JSON. Inputs are deterministic synthetic tensors with model-ready geometry, not a validated real EEG preprocessing pipeline. BENDR's 20 inputs deliberately have placeholder names in this runtime fixture; these must not be mistaken for a recommended montage. EEGPT tests four selected electrodes and sixteen 64-sample windows; BIOT tests the full checkpoint channel vocabulary and ten seconds; BENDR tests four seconds of 20-input data.

## Checks and limits

For each declared site at batch sizes 1 and 2: exact unhooked-vs-cached output, exact identity replacement, exact agreement between EEGLens zero ablation and an independently registered native zero hook, finite outputs, and cleanup. Final-site replacement on corrupted inputs with reversed trial IDs must recover the correctly reordered clean outputs. Saved records include native/exposed shapes, source/runner/checkpoint hashes and intervention output changes. All 68 site/batch conditions passed, and all tested ablations changed the output.

This establishes the tested runtime integration, not task performance. New sites use conservative `batch` semantics, so electrode/time-specific selection and corresponding sweeps remain unsupported. Real EEG preprocessing, different architectures/widths, fine-tuned wrappers, training/autograd and GPU execution are outside this validation. The initial BENDR result covers only the convolutional encoder; the subsequent full-composition validation is recorded separately. Known upstream warnings concern CPU autocast, STFT's rectangular window and eval-mode Dropout2d; no warning-driven architecture change was made.

## Subsequent expansion

The section above records the initial EEGPT/BIOT/BENDR encoder validation. See [EXPANSION.md](EXPANSION.md) and [coverage](../docs/model-coverage.md) for the current nine-family scope. Additional runners are `validate_brainomni.py`, `validate_csbrain.py`, and `validate_neurorvq.py`; run them from the parent `bcilab` workspace with EEGLens installed in the Python environment. They use the external root `research/eeglens_model_validation` and save to its `expansion/results` directory. Copied results in this repository are evidence snapshots.

`validate_reve_architecture.py` deliberately uses random initialization of the official base architecture and labels its output `pretrained: false`; it is an architecture contract check, not a substitute for the unavailable gated checkpoint. SignalJEPA has passed strict full checkpoint tests through the direct-source import route below. Full BENDR passed strict checkpoint loading and 22 native conditions, including reordered donor recovery.

### SignalJEPA direct-source validation

From `bcilab`, with Torch 2.14 and Braindecode 1.8.1 source installed:

```bash
MPLCONFIGDIR=research/eeglens_build_evidence/mpl-cache research/eeglens_survey_2026-09-09/.venv/bin/python eeglens/validation/validate_signaljepa.py --direct-source research/eeglens_survey_2026-09-09/.venv/lib/python3.12/site-packages/braindecode
```

This imports the original model, base class, utilities and interpolation modules unchanged. It bypasses only package `__init__` imports that pull in unrelated architectures and Torchaudio dependencies; no computation is mocked. The result records import mode, source and checkpoint hashes. Official-maintainer checkpoint: `braindecode/signal-jepa`, revision `51232ee0795a60e4378c17befe1e2ea5e94450c4`, full `model.safetensors` and `config.json` stored under external `expansion/braindecode-signal-jepa`. All keys load strictly; tests use the complete configured channel order and 128 Hz synthetic input. Omit `--direct-source` when using a complete standard Braindecode installation.

### Full BENDR and standard SignalJEPA

The official contextualizer is downloaded from `https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/contextualizer.pt` into external `expansion/bendr-contextualizer.pt`. `validate_bendr_context.py` strictly loads it and the previously validated encoder, then composes their native forwards with `nn.Sequential`. Results record hashes of both weights, source and runner. Model inputs in this fixture retain placeholder names; no real montage/preprocessing claim is made.

From `bcilab`:

```bash
MPLCONFIGDIR=research/eeglens_build_evidence/mpl-cache research/eeglens_survey_2026-09-09/.venv/bin/python eeglens/validation/validate_bendr_context.py
PYTHONPATH=eeglens/src MPLCONFIGDIR=research/eeglens_build_evidence/mpl-cache research/eeglens_model_validation/.venv-braindecode/bin/python eeglens/validation/validate_signaljepa.py
```

The second command uses Braindecode 1.8.1, Torch/Torchaudio 2.8.0 and standard package imports. Its result is `signaljepa.json`; the Torch 2.14 direct-source result is preserved as `signaljepa-direct-source.json`. Both passed.

## DIVER-1 / ST-EEGFormer follow-up

Two additional checkpoint-backed families bring current coverage to eleven. See [sources, commands, native RNG limitations and results](DIVER_STEEGFORMER.md). The earlier nine-family audit describes its own completed expansion.
