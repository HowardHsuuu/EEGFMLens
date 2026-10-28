# Current checkpoint checks

Fourteen component views across all eleven adapter families were executed with
this source release: 638 site/batch records, batch sizes 1 and 2, eager CPU
float32. Every retained native baseline, identity, independent intervention and
cleanup comparison was exactly equal. This is bounded integration evidence,
not a claim about all variants, devices, inputs or task quality.

CBraMod has feature and explicitly masked reconstruction views; LaBraM has its
pretrained student feature view. The catalog runner covers EEGPT's target
encoder, both BIOT encoders, BENDR encoder and contextualizer, BrainOmni tiny,
CSBrain unmasked forward, NeuroRVQ's four raw branches, SignalJEPA's encoders,
DIVER's EEG backbone and ST-EEGFormer's small encoder. Reconstruction admissions
for the six-model study remain a separate artifact: these encoder checks do not
replace them. All declared sites in these views are tested at both batch sizes.

## Verify retained evidence

From the repository root, with Python's standard library only:

```bash
python validation/verify_current_checkpoints.py
```

The verifier checks report hashes, the complete current runtime source inventory,
runner/helper bindings, expected site/batch counts, invocation and numerical
results. It does not load weights or rerun inference. Reports from earlier
releases remain unchanged in `../checkpoint_reports` and use their own verifier.

## Repeat model execution

Install this package and the dependencies recorded in the chosen report's
`environment`. Supply trusted local upstream source and weights; neither runner
downloads anything. Use a fresh process per component and a new output filename.
`../run_checkpoint_checks.py` accepts pinned CBraMod or LaBraM source:

```bash
python validation/run_checkpoint_checks.py --model cbramod --component features \
  --source /path/to/CBraMod --checkpoint /path/to/cbramod.pth \
  --checkpoint-sha256 EXPECTED_SHA256 --report /path/to/new-report.json
```

Use `--component reconstruction` for CBraMod's explicit-mask view, or
`--model labram --component features` with LaBraM source and weights. Expected
weight hashes and upstream revisions are in the corresponding reports.

For the other components, arrange assets at the relative paths in
`catalog-assets.json` (the `assets:` prefix is relative to your asset root;
`braindecode:` is relative to that installed source package):

```bash
python validation/run_catalog_checkpoints.py --model eegpt \
  --asset-root /path/to/assets \
  --manifest validation/current_checkpoints/catalog-assets.json \
  --report /path/to/new-report.json
```

For SignalJEPA also supply `--braindecode-source /path/to/braindecode`. The model
choices are listed by `--help`. The manifest binds source, configuration and
weight bytes before import. Obtain models according to `docs/models.md`; exact
converted/mirror weight identities are retained rather than silently substituting
another release. Asset filenames and hashes are the execution contract. No
participant recordings, preprocessing or optimization are involved.

Two existing source compatibility changes must be reproduced in a separate
validation checkout before manifest verification: in BrainOmni's
`model_utils/vq.py`, replace `import deepspeed.comm as dist` with
`import torch.distributed as dist`; in DN3's `dn3/utils.py`, replace
`from collections import Iterable` with `from collections.abc import Iterable`.
These are CPU/Python compatibility checks, not unmodified-upstream claims.
SignalJEPA imports unchanged model/modules source without package initialization.
NeuroRVQ explicitly removes absent downstream norms and excludes pretraining
heads; ST excludes decoder keys and an unused parent image-position parameter.
The constructors strictly load the remaining declared state and record exclusions.

## Oracle and interpretation

CBraMod/LaBraM compare a C4/patch-0 edit to hand-indexed native hooks and require
all untouched exposed entries to remain exact. The other components compare
whole-state zero ablation to independent native hooks at the declared invocation.
Those whole-state tests do not independently certify every physical sub-selection.
Native calls avoid adapter forward/layout conversion, but use declared geometry
metadata and the same loaded model. BrainOmni and DIVER pair execution RNG because
native evaluation contains dropout. Zero endpoint effects are retained; an
executable hook need not affect every readout. Fixture tests and study-specific
admissions supply their own coordinate and scientific-operation evidence.

All reports describe the measured environment and tolerance (atol 1e-6,
rtol 1e-5). Failures abort that run and existing reports are never overwritten.
Re-execution can vary with dependencies or hardware; do not edit retained
reports to make a different execution appear identical.
