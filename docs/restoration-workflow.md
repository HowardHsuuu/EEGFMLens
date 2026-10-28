# Activation-restoration workflow

The flagship workflow asks a concrete mechanistic question: after a declared
sensor/time corruption changes a model's output, which internal layer and input
coordinate contain clean-run information that is sufficient to move that output
back toward its clean value?

`restoration_sweep` runs the clean and corrupted input independently for every
trial, caches only the requested sites, and patches clean activations into the
corrupted run. Its score is negative normalized L2 output error relative to a
separately executed clean reference. A positive `clean_error_reduction` therefore
means that the patch moved the native output toward the clean output.

```python
from eegfmlens import restoration_sweep

result = restoration_sweep(
    lens,
    clean_batch,
    corrupted_batch,
    sites=("embedding.output", "blocks.5.output", "blocks.11.output"),
    targets=targets,
    recovery_site="blocks.11.output",
    seed=91,
)
```

This is a reusable default for tensor-valued encoders. For task logits, a frozen
readout margin, or a prespecified physiological score, use the lower-level
`patching_sweep` and supply that score directly.

## Run the complete example

The deterministic demo has a known causal coordinate and requires no download:

```bash
python -m pip install '.[visualization]'
python examples/restoration_workflow.py --demo --output /tmp/eegfmlens-restoration
```

For a pinned CBraMod or LaBraM checkout, checkpoint, and public EEGMMIDB file:

```bash
python -m pip install '.[workflow,labram]'
python examples/restoration_workflow.py --model cbramod \
  --upstream /path/to/CBraMod \
  --checkpoint /path/to/cbramod.pth \
  --edf /path/to/S001R04.edf \
  --output /path/to/new-output-directory
```

The real example uses the explicit EEGMMIDB recipe documented in the
[real-EEG guide](validation.md). It scans embedding and early/middle/final blocks
over C3, Cz, and C4 by default. The public Python workflow itself works with any
adapter whose selected sites expose validated physical sensor/patch axes.

## Controls and output

Every run includes identity replacement at every site and a cleanup forward after
interventions. Random locations are disjoint from the target and matched to the
target activation-delta norm for each trial. Unmatchable or zero-delta controls
remain in the record with null effects and an explicit status. Setting
`recovery_site` additionally verifies that full-site replacement at a known
sufficient downstream site exactly recovers the clean output.

The example atomically creates a new directory containing:

- `sweep.json`: every baseline, intervention, control, selector, effect, validity
  status, model manifest, and input digest;
- `report.json`: the input/checkpoint hashes, preprocessing ID, capability list,
  sweep scope, row counts, and artifact hashes;
- `restoration_heatmap.png`: mean valid event restoration by layer and coordinate.

Existing destinations are rejected. The report deliberately labels the analysis
as an activation-restoration diagnostic. Restoration can show where clean-run
information is sufficient under the chosen corruption; by itself it does not
identify a physiological source, prove that a feature is necessary, establish a
human-interpretable concept, or support a clinical claim.

## Cross-model use

Run the same declared corruption, target grid, trial pairing, and downstream score
separately for each model. Compare prespecified summaries such as the layer at
which restoration first appears, sensor/time specificity relative to controls,
and the fraction of valid trials. Raw layer indices and activation magnitudes are
not automatically aligned across architectures. `model_info(...).evidence` and
`lens.capabilities()` state which integration and coordinate semantics are
actually validated before a model enters such a comparison.
