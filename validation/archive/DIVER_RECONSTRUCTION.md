# DIVER native reconstruction source-API validation

This unreleased source change adds explicit output selection to DIVERAdapter.
Default unmasked features remain supported. Reconstruction uses the released
native time head; it does not fit, replace or recalibrate a decoder.

- 89 non-integration tests passed; two checkpoint-dependent tests deselected.
- Official native checkpoint:30 site/batch conditions, independent native zero
  hooks, identity replacement, physical final-output selection and cleanup pass.
  See `results/diver-native-reconstruction-v2.json`. CPU float32, synthetic three
  sensors/five patches, batch1/2, paired RNG9417. This is not all-geometry/device
  validation or downstream scientific efficacy.
- v1 native evidence predates an output-argument type guard. Its exact adapter
  snapshot is retained. v2 reruns and hashes the final adapter source.
- Isolated build produced an sdist and wheel. A fresh package venv installed the
  wheel with no dependencies, sharing the existing validated dependency directory.
  The isolated-mode installed checker verified33 package Python files,89 tests,
  quickstart,24 analytic sweep effects and dependency consistency. The portable
  report is `results/diver-reconstruction-wheel-v1.json`; its import-path prefix
  is explicitly normalized. This is not a wholly fresh dependency installation.
- No PyPI release or new version tag is claimed. The temporary wheel retains
  development version0.1.0a9; pin a Git commit for this unreleased API. Existing
  scientific runs remain pinned to their original commit and were not upgraded.

Reproduce the native check from the tool checkout with the existing native-source
and checkpoint layout documented in DIVER_STEEGFORMER.md:

```bash
PYTHONPATH=src python validation/validate_diver_reconstruction.py \
  --root /path/to/native-validation-assets --output /path/to/new-result.json
```

The independent mask oracle advances the native mask generator's RNG, explicitly
masks the physical input, and runs the native unmasked model. The instrumented
path instead overrides the native generated mask and invokes native masking.
Agreement therefore checks both output selection and mask timing. The new
terminal output site is a control surface, not a claim of a discovered circuit.
