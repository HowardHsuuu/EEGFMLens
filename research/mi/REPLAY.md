# Portable contrast-task reanalysis

The completed three-model response analysis is available as a local, unpublished replay archive at parent-workspace path `research/eeglens_mi/contrast-replay-v1.zip` (approximately 8.8 MB). It contains nine fold response artifacts and manifests, six expected summary/readout files, four analysis scripts, the fixed protocol, dependency versions and attribution. Raw EEG and encoder weights are not included. It is separate from the EEGLens wheel.

Extract the archive, inspect its code, and run from the extracted directory in a Python 3.12 environment:

```bash
python -m pip install -r requirements.txt
python code/replay_contrast.py --bundle . --output /absolute/path/new-results
```

The output directory must not exist. The runner verifies required members and all file hashes before computation, rebuilds all three summaries and clean-readout audits, and compares every non-input-path result field against the retained expected files. Numeric tolerance is absolute/relative 1e-10. Original inference provenance remains compared; relocated input paths are excluded because their files are checked by hash. PNG/PDF figures are regenerated, numerically checked by the plotting script and hashed; rendering bytes need not match across environments.

Validation extracted the actual ZIP outside the checkout and replayed all three models successfully using the existing installed-dependency environment. Two negative checks also passed: an altered file was rejected by checksum, and an altered expected accuracy with its manifest checksum updated was rejected by numerical comparison. Neither failure left a successful output directory. Source formatting/lint checks passed. The archive digest and full replay evidence are in [validation record](results/contrast-replay-validation-v1.json).

This establishes portable reanalysis of retained responses on the same macOS ARM platform. It does not rerun the EEG encoders, constitute a fresh-machine dependency installation, validate another operating system, or provide an independent scientific replication. Source EEG attribution and code licensing are included in the archive. Checksums detect changes; they are not a publisher signature. The [completed results](CONTRAST_TASK_RESULTS.md) retain the physiological-readout and exploratory-analysis limitations.
