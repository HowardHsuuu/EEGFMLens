# Real EEG example

Install the package plus the example dependencies, and set up the pinned upstream
checkout in [model setup](models.md):

```bash
python -m pip install '.[eeg,labram]'
python examples/real_eeg.py --model cbramod \
  --upstream /path/to/CBraMod --checkpoint /path/to/cbramod.pth \
  --edf /path/to/S001R04.edf --output /path/to/new-result.json
```

For LaBraM, use `--model labram`, its checkout and checkpoint. The example checks
the upstream Git revision before executing the supplied source. Use a trusted
checkout. No model source, weights or EEG are downloaded by the script.

A suitable public example is [EEGMMIDB S001R04](https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf).
Its SHA256 is `3d161f88e1c00632585287d2ce584c2bc0f08862438eb255ea8723e00fac693d`.
Download it separately and follow the dataset's terms.

The script selects 19 channels, filters 0.5–75 Hz, resamples to 200 Hz, scales
microvolts by 1/100, and constructs two four-second windows without rereferencing.
This is an instrumentation recipe, not a universal upstream preprocessing recipe.
It compares direct native outputs, identity replacement, final-site recovery and
selected C3/second-patch ablation against an independent raw-coordinate hook.
The JSON output records the actual inputs, checkpoint, software and errors.

An existing output file is rejected. Use a new filename for each run.
For the broader model-specific test suite see [integration validation](../validation/README.md).
