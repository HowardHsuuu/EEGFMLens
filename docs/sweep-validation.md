# Sweep API validation

`tests/test_sweep.py` and `examples/patching_sweep.py` validate the public sweep
interface without scientific datasets or fitted readouts.

- The analytic example checks all 24 layer/channel/time/trial effects, including irrelevant zero-effect coordinates.
- Tests cover token/CLS geometry, donor trial alignment, batch/order consistency and per-trial norm matching.
- Invalid or overlapping controls, score-shape errors and zero-norm cases produce explicit failures or invalid records rather than fabricated effects.
- Identity and cleanup checks preserve caller-owned model state and hooks.

```bash
python -m pytest -q tests/test_sweep.py
python examples/patching_sweep.py
```

The installed-wheel CI executes the analytic example outside the source checkout
and uploads its report with the workflow artifacts.
Native model conformance is documented separately in [validation](../validation/README.md).
These checks establish intervention execution, not a physiological mechanism.
