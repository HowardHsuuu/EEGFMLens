# Migrating to 0.1.0a17

a17 is additive. Existing runtime, attribution, concept and intervention APIs retain
their a16 behavior.

`source_attribution` applies the EEG-PRISM linear propagation rule to an additive
input attribution:

```python
from eegfmlens import source_attribution

mapped = source_attribution(
    input_attribution_result,
    source_delta,  # [trial, source, patch, sample]
    forward_matrix,  # [channel, source]
    source_names,
)
```

The result must come from input × gradient or integrated gradients. For input ×
gradient, `source_delta` is the estimated source activity relative to zero. For
integrated gradients, it is the difference between observed and baseline source
estimates. `forward_matrix` must synthesize channels in the exact order recorded by
the attribution result.

EEGFMLens deliberately does not solve the EEG inverse problem. Supply source
estimates and a matching forward matrix from a prespecified inverse pipeline. The
result reports per-trial forward reconstruction RMSE, relative reconstruction error
and attribution conservation error, so regularization or model mismatch remains
visible. Exact reconstruction yields additive conservation up to numerical error;
an approximate source solution need not.

Raw signed source attribution has shape `[trial, source, patch, sample]`.
`sum_over_time()` and `mean_over_time()` retain trial and source axes. Choose signed,
positive-only or absolute aggregation in the study rather than allowing the package
to silently discard negative evidence.

Run `python examples/source_attribution.py` for an exact known-answer experiment.
