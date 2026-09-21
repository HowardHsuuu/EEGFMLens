# Migrating to 0.1.0a18

a18 is additive. Existing runtime, attribution, concept, SAE and intervention APIs
retain their a17 behavior.

## Controlled cumulative SAE intervention

`sae_feature_sweep` turns a prespecified feature ranking into trial-level intervention
curves. Supply one or more named callbacks that map the native model output and the
current one-trial `SignalBatch` to a floating tensor with shape `[1]`:

```python
from eegfmlens import sae_feature_sweep

result = sae_feature_sweep(
    lens,
    evaluation_batch,
    "blocks.5.output",
    sae,
    feature_ranking=(17, 4, 91, 8),
    feature_counts=(0, 1, 2, 4),
    metrics={
        "target": target_score,
        "off_target": off_target_score,
    },
    random_draws=100,
    seed=23,
)
```

Use `mode="clamp"` with a separately fitted `SAECodeReference` to replace selected
codes by a reference-cohort centroid. The default `mode="ablate"` subtracts their
decoder contributions while retaining the native SAE residual.

The returned raw score tensors have shape `[trial, step]`; random controls have shape
`[draw, trial, step]`. `mean_delta`, `integrated_mean_delta`, `area_between`,
`random_integrated_mean_delta` and `random_area_between` are descriptive summaries.
The package preserves metric values and directions and does not infer a scientific
label or a hypothesis-test unit.

Rank features, train the SAE, fit code references and train any learned metric without
using the evaluation trials. Use subject-aware uncertainty and multiplicity control in
the study. If the native model is stochastic, the caller must establish paired,
reproducible execution; the sweep's seed controls only random feature rankings.
