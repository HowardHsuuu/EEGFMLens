# Migrating to 0.1.0a19

a19 is additive. Existing runtime, spectral, SAE and intervention APIs retain their
a18 behavior.

## Held-out spectral readout

Build amplitude targets from the exact model-ready batch, align a cached activation,
then fit on a caller-declared split:

```python
from eegfmlens import (
    FrequencyBand,
    activation_spectral_matrix,
    amplitude_spectral_targets,
    fit_spectral_readout,
)

targets = amplitude_spectral_targets(
    batch,
    scope="patch",
    transform="log1p_amplitude",
    f_min=0.5,
    f_max=45,
)
features = activation_spectral_matrix(cached_activation, targets)
result = fit_spectral_readout(
    features,
    targets,
    train_mask=train_rows,
    test_mask=test_rows,
    alpha=1.0,
)

sae_signatures = result.readout.direction_signature(sae.decoder_weight)
alpha_signatures = result.readout.band_signature(
    sae.decoder_weight,
    FrequencyBand(8, 13, "alpha"),
)
```

Targets use channel-mean one-sided amplitude with trial/patch row coordinates.
`activation_spectral_matrix` supports physical sites and declared channel-major token
layouts at patch scope; trial scope accepts the existing mean, flatten or CLS pooling
rules. The fit fails when provenance, geometry, device or dtype differ.

The readout's direction signature is a signed prediction in the chosen amplitude
coordinate, not a measured spectrum. Interpret directions only after reporting
held-out per-frequency fidelity. Construct train/test masks by subject or recording;
patch rows from the same recording must not cross the split.

## Score post-intervention internal states

`sae_feature_sweep` now accepts `run_metrics` and `cache_sites` alongside ordinary
native-output metrics:

```python
sweep = sae_feature_sweep(
    lens,
    evaluation_batch,
    site,
    sae,
    feature_ranking,
    feature_counts,
    metrics={"task": task_score},
    run_metrics={"decoded_alpha": decoded_alpha_score},
    cache_sites=(site,),
)
```

Each run metric receives `(RunResult, one_trial_batch)`. Requested caches contain the
post-intervention activation, so a fixed held-out readout can be evaluated at every
feature count. Output and run metric names must be disjoint. The result records both
metric sets and every cached site.
