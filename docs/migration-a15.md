# Migrating to 0.1.0a15

This alpha adds covariance-aware least-squares concept erasure without changing
existing cache, adapter or intervention contracts.

`fit_cross_covariance_subspace` remains the Euclidean SVD construction introduced
in a12. It returns an orthonormal basis for `SubspaceAblation` and is not labeled
LEACE. Use the new reference-fit path when feature covariance should determine the
minimum-displacement eraser:

```python
from eegfmlens import LEACEAblation, fit_leace_eraser

eraser = fit_leace_eraser(reference_features, reference_concepts)
intervention = LEACEAblation(site, eraser)
result = lens.run_with_interventions(batch, interventions=(intervention,))
```

`reference_features` and `reference_concepts` must be finite floating matrices with
the same rows, dtype and device. One-dimensional continuous concepts are accepted;
categorical concepts should be encoded explicitly with `categorical_targets`.
Retain the fitted `LEACEEraser` with the study artifacts. Run manifests contain its
fit dimensions, rank, numerical settings and parameter hashes, but not the fitted
tensors or reference data.

The empirical feature covariance is the default. A nonzero
`covariance_shrinkage` mixes it with an isotropic covariance of equal trace and
changes the displacement metric, so treat the value as part of the analysis plan.
Use `fit_random_subspace_control(..., rank=eraser.rank, seed=...)` with several
prespecified seeds to measure same-rank representational damage.

`LEACEAblation` applies the complete affine map along the final feature axis. It
accepts sensor, patch or non-feature axis selections but rejects partial selection
of the transformed feature axis because that operation is no longer LEACE.

Run `python examples/concept_erasure.py` for an offline known-answer experiment.
