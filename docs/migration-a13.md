# Migrating from a12 to a13

a13 is additive. Existing cache, intervention, spectral, probe, SAE and path APIs
retain their a12 behavior.

New `attribute` calls run each trial independently and require a scalar objective.
Integrated gradients requires an explicit, trial-aligned `SignalBatch` baseline;
there is no implicit zero or population-mean reference. Input results include the
attribution, raw or path-averaged multiplier, delta, objective values and completeness
error. Requested internal sites additionally return path conductance, multipliers and
activation deltas.

Use `spectral_attribution` to propagate input × gradient or integrated gradients to
one-sided frequency coordinates. Use `occlusion_curve` or
`spectral_perturbation_curve` to test a predeclared target order. Their AOPC assumes a
higher-is-better score and should be compared across models only under the same score
definition and scale.

This release does not turn every declared adapter site into a checkpoint-validated
gradient site. Exact native models may contain nondifferentiable paths; the API fails
when the objective disconnects from an input or requested site.
