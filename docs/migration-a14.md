# Migrating from a13 to a14

a14 is additive. Existing runtime, intervention, attribution, probe, SAE and path
APIs retain their a13 behavior.

`welch_power_spectrum` complements the existing periodogram with explicit Hann
segment and overlap settings. `time_domain_features` and `spectral_features` return
`EEGFeatureSet`; choose `.matrix(aggregation="flatten")` to retain all named
channel/patch coordinates or `"mean"` to average them. `channel_correlation` and
`band_connectivity` return structured channel matrices rather than flattening them.

Install `eegfmlens[aperiodic]` to use `fit_aperiodic_decomposition`. Fit once on a
declared reference cohort and pass the result to `remove_fitted_spectral_component`;
the library does not refit implicitly during application.

`cross_model_similarity` requires `Activation` records from exactly the same trial-ID
set. It reorders the right collection to the left order and rejects missing or extra
trials. Pass subject IDs through `groups` to compare within-subject-centered geometry.
