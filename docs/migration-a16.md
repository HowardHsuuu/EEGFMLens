# Migrating to 0.1.0a16

a16 is additive. Existing cache, intervention, attribution, probe, concept-erasure
and SAE APIs retain their a15 behavior.

Fit a binary ridge concept activation vector (CAV) on an explicit reference split,
then measure an objective's local sensitivity along that direction using gradients
from a declared native site:

```python
from eegfmlens import attribute, fit_concept_direction, tcav_score

concept = fit_concept_direction(
    reference_features,
    reference_labels,
    train_mask=train_mask,
    test_mask=test_mask,
)
gradients = attribute(
    lens,
    evaluation_batch,
    objective,
    method="gradient",
    sites=(site,),
)
score = tcav_score(gradients, site, concept, sample_unit="trial")
```

`tcav_permutation_test` fits the observed CAV and same-split random-label CAVs,
then compares their classic positive-sign fractions using the already computed site
gradients. It returns every directional sensitivity and the mean as well as the
classic score. The package does not infer exchangeability blocks or correct across
sites, concepts or objectives; subject-level resampling and multiple-comparison
control remain part of the study design.

`sae_concept_profile` combines positive/negative feature firing rates with cosine
alignment between every SAE decoder row and a supplied activation-space direction.
These values are descriptive. Fit reference target codes separately and clamp
selected features while preserving the native reconstruction residual:

```python
from eegfmlens import SAEFeatureClamping, fit_sae_code_reference

reference = fit_sae_code_reference(sae, target_cohort_activations)
intervention = SAEFeatureClamping(site, sae, selected_features, reference)
result = lens.run_with_interventions(batch, interventions=(intervention,))
```

The reference is bound to hashes of the exact SAE parameters. Changing the SAE after
fitting causes application to fail. A feature-axis `AxisSelection` is rejected because
it would apply only part of a decoder contribution. Run
`python examples/concept_attribution.py` for the offline known-answer chain.
