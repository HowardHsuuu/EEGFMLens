# Interpretability methods

EEGFMLens exposes operations that can be combined around one scientific question.
EEG frequency, phase, channel and time structure enter the same experiment contract
as activation replacement, probing, sparse features and circuit tests. They are not
a separate post-processing tier.

## From signal property to mechanism

A typical experiment starts with a paired input intervention and follows its effect:

```python
from eegfmlens import (
    CANONICAL_BANDS,
    patch_frequency_band,
    power_spectrum,
)

recipient = patch_frequency_band(
    clean,
    phase_scrambled,
    CANONICAL_BANDS["alpha"],
    component="phase",
    scope="trial",
)
before = lens.run_with_cache(clean, sites=sites)
after = lens.run_with_cache(recipient, sites=sites)
```

The spectral edit is recorded in `SignalBatch.transforms` and copied into run
metadata. Donor rows are matched by trial ID. Channel order, sampling rate,
preprocessing identity, unit, patch geometry, dtype and device must match. Trial
scope is accepted only for contiguous nonoverlapping patches; patch scope never
pretends that overlapping windows form one continuous signal.

`power_spectrum` returns a one-sided FFT periodogram in signal-unit squared per Hz.
It is not Welch averaging or a periodic/aperiodic fit. `scale_frequency_band`
multiplies complex coefficients by a nonnegative scalar and therefore retains phase.
`patch_frequency_band` can replace amplitude, phase, or the full complex coefficient.
Bands use a lower-inclusive, upper-exclusive interval, except that a band ending at
Nyquist includes the Nyquist bin. Sharp spectral boundaries can cause ringing and
should be paired with appropriate control bands.

## What is represented and what is used

`activation_matrix` makes pooling explicit. `fit_ridge_probe` fits only the matrix
it receives, and `layerwise_ridge_probe` requires disjoint train and test masks.
These tools measure linear availability. They do not show that a downstream head
uses the decoded property.

`fit_cross_covariance_subspace` obtains orthonormal feature directions from the SVD
of centered feature–concept cross-covariance. Its `basis` and `center` plug directly
into `SubspaceAblation`, turning a descriptive representation into a causal removal
test. Fit the subspace on training subjects and evaluate both residual decodability
and task behavior on held-out subjects. This is the simplified Euclidean construction
used in BrainPEC. It is not covariance-whitened LEACE and the API names it accordingly.

`group_variance_decomposition` exactly separates between-group and within-group sums
of squares. `within_group_contrast_consistency` compares a binary condition-contrast
direction across groups and reports mean pairwise cosine plus a precisely defined
contrast SNR. For EEG-FM audits, groups will often be subjects. Both are descriptive;
the study must estimate uncertainty at the subject level.

## Sparse features

`TopKSAE` uses ReLU Top-K codes and unit-norm decoder directions. `train_sae` is a
small deterministic reference loop over a caller-supplied training activation
matrix. `sae_metrics` reports reconstruction MSE, explained variance, mean active
features and the fraction of features activated at least once. Dataset splitting,
dead-feature resampling and feature naming are deliberately explicit study choices.

`SAEFeatureAblation` subtracts only selected decoder contributions from the native
activation. It preserves the SAE reconstruction residual instead of replacing the
whole activation with an imperfect reconstruction. `SAEFeatureSteering` adds fixed
multiples of unit-norm decoder directions. Runs hash every SAE parameter and record
feature IDs and steering coefficients.

## Proposed paths

`path_patch` tests one source-to-mediator hypothesis. It first patches donor state at
the source into a recipient run and caches the induced mediator state. It then patches
that mediator state into a fresh recipient. The result contains per-trial source and
mediated effects, and the workflow verifies identity patches at both sites in score
space. The chosen mediator must execute downstream of the source. A recovered effect
supports the tested path under these interventions; it does not discover a complete
circuit or establish physiological localization.

## Method provenance

The implementation is original MIT-licensed package code informed by the following
papers and their public repositories:

- [The Identity Trap in EEG Foundation Models: A Diagnostic Audit](https://arxiv.org/abs/2606.06647) and [FMScope](https://github.com/Jimmy110101013/fmscope): variance, subject-direction and spectral-ablation diagnostics.
- [What Do EEG Foundation Models Capture from Human Brain Signals?](https://arxiv.org/abs/2605.11410) and [BrainPEC](https://github.com/Kian-Chen/BrainPEC): feature-family probing, cross-covariance erasure and residual controls.
- [Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders](https://arxiv.org/abs/2605.13930) and its [companion repository](https://github.com/BrainCapture/mechanistic-interpretability-for-eeg-foundation-models): Top-K SAE, spectral decoding and feature interventions. The companion code is PolyForm Noncommercial; no code was copied into EEGFMLens.
- [Beyond Accuracy: Robustness, Interpretability and Expressiveness of EEG Foundation Models](https://arxiv.org/abs/2605.17562) and its [repository](https://github.com/urbansirca/Beyond-Accuracy-Robustness-Interpretability-and-Expressiveness-of-EEG-Foundation-Models): channel perturbation, attribution and block-wise probing controls.
- [EEG-PRISM](https://arxiv.org/abs/2608.13676) and [EEG-Xplain](https://arxiv.org/abs/2609.15687): physiologically grounded and space/time/frequency attribution directions tracked for future gradient APIs.

The current public API does not claim full LEACE, FOOOF/specparam decomposition,
gradient attribution, attention LRP, Q/K/V intervention, circuit discovery or a
cross-layer transcoder. Those names will be exposed only after their mathematical and
native-model contracts have dedicated correctness evidence.
