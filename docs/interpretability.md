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

## Named EEG concepts and connectivity

`time_domain_features` returns nine per-channel descriptors: Hjorth activity,
mobility and complexity; population standard deviation; RMS; zero-crossing rate;
line length; derivative standard deviation; and peak-to-peak amplitude. Flat signals
use zero mobility and complexity by definition. `spectral_features` returns log10
and relative power for caller-supplied bands plus normalized spectral entropy,
centroid and edge frequency over an explicit broadband range. It uses the same
one-sided periodogram contract as the intervention methods and does not present a
broadband slope as an aperiodic fit.

Both functions return `EEGFeatureSet`, retaining trial, channel and optional patch
coordinates. `.matrix(aggregation="flatten")` preserves those coordinates as named
probe columns; `aggregation="mean"` returns one value per descriptor. This makes the
same target usable for layer-wise probing, concept-subspace fitting and sparse-feature
annotation without embedding a dataset split in the extractor.

`welch_power_spectrum` exposes Hann segment length, overlap and number of averaged
segments. `channel_correlation` computes Pearson matrices. `band_connectivity`
computes phase-lag index, phase-locking value or magnitude-squared coherence after
rectangular FFT band selection and an analytic-signal transform. Results preserve
the trial axis and channel order. Patch scope evaluates each patch separately; trial
scope requires contiguous nonoverlapping patches. These sensor-space relationships
are descriptive and must not be interpreted as source-level or directed connectivity.

The initial feature set deliberately excludes BrainPEC quantities labeled as proxies,
including its compact sample-entropy, Lempel-Ziv, Higuchi and DFA approximations.
Adding a familiar scientific name requires a reference implementation and
known-answer evidence for that exact estimator.

## Periodic and aperiodic intervention

`fit_aperiodic_decomposition` is an optional FOOOF 1.1 integration modeled on the
FMScope diagnostic. It fits a fixed aperiodic component and periodic peaks per channel
from a Welch PSD averaged over an explicit reference `SignalBatch`. The returned
`AperiodicDecomposition` records the channel order, preprocessing, units, fit range,
window geometry, estimator version, exact fitted Gaussian components and a stable fit
digest.

`remove_fitted_spectral_component` reuses that fit on another compatible batch. It
divides periodic, aperiodic or both amplitude factors only inside the fitted range,
retains phase and records the fit digest in `SignalBatch.transforms`. Fit on a
training/reference cohort and apply the same object to held-out data. Refitting on
each evaluation condition changes the intervention and can leak cohort information.
The removal tests dependence on the fitted multiplicative component; reconstructed
signals are not guaranteed to be physiologically free of periodic or aperiodic
activity.

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
used in BrainPEC. It remains useful as a transparent baseline, but correlated feature
coordinates can make its minimum-change geometry inappropriate.

`fit_leace_eraser` implements covariance-aware least-squares concept erasure. It
whitens the reference feature covariance, finds the column space of the whitened
feature–concept cross-covariance, and returns a low-rank affine map centered on the
reference mean. The map is idempotent, preserves that mean, and removes linear
cross-covariance with the supplied concept on the fitted support. Pass categorical
concepts through `categorical_targets`; centering makes the effective rank at most
the number of categories minus one.

Fit the eraser on a declared training/reference cohort and apply that fixed object
through `LEACEAblation` to held-out activations. It acts on the final feature axis;
sensor, patch or non-feature axis selections apply the complete map at selected
positions. Partial selection of the transformed feature axis is rejected because it
would no longer be a LEACE erasure. The run records fit dimensions, rank, numerical
settings and parameter hashes. It does not store the reference data or fitted tensors,
so retain the fitted object with the study artifacts.

The default uses the empirical covariance. `covariance_shrinkage` explicitly mixes
it with an isotropic covariance of the same trace; this can stabilize small reference
cohorts but changes the metric whose displacement is minimized. Report the chosen
value rather than treating shrinkage as an implementation detail.

`fit_random_subspace_control` produces a seeded isotropic orthonormal basis with the
same requested rank and reference mean. Apply it with `SubspaceAblation` and compare
several prespecified seeds with the concept eraser. A behavioral drop beyond these
controls supports use of that fitted linear concept subspace. Linear erasure does not
exclude nonlinear concept information, and an intervention effect alone does not
identify a biological variable or a complete circuit.

`group_variance_decomposition` exactly separates between-group and within-group sums
of squares. `within_group_contrast_consistency` compares a binary condition-contrast
direction across groups and reports mean pairwise cosine plus a precisely defined
contrast SNR. For EEG-FM audits, groups will often be subjects. Both are descriptive;
the study must estimate uncertainty at the subject level.

## Concept attribution

`fit_concept_direction` fits a binary ridge CAV on caller-supplied training rows and
reports balanced accuracy on a disjoint test mask. Label one defines the positive
direction. The fit centers its inputs and targets but performs no implicit feature
standardization. A high validation score establishes linear separability of the
provided concept examples; it does not establish that the model uses the direction.
Split subjects or other dependent groups before calling the function.

`tcav_score` takes an `AttributionResult` computed with `method="gradient"` and dots
the native site's objective gradient with the fixed CAV along the final feature axis.
With `sample_unit="trial"`, positional directional derivatives are averaged within
each trial before scoring. With `"position"`, every non-feature coordinate is treated
as a sample. The result includes every directional sensitivity, their mean and the
classic TCAV score: the fraction that is strictly positive. The sign fraction is
threshold-sensitive, so inspect the raw distribution and mean rather than reporting
the fraction alone.

`tcav_permutation_test` independently permutes concept labels within the declared
train and test splits, refits same-procedure ridge CAVs and evaluates them using the
already computed native gradients. It preserves split class counts and reports a
finite-sample one-tailed p-value for the classic score. This is a method-level
random-label diagnostic. It cannot infer whether trials, windows or positions are
exchangeable, and it does not correct a search across layers, concepts, objectives or
seeds. Those choices and subject-level uncertainty remain part of the study.

TCAV is a local directional derivative at the observed activation. A large value
supports objective sensitivity along the fitted direction. Combine it with a fixed
intervention such as LEACE, subspace ablation or SAE feature editing when the claim
concerns causal use.

## Gradient and spectral attribution

`attribute` accepts an explicit scalar objective and runs each trial independently.
It provides raw gradient, input × gradient and integrated gradients. Integrated
gradients requires an explicit `SignalBatch` baseline with matching trial identities,
signal geometry, preprocessing, dtype and device. The path uses trapezoidal
integration over a caller-visible step count. The result records the input and
baseline hashes, objective values, execution arguments and per-trial completeness
error.

Requested internal sites use the same declared adapter layouts as caching and
intervention. For integrated gradients, each site's attribution is discrete path
conductance: the downstream gradient is integrated against that site's activation
change along the input path. This matters when internal activation paths are
nonlinear. `site_multipliers` and `site_deltas` are also returned for inspection,
but multiplying those two summaries is not the path-conductance definition.

`patch_attribution`, `channel_attribution` and `temporal_attribution` preserve the
trial axis and make signed versus absolute aggregation explicit. `spectral_attribution`
implements the additive inverse-DFT propagation used by EEG-PRISM for input ×
gradient or integrated gradients. It retains the raw or path-averaged gradient, so
it does not divide by zero-valued EEG samples. Conjugate frequency pairs are combined
into a one-sided result, and `conservation_error` compares its sum with time-domain
attribution. Trial scope requires contiguous, nonoverlapping patches; patch scope
keeps each patch separate.

`source_attribution` applies the same linear propagation rule to a caller-supplied
EEG source delta and forward matrix. The forward matrix has `[channel, source]`
geometry in the exact channel order recorded by the attribution result. Source deltas
have `[trial, source, patch, sample]` geometry. For input × gradient they are source
estimates relative to zero; for integrated gradients they are observed-minus-baseline
source estimates. The function maps input multipliers through the transpose of the
forward operator and multiplies them by the source delta, retaining raw signed
source contributions.

EEGFMLens does not estimate cortical activity or choose an inverse solver. The result
reports forward reconstruction RMSE, relative reconstruction error and attribution
conservation error for every trial. Exact synthesis conserves additive attribution;
an approximate inverse transfers its mismatch into the source explanation. Head
model, reference, orientation, regularization and source parcellation remain part of
the study and must be held fixed before comparing source maps. Source localization is
not validated merely because the propagation step conserves attribution.

Gradient evidence in this release consists of analytic synthetic cases for input
completeness, nonlinear-site conductance, semantic hook restoration and spectral
conservation. A declared adapter site does not by itself certify that every upstream
checkpoint path is differentiable. Native studies should record the exact component,
checkpoint and objective and treat a disconnected-gradient error as an unsupported
path, rather than silently substituting a different forward.

## Perturbation faithfulness

`occlusion_curve` progressively replaces caller-ordered, disjoint channel/patch
regions with an explicit trial-matched baseline. `spectral_perturbation_curve`
progressively removes nonoverlapping frequency bands while preserving retained-bin
phase. Both execute trials independently and return the complete score curve plus
area over the perturbation curve (AOPC), defined as the mean drop from the unperturbed
score. The score must be higher-is-better. Raw AOPC values are comparable only when
models use the same objective and score scale.

Target order is part of the hypothesis. Rank regions on separate data or with a
separately computed attribution map; ranking and evaluating on the same trials can
inflate faithfulness. `attribution_cosine_consistency` reports a symmetric mean
per-trial cosine matrix for already aligned maps and makes the signed/absolute choice
explicit. Agreement is a stability diagnostic, not proof that either map is causal.

## Activation-to-spectrum readouts

`amplitude_spectral_targets` converts model-ready EEG into channel-mean one-sided
amplitude targets. Patch rows use trial-major, patch-minor order; trial scope first
requires contiguous nonoverlapping patches. The FFT uses explicit sample count and
sampling rate, doubles the interior one-sided bins, and records the selected
frequency coordinates, preprocessing identity and source-input hash. The optional
`log1p_amplitude` transform improves dynamic range but remains specific to the input
unit and preprocessing scale.

`activation_spectral_matrix` aligns those rows with cached activation geometry.
Physical `[batch, channel, patch, feature]` sites average their channel positions.
`tokens` and `patch_tokens` are reshaped using the adapter's declared channel-major
order, with CLS removed when present. Trial targets instead use an explicit ordinary,
flattened or CLS activation pooling rule. Provenance or geometry mismatches fail.

`fit_spectral_readout` fits a multi-output ridge map only on the supplied train mask
and reports per-frequency and mean R² on both train and test rows. Create the masks at
the subject or recording level before expanding trials into patch rows; neighboring
patches from one recording are not independent held-out evidence. Constant target
frequencies are rejected because their R² is undefined.

`direction_signature` multiplies an activation-space direction by the fitted linear
map. For a unit-norm SAE decoder row, the result is the readout's signed predicted
amplitude or log-amplitude change for a one-unit move along that feature. It is not a
measured power spectrum, and high held-out fidelity is required before interpretation.
`band_signature` averages those predicted changes over a declared frequency band.

For causal follow-up, `sae_feature_sweep` accepts ordinary output `metrics` and
`run_metrics` over explicitly named `cache_sites`. A run metric can apply the fixed
spectral readout to the post-intervention activation while an output metric measures
task behavior. This tests whether a ranked feature set changes its predicted spectral
property selectively; it does not turn the readout association into physiology by
itself.

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

`sae_concept_profile` reports feature firing rates in caller-declared positive and
negative activation rows, their difference, and cosine alignment between each
decoder row and a supplied activation-space concept direction. These are descriptive
rankings. The function deliberately reports no p-values because the independent unit
and valid exchangeability scheme depend on the EEG cohort and windowing design.

`fit_sae_code_reference` computes a fixed mean code on a declared target cohort.
`SAEFeatureClamping` moves selected codes to those target means by adding only their
changed decoder contributions to the native activation. It therefore preserves the
SAE reconstruction residual exactly. The reference is bound to hashes of every SAE
parameter, and application fails if the SAE changes after fitting. A fixed-coefficient
`SAEFeatureSteering` tests a prescribed displacement; target-centroid clamping tests a
reference-fitted edit. Scientific studies should prespecify feature ranking, compare
target and off-target objectives across intervention strength or feature count, and
include random-feature rankings.

`sae_feature_sweep` implements that controlled comparison without assigning meaning to
the metrics. It cumulatively ablates or target-centroid clamps a supplied feature
ranking, executes each trial and feature count from the original activation, and stores
the raw `[trial, step]` score curves. Output metrics receive the native model output;
run metrics can inspect post-intervention activations in explicitly requested cache
sites. The two forms can therefore combine a task score with an EEG-derived readout or
another off-target behavior in the same intervention experiment. Every callback must
return one finite floating score for the current trial.

Seeded controls use uniform random permutations of the complete SAE dictionary at the
same cumulative counts. The actual permutations and baseline column are retained.
`mean_delta` and `integrated_mean_delta` summarize signed change from the unmodified
model; `area_between` and `random_area_between` integrate raw differences between two
named curves. Compare areas only when both metrics have a commensurate scale and a
declared direction. The result does not label a feature as selective, entangled or
causal, and random draws alone are not a valid subject-level null distribution.

Fit the SAE, concept ranking, code reference and any learned metric on data disjoint
from the evaluation trials. Preserve subject or recording blocks when estimating
uncertainty outside the package. The sweep deliberately does not reset native model
randomness: use an evaluated deterministic model, or control and document paired RNG
when the native architecture is stochastic.

## Proposed paths

`path_patch` tests one source-to-mediator hypothesis. It first patches donor state at
the source into a recipient run and caches the induced mediator state. It then patches
that mediator state into a fresh recipient. The result contains per-trial source and
mediated effects, and the workflow verifies identity patches at both sites in score
space. The chosen mediator must execute downstream of the source. A recovered effect
supports the tested path under these interventions; it does not discover a complete
circuit or establish physiological localization.

## Cross-model representation geometry

`linear_cka` compares finite paired `[trial, feature]` matrices and is invariant to
orthogonal feature rotation and isotropic scaling. `rsa_correlation` compares the
upper triangles of correlation, cosine or Euclidean trial-similarity matrices.
`cross_model_similarity` accepts named `Activation` collections, requires the exact
same set of trial IDs, reorders the right model to the left model's order and returns
the complete layer-by-layer matrix. It never takes a silent trial intersection.

Supplying `groups` removes each group mean before CKA or RSA. For subject-labeled EEG,
reporting ordinary and within-subject-centered similarity separately distinguishes
shared identity geometry from shared within-subject state geometry. Neither statistic
establishes functional interchangeability or a causal circuit. Use held-out linear
alignment or intervention when the claim concerns transport or use of a representation.

## Method provenance

The implementation is original MIT-licensed package code informed by the following
papers and their public repositories:

- [The Identity Trap in EEG Foundation Models: A Diagnostic Audit](https://arxiv.org/abs/2606.06647) and [FMScope](https://github.com/Jimmy110101013/fmscope): variance, subject-direction and spectral-ablation diagnostics.
- [What Do EEG Foundation Models Capture from Human Brain Signals?](https://arxiv.org/abs/2605.11410) and [BrainPEC](https://github.com/Kian-Chen/BrainPEC): feature-family probing, cross-covariance erasure and residual controls.
- [EEG Foundation Models for BCI Learn Diverse Features of Electrophysiology](https://arxiv.org/abs/2506.01867): probing of individual variability and electrophysiological features, including alpha activity. EEGFMLens keeps feature extraction and subject-aware splitting separate.
- [LEACE: Perfect Linear Concept Erasure in Closed Form](https://arxiv.org/abs/2306.03819) and its [reference implementation](https://github.com/EleutherAI/concept-erasure): covariance-aware affine concept erasure and same-rank random-subspace controls. EEGFMLens implements the published closed-form operator independently and exposes explicit reference-fit and intervention objects.
- [TCAV: Interpretability Beyond Feature Attribution](https://proceedings.mlr.press/v80/kim18d.html): linear concept directions and directional sensitivity of a class objective. EEGFMLens exposes a held-out ridge CAV, native-site gradients, raw sensitivities and a random-label null; study-level inference remains explicit.
- [Mechanistic Interpretability of EEG Foundation Models via Sparse Autoencoders](https://arxiv.org/abs/2605.13930) and its [companion repository](https://github.com/BrainCapture/mechanistic-interpretability-for-eeg-foundation-models): Top-K SAE, spectral decoding and feature interventions. EEGFMLens implements an independent held-out linear amplitude readout with explicit row alignment and direction effects; it does not claim the paper's nonlinear amplitude/phase decoder. The companion code is PolyForm Noncommercial; no code was copied into EEGFMLens.
- [Beyond Accuracy: Robustness, Interpretability and Expressiveness of EEG Foundation Models](https://arxiv.org/abs/2605.17562) and its [repository](https://github.com/urbansirca/Beyond-Accuracy-Robustness-Interpretability-and-Expressiveness-of-EEG-Foundation-Models): channel perturbation, attribution and block-wise probing controls.
- [EEG-PRISM](https://arxiv.org/abs/2608.13676): linear propagation of attribution into physiologically meaningful frequency and source coordinates. EEGFMLens exposes both mappings and makes inverse reconstruction error explicit.
- [EEG-Xplain](https://arxiv.org/abs/2609.15687): gradient attribution, space/time/frequency summaries, progressive perturbation and cross-method consistency.
- [From Clever Hans to Scientific Discovery: Interpreting EEG Foundational Transformers with LRP](https://arxiv.org/abs/2605.11885): model-specific attention-aware LRP for EEG transformers. EEGFMLens does not relabel ordinary gradients as LRP; native LRP support requires exact replacement forwards and relevance-conservation evidence per architecture.
- [CLT-Forge](https://arxiv.org/abs/2603.21014) and its [repository](https://github.com/LLM-Interp/CLT-Forge): cross-layer replacement models and attribution graphs. EEGFMLens does not label an ordinary SAE or hook graph as a CLT; a future implementation requires adapters to expose compatible residual inputs and component outputs plus replacement-fidelity tests.
- [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414): linear centered-kernel alignment for representations with different feature dimensions.
- [SVCCA](https://arxiv.org/abs/1706.05806): motivates cross-network representation comparison; EEGFMLens currently exposes CKA and RSA rather than claiming SVCCA without its truncation and regularization choices.
- [FOOOF](https://doi.org/10.1038/s41593-020-00744-x) and its [stable implementation](https://github.com/fooof-tools/fooof): optional fixed-mode spectral parameterization. The dependency is not bundled, and the fit/apply interface is original package code.

The current public API does not claim specparam 2 compatibility, attention LRP,
Q/K/V intervention, automatic circuit discovery or a cross-layer transcoder. Those
names will be exposed only after their mathematical and native-model contracts have
dedicated correctness evidence.
