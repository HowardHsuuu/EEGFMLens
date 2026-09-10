# Spectrum-preserving temporal perturbations

Decision record locked before evaluating model responses, 2026-09-09. This is a follow-up to the corrected DREAMS spindle experiment, not a new subject split. Keep all existing subject-held-out staging heads and hyperparameters fixed. Both pretrained models and the existing single-seed random-weight baselines are evaluated; only pretrained models receive activation-patching experiments.

## Three distinct inference questions

1. Whole-window phase perturbation preserves the 15-second Fourier magnitudes: sensitivity rejects an exclusively whole-window-spectrum readout, but does not by itself identify spindle morphology.
2. Local event and off-event phase perturbations preserve each edited three-second segment's Fourier magnitudes. The full 15-second spectrum need not match; measure and disclose its change. These comparisons test event localization, not exact global-spectrum invariance.
3. Patching clean intermediate event tokens into the perturbed run tests their contribution to the observed response. Token coordinates are not exclusive receptive fields. Compare off-event and random positions with equal patch count and equal activation perturbation norm.

## Cases and signals

Use the corrected 200-Hz, central-channel, microvolt/100 inputs from the previous run. Select reviewed N2 windows with a first-expert event lasting >=0.5 s fully contained in an integer-aligned three-second region. Choose the earliest eligible event and its centered/clipped region (start 1..11 seconds). Choose an off-event three-second region with no annotated spindle within a one-second guard, minimizing RMS difference from the event region. No model outputs enter selection. Record all exclusions.

Use three fixed seeds per case, derived from seed 8721, original row ID and replicate. Fourier phase offsets are uniform [-pi, pi], with DC and Nyquist held fixed. For local pairs, choose 80% of the smaller maximum attainable input L2 perturbation, then solve each phase multiplier by bisection so absolute event/off-event perturbation norms match. No taper is applied, because it would invalidate exact segment-spectrum preservation; boundary artifacts are audited.

Whole-window phase perturbation uses target L2 distance equal to the centered signal L2 norm (capped at 90% of the phase draw's attainable distance); report achieved relative norms. A five-second circular shift is an additional morphology-preserving, spectrum- and marginal-distribution-preserving control. It changes temporal position; it does not randomize event morphology.

Validate exact FFT magnitudes after conversion to model float32 (relative L2 tolerance 1e-6), energy/mean, local perturbation norm matching, and unchanged samples outside local edits. Measure sigma-envelope concentration in annotated intervals, whole-window envelope coefficient of variation, sample distribution distance, peak amplitude, and local join jumps. Do not assume phase modification destroyed morphology; report how often it did.

## Model execution and restoration

Inspect embedding.output and blocks.2/5/8/11.output. Save per-patch normalized activation changes and N2 margins for all signal variants, including random-weight encoders. Normalize N2 margin changes by the corresponding training-fold clean-margin SD. Never refit heads on surrogates.

For each non-shift surrogate, run clean donor replacement at the three event patches, the off-event patches, and one fixed random three-patch location outside the event region. The actual event replacement is unscaled. Scale each control's clean-minus-recipient delta to the same Frobenius norm as the event replacement, while preserving its position. Report scale factors; excessive scaling is an OOD limitation, not evidence of a clean counterfactual. LaBraM CLS stays unchanged. All-position final-block replacement is a plumbing control, not a mechanistic result. Assert native/identity/full-final-block recovery and hook cleanup.

Report raw restoration delta and reduction in absolute clean-margin error; do not report normalized recovery when clean vs surrogate degradation is absent. Aggregate replicate/case results within subject before uncertainty estimation. Use 10,000 subject bootstrap draws, seed 8721; report all sites and variants rather than select the best test outcome. This exploratory eight-subject follow-up does not constitute independent confirmation after the prior study.

## Interpretation gate

A spectrum-matched response establishes sensitivity to information beyond that spectrum only. Spindle-morphology evidence additionally requires measured morphology disruption, event-vs-off-event specificity, favorable event-vs-location-control restoration, and no adequate explanation by artifact/distribution changes or random-weight sensitivity. If these controls fail or remain ambiguous, explicitly conclude that spindle-specific interpretation is unsupported. Global and local spectral constraints must never be conflated.

Method context: Schreiber & Schmitz, [Surrogate time series](https://arxiv.org/abs/chao-dyn/9909037). Fourier-phase surrogates do not generally preserve the sample-value distribution. This experiment measures that limitation rather than assuming Gaussianization is harmless.

Before model evaluation, add a sigma-only whole-window phase condition: modify phases only in 11–16 Hz, retaining all other complex Fourier coefficients. Target 90% of the draw's attainable sigma-only perturbation norm. This preserves the full magnitude spectrum and limits altered frequency content, but can still change sigma amplitude distribution and event timing. It receives the same per-patch analysis and restoration controls as other non-shift surrogates. The full-band and sigma-only conditions need not have equal raw perturbation norm; report their differences and do not compare raw effect magnitudes as if matched.

Implementation gate before model evaluation: a local edit may leave a control activation exactly unchanged, particularly at the embedding. If a nonzero event restoration cannot be matched by the control's zero donor delta, record that comparison as unavailable rather than inventing a control direction. A zero event target yields a zero intervention and is uninformative. Report unavailable counts and do not interpret unmatched event effects as specificity.
