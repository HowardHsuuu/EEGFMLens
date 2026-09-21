# Tool roadmap

## Implemented

- Explicit adapters for eleven checkpoint-verified model families; see [coverage](model-coverage.md).
- One catalog/connection API and per-site capability discovery across all eleven families.
- Native activation caching, paired replacement, zero and orthonormal-subspace ablation.
- Validated input/selector contracts, scoped hook cleanup and versioned run bundles.
- Channel/time patching sweeps with per-trial matching and explicit invalid-control diagnostics.
- A complete activation-restoration workflow with auditable reports and figures.
- Trial- and patch-scope spectral measurement plus amplitude/phase/complex band interventions.
- Held-out ridge probes, Euclidean cross-covariance subspaces, covariance-aware
  LEACE with same-rank random controls, group-variance and contrast diagnostics.
- Top-K sparse autoencoders with residual-preserving feature ablation and steering.
- Held-out ridge CAVs, native-gradient TCAV with random-label nulls, descriptive SAE
  concept profiles and target-centroid code clamping.
- Identity-controlled source-to-mediator path patching for proposed circuits.
- Trial-independent gradients, input × gradient, integrated gradients and nonlinear-site path conductance.
- Additive inverse-DFT spectral and forward-model source attribution with explicit
  conservation/inverse diagnostics; progressive input/frequency perturbation, AOPC
  and cross-method consistency.
- Welch spectra, named time/frequency concepts, correlation, PLI, PLV and band
  magnitude-squared coherence.
- Reference-fitted periodic/aperiodic component intervention through an optional dependency.
- Exact trial-matched CKA/RSA across layers and models, including within-group centering.
- Pinned-source native checks for seven families and installed-wheel CI on three operating systems.
- Source type checks, pre-commit hooks, distribution audits and trusted-publishing release automation.

## Future capabilities

- Broader device, dtype and input-configuration validation.
- Bounded cache storage driven by actual user workflows.
- Additional BrainOmni configurations with independent native validation.
- Native gradient conformance across exact model components/checkpoints and functional attention/QKV interfaces.
- Architecture-specific attention-aware LRP after exact replacement-forward parity
  and relevance-conservation tests are available for each supported family.
- Additional validated EEG concept estimators beyond the current transparent core.
- Circuit graph discovery and component-level Q/K/V/MLP paths after native sites are validated.
- Cross-layer transcoder support after feature transport and reconstruction controls are specified.

Scientific studies and their publication schedules are maintained outside this
repository. They are not prerequisites for releasing the reusable tool. Current
supported scope and limitations are in [model coverage](model-coverage.md).
