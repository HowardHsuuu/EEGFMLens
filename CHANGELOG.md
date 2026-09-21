# Changelog

## 0.1.0a18 — controlled SAE feature sweeps

- Add `sae_feature_sweep` for cumulative ranked feature ablation or reference-code
  clamping against any set of caller-declared native model metrics.
- Preserve per-trial curves, run every trial and intervention count independently,
  and compare the observed ranking with seeded full-dictionary random rankings.
- Add signed mean-change integrals and target/off-target area summaries without
  inferring score direction, feature meaning or a scientific taxonomy.
- Record the exact ranking, feature counts, input/SAE/reference hashes, selector,
  random rankings, metric names and one-trial execution contract.
- Add analytic known-answer tests and an installed-wheel example.

See [migration instructions](docs/migration-a18.md).

## 0.1.0a17 — source-space attribution

- Add EEG-PRISM source-space propagation for input × gradient and integrated
  gradients using a caller-supplied source delta and EEG forward matrix.
- Return raw signed source attribution and source multipliers with explicit
  source/channel/trial coordinates and stable input hashes.
- Report per-trial forward reconstruction RMSE, relative reconstruction error
  and attribution conservation error instead of hiding inverse-model mismatch.
- Add exact and approximate inverse known-answer evidence plus an offline
  source-attribution example.

See [migration instructions](docs/migration-a17.md).

## 0.1.0a16 — concept attribution and selective SAE intervention

- Add leakage-explicit ridge concept directions with held-out balanced accuracy.
- Add TCAV from native internal-site gradients, including raw directional
  sensitivities, explicit trial/position sampling units and a same-split
  random-label permutation null.
- Add descriptive SAE concept profiles combining feature firing-rate contrast
  with decoder/concept-direction alignment.
- Add reference-fitted target-centroid SAE code clamping that changes only
  selected decoder contributions while preserving the native SAE residual.
- Add a known-answer concept-attribution example and rejection evidence for
  overlapping splits, partial feature-axis clamping and stale SAE references.

See [migration instructions](docs/migration-a16.md).

## 0.1.0a15 — covariance-aware concept erasure

- Add an independently implemented closed-form LEACE fit with explicit empirical
  covariance, optional isotropic shrinkage and low-rank affine application.
- Add `LEACEAblation` for fixed site interventions with fit metadata and parameter
  hashes in each run manifest.
- Add seeded same-rank random-subspace controls for separating concept-specific
  effects from generic representational damage.
- Reject partial feature-axis LEACE edits and add known-answer evidence for linear
  erasure, idempotence, oblique geometry, intervention cleanup and control behavior.

See [migration instructions](docs/migration-a15.md).

## 0.1.0a14 — EEG concepts and cross-model geometry

- Add Hann-windowed Welch PSDs with explicit segment and overlap geometry.
- Add named Hjorth/time descriptors, band-power and broadband spectral summaries,
  and probe-ready feature matrices with retained channel/patch coordinates.
- Add channel correlation and band-limited phase-lag index, phase-locking value
  and magnitude-squared coherence matrices.
- Add optional reference-fitted FOOOF decomposition and reusable, phase-preserving
  periodic/aperiodic component removal with fit provenance.
- Add trial-ID-aligned linear CKA and RSA matrices across model layers, including
  within-group centering for separating subject identity from within-subject geometry.
- Add a known-answer cross-model example and optional-dependency CI job.

See [migration instructions](docs/migration-a14.md).

## 0.1.0a13 — attribution and perturbation evidence

- Add trial-independent gradient, input × gradient and integrated-gradient APIs
  for model inputs and declared internal sites.
- Integrate internal-site attribution as discrete path conductance, including a
  nonlinear known-answer test that rejects the simpler invalid approximation.
- Add channel, patch and temporal summaries plus additive inverse-DFT frequency
  attribution with an explicit conservation diagnostic.
- Add progressive trial-matched input occlusion and phase-preserving spectral-band
  removal with full score curves and AOPC.
- Add cross-method attribution consistency and provenance hashes for attribution
  and perturbation experiments.
- Document native gradient evidence limits and the paired-site/fidelity contract
  required before claiming a cross-layer transcoder.

See [migration instructions](docs/migration-a13.md).

## 0.1.0a12 — composable interpretability methods

- Add trial- and patch-scope periodograms, band power, phase-preserving band
  scaling and trial-matched amplitude/phase/complex spectral patching.
- Record signal edits on `SignalBatch` and in every run manifest.
- Add leakage-explicit ridge probes, cross-covariance concept subspaces, exact
  group-variance decomposition and cross-group condition-direction diagnostics.
- Add a clean-room Top-K sparse autoencoder, reconstruction/feature-use metrics,
  and residual-preserving feature ablation and steering.
- Add identity-controlled source-to-mediator path patching for testing proposed
  internal routes.
- Document method provenance, mathematical scope and unsupported claims from the
  EEG-FM interpretability literature.

See [migration instructions](docs/migration-a12.md).

## 0.1.0a11 — public source alpha

- Align the distribution and import namespace with the project name: install
  `eegfmlens` and import from `eegfmlens`; the unreleased `eeglens` namespace is
  removed rather than retained as a compatibility alias.
- Add `supported_models()`, `model_info()` and `connect()` as one discoverable
  integration path for all eleven verified EEG model families.
- Add immutable per-site capability reporting through `lens.capabilities()`.
- Document every family's native component, required adapter options, input
  contract and physical-selection boundary in one table.
- Run the offline model catalog in installed-wheel verification.
- Add `restoration_sweep` and a flagship workflow that emits a controlled sweep,
  audit manifest, and heatmap from either a deterministic demo or real EEG.
- Expand pinned live-upstream CI from two to seven model families and expose each
  catalog entry's maintained evidence tiers.
- Add mypy, pre-commit, an installed-suite coverage floor, distribution auditing,
  signed-tag release documentation, GitHub release assets, and trusted PyPI
  publishing workflow.
- Remove unvalidated REVE and EEGMamba sketches from the distributed source.
- Move the sole retained third-party attribution next to its complete license in
  `LICENSES/`; no upstream model implementation is bundled.
- Remove generated validation reports, milestone documents and machine-specific
  runners from the maintained source tree.

See [migration instructions](docs/migration-a11.md).

## 0.1.0a10 — source alpha

- Remove bundled CBraMod/LaBraM implementations; checkpoint helpers now require
  an explicit `model_factory`. Strict keys, hashes and output modes are retained.
- Preserve LaBraM channel metadata with its direct MIT notice.
- Keep offline core examples/tests independent of external model code; add a
  separate pinned-source native CI job.
- Add package URLs, pin Ruff, and separate current user guides from generated evidence.
- Include the previously unreleased AxisSelection and DIVER reconstruction APIs.

See [migration instructions](docs/migration-a10.md). Historical entries below
describe their original versions; no PyPI publication is claimed.

## a9 post-release source changes

Add `AxisSelection` for explicit non-batch tensor-axis interventions, retaining
trial alignment and donor provenance without assigning physical electrode labels.
Physical patching sweeps reject raw-axis targets. Synthetic native-hook tests
cover partial edits, tuple preservation, identity, metadata export and cleanup.

DIVERAdapter can select the native masked time-domain reconstruction output.
It requires explicit boolean masks, preserves native RNG consumption, cleans up
mask hooks after failures, and exposes only sites on the selected output path.
The feature-only default remains unchanged. Official-checkpoint CPU validation
covers 30 site/batch conditions; this is not a new scientific finding.

## Repository scope

Scientific experiment code, results and planning documents are maintained
separately. Maintained tool validation lives in `tests/`, with source/checkpoint
setup documented in `validation/README.md`.

## 0.1.0a9 — public source alpha

Retains acknowledged BEiT v2, timm, DeiT and DINO licenses and notices. Six GitHub hosted OS/Python installed-wheel jobs passed with 87 non-integration tests per job; native checkpoint conformance remains separately scoped. No PyPI release is claimed.

## 0.1.0a8 — local alpha

Retains the PyTorch notice for bundled upstream transformer helpers without changing model computation.

## 0.1.0a7 — local alpha, not published

Run export normalizes tuple/list subclasses to plain schema-supported containers. Previously a native namedtuple output could be saved successfully with its class name as the tree kind, then fail to load because the v1 reader only recognizes `tuple` and `list`. Values/order round-trip without importing custom classes; namedtuple field names and subclass identity are not retained. This change is included in a7 and absent from a6.

## 0.1.0a6 — local alpha, not published

Adapters reject ambiguous hook metadata before wrapping: `tensor_index=True` can no longer silently select tuple element 1, and `writable="false"` can no longer enable edits through Python truthiness. Names/layouts must be nonempty strings, module paths must be strings, and descriptors must be `ActivationSite` objects. Empty root-module paths and negative integer tuple indices remain supported. These guards are included in a6 and absent from the retained a5 wheel. The installed verifier now avoids loading torch in its coordinator and verifies the actual import in a separate isolated subprocess, preventing the reproduced local OpenMP shared-memory failure with torch 2.6.

## 0.1.0a5 — local alpha, not published

NeuroRVQ rejects input channel names that differ only by case but map to the same native electrode. Valid case normalization remains supported. Official checkpoint boundary checks verify one/eight patches, twelve pre-forward invalid-input cases, exact four-branch recovery and valid-alias parity.

BrainOmni validates matching native window sizes and a finite overlap yielding a positive stride, records that configuration in metadata and rejects subsequent window/overlap changes before encode. Native short-input and tail padding are preserved. Official tiny-checkpoint tests cover 1/512/513/1536 samples, twelve bad-input cases and twenty configuration/geometry mutations, with exact recovery. The source suite passes 66 tests. These changes are absent from earlier a4 artifacts.

## 0.1.0a4 — local alpha, not published

EEGPT now reports unknown native channel names as `ValidationError` and rejects distinct input names that normalize to the same native channel ID. Valid native casing/trailing-dot normalization is preserved. BENDR validates the encoder/contextualizer composition before indexing its components, with explicit errors for missing, extra, non-indexable or incompatible components. Official EEGPT and full BENDR dense checks were rerun against these adapter changes.

## 0.1.0a3 — local alpha, not published

Intervention return values must preserve exposed tensor shape, dtype and device and contain finite values. Previously, a custom intervention could return a different shape with the same element count and be silently reshaped by the adapter. Invalid results now raise `ValidationError`; scoped cleanup preserves existing hooks and permits the next run. Invalid batch objects, string cache-site collections and malformed intervention objects also produce explicit errors. Non-tensor subspace basis/center inputs are rejected deliberately.

The eleven-family integration campaign informed the documented support
boundaries. Current tests cover the return-value guard and valid intervention
arithmetic; generated reports from earlier versions are not part of the
maintained source tree.

## 0.1.0a2 — local alpha, not published

Adds checkpoint-backed adapters for eleven EEG model families, invocation-specific hooks for shared modules, and expanded native semantic validation. See `docs/model-coverage.md` for exact checkpoints and limitations.

**Cache compatibility:** BENDR convolution activations now expose `[batch, time, features]` instead of `[batch, features, time]`, so feature-subspace interventions act on the intended axis. Regenerate old BENDR caches; do not reuse activation bundles from earlier builds. This version distinguishes the corrected artifact from prior `0.1.0a1` wheels. The change does not establish scientific or general device/dtype validity.

## Unreleased public patching sweeps

Added `patching_sweep`, `SweepTarget`, `SweepResult`, `patch_grid` and per-trial `MatchedReplacement`. Sweeps provide identity/cleanup checks, stable trial-ID random controls, explicit invalid and excessive-multiplier diagnostics, JSON output and grouped descriptive summaries. The known-answer example validates exact causal coordinates and effects. Validation evidence is recorded in `docs/sweep-validation.md`.

## 0.1.0a1 — local alpha

Initial implementation: CBraMod/LaBraM adapters, activation snapshots, trial-paired replacement, semantic zero/subspace ablation, strict local loaders, run bundles, paired metrics, tests and executable examples. No public release or scientific benchmark result is claimed.
