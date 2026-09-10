# Changelog

## 0.1.0a7 — local alpha, not published

Run export normalizes tuple/list subclasses to plain schema-supported containers. Previously a native namedtuple output could be saved successfully with its class name as the tree kind, then fail to load because the v1 reader only recognizes `tuple` and `list`. Values/order round-trip without importing custom classes; namedtuple field names and subclass identity are not retained. This change is included in a7 and absent from a6.

## 0.1.0a6 — local alpha, not published

Adapters reject ambiguous hook metadata before wrapping: `tensor_index=True` can no longer silently select tuple element 1, and `writable="false"` can no longer enable edits through Python truthiness. Names/layouts must be nonempty strings, module paths must be strings, and descriptors must be `ActivationSite` objects. Empty root-module paths and negative integer tuple indices remain supported. These guards are included in a6 and absent from the retained a5 wheel. The installed verifier now avoids loading torch in its coordinator and verifies the actual import in a separate isolated subprocess, preventing the reproduced local OpenMP shared-memory failure with torch 2.6.

## 0.1.0a5 — local alpha, not published

NeuroRVQ rejects input channel names that differ only by case but map to the same native electrode. Valid case normalization remains supported. Official checkpoint boundary checks verify one/eight patches, twelve pre-forward invalid-input cases, exact four-branch recovery and valid-alias parity.

BrainOmni validates matching native window sizes and a finite overlap yielding a positive stride, records that configuration in metadata and rejects subsequent window/overlap changes before encode. Native short-input and tail padding are preserved. Official tiny-checkpoint tests cover 1/512/513/1536 samples, twelve bad-input cases and twenty configuration/geometry mutations, with exact recovery. The source suite passes 66 tests. These changes are absent from earlier a4 artifacts.

## 0.1.0a4 — local alpha, not published

EEGPT now reports unknown native channel names as `ValidationError` and rejects distinct input names that normalize to the same native channel ID. Valid native casing/trailing-dot normalization is preserved. BENDR validates the encoder/contextualizer composition before indexing its components, with explicit errors for missing, extra, non-indexable or incompatible components. Official EEGPT and full BENDR dense checks were rerun against these adapter changes; earlier a3 artifacts remain historical.

## 0.1.0a3 — local alpha, not published

Intervention return values must preserve exposed tensor shape, dtype and device and contain finite values. Previously, a custom intervention could return a different shape with the same element count and be silently reshaped by the adapter. Invalid results now raise `ValidationError`; scoped cleanup preserves existing hooks and permits the next run. Invalid batch objects, string cache-site collections and malformed intervention objects also produce explicit errors. Non-tensor subspace basis/center inputs are rejected deliberately.

Eleven-family dense native validation and the three-model MI study are documented under `research/model_validation/DENSE_VALIDATION.md` and `research/mi/REPORT.md`. Those retained numerical studies predate the new runtime return-value guard; the guard does not change valid intervention arithmetic. The source checkpoint suite covers the changed runtime, while historical artifacts are not relabeled as new-version runs.

## 0.1.0a2 — local alpha, not published

Adds checkpoint-backed adapters for eleven EEG model families, invocation-specific hooks for shared modules, and expanded native semantic validation. See `docs/model-coverage.md` for exact checkpoints and limitations.

**Cache compatibility:** BENDR convolution activations now expose `[batch, time, features]` instead of `[batch, features, time]`, so feature-subspace interventions act on the intended axis. Regenerate old BENDR caches; do not reuse activation bundles from earlier builds. This version distinguishes the corrected artifact from prior `0.1.0a1` wheels. The change does not establish scientific or general device/dtype validity.

## Unreleased public patching sweeps

Added `patching_sweep`, `SweepTarget`, `SweepResult`, `patch_grid` and per-trial `MatchedReplacement`. Sweeps provide identity/cleanup checks, stable trial-ID random controls, explicit invalid and excessive-multiplier diagnostics, JSON output and grouped descriptive summaries. The phase experiment now consumes the public API; the known-answer example validates exact causal coordinates and effects. Full reproduction evidence is recorded in `docs/sweep-validation.md`.

## Unreleased research workflow

Added a reproducible DREAMS spindle/N2 study: subject-held-out frozen readouts, human event probes, spectral/amplitude adjustment, matched native interventions, signal-defined slow-wave controls and subject-level summaries. Includes explicit EDF uppercase-UV correction verified against physical-value text exports. Results and limitations are in `research/spindle/RESULTS.md`.

## 0.1.0a1 — local alpha

Initial implementation: CBraMod/LaBraM adapters, activation snapshots, trial-paired replacement, semantic zero/subspace ablation, strict local loaders, run bundles, paired metrics, tests and executable examples. No public release or scientific benchmark result is claimed.
