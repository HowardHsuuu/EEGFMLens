# Architecture

```text
model-ready SignalBatch + native model
                ↓
adapter: execution path, sites, tensor layouts
                ↓
EEGLens: scoped hooks, validation, snapshot/intervention
                ↓
RunResult: native output, cache, execution evidence
                ↓
public paired sweeps / diagnostics / descriptive summaries
                ↓
composable spectral edits / probes / SAE features / path tests
                ↓
caller-owned task metrics / study design / statistical inference
```

The runtime does not reconstruct attention or replace native forwards. Adapters define module outputs and reversible layouts. CBraMod branch axes restore batch/sensor/patch/feature; LaBraM distinguishes patch-only and CLS-prefixed sequences.

| Module | Responsibility |
|---|---|
| `types.py` | Batches, sites, snapshots and results |
| `model.py` | Scoped hooks, call counts and state checks |
| `adapters/` | Native paths and tensor semantics |
| `interventions.py` | Replacement, selection, zero/subspace ablation |
| `spectral.py` | EEG frequency measurement and trial-matched signal edits |
| `probes.py`, `diagnostics.py` | Held-out linear probes and representation diagnostics |
| `sae.py` | Top-K SAE training, metrics and feature interventions |
| `circuits.py` | Hypothesis-driven source-to-mediator path patching |
| `loading.py` | Strict local checkpoints and provenance |
| `io.py`, `provenance.py` | Local bundles and hashes |
| `metrics.py` | Paired effects and guarded normalization |
| `sweep.py` | Trial-paired sweeps, controlled replacements, validity records and grouped summaries |
| `workflows.py` | Reusable end-to-end diagnostics built from the intervention API |
| `adapters/labram_channels.py` | Retained native channel metadata; model implementations stay external |

Core depends on PyTorch and NumPy; LaBraM implementation imports and MNE I/O are
optional. EEG-specific operations share the same batch, intervention and
provenance contracts as activation-space methods; they are not a separate analysis
layer. Dataset pipelines, experimental hypotheses and statistical inference remain
study responsibilities. No placeholder adapter silently claims support.

Geometry, padding/bad-channel masks, chunked caches, gradients and functional instrumentation require future contracts. BrainOmni latent slots must not inherit electrode selectors merely because dimensions match.
