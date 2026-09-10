# Dense intervention validation

Dense rank-three, nonzero-centered feature projections now have retained native comparison results for eleven model families / thirteen component-checkpoint views. This complements the earlier coordinate-subspace checks; it does not certify all checkpoints, input geometries, devices or precisions.

Each result compares EEGLens with a separately registered native hook, deriving native indices without using `adapter.restore` or `Selection.mask`. Native downstream output comparisons require exact equality. Conditions are individual site × batch × global/local projection executions, not independent statistical samples.

| View | Conditions | Input | Physical dense selection |
| --- | ---: | --- | --- |
| CBraMod | 196 | Real development EEG | All sites; native spatial/temporal folding handled independently |
| LaBraM | 148 | Real development EEG | All sites; channel-major indices and CLS offset |
| CSBrain | 56 | Real development EEG | All sites; independent region-order permutation |
| EEGPT | 20 | Synthetic | Rejected: window/summary tokens do not expose physical selection |
| BIOT six-dataset | 18 | Synthetic | Rejected: STFT sequence |
| BIOT PREST | 18 | Synthetic | Rejected: STFT sequence |
| BENDR encoder | 12 | Synthetic | Rejected: downsampled feature sequence |
| BENDR encoder + contextualizer | 22 | Synthetic | Rejected: contextual/downsampled sequence |
| SignalJEPA | 20 | Synthetic | Rejected: contextual/local token layout |
| BrainOmni tiny | 24 | Synthetic geometry | Rejected: latent sensor units |
| DIVER EEG | 34 | Synthetic geometry | Embedding/head C4, patch 1; augmented-token sites reject selectors |
| ST-EEGFormer small | 20 | Synthetic | Rejected: conservative exposed token layout |
| NeuroRVQ | 208 | Synthetic | All four invocations of each shared module; other branches remain exact |
| **Total** | **796** | | |

The three real-EEG models use two development trials, nineteen sensors and four patches. Other checks use the configured synthetic geometries in their runners. Batch sizes are one and two. Local edits select a whole patch, not a sample or a continuous time range. Physiological meaning is not inferred from a dense random basis.

## Floating-point and stochastic execution contracts

All runs use local CPU float32. BrainOmni and DIVER upstream evaluation still samples dropout, so their comparisons pair CPU RNG state. These results do not establish arbitrary-batch or per-trial stochastic equivalence.

BENDR native convolution activations are B,D,T; contextual activations are T,B,D, while EEGLens exposes B,T,D. An initial batch-two dense check at `context.input` failed bitwise equality when the native oracle performed the matrix multiplication grouped by T instead of B. The corrected native oracle explicitly transposes to batch-first for the projection, then restores native shape and strides. This preserves the public operation's floating-point evaluation order and passes exact downstream comparisons. No package implementation or comparison tolerance was changed. Mathematically equivalent operations with different GEMM grouping are not asserted bitwise identical.

Projection-residual checks where recorded use a separate 1e−5 relative scale bound for float32 cancellation; that bound does not relax native downstream parity. Locality checks require exact unchanged values, and NeuroRVQ additionally requires exact output equality in the three non-target branches.

## Evidence and reproduction

The retained JSON files are under `results/dense-{mi,encoders,context,multimodal,neurorvq}-v1/`. `results/dense-summary.json` records their hashes and runner checks after completion. The runners are:

- `validate_dense_mi.py`: CBraMod, LaBraM, CSBrain, using the study's original environment.
- `validate_dense_encoders.py`: EEGPT, both BIOT checkpoints, BENDR encoder, original environment.
- `validate_dense_context.py`: full BENDR in the original environment; SignalJEPA with standard Braindecode imports in the modern environment.
- `validate_dense_multimodal.py`: BrainOmni in the original environment; DIVER/ST-EEGFormer in the modern environment.
- `validate_dense_neurorvq.py`: NeuroRVQ in the original environment.

Run from the parent workspace with `PYTHONPATH=eeglens/src`; external repositories and weights are under `research/eeglens_model_validation`. Runners retain checkpoint, native-source, adapter and runner hashes. Dependency environments and checkpoint provenance are documented in the existing validation README. These external sources and weights are not distributed with EEGLens.

Remaining release work includes supported-geometry boundaries, clean installation environments, public API error contracts, licensing/provenance and clear documentation of platform limitations. Eleven-family dense evidence is a substantial integration milestone, not a claim of TransformerLens feature parity.
