# Tool acceptance ledger

EEGLens 0.1.0a9 is a public source alpha. This ledger evaluates the open-source tool;
scientific studies and their data releases are separate. TransformerLens inspires
the quality standard, not a claim of feature parity or perfect support for every model configuration.

| Requirement | Evidence | Verified scope |
| --- | --- | --- |
| Public API and contracts | [API](../../docs/api.md), [input contracts](../../docs/input-contracts.md), [custom adapters](../../docs/custom-models.md) | Observe, replace, ablate, serialize and sweep module outputs with explicit tensor semantics |
| Eleven model families | [Coverage](../../docs/model-coverage.md), [native validation](../README.md), [dense checks](DENSE_VALIDATION.md) | 13 model/component views; 488 coordinate and 796 dense conditions; sampled CPU float32 configurations |
| Local intervention correctness | Independent native hooks, unselected-coordinate preservation, invalid-selector rejection and cleanup | Physical sensor/time selectors only where their mapping is verified |
| Installed distribution | [CI reports](results/ci-76f52c4/summary.json) | Ubuntu/macOS/Windows × Python 3.10/3.12; 87 non-integration tests each, 33 package Python files, dependency checks and examples |
| Reusable examples | [Quickstart](../../examples/quickstart.py), [analytic sweep](../../examples/patching_sweep.py), [real EEG recipe](VALIDATION_A9.md) | Offline synthetic examples; explicit local EEG/checkpoint paths for native checks |
| Source and licensing | [Provenance](VENDOR_PROVENANCE_A9.md), [third-party notices](THIRD_PARTY_NOTICES_A9.md) | Eleven source/license checks on a9 archives; recorded direct-source equivalence and acknowledged notices |

[GitHub run 34435514982](https://github.com/HowardHsuuu/EEGFMLens/actions/runs/34435514982)
validated source commit `76f52c4`. All six downloaded JSON/JUnit pairs were checked
for 87 passing tests, zero failures/errors/skips, JUnit digests, verifier/example
source hashes, quickstart and the 24-effect analytic sweep. Two checkpoint tests
were deliberately deselected. Repeating the suite on six environments is not 522
distinct tests and does not run the eleven native checkpoints on every platform.

## Evidence relocation

Native validation now lives in `validation/`. Historical JSON/XML evidence is
retained byte-for-byte, including original paths and runner hashes. Those hashes
identify the original execution, not a rerun of relocated scripts. The original
files remain available in Git history at `1481c55`. Current validation fixtures no
longer import scientific experiment modules. The package runtime is unchanged by
this repository separation; its post-separation checks are recorded separately.

## Limits and future work

GPU, mixed precision, compiled execution, gradients, individual attention heads,
QKV editing and arbitrary architecture variants are not validated capabilities.
Upstream historical derivation is not completely reconstructed. Checkpoints and
datasets retain their own terms and are not bundled. No PyPI release is claimed.

Scientific findings, a common physiological mechanism, and publication of our
analysis replay data are not conditions for this tool's acceptance.
