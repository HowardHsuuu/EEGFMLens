# Tool roadmap

## Implemented

- Explicit adapters for eleven checkpoint-verified model families; see [coverage](model-coverage.md).
- One catalog/connection API and per-site capability discovery across all eleven families.
- Native activation caching, paired replacement, zero and orthonormal-subspace ablation.
- Validated input/selector contracts, scoped hook cleanup and versioned run bundles.
- Channel/time patching sweeps with per-trial matching and explicit invalid-control diagnostics.
- Offline examples, independent native-hook validation and installed-wheel CI on three operating systems.

## Future capabilities

- Broader device, dtype and input-configuration validation.
- Bounded cache storage driven by actual user workflows.
- Additional BrainOmni configurations with independent native validation.
- Functional attention/QKV and gradient interfaces, each with separate correctness evidence.

Scientific studies and their publication schedules are maintained outside this
repository. They are not prerequisites for releasing the reusable tool. Current
supported scope and limitations are in [model coverage](model-coverage.md).
