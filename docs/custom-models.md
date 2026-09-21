# Connect a native PyTorch model

See [model coverage](model-coverage.md) for the verified integrations, including CBraMod, LaBraM, EEGPT, BIOT and the BENDR convolutional encoder. The runtime is extensible: `GenericAdapter` connects a user-supplied model without changing EEGLens internals. It does not download weights, infer EEG geometry or certify a new architecture.

```python
from eegfmlens import EEGLens, GenericAdapter, ActivationSite, inspect_modules

# Discover paths without running a forward pass.
for module in inspect_modules(model):
    print(module.path, module.module_type)

# Example: a continuous-input model with a native encoder module.
adapter = GenericAdapter(
    [ActivationSite("encoder.output", "encoder", layout="batch")],
    forward=lambda model, batch, **kwargs: model(batch.data.flatten(2), **kwargs),
)
lens = EEGLens(model.eval(), adapter)
result = lens.run_with_cache(batch)
```

`batch` is a `SignalBatch` with model-ready `[batch, sensor, patch, sample]` data and explicit sampling rate, channels, units and preprocessing identity. The forward callback may pass channel positions, masks or other required model arguments using batch metadata. Flattening patches to continuous time is appropriate only when they are consecutive and nonoverlapping; otherwise supply a model-specific conversion. The optional `validate(batch)` callback should reject incompatible input contracts.

## Declare what a site means

| Layout | Native tensor | Available semantic selection |
|---|---|---|
| `bcpd` | `[B,C,P,D]` | sensor and patch |
| `patch_tokens` | `[B,C*P,D]`, channel-major tokens | sensor and patch |
| `tokens` | `[B,1+C*P,D]`, leading CLS then channel-major tokens | sensor and patch; unrestricted edits include CLS |
| `spatial` | `[B*P,C,D]` | sensor and patch after unfolding |
| `temporal` | `[B*C,P,D]` | sensor and patch after unfolding |
| `batch` | any batch-first tensor | whole activation only |

Use `batch` for downsampled convolution features, pooled outputs, unfamiliar token order or multiple special tokens. It supports caching and whole-activation interventions without claiming electrode/time coordinates. Sensor/patch sweeps require a verified semantic layout. Arbitrary geometry is not automatically mapped.

Sites refer to module **outputs**. For tuple/list outputs, set `tensor_index=0` (or another index). For dictionary outputs, use a key such as `tensor_index="hidden_states"`. Standard dicts, tuples, lists and namedtuples preserve their untargeted values when edited. Nested output paths and arbitrary custom output containers are not supported.

Site names and layout names must be nonempty strings. Module paths are strings; `module_path=""` selects the root module. `writable` must be a boolean, and `tensor_index` must be an integer or string key (or `None` for the entire tensor output). Boolean indices and string boolean flags are rejected, because Python would otherwise interpret them as valid indices or truthy permissions. Negative integer tuple/list indices are supported. The stricter descriptor guards are included starting with a6.

A selected module must execute exactly once by default. For a reused module, declare `ActivationSite(..., call_index=1, expected_calls=4)` to target its second invocation and require exactly four calls. Give each invocation a distinct site name. Discovery alone does not infer invocation or branch semantics. The runtime is inference-only and preserves scoped hook cleanup and model state checks.

## Validate a new integration

Custom intervention objects must expose a site name, a `Selection`, and a callable `apply(current, batch, layout, model_id)`. The returned tensor must preserve the exposed shape, dtype and device and contain only finite values. A tensor with the same element count but a different shape is rejected before native restoration; the runtime does not infer an intended reshape. Invalid returns raise `ValidationError` and remove the runtime's hooks while preserving pre-existing hooks. Custom code must not mutate its input in place or run the underlying model concurrently. Pass cache sites as a collection such as `sites=["encoder.output"]`, not a bare string.

Compare cached and unhooked native predictions on the same batch, then run an intervention with an analytically known effect and verify hook cleanup. Check token ordering and sensor/time selection explicitly before using those semantics scientifically. `tests/test_generic.py` exercises a continuous CNN with dictionary/namedtuple outputs; it is a runtime contract test, not evidence that a third EEG foundation model has been validated.

For native models that fold a batch axis, a dedicated `Adapter` may override `expose(tensor, site, batch)` and `restore(tensor, site, native_shape)`. The pair must preserve all values and restore the exact native geometry; exposed axis zero must remain the trial axis. EEGPTAdapter demonstrates a validated window-folded mapping.
