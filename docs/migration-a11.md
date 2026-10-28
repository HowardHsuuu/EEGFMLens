# Moving from 0.1.0a10 to 0.1.0a11

The package namespace now matches the project name: replace imports from
`eeglens` with imports from `eegfmlens`. No compatibility alias is provided
because neither alpha was published to PyPI.

Existing adapters and checkpoint loader signatures otherwise remain compatible. The new
`connect(...)` API provides one checked construction path for all eleven public
families, while `supported_models()` and `model_info(...)` expose their contracts.

```python
from eegfmlens import connect

# Before
lens = EEGLens(model.eval(), BIOTAdapter(model, channels=channels))

# Recommended in a11
lens = connect(model.eval(), "biot", channels=channels)
```

Direct adapter construction remains supported for advanced use. BENDR selects
the full encoder/contextualizer integration by default; pass `variant="encoder"`
for the convolutional encoder alone.

`lens.capabilities()` reports exact per-site selectors and invocation behavior.
The root `THIRD_PARTY_NOTICES.md` was removed because no upstream model code is
bundled. Attribution and the complete license for the retained LaBraM channel
mapping now live together in `LICENSES/` and are still shipped in both artifacts.
