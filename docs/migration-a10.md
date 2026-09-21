# Migrating from 0.1.0a9

For the subsequent unified model connection API, see [the a11 guide](migration-a11.md).

0.1.0a10 removes bundled CBraMod/LaBraM model implementations. The runtime,
activation layouts, site names, checkpoint key handling and output modes remain
unchanged. Import native constructors from explicit upstream installations.

```python
from models.cbramod import CBraMod  # From your CBraMod checkout
from eeglens import load_cbramod

lens = load_cbramod("checkpoint.pth", model_factory=CBraMod)
```

For LaBraM, pass `model_factory=labram_base_patch200_200` from upstream
`modeling_finetune`. See [setup commands](models.md). `eeglens.models.CBraMod`
and private `_vendor` imports have been removed; no fallback downloads code.

`source_revision=` is optional caller-supplied provenance, not a verified claim.
Manifests separately record the validated reference revision and the constructor's
source-file hash when available. Supplying another implementation does not certify
it as the reference model. Prefer named native constructors over opaque lambdas.

The LaBraM constructor is called with `num_classes=0`, `init_values=0.1` and
`use_mean_pooling=False` to retain the pretrained norm. Both helpers construct
fresh models, load strictly, freeze parameters and select eval mode.

Core tests now exclude `native` as well as `integration`. Native source tests
remain available and run in a separate CI job. Generated validation reports and
machine-specific runners are not distributed with the package repository.
