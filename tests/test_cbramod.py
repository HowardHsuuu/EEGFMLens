import pytest
import torch

from eegfmlens import (
    Ablation,
    CBraModAdapter,
    EEGLens,
    Replacement,
    Selection,
    SignalBatch,
    attribute,
)

pytestmark = pytest.mark.native


def test_native_blocks_and_folded_branch_replacement(native_cbramod):
    CBraMod = native_cbramod.CBraMod
    torch.manual_seed(9)
    torch.set_num_threads(2)
    model = CBraMod(n_layer=2).eval()
    batch = SignalBatch(torch.randn(2, 3, 4, 200), ("a", "b"), ("C3", "Cz", "C4"), 200, "synthetic")
    lens = EEGLens(model, CBraModAdapter(model))
    baseline = lens.run_with_cache(batch)
    with torch.no_grad():
        torch.testing.assert_close(baseline.output, model(batch.data), rtol=0, atol=0)
    assert all(n == 1 for n in baseline.calls.values())
    for name in baseline.cache:
        activation = baseline.cache[name]
        assert activation.tensor.shape[:3] == (2, 3, 4)
        identity = lens.run_with_interventions(batch, interventions=[Replacement(name, activation)])
        torch.testing.assert_close(identity.output, baseline.output, rtol=0, atol=0)
        ablated = lens.run_with_interventions(
            batch, interventions=[Ablation(name, Selection(sensors=("C3",), patches=(1,)))]
        )
        assert not torch.equal(ablated.output, baseline.output)
    restored = lens.run_with_cache(batch, sites=[])
    torch.testing.assert_close(restored.output, baseline.output, rtol=0, atol=0)
    site = "blocks.0.output"
    attributed = attribute(
        lens,
        batch,
        lambda output, current: output.flatten(1)[:, 0],
        method="input_x_gradient",
        sites=(site,),
    )
    assert attributed.input_attribution.shape == batch.data.shape
    assert attributed.site_attributions[site].shape == baseline.cache[site].tensor.shape
    assert torch.isfinite(attributed.input_attribution).all()
    assert all(parameter.grad is None for parameter in model.parameters())
