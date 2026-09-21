import pytest
import torch

from eegfmlens import Ablation, EEGLens, LaBraMAdapter, Replacement, Selection, SignalBatch
from eegfmlens.errors import UnsupportedSiteError, ValidationError

pytestmark = pytest.mark.native


def test_native_labram_tokens_cls_and_effective_selection(native_labram):
    pytest.importorskip("timm")
    NeuralTransformer = native_labram.NeuralTransformer

    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = NeuralTransformer(
        depth=2, num_classes=0, init_values=0.1, use_mean_pooling=False
    ).eval()
    batch = SignalBatch(torch.randn(2, 3, 4, 200), ("a", "b"), ("C3", "CZ", "C4"), 200, "synthetic")
    lens = EEGLens(model, LaBraMAdapter(model))
    baseline = lens.run_with_cache(batch)
    for site in baseline.cache:
        identity = lens.run_with_interventions(
            batch, interventions=[Replacement(site, baseline.cache[site])]
        )
        torch.testing.assert_close(identity.output, baseline.output, atol=0, rtol=0)
        effect = lens.run_with_interventions(
            batch, interventions=[Ablation(site, Selection(sensors=("C3",), patches=(1,)))]
        )
        assert not torch.equal(effect.output, baseline.output)
    assert baseline.cache["embedding.output"].tensor.shape == (2, 12, 200)
    assert baseline.cache["blocks.0.output"].tensor.shape == (2, 13, 200)
    name = "blocks.0.output"
    identity = lens.run_with_interventions(
        batch, interventions=[Replacement(name, baseline.cache[name])]
    )
    torch.testing.assert_close(identity.output, baseline.output, atol=0, rtol=0)
    ablated = lens.run_with_interventions(
        batch,
        sites=[name],
        interventions=[Ablation(name, Selection(sensors=("C3",), patches=(1,)))],
    )
    changed = ablated.cache[name].tensor
    original = baseline.cache[name].tensor
    assert torch.equal(changed[:, 0], original[:, 0])  # CLS excluded
    assert torch.equal(changed[:, 1], original[:, 1])
    assert torch.equal(changed[:, 2], torch.zeros_like(changed[:, 2]))
    assert torch.equal(changed[:, 3:], original[:, 3:])
    assert not torch.equal(ablated.output, baseline.output)
    with pytest.raises(UnsupportedSiteError):
        lens.run_with_cache(batch, sites=["blocks.0.qkv"])
    with pytest.raises(ValidationError, match="options"):
        lens.run_with_cache(batch, sites=[], input_chans=[0, 1, 2, 3])


@pytest.mark.parametrize("output,shape", [("all_tokens", (1, 7, 200)), ("pooled", (1, 200))])
def test_other_native_output_paths(output, shape, native_labram):
    pytest.importorskip("timm")
    NeuralTransformer = native_labram.NeuralTransformer

    model = NeuralTransformer(
        depth=1, num_classes=0, init_values=0.1, use_mean_pooling=False
    ).eval()
    lens = EEGLens(model, LaBraMAdapter(model, output=output))
    batch = SignalBatch(torch.randn(1, 3, 2, 200), ("a",), ("C3", "CZ", "C4"), 200, "test")
    result = lens.run_with_cache(batch)
    assert result.output.shape == shape
    patched = lens.run_with_interventions(
        batch, interventions=[Replacement("blocks.0.output", result.cache["blocks.0.output"])]
    )
    torch.testing.assert_close(patched.output, result.output)
