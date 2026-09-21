from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eegfmlens import BENDRAdapter, EEGLens, EEGPTAdapter, SignalBatch
from eegfmlens.errors import ValidationError


class EEGPTFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Identity()
        self.patch_embed.patch_size = 64
        self.patch_embed.patch_stride = None
        self.num_patches = (2, 3)
        self.embed_num = 4
        self.blocks = nn.ModuleList()
        self.norm = nn.Identity()
        self.forward_calls = 0

    def prepare_chan_ids(self, channels):
        vocabulary = {"C3": 0, "C4": 1}
        return torch.tensor([[vocabulary[c.upper().strip(".")] for c in channels]])

    def forward(self, x, chan_ids):
        self.forward_calls += 1
        return x


def test_eegpt_rejects_unknown_and_colliding_aliases_before_forward():
    model = EEGPTFixture().eval()
    lens = EEGLens(model, EEGPTAdapter(model))
    for channels, message in [(("C3", "unknown"), "absent"), (("C3", "c3."), "unique")]:
        batch = SignalBatch(torch.randn(1, 2, 3, 64), ("a",), channels, 256, "fixture")
        with pytest.raises(ValidationError, match=message):
            lens.run_with_cache(batch)
        assert model.forward_calls == 0
        assert all(not m._forward_hooks for m in model.modules())
    good = SignalBatch(torch.randn(1, 2, 3, 64), ("a",), ("c3.", "C4"), 256, "fixture")
    torch.testing.assert_close(lens.run_with_cache(good, sites=()).output, good.data.flatten(2))
    assert model.forward_calls == 1


@pytest.mark.parametrize(
    "model",
    [[], [nn.Identity()], [nn.Identity()] * 3, nn.Identity(), [nn.Identity(), nn.Identity()]],
)
def test_bendr_malformed_composition_reports_validation_error(model):
    with pytest.raises(ValidationError, match="BENDR"):
        BENDRAdapter(model, channels=("C3", "C4"))


def test_bendr_constructor_recovery_with_valid_pair():
    encoder = SimpleNamespace(in_features=2, encoder_h=4)
    context = SimpleNamespace(in_features=4, transformer_layers=[None])
    adapter = BENDRAdapter([encoder, context], channels=("C3", "C4"))
    assert len(adapter.sites) == 4
