"""Check masked EEGPT's native tuple hook and physical time-patch selection."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eegfmlens import (
    EEGLens, EEGPTMaskedReconstructionAdapter, Replacement, Selection, SignalBatch,
)
from eegfmlens.errors import ValidationError


class Encoder(nn.Module):
    num_patches = (3, 2)
    embed_num = 1

    def __init__(self):
        super().__init__()
        self.patch_embed = SimpleNamespace(patch_size=4, patch_stride=4)

    def prepare_chan_ids(self, channels):
        return torch.arange(len(channels), dtype=torch.long)[None]

    def forward(self, data, channel_ids, *, mask_x):
        return data.mean(dim=(1, 2), keepdim=True).unsqueeze(-1).expand(-1, 1, 1, 4)


class Predictor(nn.Module):
    def forward(self, encoded, *, mask_x):
        value = encoded.mean(dim=-1, keepdim=True)
        combined = value.expand(-1, 2, 1, 4).clone()
        combined[:, 1] += 1
        return encoded, combined


class Reconstructor(nn.Module):
    def forward(self, combined, channel_ids, *, mask_y):
        return combined[:, mask_y // 3, 0, :]


class FullPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.predictor, self.reconstructor = Encoder(), Predictor(), Reconstructor()


def _batch(value):
    return SignalBatch(
        torch.full((2, 3, 2, 4), value), ("one", "two"), ("C3", "C4", "P3"), 256,
        "fixed-native-preprocessing",
    )


def test_masked_pipeline_patch_replacement_matches_native_hook_and_rejects_sensor_selection():
    model = FullPipeline().eval()
    lens = EEGLens(model, EEGPTMaskedReconstructionAdapter(model))
    kwargs = {"mask_x": [[0, 2]], "mask_y": [4]}
    recipient, donor = _batch(1.0), _batch(3.0)
    baseline = lens.run_with_cache(recipient, **kwargs)
    source = lens.run_with_cache(donor, **kwargs)
    assert baseline.cache["predictor.combined"].tensor.shape == (2, 2, 1, 4)
    capability = next(c for c in lens.capabilities() if c.name == "predictor.combined")
    assert capability.selectors == ("whole", "axis", "patch")
    patched = lens.run_with_interventions(
        recipient,
        interventions=[Replacement(
            "predictor.combined", source.cache["predictor.combined"],
            Selection(patches=(1,)),
        )],
        **kwargs,
    )
    torch.testing.assert_close(patched.output, source.output, rtol=0, atol=0)
    torch.testing.assert_close(
        lens.run_with_cache(recipient, **kwargs).output, baseline.output, rtol=0, atol=0
    )
    with pytest.raises(ValidationError, match="no sensor identity"):
        lens.run_with_interventions(
            recipient,
            interventions=[Replacement(
                "predictor.combined", source.cache["predictor.combined"],
                Selection(sensors=("C3",)),
            )],
            **kwargs,
        )
    with pytest.raises(ValidationError, match="ordered physical time rows"):
        lens.run_with_cache(recipient, mask_x=[[2, 0]], mask_y=[4])
