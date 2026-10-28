"""Feature erasure must remove a convolution channel, not a time sample."""

from types import SimpleNamespace

import torch
from torch import nn

from eegfmlens import BENDRAdapter, BENDREncoderAdapter, SignalBatch, SubspaceAblation


def test_bendr_convolution_features_are_last_for_subspace_erasure():
    encoder = SimpleNamespace(in_features=2, encoder_h=4, encoder=nn.Sequential(nn.Identity()))
    context = SimpleNamespace(in_features=4, transformer_layers=[None])
    batch = SignalBatch(torch.ones(2, 2, 1, 256), ("a", "b"), ("C3", "C4"), 256, "fixture")
    adapters = [
        BENDREncoderAdapter(encoder, channels=batch.channels),
        BENDRAdapter([encoder, context], channels=batch.channels),
    ]
    raw = torch.arange(2 * 4 * 7).reshape(2, 4, 7).float()
    for adapter in adapters:
        for site in adapter.sites.values():
            if site.module_path not in ("encoder.0", "0", "1.output_layer"):
                continue
            exposed = adapter.expose(raw, site, batch)
            assert exposed.shape == (2, 7, 4)
            intervention = SubspaceAblation(site.name, torch.eye(4)[:, :1], torch.zeros(4))
            edited = intervention.apply(exposed, batch, site.layout, "fixture-model")
            restored = adapter.restore(edited, site, raw.shape)
            expected = raw.clone()
            expected[:, 0, :] = 0
            torch.testing.assert_close(restored, expected, rtol=0, atol=0)
