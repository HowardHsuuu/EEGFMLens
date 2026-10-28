"""Offline execution example; randomly initialized weights, no scientific claim."""

from dataclasses import replace

import torch
from torch import nn

from eegfmlens import ActivationSite, Adapter, EEGLens, Replacement, SignalBatch


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = nn.Sequential(nn.Linear(200, 32), nn.GELU(), nn.Linear(32, 16)).eval()
    lens = EEGLens(model, Adapter([ActivationSite("features", "2")]))
    batch = SignalBatch(
        torch.randn(2, 3, 4, 200), ("trial-1", "trial-2"), ("C3", "CZ", "C4"), 200, "synthetic-v1"
    )
    site = "features"
    clean = lens.run_with_cache(batch, sites=[site])
    recipient = replace(batch, data=torch.zeros_like(batch.data))
    corrupt = lens.run_with_cache(recipient, sites=[])
    patched = lens.run_with_interventions(
        recipient, interventions=[Replacement(site, clean.cache[site])]
    )
    torch.testing.assert_close(patched.output, clean.output)
    assert not torch.equal(corrupt.output, clean.output)
    print(f"Cached {site}: {tuple(clean.cache[site].tensor.shape)}")
    print("Full final-block replacement restored the clean output.")


if __name__ == "__main__":
    main()
