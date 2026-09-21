"""Offline cross-model and EEG-concept analysis with known-answer models."""

import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    EEGLens,
    SignalBatch,
    cross_model_similarity,
    spectral_features,
    time_domain_features,
)


class FeatureModel(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = nn.Linear(64, 64, bias=False)
        self.transform.weight.data.copy_(transform)
        self.transform.weight.requires_grad_(False)

    def forward(self, data):
        return self.transform(data).mean(dim=(1, 2, 3))


def main():
    generator = torch.Generator().manual_seed(19)
    trials = torch.randn(12, 2, 2, 64, generator=generator)
    batch = SignalBatch(
        trials,
        tuple(f"trial-{index}" for index in range(12)),
        ("C3", "C4"),
        128,
        "cross-model-known-answer",
    )
    identity = torch.eye(64)
    rotation, _ = torch.linalg.qr(torch.randn(64, 64, generator=generator))
    first = EEGLens(
        FeatureModel(identity).eval(),
        Adapter([ActivationSite("features", "transform")]),
        model_id="known-answer:first",
    ).run_with_cache(batch, sites=["features"])
    second = EEGLens(
        FeatureModel(rotation).eval(),
        Adapter([ActivationSite("features", "transform")]),
        model_id="known-answer:second",
    ).run_with_cache(batch, sites=["features"])
    similarity = cross_model_similarity(first.cache, second.cache, pooling="flatten")
    time_features = time_domain_features(batch)
    frequency_features = spectral_features(batch)
    print(f"cross-model linear CKA: {float(similarity.values[0, 0]):.6f}")
    print(
        "neurophysiology feature matrix:",
        tuple(time_features.matrix().shape),
        "+",
        tuple(frequency_features.matrix().shape),
    )


if __name__ == "__main__":
    main()
