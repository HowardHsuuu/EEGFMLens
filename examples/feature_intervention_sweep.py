"""Known-answer cumulative SAE intervention with random-feature controls."""

import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    EEGLens,
    SignalBatch,
    TopKSAE,
    sae_feature_sweep,
)


class TwoObjectiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Identity()

    def forward(self, data):
        hidden = self.hidden(data)[:, 0, 0]
        return torch.stack((hidden[:, 0], hidden[:, 1]), dim=1)


def identity_sae():
    sae = TopKSAE(3, 3, 3).eval()
    with torch.no_grad():
        sae.encoder_weight.copy_(torch.eye(3))
        sae.encoder_bias.zero_()
        sae.decoder_weight.copy_(torch.eye(3))
        sae.decoder_bias.zero_()
    return sae


def main():
    batch = SignalBatch(
        torch.tensor([[[[3.0, 5.0, 2.0]]], [[[4.0, 6.0, 1.0]]]]),
        ("trial-a", "trial-b"),
        ("C3",),
        200,
        "feature-sweep-known-answer",
    )
    lens = EEGLens(
        TwoObjectiveModel().eval(),
        Adapter([ActivationSite("hidden", "hidden")]),
        model_id="feature-sweep-known-answer",
    )
    result = sae_feature_sweep(
        lens,
        batch,
        "hidden",
        identity_sae(),
        feature_ranking=(0, 2),
        feature_counts=(0, 1, 2),
        metrics={
            "target": lambda output, current: output[:, 0],
            "off_target": lambda output, current: output[:, 1],
        },
        random_draws=16,
        seed=17,
    )
    target_delta = result.mean_delta("target")
    off_target_delta = result.mean_delta("off_target")
    if not torch.equal(target_delta, torch.tensor([0.0, -3.5, -3.5])):
        raise RuntimeError("Known-answer target curve failed")
    if not torch.equal(off_target_delta, torch.zeros(3)):
        raise RuntimeError("Known-answer off-target curve failed")

    print("feature fractions:", result.feature_fractions.tolist())
    print("target mean changes:", target_delta.tolist())
    print("off-target mean changes:", off_target_delta.tolist())
    print(
        "target integrated change versus random median:",
        f"{result.integrated_mean_delta('target'):.3f}",
        f"vs {float(result.random_integrated_mean_delta('target').median()):.3f}",
    )


if __name__ == "__main__":
    main()
