"""Offline signal-to-circuit example with a deterministic known-answer model."""

import torch
from torch import nn

from eegfmlens import (
    CANONICAL_BANDS,
    ActivationSite,
    Adapter,
    BandTarget,
    EEGLens,
    SignalBatch,
    activation_matrix,
    attribute,
    fit_ridge_probe,
    path_patch,
    r2_score,
    scale_frequency_band,
    spectral_attribution,
    spectral_band_attribution,
    spectral_perturbation_curve,
)


class SpectralToyModel(nn.Module):
    """Expose two band amplitudes and a downstream mixing site."""

    def __init__(self):
        super().__init__()
        self.features = nn.Identity()
        self.mediator = nn.Identity()

    def forward(self, data):
        series = data.flatten(2)
        spectrum = torch.fft.rfft(series, dim=-1).abs().mean(1)
        bands = torch.stack((spectrum[:, 20], spectrum[:, 40]), dim=1)
        features = self.features(bands)
        mediated = self.mediator(features * torch.tensor([1.0, 0.25]))
        return mediated.sum(1)


def main():
    sampling_rate = 128
    time = torch.arange(256) / sampling_rate
    amplitudes = torch.arange(1, 9, dtype=torch.float32)
    trials = torch.stack(
        [
            amplitude * torch.sin(2 * torch.pi * 10 * time) + torch.sin(2 * torch.pi * 20 * time)
            for amplitude in amplitudes
        ]
    ).reshape(8, 1, 2, 128)
    clean = SignalBatch(
        trials,
        tuple(f"trial-{index}" for index in range(8)),
        ("C3",),
        sampling_rate,
        "known-answer-v1",
    )
    alpha_removed = scale_frequency_band(clean, CANONICAL_BANDS["alpha"], 0)
    model = SpectralToyModel().eval()
    lens = EEGLens(
        model,
        Adapter(
            [
                ActivationSite("features", "features", layout="batch"),
                ActivationSite("mediator", "mediator", layout="batch"),
            ]
        ),
        model_id="spectral-known-answer",
    )

    cached = lens.run_with_cache(clean, sites=["features"])
    matrix = activation_matrix(cached.cache["features"])
    probe = fit_ridge_probe(matrix[:6], amplitudes[:6], alpha=1e-4)
    held_out = r2_score(amplitudes[6:], probe.predict(matrix[6:]))

    traced = path_patch(
        lens,
        clean,
        alpha_removed,
        lambda output, batch: output,
        source_site="features",
        mediator_site="mediator",
    )
    attributed = attribute(
        lens,
        clean,
        lambda output, batch: output,
        method="input_x_gradient",
    )
    frequency_attribution = spectral_attribution(attributed)
    alpha_attribution = spectral_band_attribution(
        frequency_attribution,
        CANONICAL_BANDS["alpha"],
    )
    faithfulness = spectral_perturbation_curve(
        lens,
        clean,
        lambda output, batch: output,
        (
            BandTarget("alpha", CANONICAL_BANDS["alpha"]),
            BandTarget("beta", CANONICAL_BANDS["beta"]),
        ),
    )
    print(f"held-out amplitude probe R2: {held_out.mean_r2:.4f}")
    print("alpha attribution:", alpha_attribution.flatten().tolist())
    print("spectral attribution conservation:", frequency_attribution.conservation_error.tolist())
    print("spectral perturbation AOPC:", faithfulness.aopc.tolist())
    print("recipient scores:", traced.recipient_score.tolist())
    print("source effects:", traced.source_effect.tolist())
    print("mediated effects:", traced.mediated_effect.tolist())


if __name__ == "__main__":
    main()
