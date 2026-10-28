"""Known-answer activation-to-spectrum readout composed with an SAE sweep."""

from dataclasses import replace

import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    EEGLens,
    SignalBatch,
    TopKSAE,
    activation_spectral_matrix,
    amplitude_spectral_targets,
    fit_spectral_readout,
    sae_feature_sweep,
)


class FixedFrequencyEncoder(nn.Module):
    def __init__(self, samples=32, sampling_rate=32):
        super().__init__()
        time = torch.arange(samples, dtype=torch.float64) / sampling_rate
        frequencies = torch.arange(4, 9, dtype=torch.float64)
        self.register_buffer(
            "projector",
            (2 / samples) * torch.sin(2 * torch.pi * frequencies[:, None] * time),
        )
        self.hidden = nn.Identity()

    def forward(self, data):
        hidden = self.hidden(data @ self.projector.T)
        return hidden[:, 0, 0]


def identity_sae():
    sae = TopKSAE(5, 5, 5).double().eval()
    with torch.no_grad():
        sae.encoder_weight.copy_(torch.eye(5, dtype=torch.float64))
        sae.encoder_bias.zero_()
        sae.decoder_weight.copy_(torch.eye(5, dtype=torch.float64))
        sae.decoder_bias.zero_()
    return sae


def main():
    generator = torch.Generator().manual_seed(71)
    amplitudes = 0.5 + torch.rand(24, 5, generator=generator, dtype=torch.float64)
    time = torch.arange(32, dtype=torch.float64) / 32
    frequencies = torch.arange(4, 9, dtype=torch.float64)
    basis = torch.sin(2 * torch.pi * frequencies[:, None] * time)
    data = (amplitudes @ basis).reshape(24, 1, 1, 32)
    batch = SignalBatch(
        data,
        tuple(f"trial-{index}" for index in range(24)),
        ("Cz",),
        32,
        "fixed-frequency-known-answer",
    )
    lens = EEGLens(
        FixedFrequencyEncoder().eval(),
        Adapter([ActivationSite("hidden", "hidden")]),
        model_id="fixed-frequency-known-answer",
    )
    cached = lens.run_with_cache(batch, sites=("hidden",)).cache["hidden"]
    targets = amplitude_spectral_targets(
        batch,
        transform="amplitude",
        f_min=4,
        f_max=9,
    )
    features = activation_spectral_matrix(cached, targets)
    train = torch.zeros(24, dtype=torch.bool)
    train[:16] = True
    test = ~train
    fitted = fit_spectral_readout(
        features,
        targets,
        train_mask=train,
        test_mask=test,
        alpha=1e-8,
    )
    if fitted.test.mean_r2 < 0.999:
        raise RuntimeError("Known-answer held-out spectral readout failed")

    evaluation = replace(
        batch,
        data=batch.data[test],
        trial_ids=tuple(trial for trial, keep in zip(batch.trial_ids, test) if bool(keep)),
    )

    def cached_frequency(index):
        def metric(run, current):
            geometry = amplitude_spectral_targets(
                current,
                transform="amplitude",
                f_min=4,
                f_max=9,
            )
            rows = activation_spectral_matrix(run.cache["hidden"], geometry)
            return fitted.readout.predict(rows)[:, index]

        return metric

    sweep = sae_feature_sweep(
        lens,
        evaluation,
        "hidden",
        identity_sae(),
        feature_ranking=(0,),
        feature_counts=(0, 1),
        metrics={"task_4_hz": lambda output, current: output[:, 0]},
        run_metrics={
            "decoded_4_hz": cached_frequency(0),
            "decoded_8_hz": cached_frequency(4),
        },
        cache_sites=("hidden",),
        random_draws=16,
        seed=29,
    )
    if float(sweep.mean_delta("decoded_4_hz")[-1]) >= -0.4:
        raise RuntimeError("Known-answer spectral feature intervention failed")
    if abs(float(sweep.mean_delta("decoded_8_hz")[-1])) >= 1e-6:
        raise RuntimeError("Known-answer off-target spectral control failed")

    print("held-out mean spectral R2:", f"{fitted.test.mean_r2:.6f}")
    print(
        "feature 0 direction signature:",
        fitted.readout.direction_signature(identity_sae().decoder_weight[0]).tolist(),
    )
    print("decoded 4 Hz mean change:", sweep.mean_delta("decoded_4_hz").tolist())
    print("decoded 8 Hz mean change:", sweep.mean_delta("decoded_8_hz").tolist())


if __name__ == "__main__":
    main()
