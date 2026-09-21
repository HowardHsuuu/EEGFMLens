import pytest
import torch

pytest.importorskip("fooof")

from eegfmlens import SignalBatch, fit_aperiodic_decomposition  # noqa: E402

pytestmark = pytest.mark.optional


def test_fooof_fit_recovers_a_known_aperiodic_exponent():
    sampling_rate = 128
    samples = 1024
    frequencies = torch.fft.rfftfreq(samples, d=1 / sampling_rate)
    generator = torch.Generator().manual_seed(23)
    trials = []
    for _ in range(8):
        phase = 2 * torch.pi * torch.rand(frequencies.shape, generator=generator)
        amplitude = frequencies.clamp_min(1).pow(-0.75)
        amplitude[0] = 0
        coefficients = torch.polar(amplitude, phase)
        trials.append(torch.fft.irfft(coefficients, n=samples))
    batch = SignalBatch(
        torch.stack(trials).reshape(8, 1, 4, 256),
        tuple(f"trial-{index}" for index in range(8)),
        ("C3",),
        sampling_rate,
        "known-one-over-f",
    )
    fitted = fit_aperiodic_decomposition(
        batch,
        segment_samples=512,
        peak_width_limits=(1, 12),
        max_peaks=0,
    )
    assert fitted.fits[0].exponent == pytest.approx(1.5, abs=0.2)
    assert fitted.fits[0].r_squared > 0.9
