from dataclasses import replace

import pytest
import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    EEGLens,
    FrequencyBand,
    SignalBatch,
    band_power,
    patch_frequency_band,
    power_spectrum,
    scale_frequency_band,
)
from eegfmlens.errors import ValidationError


def sine_batch(*, amplitude=1.0, phase=0.0):
    sampling_rate = 128
    time = torch.arange(256, dtype=torch.float64) / sampling_rate
    signal = amplitude * torch.sin(2 * torch.pi * 10 * time + phase)
    signal += 0.4 * torch.sin(2 * torch.pi * 20 * time)
    data = signal.reshape(1, 1, 2, 128)
    return SignalBatch(data, ("trial",), ("C3",), sampling_rate, "synthetic")


def test_band_scaling_is_frequency_specific_and_audited():
    batch = sine_batch()
    alpha, beta = FrequencyBand(8, 13), FrequencyBand(18, 23)
    before = power_spectrum(batch)
    edited = scale_frequency_band(batch, alpha, 0)
    after = power_spectrum(edited)
    assert float(band_power(after, alpha)) < float(band_power(before, alpha)) * 1e-10
    torch.testing.assert_close(
        band_power(after, beta), band_power(before, beta), atol=1e-12, rtol=1e-10
    )
    assert edited.transforms[-1].name == "scale_frequency_band"
    identity = scale_frequency_band(batch, alpha, 1)
    torch.testing.assert_close(identity.data, batch.data, atol=1e-12, rtol=1e-12)


def test_spectral_patching_aligns_trials_and_separates_amplitude_from_phase():
    recipient = sine_batch(amplitude=1.0, phase=0.0)
    donor = replace(sine_batch(amplitude=2.0, phase=torch.pi / 2), trial_ids=("trial",))
    band = FrequencyBand(8, 13)
    amplitude = patch_frequency_band(recipient, donor, band, component="amplitude")
    recipient_fft = torch.fft.rfft(recipient.data.reshape(1, 1, -1))
    donor_fft = torch.fft.rfft(donor.data.reshape(1, 1, -1))
    amplitude_fft = torch.fft.rfft(amplitude.data.reshape(1, 1, -1))
    frequencies = torch.fft.rfftfreq(256, 1 / 128)
    mask = (frequencies >= 8) & (frequencies < 13)
    carrier = frequencies == 10
    torch.testing.assert_close(amplitude_fft[..., mask].abs(), donor_fft[..., mask].abs())
    torch.testing.assert_close(
        amplitude_fft[..., carrier].angle(), recipient_fft[..., carrier].angle()
    )
    phase = patch_frequency_band(recipient, donor, band, component="phase")
    phase_fft = torch.fft.rfft(phase.data.reshape(1, 1, -1))
    torch.testing.assert_close(phase_fft[..., mask].abs(), recipient_fft[..., mask].abs())
    torch.testing.assert_close(phase_fft[..., carrier].angle(), donor_fft[..., carrier].angle())


def test_trial_scope_rejects_overlapping_patches_and_run_records_edits():
    batch = replace(sine_batch(), patch_stride_samples=64)
    with pytest.raises(ValidationError, match="contiguous nonoverlapping"):
        power_spectrum(batch)
    edited = scale_frequency_band(sine_batch(), FrequencyBand(8, 13), 0)
    lens = EEGLens(nn.Sequential(nn.Identity()).eval(), Adapter([ActivationSite("x", "0")]))
    result = lens.run_with_cache(edited, sites=[])
    assert result.metadata["signal_transforms"][0]["name"] == "scale_frequency_band"
