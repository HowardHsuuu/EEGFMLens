from dataclasses import replace

import pytest
import torch

from eegfmlens import (
    AperiodicChannelFit,
    AperiodicDecomposition,
    FrequencyBand,
    SignalBatch,
    fit_aperiodic_decomposition,
    remove_fitted_spectral_component,
)
from eegfmlens.errors import ValidationError


def fixture():
    sampling_rate = 128
    time = torch.arange(256, dtype=torch.float64) / sampling_rate
    signal = torch.sin(2 * torch.pi * 10 * time)
    return SignalBatch(
        signal.reshape(1, 1, 2, 128),
        ("trial-a",),
        ("C3",),
        sampling_rate,
        "aperiodic-fixture",
        unit="uV",
    )


def decomposition():
    return AperiodicDecomposition(
        (AperiodicChannelFit("C3", 0.0, 1.0, ((10.0, 2.0, 2.0),), 0.99),),
        ("C3",),
        128,
        "aperiodic-fixture",
        "uV",
        FrequencyBand(1, 45),
        "trial",
        128,
        64,
        (1.0, 12.0),
        6,
        0.1,
    )


def test_fitted_component_removal_is_phase_preserving_and_traceable():
    batch = fixture()
    periodic = remove_fitted_spectral_component(batch, decomposition(), component="periodic")
    before = torch.fft.rfft(batch.data.flatten(2), dim=-1)
    after = torch.fft.rfft(periodic.data.flatten(2), dim=-1)
    assert after[..., 20].abs() < before[..., 20].abs() / 5
    torch.testing.assert_close(after[..., 20].angle(), before[..., 20].angle())
    torch.testing.assert_close(after[..., 100], before[..., 100])
    assert periodic.transforms[-1].name == "remove_fitted_spectral_component"
    assert len(decomposition().digest) == 64


def test_component_removal_checks_the_fitted_signal_contract():
    batch = fixture()
    fitted = decomposition()
    wrong = replace(fitted, preprocessing_id="different-preprocessing")
    with pytest.raises(ValidationError, match="preprocessing_id differs"):
        remove_fitted_spectral_component(batch, wrong)


def test_missing_optional_fit_dependency_has_an_actionable_error():
    try:
        import fooof  # noqa: F401
    except ImportError:
        with pytest.raises(ValidationError, match="optional 'aperiodic' dependency"):
            fit_aperiodic_decomposition(fixture())
    else:
        pytest.skip("Optional dependency is installed; covered by optional fit test")
