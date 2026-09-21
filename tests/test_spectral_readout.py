from dataclasses import replace

import pytest
import torch

from eegfmlens import (
    Activation,
    FrequencyBand,
    SignalBatch,
    activation_spectral_matrix,
    amplitude_spectral_targets,
    fit_spectral_readout,
)
from eegfmlens.errors import ValidationError


def sine_batch():
    time = torch.arange(8, dtype=torch.float64) / 8
    amplitudes = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    waves = amplitudes[..., None] * torch.sin(2 * torch.pi * time)[None, None, :]
    data = waves[:, None].repeat(1, 2, 1, 1)
    return SignalBatch(data, ("a", "b"), ("C3", "C4"), 8, "known-spectrum")


def activation(tensor, layout="bcpd"):
    return Activation(
        tensor=tensor,
        site="hidden",
        model_id="known-model",
        trial_ids=("a", "b"),
        channels=("C3", "C4"),
        preprocessing_id="known-spectrum",
        sampling_rate=8,
        stride=8,
        patch_samples=8,
        unit="model_scaled",
        layout=layout,
        native_shape=tuple(tensor.shape),
    )


def test_log_amplitude_targets_and_patch_activation_alignment_have_known_geometry():
    targets = amplitude_spectral_targets(sine_batch(), f_min=0.5, f_max=3)
    torch.testing.assert_close(targets.frequencies, torch.tensor([1.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(
        targets.values[:, 0],
        torch.log1p(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)),
    )
    assert targets.row_coordinates == (("a", 0), ("a", 1), ("b", 0), ("b", 1))

    base = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]],
        dtype=torch.float64,
    )
    bcpd = torch.stack((base - 1, base + 1), dim=1)
    expected = base.reshape(4, 2)
    torch.testing.assert_close(activation_spectral_matrix(activation(bcpd), targets), expected)

    tokens = bcpd.reshape(2, 4, 2)
    tokens = torch.cat((torch.full((2, 1, 2), 999.0, dtype=torch.float64), tokens), dim=1)
    torch.testing.assert_close(
        activation_spectral_matrix(activation(tokens, "tokens"), targets), expected
    )


def test_spectral_readout_recovers_linear_targets_and_direction_signatures():
    generator = torch.Generator().manual_seed(31)
    batch = SignalBatch(
        torch.randn(12, 1, 1, 8, generator=generator, dtype=torch.float64),
        tuple(f"trial-{index}" for index in range(12)),
        ("Cz",),
        8,
        "linear-readout",
    )
    targets = amplitude_spectral_targets(batch, f_min=0.5)
    features = torch.tensor(
        [[float(index), float((index * 5) % 7)] for index in range(12)],
        dtype=torch.float64,
    )
    weight = torch.tensor([[0.2, 0.1, -0.1, 0.3], [0.05, -0.15, 0.2, 0.1]], dtype=torch.float64)
    values = features @ weight + 10
    targets = replace(targets, values=values)
    train = torch.zeros(12, dtype=torch.bool)
    train[:8] = True
    test = ~train

    result = fit_spectral_readout(
        features,
        targets,
        train_mask=train,
        test_mask=test,
        alpha=1e-10,
    )
    assert result.train.mean_r2 == pytest.approx(1.0, abs=1e-10)
    assert result.test.mean_r2 == pytest.approx(1.0, abs=1e-8)
    torch.testing.assert_close(
        result.readout.direction_signature(torch.tensor([1.0, 0.0], dtype=torch.float64)),
        weight[0],
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        result.readout.band_signature(
            torch.tensor([1.0, 0.0], dtype=torch.float64), FrequencyBand(1, 3)
        ),
        weight[0, :2].mean(),
        atol=1e-10,
        rtol=1e-10,
    )
    assert result.train_rows == 8 and result.test_rows == 4
    assert len(result.metadata["target_sha256"]) == 64


def test_spectral_readout_contracts_reject_ambiguous_alignment_and_splits():
    batch = sine_batch()
    targets = amplitude_spectral_targets(batch, f_min=0.5, f_max=3)
    with pytest.raises(ValidationError, match="FFT geometry"):
        replace(targets, samples=targets.samples + 1)
    tensor = torch.ones(2, 2, 2, 2, dtype=torch.float64)
    wrong = replace(activation(tensor), trial_ids=("b", "a"))
    with pytest.raises(ValidationError, match="provenance"):
        activation_spectral_matrix(wrong, targets)
    with pytest.raises(ValidationError, match="only to trial"):
        activation_spectral_matrix(activation(tensor), targets, trial_pooling="flatten")

    features = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    train = torch.tensor([True, True, False, False])
    overlap = torch.tensor([False, True, True, False])
    with pytest.raises(ValidationError, match="disjoint"):
        fit_spectral_readout(
            features,
            targets,
            train_mask=train,
            test_mask=overlap,
            alpha=1,
        )
    with pytest.raises(ValidationError, match="directions"):
        good_test = ~train
        result = fit_spectral_readout(
            features,
            replace(
                targets,
                values=torch.tensor(
                    [[1.0, 2.0], [2.0, 4.0], [3.0, 5.0], [4.0, 7.0]],
                    dtype=torch.float64,
                ),
            ),
            train_mask=train,
            test_mask=good_test,
            alpha=1,
        )
        result.readout.direction_signature(torch.ones(3, dtype=torch.float64))


def test_trial_targets_reject_overlapping_patch_geometry():
    batch = replace(sine_batch(), patch_stride_samples=4)
    with pytest.raises(ValidationError, match="contiguous nonoverlapping"):
        amplitude_spectral_targets(batch, scope="trial")
