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
    attribute,
    channel_attribution,
    patch_attribution,
    spectral_attribution,
    spectral_band_attribution,
    temporal_attribution,
)
from eegfmlens.errors import ValidationError


class Quadratic(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Identity()

    def forward(self, data):
        hidden = self.hidden(2 * data)
        return hidden.square().flatten(1).sum(1)


class NonlinearSite(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Identity()

    def forward(self, data):
        hidden = self.hidden(data.square())
        return hidden.square().flatten(1).sum(1)


class FoldedSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Identity()

    def forward(self, data):
        batch, channels, patches, features = data.shape
        folded = data.permute(0, 2, 1, 3).reshape(batch * patches, channels, features)
        folded = self.spatial(folded)
        restored = folded.reshape(batch, patches, channels, features).permute(0, 2, 1, 3)
        return restored.square().flatten(1).sum(1)


def fixture():
    time = torch.arange(64, dtype=torch.float64) / 64
    first = torch.sin(2 * torch.pi * 10 * time)
    second = 0.5 * torch.sin(2 * torch.pi * 10 * time)
    batch = SignalBatch(
        torch.stack((first, second)).reshape(2, 1, 2, 32),
        ("a", "b"),
        ("C3",),
        64,
        "attribution-test",
    )
    baseline = replace(
        batch,
        data=torch.zeros_like(batch.data).flip(0),
        trial_ids=("b", "a"),
    )
    lens = EEGLens(
        Quadratic().double().eval(),
        Adapter([ActivationSite("hidden", "hidden")]),
    )
    return lens, batch, baseline


def objective(output, batch):
    return output


def test_integrated_gradients_is_trial_independent_complete_and_site_aware():
    lens, batch, baseline = fixture()
    result = attribute(
        lens,
        batch,
        objective,
        baseline=baseline,
        sites=("hidden",),
        steps=8,
    )
    expected = 4 * batch.data.square()
    torch.testing.assert_close(result.input_attribution, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(result.site_attributions["hidden"], expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        result.input_attribution.flatten(1).sum(1),
        result.objective - result.baseline_objective,
        atol=1e-10,
        rtol=1e-10,
    )
    assert max(abs(value) for value in result.metadata["completeness_error"]) < 1e-10
    assert patch_attribution(result).shape == (2, 1, 2)
    assert channel_attribution(result).shape == (2, 1)
    assert temporal_attribution(result).shape == (2, 2)
    assert not lens.model.hidden._forward_hooks


def test_gradient_and_input_x_gradient_have_distinct_declared_semantics():
    lens, batch, _ = fixture()
    gradient = attribute(lens, batch, objective, method="gradient")
    product = attribute(lens, batch, objective, method="input_x_gradient")
    torch.testing.assert_close(gradient.input_attribution, 8 * batch.data)
    torch.testing.assert_close(product.input_attribution, 8 * batch.data.square())
    with pytest.raises(ValidationError, match="requires input×gradient"):
        spectral_attribution(gradient)


def test_prism_spectral_mapping_conserves_ig_and_localizes_carrier():
    lens, batch, baseline = fixture()
    integrated = attribute(lens, batch, objective, baseline=baseline, steps=8)
    spectrum = spectral_attribution(integrated)
    torch.testing.assert_close(
        spectrum.conservation_error,
        torch.zeros_like(spectrum.conservation_error),
        atol=1e-10,
        rtol=0,
    )
    alpha = spectral_band_attribution(spectrum, FrequencyBand(8, 13))
    outside = spectral_band_attribution(spectrum, FrequencyBand(1, 8))
    assert bool((alpha > outside * 1e6).all())
    assert spectrum.mean_over_channels().shape == (2, 33)


def test_site_integrated_attribution_follows_nonlinear_activation_path():
    _, batch, baseline = fixture()
    lens = EEGLens(
        NonlinearSite().double().eval(),
        Adapter([ActivationSite("hidden", "hidden")]),
    )
    result = attribute(
        lens,
        batch,
        objective,
        baseline=baseline,
        sites=("hidden",),
        steps=8,
    )
    torch.testing.assert_close(
        result.site_attributions["hidden"].flatten(1).sum(1),
        result.objective - result.baseline_objective,
        atol=1e-10,
        rtol=1e-10,
    )


def test_site_gradient_uses_declared_semantic_layout_and_restores_native_output():
    _, batch, _ = fixture()
    lens = EEGLens(
        FoldedSpatial().double().eval(),
        Adapter([ActivationSite("spatial", "spatial", layout="spatial")]),
    )
    result = attribute(
        lens,
        batch,
        objective,
        method="input_x_gradient",
        sites=("spatial",),
    )
    assert result.site_attributions["spatial"].shape == batch.data.shape
    torch.testing.assert_close(result.objective, batch.data.square().flatten(1).sum(1))
    torch.testing.assert_close(
        result.site_attributions["spatial"],
        2 * batch.data.square(),
    )


def test_attribution_requires_explicit_valid_objective_and_baseline():
    lens, batch, _ = fixture()
    with pytest.raises(ValidationError, match="explicit SignalBatch baseline"):
        attribute(lens, batch, objective)
    with pytest.raises(ValidationError, match="one finite floating"):
        attribute(lens, batch, lambda output, current: output[:, None], method="gradient")
