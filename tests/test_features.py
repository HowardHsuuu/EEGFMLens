from dataclasses import replace

import pytest
import torch

from eegfmlens import (
    CANONICAL_BANDS,
    SignalBatch,
    band_connectivity,
    channel_correlation,
    spectral_features,
    time_domain_features,
)
from eegfmlens.errors import ValidationError


def fixture():
    sampling_rate = 128
    time = torch.arange(128, dtype=torch.float64) / sampling_rate
    first = torch.sin(2 * torch.pi * 10 * time)
    second = torch.sin(2 * torch.pi * 10 * time + torch.pi / 2)
    return SignalBatch(
        torch.stack((first, second)).reshape(1, 2, 2, 64),
        ("trial-a",),
        ("C3", "C4"),
        sampling_rate,
        "feature-fixture",
    )


def test_time_features_retain_coordinates_and_export_probe_matrix():
    features = time_domain_features(fixture())
    assert features.values.shape == (1, 2, 9)
    assert features.matrix().shape == (1, 18)
    assert features.matrix(aggregation="mean").shape == (1, 9)
    assert features.column_names()[0] == "C3:hjorth.activity"
    torch.testing.assert_close(
        features.values[..., features.feature_names.index("hjorth.activity")],
        torch.full((1, 2), 0.5, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    patch = time_domain_features(fixture(), scope="patch")
    assert patch.values.shape == (1, 2, 2, 9)
    assert len(patch.column_names()) == 36


def test_spectral_features_localize_known_alpha_carrier():
    features = spectral_features(fixture())
    alpha_relative = features.values[..., features.feature_names.index("alpha.relative_power")]
    entropy = features.values[..., features.feature_names.index("spectral_entropy")]
    centroid = features.values[..., features.feature_names.index("spectral_centroid_hz")]
    edge = features.values[..., features.feature_names.index("spectral_edge_0.95_hz")]
    assert bool((alpha_relative > 0.999999).all())
    assert bool((entropy < 1e-10).all())
    torch.testing.assert_close(centroid, torch.full_like(centroid, 10), atol=1e-10, rtol=0)
    torch.testing.assert_close(edge, torch.full_like(edge, 10), atol=0, rtol=0)
    with pytest.raises(ValidationError, match="zero broadband power"):
        spectral_features(replace(fixture(), data=torch.zeros_like(fixture().data)))


def test_channel_and_band_connectivity_have_known_phase_behavior():
    batch = fixture()
    correlation = channel_correlation(batch)
    assert correlation.values.shape == (1, 2, 2)
    assert abs(float(correlation.values[0, 0, 1])) < 1e-12
    pli = band_connectivity(batch, CANONICAL_BANDS["alpha"])
    plv = band_connectivity(
        batch,
        CANONICAL_BANDS["alpha"],
        measure="phase_locking_value",
    )
    coherence = band_connectivity(
        batch,
        CANONICAL_BANDS["alpha"],
        measure="magnitude_squared_coherence",
        scope="patch",
    )
    assert pli.values[0, 0, 1] == pytest.approx(1, abs=1e-12)
    assert plv.values[0, 0, 1] == pytest.approx(1, abs=1e-12)
    assert coherence.values.shape == (1, 2, 2, 2)
    assert coherence.values[0, 0, 0, 1] == pytest.approx(1, abs=1e-12)
