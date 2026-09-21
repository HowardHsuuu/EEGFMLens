from dataclasses import replace

import pytest
import torch
from torch import nn

from eegfmlens import (
    CANONICAL_BANDS,
    Adapter,
    BandTarget,
    EEGLens,
    InputTarget,
    Selection,
    SignalBatch,
    attribution_cosine_consistency,
    occlusion_curve,
    spectral_perturbation_curve,
)
from eegfmlens.errors import ValidationError


def sum_lens():
    return EEGLens(nn.Sequential(nn.Identity()).eval(), Adapter([]))


def score(output, batch):
    return output.flatten(1).sum(1)


def test_progressive_occlusion_is_trial_aligned_and_has_exact_aopc():
    batch = SignalBatch(
        torch.tensor([[[[1.0], [3.0]]], [[[2.0], [4.0]]]]),
        ("a", "b"),
        ("C3",),
        100,
        "fixture",
    )
    baseline = replace(
        batch,
        data=torch.zeros_like(batch.data).flip(0),
        trial_ids=("b", "a"),
    )
    curve = occlusion_curve(
        sum_lens(),
        batch,
        baseline,
        score,
        (
            InputTarget("late", Selection(patches=(1,))),
            InputTarget("early", Selection(patches=(0,))),
        ),
    )
    torch.testing.assert_close(curve.scores, torch.tensor([[4.0, 1.0, 0.0], [6.0, 2.0, 0.0]]))
    torch.testing.assert_close(curve.aopc, torch.tensor([3.5, 5.0]))
    assert curve.target_names == ("late", "early")
    with pytest.raises(ValidationError, match="disjoint"):
        occlusion_curve(
            sum_lens(),
            batch,
            baseline,
            score,
            (
                InputTarget("one", Selection(patches=(0,))),
                InputTarget("same", Selection(patches=(0,))),
            ),
        )


class SpectralScore(nn.Module):
    def forward(self, data):
        spectrum = torch.fft.rfft(data.flatten(2), dim=-1).abs().mean(1)
        return spectrum[:, 10] + 0.25 * spectrum[:, 20]


def test_spectral_perturbation_reports_ordered_frequency_faithfulness():
    sampling_rate = 64
    time = torch.arange(64) / sampling_rate
    signal = torch.sin(2 * torch.pi * 10 * time) + torch.sin(2 * torch.pi * 20 * time)
    batch = SignalBatch(signal.reshape(1, 1, 2, 32), ("a",), ("C3",), 64, "fixture")
    lens = EEGLens(SpectralScore().eval(), Adapter([]))
    curve = spectral_perturbation_curve(
        lens,
        batch,
        lambda output, current: output,
        (
            BandTarget("alpha", CANONICAL_BANDS["alpha"]),
            BandTarget("beta", CANONICAL_BANDS["beta"]),
        ),
    )
    assert curve.scores[0, 0] > curve.scores[0, 1] > curve.scores[0, 2] - 1e-5
    assert float(curve.aopc[0]) > 0


def test_cross_method_consistency_is_symmetric_and_trial_aware():
    first = torch.tensor([[[1.0, 2.0]], [[2.0, 1.0]]])
    second = 2 * first
    third = first.flip(-1)
    consistency = attribution_cosine_consistency((first, second, third))
    torch.testing.assert_close(consistency, consistency.T)
    torch.testing.assert_close(consistency.diag(), torch.ones(3))
    assert consistency[0, 1] == pytest.approx(1)
    assert consistency[0, 2] < 1
