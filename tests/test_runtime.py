from dataclasses import replace

import pytest
import torch
from torch import nn

from eegfmlens import (
    Ablation,
    ActivationSite,
    Adapter,
    EEGLens,
    Replacement,
    Selection,
    SignalBatch,
    paired_effect,
)
from eegfmlens.errors import HookExecutionError, UnsupportedSiteError, ValidationError


class KnownModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.signal = nn.Identity()
        self.unused = nn.Identity()

    def forward(self, x, fail=False):
        x = self.signal(x * 2)
        if fail:
            raise RuntimeError("intentional")
        return x.sum((1, 2, 3))


@pytest.fixture
def batch():
    return SignalBatch(
        torch.arange(24, dtype=torch.float32).reshape(2, 2, 3, 2),
        ("trial-a", "trial-b"),
        ("C3", "C4"),
        200,
        "fixture-v1",
    )


@pytest.fixture
def lens():
    return EEGLens(
        KnownModel().eval(),
        Adapter([ActivationSite("signal", "signal"), ActivationSite("unused", "unused")]),
    )


def test_observation_identity_and_trial_reordering(lens, batch):
    clean = lens.run_with_cache(batch, sites=["signal"])
    torch.testing.assert_close(clean.output, lens.model(batch.data), rtol=0, atol=0)
    recipient = replace(
        batch, data=torch.zeros_like(batch.data).flip(0), trial_ids=batch.trial_ids[::-1]
    )
    out = lens.run_with_interventions(
        recipient, interventions=[Replacement("signal", clean.cache["signal"])]
    )
    torch.testing.assert_close(out.output, clean.output.flip(0), rtol=0, atol=0)
    assert not lens.model.signal._forward_hooks


def test_semantic_selection_has_exact_known_effect(lens, batch):
    site = Ablation("signal", Selection(sensors=("C4",), patches=(1,)))
    result = lens.run_with_interventions(batch, interventions=[site])
    expected = lens.model(batch.data) - 2 * batch.data[:, 1, 1].sum(-1)
    torch.testing.assert_close(result.output, expected, rtol=0, atol=0)


def test_cleanup_preserves_existing_hooks_after_exception(lens, batch):
    calls = []
    handle = lens.model.signal.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(RuntimeError, match="intentional"):
        lens.run_with_cache(batch, sites=["signal"], fail=True)
    assert len(lens.model.signal._forward_hooks) == 1
    lens.run_with_cache(batch, sites=["signal"])
    assert calls == [1, 1]
    handle.remove()


def test_unused_and_unknown_sites_fail(lens, batch):
    with pytest.raises(HookExecutionError, match="did not execute"):
        lens.run_with_cache(batch, sites=["unused"])
    with pytest.raises(UnsupportedSiteError):
        lens.run_with_cache(batch, sites=["does-not-exist"])
    assert not lens.model.unused._forward_hooks


def test_donor_mismatch_fails_and_cleans_up(lens, batch):
    donor = lens.run_with_cache(batch, sites=["signal"]).cache["signal"]
    for incompatible in [
        replace(donor, model_id="other-model"),
        replace(donor, preprocessing_id="other-preprocessing"),
        replace(donor, trial_ids=("wrong", "ids")),
        replace(donor, tensor=donor.tensor.double()),
    ]:
        with pytest.raises(ValidationError):
            lens.run_with_interventions(batch, interventions=[Replacement("signal", incompatible)])
        assert not lens.model.signal._forward_hooks


def test_repeated_site_is_rejected(batch):
    class Repeated(KnownModel):
        def forward(self, x):
            return self.signal(self.signal(x))

    model = Repeated().eval()
    lens = EEGLens(model, Adapter([ActivationSite("signal", "signal")]))
    with pytest.raises(HookExecutionError, match="more than once"):
        lens.run_with_cache(batch)
    assert not model.signal._forward_hooks


def test_cache_is_not_changed_by_later_inplace_operation(batch):
    class Inplace(KnownModel):
        def forward(self, x):
            x = self.signal(x.clone())
            return x.add_(10)

    lens = EEGLens(Inplace().eval(), Adapter([ActivationSite("signal", "signal")]))
    run = lens.run_with_cache(batch)
    torch.testing.assert_close(run.cache["signal"].tensor, batch.data)
    assert run.cache["signal"].tensor.data_ptr() != run.output.data_ptr()


def test_eval_required(lens, batch):
    lens.model.train()
    with pytest.raises(ValidationError, match="eval"):
        lens.run_with_cache(batch, sites=["signal"])


def test_duplicate_trials_and_bad_coordinates(batch):
    with pytest.raises(ValidationError):
        replace(batch, trial_ids=("same", "same"))
    with pytest.raises(ValidationError):
        Selection(sensors=("unknown",)).mask(batch.data, batch, "bcpd")


def test_recovery_is_neither_fabricated_nor_clipped():
    assert paired_effect(1, 1, 1).recovery is None
    assert paired_effect(1, 2, 1).recovery is None
    assert paired_effect(5, 1, 7).recovery == 1.5
    assert paired_effect(5, 1, 0).recovery == -0.25
    with pytest.raises(ValueError):
        paired_effect(float("nan"), 1, 2)
