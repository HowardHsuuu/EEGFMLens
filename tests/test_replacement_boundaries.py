"""Replacement must preserve trial identity and reject incompatible coordinates."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from eeglens import ActivationSite, Adapter, EEGLens, Replacement, Selection, SignalBatch
from eeglens.errors import ValidationError


@pytest.mark.parametrize(
    "changed",
    [
        {"channels": ("C4", "C3")},
        {"sampling_rate": 100},
        {"stride": 2},
        {"patch_samples": 8},
        {"unit": "volts"},
        {"layout": "tokens"},
        {"site": "another-site"},
        {"trial_ids": ("donor-a", "donor-a")},
        {"trial_ids": ("donor-a",)},
        {"tensor": torch.zeros(2, 2, 2, 3)},
        {"tensor": torch.full((2, 2, 2, 4), float("nan"))},
        {"tensor": torch.full((2, 2, 2, 4), float("inf"))},
    ],
)
def test_invalid_donor_recovers_to_trial_aligned_local_replacement(changed):
    model = nn.Sequential(nn.Identity()).eval()
    lens = EEGLens(model, Adapter([ActivationSite("hidden", "0")]))
    data = torch.arange(32, dtype=torch.float32).reshape(2, 2, 2, 4) + 1
    batch = SignalBatch(data.clone(), ("donor-a", "donor-b"), ("C3", "C4"), 200, "fixture")
    donor = lens.run_with_cache(batch).cache["hidden"]
    donor_before = donor.tensor.clone()
    # Recipient rows are reversed and values are distinct from the donor.
    recipient = replace(batch, data=-data.flip(0), trial_ids=("donor-b", "donor-a"))
    recipient_before = recipient.data.clone()
    calls = []
    handle = model[0].register_forward_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(ValidationError):
            lens.run_with_interventions(
                recipient,
                interventions=[Replacement("hidden", replace(donor, **changed))],
            )
        assert len(model[0]._forward_hooks) == 1
        result = lens.run_with_interventions(
            recipient,
            interventions=[Replacement("hidden", donor, Selection(sensors=("C3",), patches=(1,)))],
            sites=["hidden"],
        )
        expected = recipient_before.clone()
        expected[0, 0, 1] = data[1, 0, 1]
        expected[1, 0, 1] = data[0, 0, 1]
        torch.testing.assert_close(result.output, expected, atol=0, rtol=0)
        torch.testing.assert_close(result.cache["hidden"].tensor, expected, atol=0, rtol=0)
        torch.testing.assert_close(recipient.data, recipient_before, atol=0, rtol=0)
        torch.testing.assert_close(donor.tensor, donor_before, atol=0, rtol=0)
        clean = lens.run_with_cache(recipient)
        torch.testing.assert_close(clean.output, recipient_before, atol=0, rtol=0)
        assert len(calls) == 3 and len(model[0]._forward_hooks) == 1
    finally:
        handle.remove()
