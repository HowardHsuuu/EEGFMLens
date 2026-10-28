"""Offline contract demonstration: reject wrong coordinates and escaping edits."""

from dataclasses import dataclass

import torch
from torch import nn

from eegfmlens import ActivationSite, Adapter, EEGLens, Selection, SignalBatch
from eegfmlens.errors import ValidationError


@dataclass
class AddOne:
    selection: Selection
    escape: bool = False
    site: str = "hidden"

    def apply(self, current, batch, layout, model_id):
        if self.escape:
            return current + 1
        return current + self.selection.mask(current, batch, layout)


def main():
    model = nn.Sequential(nn.Identity()).eval()
    lens = EEGLens(model, Adapter([ActivationSite("hidden", "0")]))
    batch = SignalBatch(torch.zeros(1, 2, 2, 4), ("example",), ("C3", "C4"), 200, "synthetic")
    for edit, reason in [
        (AddOne(Selection(sensors=("C9",), patches=(0,))), "unknown sensor"),
        (AddOne(Selection(sensors=("C3",), patches=(0,)), escape=True), "outside selection"),
    ]:
        try:
            lens.run_with_interventions(batch, interventions=[edit])
        except ValidationError:
            print("Rejected:", reason)
        else:
            raise AssertionError("Invalid edit was accepted")
        assert not model[0]._forward_hooks and torch.count_nonzero(batch.data) == 0
    result = lens.run_with_interventions(
        batch, interventions=[AddOne(Selection(sensors=("C3",), patches=(0,)))]
    )
    expected = torch.zeros_like(batch.data)
    expected[:, 0, 0] = 1
    assert torch.equal(result.output, expected)
    assert result.metadata["interventions"][0]["selection_support_verified"]
    print("Accepted: C3 patch 0 changed; all other coordinates stayed identical.")


if __name__ == "__main__":
    main()
