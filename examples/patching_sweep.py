"""Offline calibration with two causal coordinates and irrelevant controls."""

import argparse
from dataclasses import replace

import torch
from torch import nn

from eeglens import ActivationSite, Adapter, EEGLens, SignalBatch, patching_sweep


class KnownCircuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Identity()
        self.late = nn.Identity()

    def forward(self, x):
        x = self.early(x)
        x = self.late(2 * x)
        return 3 * x[:, 0, 1, 0] - x[:, 1, 2, 0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", help="Optional new JSON file; no overwrite")
    args = p.parse_args()
    lens = EEGLens(
        KnownCircuit().eval(),
        Adapter([ActivationSite("early", "early"), ActivationSite("late", "late")]),
    )
    clean = SignalBatch(
        torch.arange(1, 13, dtype=torch.float32).reshape(2, 2, 3, 1),
        ("trial-a", "trial-b"),
        ("C3", "C4"),
        200,
        "analytic-example",
    )
    recipient = replace(clean, data=torch.zeros_like(clean.data))
    result = patching_sweep(
        lens, clean, recipient, lambda output, batch: output, random_controls=False
    )
    checked = 0
    for row in result.rows:
        if row["kind"] != "event":
            continue
        index = clean.trial_ids.index(row["trial_id"])
        expected = {
            "C3:patch1": 6 * float(clean.data[index, 0, 1, 0]),
            "C4:patch2": -2 * float(clean.data[index, 1, 2, 0]),
        }.get(row["target"], 0)
        assert row["delta"] == expected
        checked += 1
    assert checked == 24, f"Expected all 24 effects, received {checked}"
    if args.output:
        result.save(args.output)
    print("All 24 layer/channel/time/trial effects match the analytic answer.")
    print("Only C3 patch 1 and C4 patch 2 affect the score, at both layers.")


if __name__ == "__main__":
    main()
