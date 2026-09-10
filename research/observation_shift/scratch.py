"""Small supervised EEG comparator; explicitly not an architecture-matched ablation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 8, 25, stride=4, padding=12),
            nn.GELU(),
            nn.Conv1d(8, 16, 15, stride=4, padding=7),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x).flatten()


def transform(x, mode, gain=1.0):
    x = x * gain
    if mode == "normalized":
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, correction=0, keepdim=True).clamp_min(1e-8)
    return x


def run(previous, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rows = json.loads((previous / "prepared/manifest.json").read_text())["rows"]
    subjects = np.array([r["subject"] for r in rows])
    x = torch.from_numpy(np.load(previous / "prepared/inputs.npz")["x"].reshape(len(rows), 1, -1))
    y = torch.tensor([r["n2"] for r in rows], dtype=torch.float32)
    results = []
    for held in range(1, 9):
        vals = [held % 8 + 1, (held + 1) % 8 + 1]
        ids = torch.from_numpy(np.flatnonzero(~np.isin(subjects, [held, *vals])))
        test = np.flatnonzero(subjects == held)
        positive = y[ids].sum()
        criterion = nn.BCEWithLogitsLoss(pos_weight=(len(ids) - positive) / positive)
        for seed in [9137, 9138, 9139]:
            for mode in ["clean", "augmented", "normalized"]:
                torch.manual_seed(seed + held * 100)
                model = SmallCNN()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                best = (-float("inf"), -float("inf"))
                stale = 0
                best_state = None
                for epoch in range(40):
                    model.train()
                    order = ids[torch.randperm(len(ids))]
                    for begin in range(0, len(order), 32):
                        batch = order[begin : begin + 32]
                        gain = (
                            torch.where(torch.rand(len(batch), 1, 1) < 0.5, 0.5, 2.0)
                            if mode == "augmented"
                            else 1.0
                        )
                        optimizer.zero_grad()
                        loss = criterion(model(transform(x[batch], mode, gain)), y[batch])
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    bas, losses = [], []
                    with torch.no_grad():
                        for v in vals:
                            mask = np.flatnonzero(subjects == v)
                            for gain in [0.5, 2.0] if mode == "augmented" else [1.0]:
                                scores = model(transform(x[mask], mode, gain))
                                bas.append(
                                    balanced_accuracy_score(y[mask].numpy(), scores.numpy() >= 0)
                                )
                                losses.append(float(criterion(scores, y[mask])))
                    value = (float(np.mean(bas)), -float(np.mean(losses)))
                    if value > best:
                        best = value
                        stale = 0
                        best_epoch = epoch + 1
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    else:
                        stale += 1
                    if stale >= 6:
                        break
                model.load_state_dict(best_state)
                model.eval()
                torch.save(best_state, output / f"s{held}-{seed}-{mode}.pt")
                with torch.no_grad():
                    for gain in [0.5, 0.75, 1.0, 1.5, 2.0]:
                        scores = model(transform(x[test], mode, gain)).numpy()
                        results.append(
                            dict(
                                subject=held,
                                seed=seed,
                                mode=mode,
                                gain=gain,
                                best_epoch=best_epoch,
                                epochs_run=epoch + 1,
                                validation_ba=best[0],
                                balanced_accuracy=float(
                                    balanced_accuracy_score(y[test].numpy(), scores >= 0)
                                ),
                                auc=float(roc_auc_score(y[test].numpy(), scores)),
                                test_rows=test.tolist(),
                                margins=scores.tolist(),
                            )
                        )
        print("scratch held subject", held, "complete", flush=True)
    doc = dict(
        results=results,
        parameters=sum(p.numel() for p in model.parameters()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(
            Path(__file__).with_name("PROTOCOL.md").read_bytes()
        ).hexdigest(),
    )
    (output / "results.json").write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.previous, a.output)
