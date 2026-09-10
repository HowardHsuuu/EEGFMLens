"""Explicit normalization baseline with held-out task accuracy, not only invariance."""

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eeglens import SignalBatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase"))
from evaluate import model_loader, pooled


def run(model, checkpoint, previous, output, random=False):
    torch.set_num_threads(2)
    warnings.filterwarnings("error", category=ConvergenceWarning)
    label = model + ("-random" if random else "")
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / f"{label}.json").exists()
    manifest = json.loads((previous / "prepared/manifest.json").read_text())
    assert "UV-text-verified" in manifest["recipe"]
    hashes = json.loads((previous / "artifact_hashes.json").read_text())
    assert (
        hashlib.sha256((previous / "prepared/inputs.npz").read_bytes()).hexdigest()
        == hashes["prepared/inputs.npz"]
    )
    rows = manifest["rows"]
    subjects = np.array([r["subject"] for r in rows])
    y = np.array([r["n2"] for r in rows])
    raw = np.load(previous / "prepared/inputs.npz")["x"]
    gains = [0.75, 1.0, 1.5]
    features = np.empty((3, len(raw), 200), dtype=np.float32)
    lens = model_loader(model, checkpoint, random)
    for gi, gain in enumerate(gains):
        x = raw * gain
        x = (x - x.mean((2, 3), keepdims=True)) / x.std((2, 3), keepdims=True).clip(1e-8)
        for subject in range(1, 9):
            indices = np.flatnonzero(subjects == subject)
            for start in range(0, len(indices), 8):
                ids = indices[start : start + 8]
                batch = SignalBatch(
                    torch.from_numpy(x[ids]),
                    tuple(str(i) for i in ids),
                    (rows[ids[0]]["channel"],),
                    200,
                    manifest["recipe"] + ":unit-SD",
                )
                features[gi, ids] = pooled(lens.run_with_cache(batch, sites=[]).output).numpy()
    records = []
    with threadpool_limits(limits=2):
        for held in range(1, 9):
            vals = [held % 8 + 1, (held + 1) % 8 + 1]
            train = ~np.isin(subjects, [held, *vals])
            test = subjects == held
            scaler = StandardScaler().fit(features[1, train])
            candidates = []
            for c in [0.1, 1.0, 10.0]:
                head = LogisticRegression(
                    C=c, class_weight="balanced", max_iter=3000, solver="lbfgs", random_state=4311
                ).fit(scaler.transform(features[1, train]), y[train])
                value = np.mean(
                    [
                        balanced_accuracy_score(
                            y[subjects == v],
                            head.predict(scaler.transform(features[1, subjects == v])),
                        )
                        for v in vals
                    ]
                )
                candidates.append((value, c, head))
            value, c, head = max(candidates, key=lambda r: r[0])
            base = head.decision_function(scaler.transform(features[1, test]))
            for gi, gain in enumerate(gains):
                scores = head.decision_function(scaler.transform(features[gi, test]))
                records.append(
                    dict(
                        subject=held,
                        gain=gain,
                        C=c,
                        validation_score=float(value),
                        balanced_accuracy=float(balanced_accuracy_score(y[test], scores >= 0)),
                        auc=float(roc_auc_score(y[test], scores)),
                        normalized_prediction_flips=int(np.sum((scores >= 0) != (base >= 0))),
                        max_margin_difference=float(max(abs(scores - base))),
                    )
                )
    doc = dict(
        records=records,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(
            Path(__file__).with_name("PROTOCOL.md").read_bytes()
        ).hexdigest(),
        max_feature_difference=float(np.max(abs(features - features[1:2]))),
        model=lens.manifest,
    )
    np.savez_compressed(output / f"{label}.npz", features=features)
    (output / f"{label}.json").write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(label, "normalization complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cbramod", "labram"], required=True)
    p.add_argument("--random", action="store_true")
    for key in ["checkpoint", "previous", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    run(a.model, a.checkpoint, a.previous, a.output, a.random)
