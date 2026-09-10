"""Training-only readout adaptation, with token-preserving controls."""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def run(features_dir, previous, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((previous / "prepared/manifest.json").read_text())
    subjects = np.array([r["subject"] for r in manifest["rows"]])
    y = np.array([r["n2"] for r in manifest["rows"]])
    results, parameters = {}, {}
    warnings.filterwarnings("error", category=ConvergenceWarning)
    with threadpool_limits(limits=2):
        for label in ["cbramod", "labram", "cbramod-random", "labram-random"]:
            bundle = np.load(features_dir / f"{label}.npz")
            gains = bundle["gains"]
            tokens = bundle["features"]
            for pooling in ["mean", "flatten"]:
                x = tokens.mean(2) if pooling == "mean" else tokens.reshape(len(gains), len(y), -1)
                for domain, gi in [("clean", [2]), ("augmented", [0, 4])]:
                    key = f"{label}:{pooling}:{domain}"
                    out = np.empty((len(gains), len(y)))
                    folds = []
                    for held in range(1, 9):
                        validation = [held % 8 + 1, (held + 1) % 8 + 1]
                        train = ~np.isin(subjects, [held, *validation])
                        test = subjects == held
                        assert not (train & test).any()
                        xx = np.concatenate([x[g, train] for g in gi])
                        yy = np.tile(y[train], len(gi))
                        sample_weight = np.full(len(yy), 1 / len(gi))
                        scaler = StandardScaler().fit(xx)
                        scaled = scaler.transform(xx)
                        candidates = []
                        for c in [0.1, 1.0, 10.0]:
                            head = LogisticRegression(
                                C=c,
                                class_weight="balanced",
                                solver="lbfgs",
                                max_iter=3000,
                                random_state=4311,
                            )
                            head.fit(scaled, yy, sample_weight=sample_weight)
                            scores = []
                            for v in validation:
                                mask = subjects == v
                                for g in gi:
                                    scores.append(
                                        balanced_accuracy_score(
                                            y[mask], head.predict(scaler.transform(x[g, mask]))
                                        )
                                    )
                            candidates.append((float(np.mean(scores)), c, head))
                        validation_score, c, head = max(candidates, key=lambda z: z[0])
                        weight = head.coef_[0] / scaler.scale_
                        intercept = float(head.intercept_[0] - scaler.mean_ @ weight)
                        parameters[key + f":s{held}:weight"] = weight
                        parameters[key + f":s{held}:intercept"] = np.array(intercept)
                        for g, gain in enumerate(gains):
                            score = x[g, test] @ weight + intercept
                            out[g, test] = score
                            folds.append(
                                dict(
                                    subject=held,
                                    gain=float(gain),
                                    n=int(test.sum()),
                                    C=c,
                                    validation_score=validation_score,
                                    balanced_accuracy=float(
                                        balanced_accuracy_score(y[test], score >= 0)
                                    ),
                                    auc=float(roc_auc_score(y[test], score)),
                                )
                            )
                    results[key] = folds
                    parameters[key + ":oof"] = out
                    print(key, "complete", flush=True)
            del tokens, bundle
    np.savez_compressed(output / "readouts.npz", **parameters)
    doc = dict(
        results=results,
        protocol_sha256=hashlib.sha256(
            Path(__file__).with_name("PROTOCOL.md").read_bytes()
        ).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        feature_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in features_dir.glob("*.npz")
        },
    )
    (output / "results.json").write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["features", "previous", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.features, a.previous, a.output)
