"""Check clean final-readout quality before interpreting physiological loss changes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r2(y, p):
    denominator = ((y - y.mean(0)) ** 2).sum(0)
    if np.any(denominator <= 1e-12):
        raise ValueError("Degenerate descriptor variance")
    return (1 - ((y - p) ** 2).sum(0) / denominator).tolist()


def run(root, output):
    if output.exists():
        raise FileExistsError(output)
    targets, predictions, subjects, inputs = [], [], [], []
    model = None
    for fold in range(3):
        path = root / f"fold{fold}/responses.npz"
        meta = json.loads(path.with_name("manifest.json").read_text())
        if sha(path) != meta["artifact_sha256"] or meta["fold"] != fold:
            raise ValueError("Response provenance mismatch")
        if model is not None and model != meta["model"]:
            raise ValueError("Mixed models")
        model = meta["model"]
        with np.load(path, allow_pickle=False) as x:
            if x["conditions"][0] != "clean":
                raise ValueError("First condition must be clean")
            current = set(x["subjects"])
            if current != set(meta["roles"]["evaluation_subjects"]) or current & set(
                np.concatenate(subjects) if subjects else []
            ):
                raise ValueError("Invalid evaluation subject roles")
            targets.append(x["descriptors"])
            predictions.append(x["predictions"][:, 0, 1:])
            subjects.append(x["subjects"])
        inputs.append(dict(file=str(path), sha256=sha(path)))
    y, p, s = map(np.concatenate, (targets, predictions, subjects))
    if len(y) != 810 or len(np.unique(s)) != 18 or y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Incomplete or invalid clean predictions")
    rows = [
        dict(subject=int(subject), r2=r2(y[s == subject], p[s == subject]))
        for subject in np.unique(s)
    ]
    values = np.array([r["r2"] for r in rows])
    report = dict(
        model=model,
        scope="Clean frozen final readouts, exploratory original-training cross-fitting; not middle-layer concept scores",
        targets=["mu", "beta", "occipital_alpha", "global_rms"],
        pooled_r2=r2(y, p),
        subject_median_r2=np.median(values, 0).tolist(),
        positive_subject_counts=(values > 0).sum(0).tolist(),
        subject_r2=rows,
        inputs=inputs,
        runner_sha256=sha(Path(__file__)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(model, report["pooled_r2"], report["positive_subject_counts"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.results, a.output)
