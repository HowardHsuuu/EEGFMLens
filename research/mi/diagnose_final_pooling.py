"""Exploratory clean-readout pooling diagnostic; does not alter native study readouts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from probes import fit_ridge, load_features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r2(y, p):
    return (1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).tolist()


def run(name, root, output):
    if output.exists():
        raise FileExistsError(output)
    features = root / "eeglens_mi/features-v1/train" / f"{name}.npz"
    data = load_features(features, "train")
    source = root / "eeglens_mi/iterative-recovery-v1" / name / "summary.json"
    history = json.loads(source.read_text())
    if sha(features) != history["feature_sha256"]:
        raise ValueError("Feature mismatch")
    c3, c4 = [list(data["channels"]).index(c) for c in ("C3", "C4")]
    arrays = {
        "final_c4_minus_c3": data["final"][:, c4] - data["final"][:, c3],
        "final_c3_c4": data["final"][:, [c3, c4]].reshape(len(data["subjects"]), -1),
    }
    y = data["descriptors"][:, :2]
    outputs = {}
    roles = []
    for name_x, x in arrays.items():
        prediction = np.full_like(y, np.nan, dtype=np.float64)
        for fold in history["folds"]:
            fit, val, test = [
                np.isin(data["subjects"], fold[k])
                for k in ("fitting_subjects", "validation_subjects", "evaluation_subjects")
            ]
            fitted = fit_ridge(x[fit], y[fit], x[val], y[val])
            prediction[test] = x[test] @ fitted["coef"] + fitted["intercept"]
            roles.append(
                dict(representation=name_x, fold=fold["fold"], alpha=fitted["alpha"].tolist())
            )
        assert np.isfinite(prediction).all()
        subjects = [
            dict(subject=int(s), r2=r2(y[data["subjects"] == s], prediction[data["subjects"] == s]))
            for s in np.unique(data["subjects"])
        ]
        scores = np.array([s["r2"] for s in subjects])
        outputs[name_x] = dict(
            pooled_r2=r2(y, prediction),
            positive_subject_counts=(scores > 0).sum(0).tolist(),
            subject_median_r2=np.median(scores, 0).tolist(),
            subjects=subjects,
        )
    report = dict(
        model=name,
        scope="Post-result exploratory diagnostic, final-layer 200D contrast and 400D sensor concatenation; original 9/3/6 roles; no replacement of native-study readouts",
        feature_sha256=sha(features),
        fold_source_sha256=sha(source),
        runner_sha256=sha(Path(__file__)),
        fitter_sha256=sha(Path(__file__).with_name("probes.py")),
        results=outputs,
        selected_alphas=roles,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(name, {k: (v["pooled_r2"], v["positive_subject_counts"]) for k, v in outputs.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cbramod", "labram", "csbrain"], required=True)
    p.add_argument("--root", type=Path, default=Path("research"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.model, a.root, a.output)
