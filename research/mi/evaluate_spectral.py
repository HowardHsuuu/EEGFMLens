"""Apply the previously fitted spectral baseline to the reserved test subjects."""

import argparse
import json
from pathlib import Path

import numpy as np
from extract import sha
from spectral_baseline import balanced_accuracy, spectral_features


def run(root, output):
    folder = root / "spectral-v1"
    meta = json.loads((folder / "summary.json").read_text())
    for partition in ("train", "validation"):
        if meta[f"{partition}_sha256"] != sha(root / f"prepared-v1/{partition}/trials.npz"):
            raise ValueError("Baseline fitting data hash mismatch")
    test = root / "prepared-v1/test"
    tm = json.loads((test / "manifest.json").read_text())
    if tm["partition"] != "test" or tm["artifact_sha256"] != sha(test / "trials.npz"):
        raise ValueError("Test artifact mismatch")
    x = np.load(test / "trials.npz", allow_pickle=False)
    for partition in ("train", "validation"):
        other = np.load(root / f"prepared-v1/{partition}/trials.npz", allow_pickle=False)
        if set(x["subjects"]) & set(other["subjects"]):
            raise ValueError("Test subjects overlap fitting or selection subjects")
    fitted = np.load(folder / "baseline.npz", allow_pickle=False)
    margins = (
        spectral_features(x["volts"], list(x["channels"])) @ fitted["coef"] + fitted["intercept"]
    ).ravel()
    subjects = [
        dict(
            subject=int(s),
            balanced_accuracy=balanced_accuracy(
                x["labels"][x["subjects"] == s], margins[x["subjects"] == s]
            ),
        )
        for s in np.unique(x["subjects"])
    ]
    report = dict(
        partition="test",
        subjects=subjects,
        mean_subject_balanced_accuracy=float(np.mean([s["balanced_accuracy"] for s in subjects])),
        baseline_sha256=sha(folder / "baseline.npz"),
        baseline_summary_sha256=sha(folder / "summary.json"),
        test_manifest_sha256=sha(test / "manifest.json"),
        runner_sha256=sha(__file__),
        scope="Existing train-fitted, validation-selected coefficients; no test fitting. Baseline artifact was not hash-pinned in the intervention protocol.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Spectral test subject-mean BA", report["mean_subject_balanced_accuracy"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.output)
