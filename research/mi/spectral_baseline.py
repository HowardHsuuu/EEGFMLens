"""Fixed-band, electrode-preserving task baseline on the shared raw trials."""

import argparse
import json
from pathlib import Path

import numpy as np
from extract import CHANNELS, sha
from probes import fit_ridge
from scipy.signal import welch

BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def spectral_features(volts, channels):
    selected = volts[:, [channels.index(c) for c in CHANNELS]]
    frequencies, psd = welch(selected, fs=160, nperseg=320, noverlap=160, axis=-1)
    features = [
        np.log(
            np.maximum(
                psd[..., (frequencies >= lo) & (frequencies < hi)].mean(-1), np.finfo(float).tiny
            )
        )
        for lo, hi in BANDS
    ]
    return np.stack(features, axis=-1).reshape(len(volts), -1)


def balanced_accuracy(labels, margins):
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both classes required for balanced accuracy")
    predicted = margins >= 0
    return float(np.mean([np.mean(predicted[labels == label] == label) for label in (0, 1)]))


def run(train, validation, output):
    data = []
    for folder, partition in ((train, "train"), (validation, "validation")):
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest.get("partition") != partition or manifest["artifact_sha256"] != sha(
            folder / "trials.npz"
        ):
            raise ValueError("Partition or artifact hash mismatch")
        data.append(np.load(folder / "trials.npz", allow_pickle=False))
    t, v = data
    if set(t["subjects"]) & set(v["subjects"]):
        raise ValueError("Subject leakage")
    tx = spectral_features(t["volts"], list(t["channels"]))
    vx = spectral_features(v["volts"], list(v["channels"]))
    fitted = fit_ridge(tx, (t["labels"] * 2 - 1)[:, None], vx, (v["labels"] * 2 - 1)[:, None])
    margins = (vx @ fitted["coef"] + fitted["intercept"]).ravel()
    subject_results = [
        dict(
            subject=int(s),
            balanced_accuracy=balanced_accuracy(
                v["labels"][v["subjects"] == s], margins[v["subjects"] == s]
            ),
        )
        for s in np.unique(v["subjects"])
    ]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "baseline.npz",
        **fitted,
        validation_margins=margins,
        trial_ids=v["trial_ids"],
        subjects=v["subjects"],
        labels=v["labels"],
    )
    result = dict(
        runner_sha256=sha(__file__),
        train_sha256=sha(train / "trials.npz"),
        validation_sha256=sha(validation / "trials.npz"),
        channels=CHANNELS,
        bands=BANDS,
        reference="as recorded; same 19 electrode locations as models",
        scope="Validation-selected baseline; not held-out test evidence",
        alpha=fitted["alpha"].tolist(),
        subjects=subject_results,
        mean_subject_balanced_accuracy=float(
            np.mean([r["balanced_accuracy"] for r in subject_results])
        ),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Spectral validation mean subject BA:", result["mean_subject_balanced_accuracy"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.train, a.validation, a.output)
