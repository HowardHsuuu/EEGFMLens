"""Evaluate frozen probes without refitting; retain subject identities and provenance."""

import argparse
import json
from pathlib import Path

import numpy as np
from extract import sha
from probes import load_features
from spectral_baseline import balanced_accuracy


def run(features, training_features, probes, partition, output):
    fm = json.loads(features.with_suffix(".json").read_text())
    tm = json.loads(training_features.with_suffix(".json").read_text())
    pm = json.loads((probes / "manifest.json").read_text())
    if pm["artifact_sha256"] != sha(probes / "probes.npz"):
        raise ValueError("Probe artifact hash mismatch")
    if pm["train_sha256"] != sha(training_features):
        raise ValueError("Probe training provenance mismatch")
    if partition == "validation" and pm["validation_sha256"] != sha(features):
        raise ValueError("Probe validation provenance mismatch")
    for key in ("model", "checkpoint_sha256", "input_recipe", "sites", "pooling"):
        if fm[key] != tm[key]:
            raise ValueError(f"Feature contract mismatch: {key}")
    for key, default in (("initialization", "pretrained"), ("random_seed", None)):
        if fm.get(key, default) != tm.get(key, default):
            raise ValueError(f"Initialization mismatch: {key}")
    for key in ("model", "checkpoint_sha256"):
        if pm[key] != tm[key]:
            raise ValueError(f"Probe identity mismatch: {key}")
    x = load_features(features, partition)
    train = load_features(training_features, "train")
    if set(x["subjects"]) & set(train["subjects"]):
        raise ValueError("Evaluation subjects overlap training")
    if not np.array_equal(x["channels"], train["channels"]):
        raise ValueError("Electrode order mismatch")
    if len(set(x["trial_ids"])) != len(x["trial_ids"]):
        raise ValueError("Duplicate evaluation trial IDs")
    p = np.load(probes / "probes.npz", allow_pickle=False)
    prediction = (
        x["final"].reshape(len(x["labels"]), -1) @ p["readout_coef"] + p["readout_intercept"]
    )
    channels = list(x["channels"])
    contrast = x["middle"][:, channels.index("C4")] - x["middle"][:, channels.index("C3")]
    concept = contrast @ p["concept_coef"] + p["concept_intercept"]
    target = x["descriptors"][:, :2]
    variance = ((target - target.mean(0)) ** 2).sum(0)
    if (
        (variance <= 1e-12).any()
        or not np.isfinite(prediction).all()
        or not np.isfinite(concept).all()
    ):
        raise ValueError("Degenerate targets or non-finite predictions")
    subjects = [
        dict(
            subject=int(s),
            balanced_accuracy=balanced_accuracy(
                x["labels"][x["subjects"] == s], prediction[x["subjects"] == s, 0]
            ),
        )
        for s in np.unique(x["subjects"])
    ]
    report = dict(
        model=fm["model"],
        partition=partition,
        initialization=fm.get("initialization", "pretrained"),
        seed=fm.get("random_seed"),
        mean_subject_balanced_accuracy=float(np.mean([s["balanced_accuracy"] for s in subjects])),
        subjects=subjects,
        pooled_middle_concept_r2=(1 - ((concept - target) ** 2).sum(0) / variance).tolist(),
        feature_manifest_sha256=sha(features.with_suffix(".json")),
        probe_manifest_sha256=sha(probes / "manifest.json"),
        runner_sha256=sha(__file__),
        scope="Frozen validation-selected probes; pooled R² differs from subject-mean task accuracy. Random controls use one initialization seed.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        fm["model"], report["initialization"], partition, report["mean_subject_balanced_accuracy"]
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--training-features", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--partition", choices=("validation", "test"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.features, a.training_features, a.probes, a.partition, a.output)
