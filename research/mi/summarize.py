"""Subject-level intervention summaries; uncertainty resamples subjects, not trials."""

import argparse
import json
from pathlib import Path

import numpy as np
from extract import sha
from spectral_baseline import balanced_accuracy


def bootstrap_mean(values, seed=193, repeats=10000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite subject effects")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(len(values), size=(repeats, len(values)))].mean(1)
    return dict(
        mean=float(values.mean()),
        low=float(np.quantile(sampled, 0.025)),
        high=float(np.quantile(sampled, 0.975)),
        subjects=len(values),
    )


def run(responses, probes, training_features, output):
    meta = json.loads((responses / "manifest.json").read_text())
    pm = json.loads((probes / "manifest.json").read_text())
    if meta["artifact_sha256"] != sha(responses / "responses.npz"):
        raise ValueError("Response hash mismatch")
    if meta["probes_manifest_sha256"] != sha(probes / "manifest.json") or pm["train_sha256"] != sha(
        training_features
    ):
        raise ValueError("Probe/training provenance mismatch")
    x = np.load(responses / "responses.npz", allow_pickle=False)
    t = np.load(training_features, allow_pickle=False)
    if set(x["subjects"]) & set(t["subjects"]):
        raise ValueError("Evaluation subjects overlap training")
    variance = t["descriptors"].var(0)
    if (variance <= 1e-12).any():
        raise ValueError("Degenerate training descriptor variance")
    conditions = list(x["conditions"])
    if conditions[0] != "clean":
        raise ValueError("Expected clean baseline first")
    prediction = x["predictions"]
    targets = x["descriptors"]
    normalized_loss = (prediction[..., 1:] - targets[:, None, :]) ** 2 / variance
    loss_change = normalized_loss - normalized_loss[:, :1]
    rows = []
    for index, name in enumerate(conditions):
        subjects = []
        for subject in np.unique(x["subjects"]):
            mask = x["subjects"] == subject
            ba = balanced_accuracy(x["labels"][mask], prediction[mask, index, 0])
            clean_ba = balanced_accuracy(x["labels"][mask], prediction[mask, 0, 0])
            subjects.append(
                dict(
                    subject=int(subject),
                    balanced_accuracy=ba,
                    accuracy_change=ba - clean_ba,
                    target_loss_change=float(loss_change[mask, index, :2].mean()),
                    off_target_loss_change=float(loss_change[mask, index, 2:].mean()),
                    signed_margin_change=float(
                        (
                            (prediction[mask, index, 0] - prediction[mask, 0, 0])
                            * (x["labels"][mask] * 2 - 1)
                        ).mean()
                    ),
                )
            )
        rows.append(
            dict(
                condition=name,
                subjects=subjects,
                **{
                    metric: bootstrap_mean([s[metric] for s in subjects])
                    for metric in (
                        "balanced_accuracy",
                        "accuracy_change",
                        "target_loss_change",
                        "off_target_loss_change",
                        "signed_margin_change",
                    )
                },
                multiplier_max=float(x["multipliers"][:, index].max()),
            )
        )
    # Paired comparison uses the same subject, averaging random seeds before bootstrap.
    comparisons = []
    for dose in (0.5, 1.0):
        concept = next(r for r in rows if r["condition"] == f"concept-{dose}")
        controls = [
            next(r for r in rows if r["condition"] == f"matched-{seed}-{dose}")
            for seed in (31, 71, 113)
        ]
        comparison = dict(dose=dose)
        for metric in ("accuracy_change", "target_loss_change", "off_target_loss_change"):
            differences = [
                s[metric] - np.mean([c["subjects"][i][metric] for c in controls])
                for i, s in enumerate(concept["subjects"])
            ]
            comparison[metric] = bootstrap_mean(differences)
        comparisons.append(comparison)
    report = dict(
        model=meta["model"],
        partition=meta["partition"],
        conditions=rows,
        concept_minus_matched_random=comparisons,
        runner_sha256=sha(__file__),
        response_manifest_sha256=sha(responses / "manifest.json"),
        caveat="Exploratory subject-bootstrap intervals; no multiplicity correction. Validation is not test evidence.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Subject-level response summary saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--training-features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.responses, a.probes, a.training_features, a.output)
