"""Descriptive subject-level summaries requiring all three completed outer folds."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ba(y, margin):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Need both labels for balanced accuracy")
    return float(np.mean([np.mean((margin[y == k] > 0) == k) for k in (0, 1)]))


def run(root, output):
    if output.exists():
        raise FileExistsError(output)
    rows, inputs, trial_ids, seen_subjects = [], [], set(), set()
    expected_conditions = None
    protocol_hash = None
    provenance = None
    subject_universe = None
    for fold in range(3):
        folder = root / f"fold{fold}"
        meta = json.loads((folder / "manifest.json").read_text())
        artifact = folder / "responses.npz"
        if sha(artifact) != meta["artifact_sha256"] or meta["fold"] != fold:
            raise ValueError("Fold artifact mismatch")
        if protocol_hash is not None and protocol_hash != meta["protocol_sha256"]:
            raise ValueError("Mixed protocols")
        protocol_hash = meta["protocol_sha256"]
        identity = {
            key: meta[key]
            for key in (
                "model",
                "checkpoint_sha256",
                "feature_sha256",
                "basis_sha256",
                "runner_sha256",
                "helper_sha256",
                "fitter_sha256",
                "builder_sha256",
            )
        }
        if provenance is not None and identity != provenance:
            raise ValueError("Mixed model, input or implementation provenance")
        provenance = identity
        x = np.load(artifact, allow_pickle=False)
        conditions = list(x["conditions"])
        if len(conditions) != 71 or len(set(conditions)) != 71 or conditions[0] != "clean":
            raise ValueError("Expected 71 unique conditions beginning with clean")
        if expected_conditions is not None and conditions != expected_conditions:
            raise ValueError("Mixed condition orders")
        expected_conditions = conditions
        subjects = set(x["subjects"])
        roles = meta["roles"]
        fitting = set(roles["fitting_subjects"])
        validation = set(roles["validation_subjects"])
        universe = fitting | validation | subjects
        if (
            (len(fitting), len(validation), len(subjects)) != (9, 3, 6)
            or fitting & validation
            or (subject_universe is not None and universe != subject_universe)
        ):
            raise ValueError("Inconsistent 9/3/6 subject partition")
        subject_universe = universe
        if (
            subjects != set(roles["evaluation_subjects"])
            or subjects & (set(roles["fitting_subjects"]) | set(roles["validation_subjects"]))
            or subjects & seen_subjects
        ):
            raise ValueError("Overlapping or inconsistent subject roles")
        if trial_ids & set(x["trial_ids"]):
            raise ValueError("Duplicate evaluation trials")
        seen_subjects.update(subjects)
        trial_ids.update(x["trial_ids"])
        pred = x["predictions"]
        targets = x["descriptors"]
        variance = x["training_target_variance"][1:]
        if (
            pred.shape != (270, 71, 5)
            or targets.shape != (270, 4)
            or variance.shape != (4,)
            or not all(
                np.isfinite(v).all() for v in (pred, targets, variance, x["norms"], x["gains"])
            )
            or np.any(variance <= 1e-12)
        ):
            raise ValueError("Invalid predictions or target variance")
        if x["norms"].shape != (270, 71) or x["gains"].shape != (270, 71):
            raise ValueError("Invalid perturbation audit shape")
        for method in ("contrast", "sensor"):
            for rank in (2, 4, 8, 16, 32):
                target = x["norms"][:, conditions.index(f"{method}-concept-{rank}")]
                for seed in (31, 71, 113):
                    matched = x["norms"][:, conditions.index(f"{method}-matched-{seed}-{rank}")]
                    np.testing.assert_allclose(matched, target, atol=1e-5, rtol=1e-5)
        loss = (pred[:, :, 1:] - targets[:, None, :]) ** 2 / variance
        for subject in sorted(subjects):
            mask = x["subjects"] == subject
            clean_ba = ba(x["labels"][mask], pred[mask, 0, 0])
            for i, condition in enumerate(conditions):
                change = (loss[mask, i] - loss[mask, 0]).mean(0)
                rows.append(
                    dict(
                        subject=int(subject),
                        fold=fold,
                        condition=condition,
                        balanced_accuracy=ba(x["labels"][mask], pred[mask, i, 0]),
                        accuracy_change=ba(x["labels"][mask], pred[mask, i, 0]) - clean_ba,
                        target_loss_change=float(change[:2].mean()),
                        off_target_loss_change=float(change[2:].mean()),
                        mean_norm=float(x["norms"][mask, i].mean()),
                        max_gain=float(x["gains"][mask, i].max()),
                    )
                )
        inputs.append(
            dict(
                file=str(artifact),
                sha256=sha(artifact),
                manifest_sha256=sha(folder / "manifest.json"),
            )
        )
    if len(seen_subjects) != 18 or len(trial_ids) != 810:
        raise ValueError("Incomplete 18-subject/810-trial evaluation")
    aggregate = []
    for condition in expected_conditions:
        selected = [r for r in rows if r["condition"] == condition]
        aggregate.append(
            dict(
                condition=condition,
                **{
                    k: float(np.mean([r[k] for r in selected]))
                    for k in (
                        "balanced_accuracy",
                        "accuracy_change",
                        "target_loss_change",
                        "off_target_loss_change",
                        "mean_norm",
                    )
                },
                max_gain=max(r["max_gain"] for r in selected),
            )
        )
    comparisons = []
    for method in ("contrast", "sensor"):
        for rank in (2, 4, 8, 16, 32):
            differences = []
            for subject in sorted(seen_subjects):

                def get(condition):
                    return next(
                        r for r in rows if r["subject"] == subject and r["condition"] == condition
                    )

                concept = get(f"{method}-concept-{rank}")
                controls = [get(f"{method}-matched-{seed}-{rank}") for seed in (31, 71, 113)]
                differences.append(
                    dict(
                        subject=int(subject),
                        **{
                            k: concept[k] - float(np.mean([c[k] for c in controls]))
                            for k in (
                                "accuracy_change",
                                "target_loss_change",
                                "off_target_loss_change",
                            )
                        },
                    )
                )
            comparisons.append(
                dict(
                    method=method,
                    rank=rank,
                    subjects=differences,
                    means={
                        k: float(np.mean([r[k] for r in differences]))
                        for k in ("accuracy_change", "target_loss_change", "off_target_loss_change")
                    },
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                scope="Descriptive cross-fitted subject means; no confirmatory p-values or independence assumption across overlapping fitted folds",
                inputs=inputs,
                provenance=provenance,
                protocol_sha256=protocol_hash,
                runner_sha256=sha(Path(__file__)),
                subjects=18,
                trials=810,
                rows=rows,
                aggregate=aggregate,
                concept_minus_matched_random=comparisons,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.results, a.output)
