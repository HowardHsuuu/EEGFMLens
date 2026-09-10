"""Audit retained predictions and plot subject-level recovery without fitting models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ("cbramod", "labram", "csbrain")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r2(y, prediction):
    denominator = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    if np.any(denominator <= 1e-12):
        raise ValueError("Degenerate target variance")
    return 1 - ((y - prediction) ** 2).sum(axis=0) / denominator


def run(root, output):
    rows, evidence = [], []
    for model in MODELS:
        folder = root / model
        summary = json.loads((folder / "summary.json").read_text())
        artifact = folder / "predictions.npz"
        if sha(artifact) != summary["artifact_sha256"]:
            raise ValueError(f"Prediction digest mismatch: {model}")
        with np.load(artifact, allow_pickle=False) as data:
            y, subjects = data["targets"], data["subjects"]
            if not np.isfinite(y).all() or len(np.unique(data["trial_ids"])) != len(y):
                raise ValueError("Invalid targets or duplicate trial IDs")
            recorded = {s["subject"]: s["scores"] for s in summary["subject_r2"]}
            if set(recorded) != set(np.unique(subjects)):
                raise ValueError("Subject inventory mismatch")
            for condition, expected in summary["pooled_out_of_fold_r2"].items():
                prediction = data[condition]
                if prediction.shape != y.shape or not np.isfinite(prediction).all():
                    raise ValueError("Invalid prediction")
                np.testing.assert_allclose(r2(y, prediction), expected, atol=1e-12, rtol=0)
                for subject in np.unique(subjects):
                    mask = subjects == subject
                    score = r2(y[mask], prediction[mask])
                    np.testing.assert_allclose(
                        score, recorded[int(subject)][condition], atol=1e-12, rtol=0
                    )
                    rows.append(
                        dict(
                            model=model,
                            subject=int(subject),
                            condition=condition,
                            mu_r2=float(score[0]),
                            beta_r2=float(score[1]),
                        )
                    )
        evidence.append(
            dict(
                model=model,
                summary_sha256=sha(folder / "summary.json"),
                predictions_sha256=sha(artifact),
            )
        )
    output.mkdir(parents=True, exist_ok=False)
    table = output / "subject_scores.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for col, model in enumerate(MODELS):
        for row, target in enumerate(("mu", "beta")):
            ax = axes[row, col]
            subset = [r for r in rows if r["model"] == model]
            for subject in sorted({r["subject"] for r in subset}):
                pair = [
                    next(
                        r[f"{target}_r2"]
                        for r in subset
                        if r["subject"] == subject and r["condition"] == condition
                    )
                    for condition in ("clean", "recovered")
                ]
                ax.plot([0, 1], pair, color="#8294a6", alpha=0.6, linewidth=0.8, marker=".")
            for x, condition in enumerate(("clean", "recovered")):
                values = [r[f"{target}_r2"] for r in subset if r["condition"] == condition]
                ax.scatter(x, np.median(values), marker="D", color="#b13c31", s=40, zorder=4)
                ax.text(
                    x,
                    0.98,
                    f"{sum(v > 0 for v in values)}/{len(values)} > 0",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=9,
                )
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlim(-0.3, 1.3)
            ax.margins(y=0.2)
            ax.set_xticks([0, 1], ["Clean", "Refitted after erasure"])
            ax.set_title(f"{model} — {target}")
            ax.set_ylabel("Within-subject R²")
    fig.suptitle("Recoverable information varies across subjects", fontsize=15)
    fig.text(
        0.5,
        0.015,
        "Each line: one held-out training subject; red diamond: median. "
        "Exploratory 3-fold analysis, 18 subjects.\n"
        "Rank-two coefficient-span erasure; separate panel scales; no values clipped. "
        "This is not downstream causal evidence.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"subject_recovery.{extension}", dpi=180)
    plt.close(fig)
    report = dict(
        inputs=evidence,
        runner_sha256=sha(Path(__file__)),
        numpy=np.__version__,
        matplotlib=matplotlib.__version__,
        rows=len(rows),
        comparison_tolerance=dict(atol=1e-12, rtol=0),
        outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), verified_subject_condition_rows=len(rows))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.results, args.output)
