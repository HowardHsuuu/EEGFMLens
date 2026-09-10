"""Show all fixed ranks and individual subject effects from a completed model summary."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(summary, output):
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(summary.read_text())
    if data["subjects"] != 18 or data["trials"] != 810:
        raise ValueError("Expected complete three-fold analysis")
    comparisons = data["concept_minus_matched_random"]
    metrics = [
        ("accuracy_change", "Balanced accuracy difference (percentage points)", 100),
        ("target_loss_change", "Target normalized loss difference", 1),
        ("off_target_loss_change", "Non-target normalized loss difference", 1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    ranks = [2, 4, 8, 16, 32]
    for ax, (metric, label, scale) in zip(axes, metrics):
        for method, offset, color in [("contrast", -0.09, "#236c9e"), ("sensor", 0.09, "#b24b37")]:
            positions = np.arange(5) + offset
            means = []
            for position, rank in zip(positions, ranks):
                record = next(c for c in comparisons if c["method"] == method and c["rank"] == rank)
                values = np.array([s[metric] for s in record["subjects"]]) * scale
                if len(values) != 18 or not np.isfinite(values).all():
                    raise ValueError("Invalid subject effects")
                np.testing.assert_allclose(
                    values.mean(), record["means"][metric] * scale, atol=1e-12, rtol=0
                )
                means.append(values.mean())
                ax.scatter(np.full(18, position), values, color=color, alpha=0.32, s=13)
            ax.plot(positions, means, color=color, marker="D", label=method, linewidth=1.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
        ax.set_xticks(np.arange(5), [str(r) for r in ranks])
        ax.set_xlabel("Erased rank")
        ax.set_ylabel(label)
        ax.legend(frameon=False)
    fig.suptitle(f"{data['provenance']['model']}: concept minus norm-matched random controls")
    fig.text(
        0.5,
        0.015,
        "Dots: 18 individual subjects; diamonds: descriptive mean. All fixed ranks shown.\n"
        "Exploratory cross-fitting; no confidence intervals or confirmatory significance implied.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    output.mkdir(parents=True)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"contrast_task.{extension}", dpi=180)
    plt.close(fig)
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                summary_sha256=sha(summary),
                runner_sha256=sha(Path(__file__)),
                matplotlib=matplotlib.__version__,
                numpy=np.__version__,
                outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.summary, a.output)
