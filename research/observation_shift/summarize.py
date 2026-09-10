"""Equal-subject descriptive gain and probe comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np


def estimate(values):
    values = np.asarray(values, dtype=float)
    assert values.shape == (8,) and np.isfinite(values).all()
    rng = np.random.default_rng(9137)
    samples = rng.choice(values, (10000, 8), replace=True).mean(1)
    return dict(
        mean=float(values.mean()),
        ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
        subjects=values.tolist(),
    )


def summarize(gate, probes, output):
    output.mkdir(exist_ok=True, parents=True)
    probe = json.loads((probes / "results.json").read_text())["results"]
    results = {}
    for model in ["cbramod", "labram", "cbramod-random", "labram-random"]:
        source = json.loads((gate / f"{model}.json").read_text())
        model_rows = []
        for gain in source["gains"]:
            rs = sorted(
                [r for r in source["results"] if r["gain"] == gain], key=lambda r: r["subject"]
            )
            record = dict(
                gain=gain,
                fixed_head={
                    k: estimate([r[k] for r in rs])
                    for k in [
                        "balanced_accuracy",
                        "auc",
                        "flip_fraction",
                        "mean_absolute_margin_change",
                    ]
                },
            )
            for pooling in ["mean", "flatten"]:
                by_domain = {}
                for domain in ["clean", "augmented"]:
                    rows = sorted(
                        [r for r in probe[f"{model}:{pooling}:{domain}"] if r["gain"] == gain],
                        key=lambda r: r["subject"],
                    )
                    by_domain[domain] = np.array([r["balanced_accuracy"] for r in rows])
                record[pooling] = dict(
                    clean=estimate(by_domain["clean"]),
                    augmented=estimate(by_domain["augmented"]),
                    paired_gain=estimate(by_domain["augmented"] - by_domain["clean"]),
                )
            model_rows.append(record)
        results[model] = model_rows
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "# Gain and readout pilot: preliminary results\n",
        "These are exploratory results on the previously used eight-subject DREAMS cohort. They do not establish a foundation-model mechanism, genuine device generalization, or information loss. Internal correction and multichannel/independent-cohort experiments remain outstanding.\n",
        "## Fixed historical head\n",
        "| Model | Gain 0.5 | Gain 0.75 | Clean | Gain 1.5 | Gain 2 |\n|---|---:|---:|---:|---:|---:|\n",
    ]
    for model, rows in results.items():
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(f"{r['fixed_head']['balanced_accuracy']['mean']:.3f}" for r in rows)
            + " |\n"
        )
    lines.extend(
        [
            "\n## Paired effect of training-domain readout adaptation\n",
            "Balanced-accuracy differences: augmented-head minus clean-trained head, with matched model/pooling/split/solver. Intervals are descriptive paired subject bootstraps (10,000 draws); no multiple-comparison claim. Augmented training sees gains 0.5 and 2 only.\n",
            "| Model / pooling | Unseen gain 0.75 | Clean off-target effect | Unseen gain 1.5 |\n|---|---:|---:|---:|\n",
        ]
    )
    for model, rows in results.items():
        for pooling in ["mean", "flatten"]:
            cells = []
            for i in [1, 2, 3]:
                d = rows[i][pooling]["paired_gain"]
                cells.append(f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
            lines.append("| " + model + " / " + pooling + " | " + " | ".join(cells) + " |\n")
    lines.extend(
        [
            "\n## Interpretation\n",
            "CBraMod's historical fixed-head balanced accuracy falls from 0.687 at gain 1 to 0.481 at gain 0.5. A frozen-encoder mean-pooled probe trained using shifted training subjects reaches 0.696 at gain 0.5 on held-out subjects. This shows that the collapse of this particular fixed head is not sufficient evidence that the representation has lost all N2-discriminative information. It does not show complete information preservation.\n",
            "The more demanding unseen-strength comparison is less decisive, and augmented training can harm clean-condition performance. Flattening all patch tokens does not automatically help on this small cohort. LaBraM's balanced accuracy stays near 0.60 while many individual decisions change: aggregate performance stability is not prediction invariance. Random-encoder controls also change with gain, so sensitivity alone cannot be attributed to pretraining.\n",
            "Positive global calibration gain is invertible and is canceled numerically by per-window standardization (maximum input relative error below 8e-8). This is an input-equivalence check, not a measured improvement in normalized-model task accuracy; normalization may itself remove useful amplitude information. Known gain inversion is a trivial calibration baseline.\n",
            "Next: test a training-derived correction at prespecified intermediate sites, with clean-condition harm, mismatched-pair controls, and unseen subjects/strengths. Oracle donor restoration by itself will not resolve whether the effect is transportable. No new physiology or publication-level novelty is claimed by this gate.\n",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines))
    print("Gain gate and probe summaries saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["gate", "probes", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    summarize(a.gate, a.probes, a.output)
