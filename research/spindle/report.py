"""Generate the aggregate report and figure from a completed, verified run."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(work, destination):
    summary = json.loads((work / "results/summary.json").read_text())
    manifest = json.loads((work / "prepared/manifest.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)

    def fmt(record):
        return f"{record['mean']:.3f} [{record['low']:.3f}, {record['high']:.3f}]"

    lines = [
        "# EEGLens spindle experiment: results",
        "",
        "**The pilot does not establish a shared spindle-specific mechanism.** Both models contain linearly decodable spindle-related information, but spectral/amplitude features are stronger. LaBraM shows some direction-specific N2-margin effects compared with random controls; its effect is not distinguishable from the sigma direction in this small sample. CBraMod does not show consistent spindle-specific intervention evidence. Neither model shows a reliable overall balanced-accuracy reduction from the spindle erasure.",
        "",
        "All values below come from the corrected, text-verified microvolt pipeline. The initial reader-scale run was invalidated and is not included. Intervals are 95% subject-bootstrap descriptive intervals (8 subjects, 10,000 resamples), not corrected confirmatory significance tests.",
        "",
        "## Prediction baseline",
        "",
        "| Features | N2 balanced accuracy | N2 AUROC |",
        "|---|---|---|",
    ]
    for name in ["majority", "spectral", "cbramod-random", "cbramod", "labram-random", "labram"]:
        entry = summary["baseline"][name]
        lines.append(f"| {name} | {fmt(entry['balanced_accuracy'])} | {fmt(entry['auc'])} |")
    lines += ["", "Paired pretrained minus random-weight balanced accuracy:", ""]
    for model, value in summary["pretrained_minus_random"].items():
        lines.append(f"- {model}: {fmt(value)}. The random encoder uses one initialization seed.")
    lines += [
        "",
        "## Spindle information within N2",
        "",
        "All concept fitting uses reviewed training N2 windows. Layer/C selection uses validation subjects. Metrics are averaged over held-out subjects, not pooled epochs.",
        "",
        "| Probe | Held-out AUROC |",
        "|---|---|",
    ]
    for name, value in summary["concept"].items():
        lines.append(f"| {name} | {fmt(value)} |")
    lines += [
        "",
        "`residual_concept_score` is a probe on activations after train-fitted spectral/amplitude regression. `adjusted_direction_score` tests the resulting direction after removal of the fitted nuisance span. `expert2` uses independent second-rater labels on the six available subjects, with the original expert-1-trained probes and no reselection.",
        "",
        "The residual results do not establish reliable spindle information beyond the selected spectral/amplitude controls. This does not prove its absence: linear residualization may remove genuine spindle signal, and there are few training subjects.",
        "",
        "## Internal intervention",
        "",
        "All directions are rank one. Each control matches the raw spindle erasure norm per window. LaBraM CLS is preserved. The unchanged native remainder and fixed staging readout produce the edited prediction. Identity replacement and cached-feature parity are asserted in each test batch.",
        "",
        "Selectivity is the positive-minus-negative matched N2 margin change, normalized by the training margin standard deviation. More negative values indicate preferential suppression on spindle-positive windows. Control contrasts below subtract the control selectivity from spindle selectivity.",
        "",
        "| Model | Spindle vs random mean | Spindle vs sigma | Spindle vs slow-wave control |",
        "|---|---|---|---|",
    ]
    for model in ["cbramod", "labram"]:
        c = summary["paired_control_differences"][model]
        lines.append(
            f"| {model} | {fmt(c['random_mean'])} | {fmt(c['sigma'])} | {fmt(c['slow_wave'])} |"
        )
    lines += [
        "",
        "Raw accuracy-drop and perturbation evidence:",
        "",
        "| Model/control | Balanced accuracy drop | Maximum relative norm mismatch |",
        "|---|---|---|",
    ]
    for model in ["cbramod", "labram"]:
        for control in [
            "spindle",
            "adjusted_spindle",
            "sigma",
            "slow_wave",
            "shuffled",
            "random_mean",
        ]:
            entry = summary["interventions"][model][control]
            norm = entry["maximum_relative_norm_error"]
            lines.append(
                f"| {model}/{control} | {fmt(entry['accuracy_drop'])} | {norm:.2e} |"
                if norm is not None
                else f"| {model}/{control} | {fmt(entry['accuracy_drop'])} | mean of 10 controls |"
            )
    counts = [r["pairs"] for r in summary["matching"]]
    balance = [
        max(abs(v) for v in r["mean_standardized_covariate_difference"])
        for r in summary["matching"]
    ]
    lines += [
        "",
        "## Coverage and limitations",
        "",
        f"- {len(manifest['rows'])} stable-stage 15-second windows from eight DREAMS patients; no full-night or clinical staging claim.",
        f"- Matched pairs by subject: {counts}, total {sum(counts)}. Maximum absolute mean standardized covariate differences by subject: {[round(v, 3) for v in balance]}. Matching is imperfect and does not establish complete control of confounding.",
        "- First-rater concept coverage ends conservatively at 990 seconds. Unreviewed or ambiguous windows are not negative spindle examples.",
        "- The slow-wave morphology control is signal-defined. No matched human K-complex control was available; separate DREAMS subsets cannot be joined by excerpt number.",
        "- Single central electrode, 0.5–20 Hz bandwidth, short independent windows and mean pooling constrain what this pilot can reveal about the full pretrained models.",
        "- Subject bootstrap intervals are exploratory; overlapping fold training sets, eight subjects, multiple controls and one random-weight seed limit inference.",
        "- Layer-wise linear probes and rank-one interventions do not exhaust possible nonlinear or distributed mechanisms. Model/readout dependence is not a biological causal claim.",
        "",
        "## What this establishes",
        "",
        "EEGLens can execute a complete, falsifiable experiment linking human annotations, held-out readouts, native internal interventions and matched perturbation controls. This experiment separates decodability from selective use: a spindle-related probe alone would have supported a stronger story than the controls justify. The next scientific question is whether richer event/time representations yield evidence beyond spectral statistics on a larger, independently annotated cohort; it is not answered here.",
        "",
        "Reproduce with [README.md](README.md); fixed decisions and the unit correction are in [PROTOCOL.md](PROTOCOL.md). Machine-readable aggregate outcomes are in [results/summary.json](results/summary.json).",
        "",
        "Sources: [DREAMS](https://zenodo.org/records/2650142), [physiological-feature audit](https://arxiv.org/abs/2605.11410), [EEG SAE interpretation](https://arxiv.org/abs/2605.13930).",
        "",
        "![Aggregate pilot results](results/figure.png)",
        "",
    ]
    (destination / "RESULTS.md").write_text("\n".join(lines))
    resultdir = destination / "results"
    resultdir.mkdir(exist_ok=True)
    (resultdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    panels = [
        (
            "N2 prediction",
            [
                (n, summary["baseline"][n]["balanced_accuracy"])
                for n in ["spectral", "cbramod", "labram"]
            ],
            0.5,
        ),
        (
            "Spindle decoding within N2",
            [(n, summary["concept"][n]) for n in ["spectral", "cbramod", "labram"]],
            0.5,
        ),
        (
            "Spindle minus control selectivity",
            [
                (f"{m} vs {c}", summary["paired_control_differences"][m][c])
                for m in ["cbramod", "labram"]
                for c in ["random_mean", "sigma"]
            ],
            0,
        ),
    ]
    for ax, (title, entries, zero) in zip(axes, panels):
        for i, (label, v) in enumerate(entries):
            ax.errorbar(
                v["mean"],
                i,
                xerr=np.array([[v["mean"] - v["low"]], [v["high"] - v["mean"]]]),
                fmt="o",
                capsize=4,
                color="#2563a6",
            )
        ax.set_yticks(range(len(entries)), [e[0] for e in entries])
        ax.invert_yaxis()
        ax.axvline(zero, color="gray", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel("Balanced accuracy")
    axes[1].set_xlabel("AUROC")
    axes[2].set_xlabel("Normalized margin contrast (negative = more selective)")
    fig.suptitle(
        "EEGLens / DREAMS pilot — 8 held-out subjects; exploratory 95% subject intervals",
        fontsize=14,
    )
    fig.savefig(resultdir / "figure.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    render(a.work, a.destination)
