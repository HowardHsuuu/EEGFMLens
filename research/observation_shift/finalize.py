"""Summarize the completed exploratory calibration study without selecting a best site."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from summarize import estimate


def run(root, output):
    output.mkdir(exist_ok=True, parents=True)
    scratch = json.loads((root / "scratch-v1/results.json").read_text())
    assert len(scratch["results"]) == 360
    assert len(list((root / "scratch-v1").glob("*.pt"))) == 72
    summary = {"normalized": {}, "scratch": {}, "correction_null_contrasts": []}
    for model in ["cbramod", "labram", "cbramod-random", "labram-random"]:
        doc = json.loads((root / f"normalized-v1/{model}.json").read_text())
        assert len(doc["records"]) == 24
        result = {}
        for gain in [0.75, 1.0, 1.5]:
            rows = sorted(
                [r for r in doc["records"] if r["gain"] == gain], key=lambda r: r["subject"]
            )
            assert [r["subject"] for r in rows] == list(range(1, 9))
            assert all(r["normalized_prediction_flips"] == 0 for r in rows)
            result[str(gain)] = estimate([r["balanced_accuracy"] for r in rows])
        summary["normalized"][model] = dict(
            scores=result, max_feature_difference=doc["max_feature_difference"]
        )
    for mode in ["clean", "augmented", "normalized"]:
        result = {}
        for gain in [0.5, 0.75, 1.0, 1.5, 2.0]:
            values = []
            for subject in range(1, 9):
                rows = [
                    r
                    for r in scratch["results"]
                    if r["mode"] == mode and r["gain"] == gain and r["subject"] == subject
                ]
                assert sorted(r["seed"] for r in rows) == [9137, 9138, 9139]
                values.append(np.mean([r["balanced_accuracy"] for r in rows]))
            result[str(gain)] = estimate(values)
        summary["scratch"][mode] = result
    correction = json.loads((root / "report-v1/corrections.json").read_text())
    assert len(correction) == 120 and all(r["invalid"] == 0 for r in correction)
    for r in correction:
        if r["kind"] not in ["mean", "affine"]:
            continue
        null_kind = "random_mean" if r["kind"] == "mean" else "permuted_affine"
        null = next(
            n
            for n in correction
            if all(n[k] == r[k] for k in ["model", "gain", "site"]) and n["kind"] == null_kind
        )
        assert [s["subject"] for s in r["subjects"]] == [s["subject"] for s in null["subjects"]]
        summary["correction_null_contrasts"].append(
            dict(
                model=r["model"],
                gain=r["gain"],
                site=r["site"],
                kind=r["kind"],
                null_kind=null_kind,
                paired_ba_difference=estimate(
                    np.array(r["corrected"]["subjects"]) - null["corrected"]["subjects"]
                ),
            )
        )
    (output / "FINAL_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Calibration study: completed pilot and decision\n",
        "Global gain repair is not supported as the main scientific contribution. Input normalization is a sufficient decision-invariance baseline here; internal correction effects are inconsistent and can harm clean predictions. This is a negative feasibility decision on a reused eight-subject cohort, not evidence that EEG FM mechanisms are solved.\n",
        "## Simple comparators\n",
        "Balanced accuracy is averaged equally across subjects. Scratch seeds are averaged within each subject first (eight independent subject units, not 24). All heads/checkpoints use subject-disjoint training and validation.\n",
        "| Model / training | Gain 0.5 | 0.75 | 1 | 1.5 | 2 |\n|---|---:|---:|---:|---:|---:|",
    ]
    for mode, scores in summary["scratch"].items():
        cells = [f"{scores[str(g)]['mean']:.4f}" for g in [0.5, 0.75, 1.0, 1.5, 2.0]]
        lines.append("| Small CNN / " + mode + " | " + " | ".join(cells) + " |")
    for model, doc in summary["normalized"].items():
        cells = [f"{doc['scores'][str(g)]['mean']:.4f}" for g in [0.75, 1.0, 1.5]]
        lines.append("| " + model + " / normalized | — | " + " | ".join(cells) + " | — |")
    lines += [
        "\nAll four normalized encoders have zero decision flips on all 808 windows across the three evaluated gains. Features are numerically close, not bitwise equal. This preprocessing removes absolute amplitude and is not a general clinical preprocessing recommendation.\n",
        "## Functional interventions\n",
        "The completed native evaluation contains 96,960 intervention records across two pretrained models, five sites, four methods and three gains. Every intervention was valid. Maps use only training paired activations; gain 0.75 and 1.5 are unseen strengths. Clean donor activations are used only for oracle diagnostics. Identity and full-final-site checks ran inside the native experiment. See CORRECTIONS.md for all sites; FINAL_SUMMARY.json additionally contains all 60 paired true-versus-null contrasts. Intervals are exploratory subject bootstraps without multiplicity correction. No best layer is selected.\n",
        "## What the evidence resolves\n",
        "- A failing historical head does not establish absent task information: training-domain augmentation can improve a new held-out readout. Gains vary by condition and do not prove a universal correction.\n",
        "- Tested intermediate mean/affine maps do not provide a stable repair advantage with clean-condition preservation. Their failure cannot establish information destruction.\n",
        "- Normalization makes this particular positive global scaling nuisance trivial while retaining useful task accuracy. It weakens the motivation for expanding gain repair.\n",
        "- The small supervised CNN is a practical comparator only. Its three seeds do not replace multiple random-encoder seeds or an architecture-matched pretraining ablation.\n",
        "## Limits and next decision\n",
        "The cohort has already informed earlier analyses; the gain change is synthetic and single-channel. No real-device, montage, independent-cohort, cross-task, or causal-pretraining claim is justified. The broader shared-computation question requires a separately fixed protocol and data/novelty gate. Keep this pilot as a reproducible negative result and EEGLens validation; do not scale the same calibration experiment merely to seek significance.\n",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines))
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in [".json", ".npz", ".pt", ".md"]
        and p.name != "artifact_manifest.json"
    ]
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)
    }
    (output / "artifact_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
