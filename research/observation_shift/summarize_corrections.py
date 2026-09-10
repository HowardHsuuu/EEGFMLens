"""Subject-paired native correction effects, including off-target harm."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from summarize import estimate


def summarize(source, output):
    output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for model in ["cbramod", "labram"]:
        doc = json.loads((source / f"{model}.json").read_text())
        assert len(doc["records"]) == 808 * 3 * 5 * 4
        grouped = defaultdict(list)
        for row in doc["records"]:
            grouped[(row["gain"], row["site"], row["kind"])].append(row)
        for (gain, site, kind), rows in grouped.items():
            corrected, base, clean, oracle = [], [], [], []
            by_subject = []
            for subject in range(1, 9):
                rs = [r for r in rows if r["subject"] == subject and r["valid"]]
                assert rs, (model, subject, gain, site, kind)
                y = [r["n2"] for r in rs]
                metrics = {
                    key: float(balanced_accuracy_score(y, np.array([r[key] for r in rs]) >= 0))
                    for key in ["margin", "clean_margin", "corrupt_margin", "oracle_margin"]
                }
                corrected.append(metrics["margin"])
                base.append(metrics["corrupt_margin"])
                clean.append(metrics["clean_margin"])
                oracle.append(metrics["oracle_margin"])
                by_subject.append(dict(subject=subject, n=len(rs), **metrics))
            all_results.append(
                dict(
                    model=model,
                    gain=gain,
                    site=site,
                    kind=kind,
                    corrected=estimate(corrected),
                    baseline=estimate(base),
                    clean=estimate(clean),
                    oracle=estimate(oracle),
                    paired_improvement=estimate(np.array(corrected) - base),
                    subjects=by_subject,
                    invalid=sum(not r["valid"] for r in rows),
                    multiplier_quantiles=np.quantile(
                        [r["multiplier"] for r in rows], [0, 0.5, 0.95, 1]
                    ).tolist(),
                    relative_edit_quantiles=np.quantile(
                        [r["relative_edit"] for r in rows], [0, 0.5, 0.95, 1]
                    ).tolist(),
                )
            )
    (output / "corrections.json").write_text(json.dumps(all_results, indent=2) + "\n")
    lines = [
        "# Internal correction experiment\n",
        "All effects below are paired equal-subject balanced-accuracy changes relative to the original fixed head on the same inputs. Maps were fitted only on five training subjects at gains 0.5 and 2, selected on two validation subjects, and evaluated on the held-out subject. The reported gains 0.75 and 1.5 were not used for fitting or selection. Gain 1 measures clean-condition harm. No learned correction uses a clean test donor.\n",
        "The table reports every site and method; intervals are descriptive 95% subject bootstraps with only eight subjects, not corrected for multiple testing. Identity, oracle and full-final-site recovery checks passed during native execution. Oracle results are stored separately in the machine-readable file; restoring clean activations is not itself a learned repair.\n",
        "| Model / site / correction | Gain 0.75 | Clean effect | Gain 1.5 |\n|---|---:|---:|---:|\n",
    ]
    for model in ["cbramod", "labram"]:
        for site in ["embedding.output", *[f"blocks.{i}.output" for i in [2, 5, 8, 11]]]:
            for kind in ["mean", "random_mean", "affine", "permuted_affine"]:
                cells = []
                for gain in [0.75, 1.0, 1.5]:
                    r = next(
                        r
                        for r in all_results
                        if r["model"] == model
                        and r["site"] == site
                        and r["kind"] == kind
                        and r["gain"] == gain
                    )["paired_improvement"]
                    cells.append(f"{r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
                lines.append(
                    "| " + model + " / " + site + " / " + kind + " | " + " | ".join(cells) + " |\n"
                )
    (output / "CORRECTIONS.md").write_text("\n".join(lines))
    print("All correction conditions summarized")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    summarize(a.source, a.output)
