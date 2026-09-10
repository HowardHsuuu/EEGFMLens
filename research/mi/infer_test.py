"""Exact subject sign-flip and Holm correction for the frozen test endpoints."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from extract import sha


def sign_flip_greater(effects):
    values = np.asarray(effects, dtype=float)
    if values.ndim != 1 or not 2 <= len(values) <= 16 or not np.isfinite(values).all():
        raise ValueError("Expected 2–16 finite subject effects")
    signs = np.array(list(itertools.product((-1, 1), repeat=len(values))))
    null = (signs * values).mean(1)
    return float(np.mean(null >= values.mean() - 1e-14))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p-values")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def run(root, protocol, output):
    rows = []
    for name in ("cbramod", "labram", "csbrain"):
        folder = root / name
        manifest = json.loads((folder / "manifest.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        if manifest["partition"] != "test" or manifest["protocol_sha256"] != sha(protocol):
            raise ValueError("Not a response from the frozen test protocol")
        if summary["response_manifest_sha256"] != sha(folder / "manifest.json"):
            raise ValueError("Summary provenance mismatch")
        by_condition = {r["condition"]: r for r in summary["conditions"]}
        concept = by_condition["concept-1.0"]["subjects"]
        controls = [by_condition[f"matched-{seed}-1.0"]["subjects"] for seed in (31, 71, 113)]
        for metric, direction in (("target_loss_change", 1), ("accuracy_change", -1)):
            values = []
            for i, subject in enumerate(concept):
                if any(c[i]["subject"] != subject["subject"] for c in controls):
                    raise ValueError("Subject order mismatch")
                values.append(
                    direction * (subject[metric] - np.mean([c[i][metric] for c in controls]))
                )
            rows.append(
                dict(
                    model=name,
                    endpoint=metric,
                    direction=direction,
                    directional_subject_effects=values,
                    raw_p=sign_flip_greater(values),
                )
            )
    adjusted = holm([r["raw_p"] for r in rows])
    for row, value in zip(rows, adjusted):
        row["holm_p"] = float(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                tests=rows,
                family_size=6,
                protocol_sha256=sha(protocol),
                runner_sha256=sha(__file__),
                caveat="Sign-flip inference assumes exchangeable signs under the null. Six subjects give coarse p-values; nonsignificance does not establish equivalence.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responses-root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.responses_root, a.protocol, a.output)
