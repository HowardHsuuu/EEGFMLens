"""Independent signal regeneration and one-case-per-subject model repeat."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from prepare import prepare


def verify(work, previous, events, checkpoints, output):
    output.mkdir(parents=True, exist_ok=False)
    prepare(previous, events, output / "regenerated")
    for name in ["cases.json", "quality.json"]:
        assert json.loads((work / "prepared" / name).read_text()) == json.loads(
            (output / "regenerated" / name).read_text()
        )
    values = np.load(work / "prepared/signals.npz")["signals"]
    np.testing.assert_array_equal(values, np.load(output / "regenerated/signals.npz")["signals"])
    doc = json.loads((work / "prepared/cases.json").read_text())
    first = {}
    for c in doc["cases"]:
        first.setdefault(c["subject"], c)
    subset = output / "subset"
    subset.mkdir()
    doc["cases"] = list(first.values())
    ids = {c["case"] for c in doc["cases"]}
    (subset / "cases.json").write_text(json.dumps(doc) + "\n")
    np.savez_compressed(subset / "signals.npz", signals=values)
    comparisons = {}
    for model in ["cbramod", "labram"]:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "evaluate.py"),
                "--model",
                model,
                "--checkpoint",
                str(checkpoints[model]),
                "--previous",
                str(previous),
                "--prepared",
                str(subset),
                "--output",
                str(output / "results"),
            ],
            check=True,
        )
        for kind in ["responses", "restorations"]:
            original = json.loads((work / f"results/{model}-{kind}.json").read_text())
            selected = [r for r in original if r["case"] in ids]
            repeated = json.loads((output / f"results/{model}-{kind}.json").read_text())
            assert selected == repeated, (model, kind)
            comparisons[f"{model}-{kind}"] = len(repeated)
    record = dict(
        signal_regeneration="every sample exactly equal",
        case_and_quality_regeneration="exact equality",
        repeated_case_ids=sorted(ids),
        repeated_rows_exactly_equal=comparisons,
    )
    (output / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["work", "previous", "events", "cbramod", "labram", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    verify(
        a.work, a.previous, a.events, {k: getattr(a, k) for k in ["cbramod", "labram"]}, a.output
    )
