"""Inventory existing dual labels without refitting or treating unreviewed time as negatives."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def run(manifest, output):
    doc = json.loads(manifest.read_text())
    rows = doc["rows"]
    records = []
    for subject in range(1, 9):
        rs = [r for r in rows if r["subject"] == subject]
        valid = [r for r in rs if r["spindle"] >= 0]
        assert all(r["reviewed"] and r["start"] + 15 <= 990 for r in valid)
        joint = Counter((r["stage"], r["spindle"]) for r in valid)
        n2 = [r for r in valid if r["n2"]]
        vals = [subject % 8 + 1, (subject + 1) % 8 + 1]
        records.append(
            dict(
                subject=subject,
                windows=len(rs),
                reviewed_labeled=len(valid),
                unknown_or_boundary=len(rs) - len(valid),
                n2_spindle_negative=sum(r["spindle"] == 0 for r in n2),
                n2_spindle_positive=sum(r["spindle"] == 1 for r in n2),
                joint=[dict(stage=k[0], spindle=k[1], count=v) for k, v in sorted(joint.items())],
                split=dict(
                    test=[subject],
                    validation=vals,
                    train=[s for s in range(1, 9) if s not in [subject, *vals]],
                ),
            )
        )
    result = dict(
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        recipe=doc["recipe"],
        annotation_policy=doc["annotation_policy"],
        subjects=records,
        decision="Usable only for exploratory plumbing; already reused cohort, sparse positives and strong stage association. Not an independent cross-task mechanism test.",
    )
    output.mkdir(exist_ok=True, parents=True)
    (output / "DATA_INVENTORY.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Existing dual-task data gate\n",
        "This audit reuses existing annotations and baseline outcomes; no new head is fitted or selected. Unknown/boundary spindle labels remain excluded.\n",
        "| Subject | All windows | Reviewed labeled | N2 negative | N2 positive |\n|---|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['subject']} | {r['windows']} | {r['reviewed_labeled']} | {r['n2_spindle_negative']} | {r['n2_spindle_positive']} |"
        )
    lines += [
        "\nEvery subject has both spindle labels within N2, but subjects 3 and 4 have only four and five positive windows. The effective independent sample is eight subjects. The complete stage×spindle counts and original subject-disjoint split assignments are in DATA_INVENTORY.json.\n",
        "Existing conditional spindle probes already ran in the earlier spindle study. Their equal-subject AUROC was spectral 0.8581, CBraMod 0.6750, LaBraM 0.7728. These are intermediate-feature probes selected using training/validation data, not independently validated final task-B heads. Repeating them as a new experiment would add no evidence.\n",
        "Decision: retain DREAMS as a feasibility example, but do not use it as independent confirmation or infer shared computations from stage/event co-degradation. A prospective cross-task experiment still needs a frozen intervention, target-side selection exclusion, conditional controls and independent data.\n",
        "Source: [DREAMS depositor record](https://doi.org/10.5281/ZENODO.2650141). The accompanying source PDF identifies CC BY-NC-ND 3.0; this inventory does not redistribute signals. Window filtering and annotation boundaries are specified in the prepared manifest.\n",
    ]
    (output / "DATA_INVENTORY.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.manifest, a.output)
