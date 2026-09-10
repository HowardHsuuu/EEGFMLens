"""Compare a full public-sweep rerun against preserved pre-refactor artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(reference, rerun, output):
    old_hashes = json.loads((reference / "artifact_hashes.json").read_text())
    record = {}
    for name in ["prepared/inputs.npz"]:
        # Phase runs store surrogates; clean inputs remain in the parent spindle run.
        assert not (reference / name).exists()
    for name in ["prepared/signals.npz", "prepared/cases.json", "prepared/quality.json"]:
        assert hashlib.sha256((reference / name).read_bytes()).hexdigest() == old_hashes[name]
        if name.endswith(".npz"):
            np.testing.assert_array_equal(
                np.load(reference / name)["signals"], np.load(rerun / name)["signals"]
            )
        else:
            assert json.loads((reference / name).read_text()) == json.loads(
                (rerun / name).read_text()
            ), name
    for model in ["cbramod", "labram", "cbramod-random", "labram-random"]:
        for kind in ["responses", "restorations"]:
            name = f"results/{model}-{kind}.json"
            assert hashlib.sha256((reference / name).read_bytes()).hexdigest() == old_hashes[name]
            old = json.loads((reference / name).read_text())
            new = json.loads((rerun / name).read_text())
            assert len(old) == len(new), (name, len(old), len(new))
            for i, (a, b) in enumerate(zip(old, new)):
                assert a == b, (name, i, [(k, a[k], b.get(k)) for k in a if a[k] != b.get(k)])
            record[f"{model}-{kind}"] = len(new)
    assert json.loads((reference / "results/summary.json").read_text()) == json.loads(
        (rerun / "results/summary.json").read_text()
    )
    audit = []
    for model in ["cbramod", "labram", "cbramod-random", "labram-random"]:
        rows = json.loads((rerun / f"results/{model}-sweep-audit.json").read_text())
        controls = [r for r in rows if r["kind"] != "identity"]
        for r in controls:
            assert r["status"] == "matched" and r["valid"]
            assert np.isclose(r["actual_delta_norm"], r["target_norm"], rtol=1e-4, atol=1e-6)
            assert r["excessive_multiplier"] == (r["multiplier"] > 2)
        audit.append(
            dict(
                model=model,
                identity_checks=sum(r["kind"] == "identity" for r in rows),
                interventions=len(controls),
                large_multiplier_flags=sum(r["excessive_multiplier"] for r in controls),
            )
        )
    result = dict(
        comparison="exact equality across every original record and aggregate estimate",
        matched_records=record,
        prepared_signals_and_quality="exact equality",
        public_api_audit=audit,
        reference_artifact_manifest_sha256=hashlib.sha256(
            (reference / "artifact_hashes.json").read_bytes()
        ).hexdigest(),
    )
    with output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["reference", "rerun", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    compare(a.reference, a.rerun, a.output)
