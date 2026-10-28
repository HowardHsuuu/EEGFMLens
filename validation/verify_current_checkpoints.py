"""Verify retained current-checkpoint evidence and its shipped code bindings.

Standard library only. This checks report integrity/consistency, not model inference.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED = {
    "cbramod-features.json": 100, "cbramod-reconstruction.json": 100,
    "labram-features.json": 74, "eegpt-current.json": 20,
    "biot-current.json": 18, "biot-prest-current.json": 18,
    "bendr-current.json": 12, "bendr-context-current.json": 22,
    "brainomni-current.json": 24, "csbrain-current.json": 28,
    "neurorvq-current.json": 104, "signaljepa-current.json": 20,
    "diver-current.json": 78, "steegformer-current.json": 20,
}


def require(ok, description):
    if not ok:
        raise ValueError(description)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(root):
    folder = root / "validation/current_checkpoints"
    manifest = json.loads((folder / "manifest.json").read_text())
    require(set(manifest["reports"]) == set(EXPECTED), "Component inventory mismatch")
    runtime = {
        p.relative_to(root / "src/eegfmlens").as_posix(): digest(p)
        for p in sorted((root / "src/eegfmlens").rglob("*.py"))
    }
    require(runtime == manifest["runtime_sha256"], "Runtime differs from executed source")
    for name, expected in manifest["support_files_sha256"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "Invalid support path")
        require(digest(path) == expected, f"Support file digest mismatch: {name}")
    assets = json.loads((folder / "catalog-assets.json").read_text())
    total = 0
    for name, expected in EXPECTED.items():
        path = folder / name
        require(digest(path) == manifest["reports"][name]["sha256"], f"Report digest: {name}")
        r = json.loads(path.read_text())
        require(r["status"] == "pass", f"Failed report: {name}")
        require(r["runtime_sha256"] == runtime, f"Report runtime: {name}")
        require(r["environment"]["device"] == "cpu" and r["environment"]["dtype"] == "float32",
                f"Environment: {name}")
        require(r["input"]["batches"] == [1, 2], f"Input batches: {name}")
        generic = "assets" in r
        runner = "run_catalog_checkpoints.py" if generic else "run_checkpoint_checks.py"
        require(r["runner_sha256"] == digest(root / "validation" / runner), f"Runner: {name}")
        if generic:
            require(r["assets"] == assets[r["model"]], f"Asset manifest: {name}")
            require(r["builder_sha256"] == digest(root / "validation/checkpoint_components.py"),
                    f"Builder: {name}")
            require(r["helper_sha256"] == digest(root / "validation/run_checkpoint_checks.py"),
                    f"Helper: {name}")
        records = r["records"]
        require(len(records) == expected == manifest["reports"][name]["records"], f"Count: {name}")
        keys = [(v["batch"], v["site"]) for v in records]
        require(len(keys) == len(set(keys)), f"Duplicate site/batch: {name}")
        require({s for b, s in keys if b == 1} == {s for b, s in keys if b == 2}, f"Site sets: {name}")
        require(all(b in (1, 2) for b, _ in keys), f"Unexpected batch: {name}")
        for v in records:
            for metric in ("baseline_max_abs_error", "identity_max_abs_error",
                           "independent_edit_max_abs_error", "cleanup_max_abs_error"):
                # All retained comparisons were exactly equal. Reruns use the stated allclose tolerance.
                require(v[metric] == 0, f"Retained comparison mismatch: {name}/{v['site']}/{metric}")
            require(math.isfinite(v["effect_max_abs"]) and v["effect_max_abs"] >= 0, "Invalid effect")
            require(v["hooks_restored"] is True, "Hook cleanup failed")
            if generic:
                require(v["observed_calls"] == v["expected_calls"] > v["call_index"] >= 0,
                        "Invocation mismatch")
                require(v["selected_zero_exact"] is True, "Edit support mismatch")
            else:
                require(v["single_call"] is True and v["selected_zero_and_untouched_exact"] is True,
                        "Partial edit or invocation mismatch")
        total += expected
    require((manifest["families"], manifest["component_views"], manifest["site_batch_records"])
            == (11, 14, total), "Summary inventory mismatch")
    return {"status": "pass", "families": 11, "component_views": 14,
            "site_batch_records": total, "neural_inference_repeated": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify(p.parse_args().root)))
