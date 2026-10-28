"""Check the shipped evidence inventory; never load weights or run a model."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != "eegfmlens-checkpoint-manifest-v1":
        raise ValueError("Unknown manifest schema")
    reports = manifest["reports"]
    families, ids, count = set(), set(), 0
    for entry in reports:
        name = entry["file"]
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Report path must be a local JSON basename")
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Report bytes differ: {name}")
        r = json.loads(path.read_text())
        if r["schema"] != "eegfmlens-checkpoint-report-v1" or r["id"] in ids:
            raise ValueError("Unknown schema or duplicate report")
        ids.add(r["id"])
        families.add(r["family"])
        if (r["id"], r["family"]) != (entry["id"], entry["family"]):
            raise ValueError("Report identity mismatch")
        if r["evidence_layer"] != "historical checkpoint-backed synthetic integration":
            raise ValueError("Historical scope must be retained")
        for digest in [r["checkpoint"]["sha256"], *r["implementation"].values()]:
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Missing checkpoint or implementation identity")
        e = r["execution"]
        if (e["device"], e["dtype"], e["mode"]) != ("cpu", "float32", "eager eval"):
            raise ValueError("Unexpected execution contract")
        if len(e["input_shape_without_batch"]) != 3 or any(
            not isinstance(n, int) or n <= 0 for n in e["input_shape_without_batch"]
        ):
            raise ValueError("Invalid input geometry")
        if len(e["channels"]) != e["input_shape_without_batch"][0]:
            raise ValueError("Channel count differs from geometry")
        if not r["component"] or not r["oracle"]["baseline"] or not r["oracle"]["intervention"]:
            raise ValueError("Missing component or oracle")
        if r["result"] != "recorded checks passed":
            raise ValueError("Unexpected recorded outcome")
        records = r["records"]
        if len(records) != entry["site_batch_records"] or len(records) != r["site_batch_records"]:
            raise ValueError("Record count mismatch")
        conditions = set()
        for record in records:
            batch = record.get("batch", record.get("batch_size"))
            key = (batch, record["site"], record.get("call_index"))
            if batch not in e["batch_sizes"] or key in conditions:
                raise ValueError("Invalid or duplicate site/batch/call")
            conditions.add(key)
            for k, value in record.items():
                if isinstance(value, bool) and not value:
                    raise ValueError(f"Failed recorded assertion: {r['id']}/{k}")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Nonfinite recorded metric")
        count += len(records)
    if len(ids) != manifest["components"] or len(families) != manifest["families"]:
        raise ValueError("Family/component inventory differs")
    if count != manifest["historical_records"]:
        raise ValueError("Record inventory differs")
    entry = manifest["study_summary"]
    path = root / entry["file"]
    if Path(entry["file"]).name != entry["file"]:
        raise ValueError("Invalid study summary path")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Study summary bytes differ")
    study = json.loads(path.read_text())
    if {r["model"] for r in study["models"]} != {
        "cbramod", "diver", "steegformer", "eegpt", "csbrain", "labram"
    }:
        raise ValueError("Study summary must retain all six paths")
    return {
        "status": "pass_report_integrity_and_inventory",
        "families": len(families), "historical_components": len(ids),
        "historical_site_batch_records": count, "study_paths": len(study["models"]),
        "native_inference_repeated": False,
        "scope": "Report bytes, geometry and recorded assertions; not independent neural validation",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).with_name("checkpoint_reports"))))
