"""Run the fixed phase experiment in a fresh directory, sequentially on CPU."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from prepare import prepare


def provenance(work, previous, checkpoints):
    source = Path(__file__).resolve().parent
    record = dict(
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.iterdir()
            if p.suffix in {".py", ".md"}
        },
        checkpoint_sha256={
            k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in checkpoints.items()
        },
        parent_artifact_manifest_sha256=hashlib.sha256(
            (previous / "artifact_hashes.json").read_bytes()
        ).hexdigest(),
        versions={
            n: importlib.metadata.version(n) for n in ["eeglens", "torch", "numpy", "scipy", "timm"]
        },
        random_seed=4311,
    )
    (work / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["previous", "events", "work", "cbramod", "labram"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    provenance(a.work, a.previous, {n: getattr(a, n) for n in ["cbramod", "labram"]})
    prepare(a.previous, a.events, a.work / "prepared")
    source = Path(__file__).resolve().parent
    for name in ["cbramod", "labram"]:
        for random in [False, True]:
            cmd = [
                sys.executable,
                str(source / "evaluate.py"),
                "--model",
                name,
                "--checkpoint",
                str(getattr(a, name)),
                "--previous",
                str(a.previous),
                "--prepared",
                str(a.work / "prepared"),
                "--output",
                str(a.work / "results"),
            ]
            subprocess.run(cmd + (["--random"] if random else []), check=True)
    subprocess.run(
        [sys.executable, str(source / "summarize.py"), "--work", str(a.work)], check=True
    )
    subprocess.run([sys.executable, str(source / "report.py"), "--work", str(a.work)], check=True)
    hashes = {
        str(f.relative_to(a.work)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in a.work.rglob("*")
        if f.is_file()
    }
    (a.work / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
