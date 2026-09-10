"""Run the complete experiment into a new output directory."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import baselines
import extract
import intervene
import prepare
import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "work", "cbramod", "labram"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.work.exists():
        raise FileExistsError("Choose a new work directory; existing runs are not overwritten")
    a.work.mkdir(parents=True)
    source = Path(__file__).parent
    record = dict(
        source_sha256={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in source.iterdir()
            if f.suffix in {".py", ".md", ".txt"}
        },
        versions={
            name: importlib.metadata.version(name)
            for name in ["eeglens", "torch", "numpy", "scipy", "scikit-learn", "mne", "timm"]
        },
    )
    (a.work / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    prepared, features, results = [a.work / name for name in ["prepared", "features", "results"]]
    prepare.prepare(a.data, prepared)
    for model in ["cbramod", "labram"]:
        for random in [False, True]:
            extract.extract(model, getattr(a, model), prepared, features, random_weights=random)
    baselines.run(prepared, features, results)
    for model in ["cbramod", "labram"]:
        intervene.run(model, getattr(a, model), prepared, features, results)
    summarize.run(prepared, features, results)
    hashes = {
        str(f.relative_to(a.work)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in a.work.rglob("*")
        if f.is_file()
    }
    (a.work / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
