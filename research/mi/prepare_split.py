"""Prepare exactly one partition from a frozen subject split, rejecting incomplete input."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare import prepare


def partition_files(split_path, partition, data):
    split = json.loads(split_path.read_text())
    groups = [split[name] for name in ("train", "validation", "test")]
    members = sum(groups, [])
    if len(members) != len(set(members)) or any(not group for group in groups):
        raise ValueError("Split has duplicate subjects or an empty partition")
    if partition not in ("train", "validation", "test"):
        raise ValueError("Unknown partition")
    paths = [
        data / f"S{s:03d}" / f"S{s:03d}R{r:02d}.edf"
        for s in split[partition]
        for r in split["runs"]
    ]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError(
            f"Partition incomplete: {len(missing)} recordings missing; first: {missing[0]}"
        )
    return split, paths


def run(split_path, partition, data, output):
    split, paths = partition_files(split_path, partition, data)
    prepare(paths, output)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    actual = set(record["subject"] for record in manifest["records"])
    if actual != set(split[partition]):
        raise ValueError("Prepared subjects do not match the full frozen partition")
    manifest.update(
        partition=partition,
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        split_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Frozen subject partition; raw-voltage preparation, no fitted transforms or model inference",
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.split, args.partition, args.data, args.output)
