"""Explicit, resumable download of the EEGMMIDB recordings in a fixed split."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def fetch_one(root, subject, run):
    relative = f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
    url = f"https://physionet.org/files/eegmmidb/1.0.0/{relative}"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".edf.part")
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as target:
            expected = response.headers.get("Content-Length")
            while block := response.read(1024 * 1024):
                target.write(block)
        if expected is not None and temporary.stat().st_size != int(expected):
            raise ValueError(f"Incomplete response: {relative}")
        # Check EDF header and record-size consistency before accepting cache.
        validate_edf(temporary)
        temporary.replace(path)
    validate_edf(path)
    return dict(
        file=relative,
        source=url,
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def validate_edf(path):
    with path.open("rb") as stream:
        header = stream.read(256)
        if header[:8].strip() != b"0":
            raise ValueError(f"Not EDF: {path.name}")
        size, records, channels = int(header[184:192]), int(header[236:244]), int(header[252:256])
        if size != 256 * (channels + 1) or records <= 0:
            raise ValueError("Unsupported/incomplete EDF header")
        channel_header = stream.read(size - 256)
        samples = sum(
            int(channel_header[216 * channels + 8 * i : 216 * channels + 8 * (i + 1)])
            for i in range(channels)
        )
    if path.stat().st_size != size + records * samples * 2:
        raise ValueError(f"EDF length does not match header: {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    subjects = split["train"] + split["validation"] + split["test"]
    if len(subjects) != len(set(subjects)):
        raise ValueError("Subject leakage in split")
    completed, failed = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {
            pool.submit(fetch_one, args.output, s, r): (s, r)
            for s in subjects
            for r in split["runs"]
        }
        for job in as_completed(jobs):
            try:
                completed.append(job.result())
                print("downloaded/verified", len(completed), "/", len(jobs), flush=True)
            except Exception as exc:
                failed.append(dict(subject=jobs[job][0], run=jobs[job][1], error=str(exc)))
    report = dict(
        split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest(),
        files=sorted(completed, key=lambda x: x["file"]),
        failures=failed,
        license="Open Data Commons Attribution License v1.0",
        citation="Schalk (2009), 10.13026/C28G6P",
    )
    (args.output / "downloads.json").write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise SystemExit(f"{len(failed)} downloads failed; completed files retained")
