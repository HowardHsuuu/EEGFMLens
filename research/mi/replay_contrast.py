"""Replay a trusted extracted contrast-analysis bundle without EEG models or weights."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

MODELS = ("cbramod", "labram", "csbrain")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(expected, actual, location="result"):
    """Compare nested scientific values; paths are excluded by the caller."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"Different keys at {location}")
        for key in expected:
            compare(expected[key], actual[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Different lengths at {location}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{location}[{index}]")
    elif isinstance(expected, float):
        if not math.isfinite(actual) or not math.isclose(
            expected, actual, abs_tol=1e-10, rel_tol=1e-10
        ):
            raise ValueError(f"Numerical mismatch at {location}")
    elif expected != actual:
        raise ValueError(f"Mismatch at {location}")


def run(bundle, output):
    bundle, output = bundle.resolve(), output.absolute()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((bundle / "bundle.json").read_text())
    required = {
        "code/replay_contrast.py",
        "code/summarize_contrast_task.py",
        "code/audit_clean_readouts.py",
        "code/plot_contrast_task.py",
    }
    for model in MODELS:
        required.update(f"expected/{model}/{name}.json" for name in ("summary", "clean"))
        for fold in range(3):
            required.update(
                f"data/{model}/fold{fold}/{name}" for name in ("manifest.json", "responses.npz")
            )
    if not required.issubset(manifest["files"]):
        raise ValueError("Incomplete replay bundle")
    for name, digest in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or sha(path) != digest:
            raise ValueError(f"Bundle path or checksum mismatch: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".contrast-replay-", dir=output.parent) as temporary:
        staging = Path(temporary) / "results"
        staging.mkdir()
        for model in MODELS:
            current = staging / model
            current.mkdir()
            for script, name in (
                ("summarize_contrast_task.py", "summary"),
                ("audit_clean_readouts.py", "clean"),
            ):
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        str(bundle / "code" / script),
                        "--results",
                        str(bundle / "data" / model),
                        "--output",
                        str(current / f"{name}.json"),
                    ],
                    check=True,
                )
                expected = json.loads((bundle / "expected" / model / f"{name}.json").read_text())
                actual = json.loads((current / f"{name}.json").read_text())
                # Inputs have relocated paths. Check every other field, including provenance.
                expected.pop("inputs")
                actual.pop("inputs")
                compare(expected, actual, f"{model}.{name}")
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(bundle / "code/plot_contrast_task.py"),
                    "--summary",
                    str(current / "summary.json"),
                    "--output",
                    str(current / "figures"),
                ],
                check=True,
            )
        report = dict(
            scope="Retained response reanalysis, not model inference or independent scientific replication",
            models=list(MODELS),
            numerical_atol=1e-10,
            numerical_rtol=1e-10,
            bundle_manifest_sha256=sha(bundle / "bundle.json"),
            versions={name: version(name) for name in ("numpy", "matplotlib")},
            outputs={
                str(p.relative_to(staging)): sha(p) for p in staging.rglob("*") if p.is_file()
            },
        )
        (staging / "replay.json").write_text(json.dumps(report, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.bundle, args.output)
