"""Verify installed wheel bytes, then run tests and quickstart outside the checkout."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(wheel, repository, output, *, coverage=False):
    wheel, repository, output = [p.resolve() for p in (wheel, repository, output)]
    output.parent.mkdir(parents=True, exist_ok=True)
    junit_output = output.with_suffix(".xml")
    # Never treat an old successful report as evidence for a failed rerun.
    if output.exists():
        raise FileExistsError(output)
    if junit_output.exists():
        raise FileExistsError(junit_output)
    distribution = importlib.metadata.distribution("eeglens")
    # Keep this coordinator free of native runtimes. Loading torch here while
    # launching another torch process can exhaust sandbox OpenMP SHM access.
    spec = importlib.util.find_spec("eeglens")
    if spec is None or spec.origin is None:
        raise RuntimeError("Installed EEGLens module cannot be resolved")
    imported = Path(spec.origin).resolve()
    installed = Path(distribution.locate_file("eeglens/__init__.py")).resolve()
    if imported != installed or repository in imported.parents:
        raise RuntimeError(
            "EEGLens import does not originate from an installed distribution outside the checkout"
        )
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("archive_info", {}).get("hashes", {}).get("sha256") != sha(wheel):
        raise RuntimeError("Installed distribution does not match the supplied wheel digest")
    count = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("eeglens/") and name.endswith(".py"):
                if archive.read(name) != Path(distribution.locate_file(name)).read_bytes():
                    raise RuntimeError(f"Installed file differs from wheel: {name}")
                count += 1
    if not count:
        raise RuntimeError("Wheel contains no package Python files")
    with tempfile.TemporaryDirectory(prefix="eeglens-installed-") as cwd:
        junit = Path(cwd) / "tests.xml"
        sweep = Path(cwd) / "known-answer-sweep.json"
        commands = [
            [
                sys.executable,
                "-I",
                "-c",
                "import sys; from pathlib import Path; import eeglens; "
                "assert Path(eeglens.__file__).resolve() == Path(sys.argv[1])",
                str(installed),
            ],
            [sys.executable, "-I", "-m", "pip", "check"],
            [
                sys.executable,
                "-I",
                "-m",
                "pytest",
                "-q",
                str(repository / "tests"),
                "-m",
                "not integration and not native",
                f"--junitxml={junit}",
            ],
            [sys.executable, "-I", str(repository / "examples/quickstart.py")],
            [
                sys.executable,
                "-I",
                str(repository / "examples/patching_sweep.py"),
                "--output",
                str(sweep),
            ],
        ]
        if coverage:
            commands[2].extend(["--cov=eeglens", "--cov-report=term-missing"])
        for command in commands:
            subprocess.run(command, cwd=cwd, check=True)
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        counts = {
            key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        if not counts["tests"] or counts["failures"] or counts["errors"]:
            raise RuntimeError("No successful installed-package test suite")
        report = dict(
            version=distribution.version,
            runner_sha256=sha(__file__),
            wheel_sha256=sha(wheel),
            imported_from=str(imported),
            verified_python_files=count,
            tests=counts,
            junit_sha256=sha(junit),
            junit_file=junit_output.name,
            python=sys.version,
            platform=platform.platform(),
            dependencies={
                d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
            },
            quickstart="passed",
            known_answer_sweep="passed",
            known_answer_sweep_sha256=sha(sweep),
            example_source_sha256={
                name: sha(repository / "examples" / name)
                for name in ("quickstart.py", "patching_sweep.py")
            },
            pip_check="passed",
            scope="Installed wheel verification and non-integration suite; checkpoint/native studies remain separate evidence.",
        )
        junit_output.write_bytes(junit.read_bytes())
        output.write_text(json.dumps(report, indent=2) + "\n")
    print("Installed wheel validation report:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--coverage", action="store_true", help="Require pytest-cov and report coverage"
    )
    args = parser.parse_args()
    wheels = list(args.dist.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one wheel in --dist")
    run(wheels[0], args.repository, args.output, coverage=args.coverage)
