"""Fail closed when a source release or built distribution is incomplete."""

import argparse
import tarfile
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

BANNED_PARTS = (
    "third_party_notices",
    "eeglens/",
    "validation/archive",
    "validation/migration",
    "validation/results",
    "adapters/eegmamba.py",
    "adapters/reve.py",
)
BANNED_SUFFIXES = (".ckpt", ".edf", ".pth", ".pt", ".safetensors")


def _audit_names(names):
    lowered = tuple(name.lower() for name in names)
    for name in lowered:
        if any(part in name for part in BANNED_PARTS):
            raise RuntimeError(f"Release contains retired or generated content: {name}")
        if name.endswith(BANNED_SUFFIXES):
            raise RuntimeError(f"Release contains a model or data artifact: {name}")


def _require_suffix(names, suffix):
    if not any(name.endswith(suffix) for name in names):
        raise RuntimeError(f"Release is missing {suffix}")


def audit_distributions(dist, version):
    files = tuple(path for path in dist.iterdir() if path.is_file())
    wheels = tuple(path for path in files if path.suffix == ".whl")
    sdists = tuple(path for path in files if path.name.endswith(".tar.gz"))
    if len(files) != 2 or len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("dist must contain exactly one wheel and one source archive")
    wheel, sdist = wheels[0], sdists[0]
    if not wheel.name.startswith(f"eegfmlens-{version}-"):
        raise RuntimeError(f"Wheel filename does not match version {version}")
    if sdist.name != f"eegfmlens-{version}.tar.gz":
        raise RuntimeError(f"Source archive filename does not match version {version}")

    with zipfile.ZipFile(wheel) as archive:
        names = tuple(archive.namelist())
        _audit_names(names)
        for suffix in (
            "eegfmlens/__init__.py",
            "eegfmlens/catalog.py",
            "eegfmlens/workflows.py",
            "eegfmlens/spectral.py",
            "eegfmlens/attribution.py",
            "eegfmlens/perturbation.py",
            "eegfmlens/features.py",
            "eegfmlens/alignment.py",
            "eegfmlens/aperiodic.py",
            "eegfmlens/probes.py",
            "eegfmlens/concepts.py",
            "eegfmlens/sae.py",
            "eegfmlens/circuits.py",
            ".dist-info/licenses/LICENSE",
            ".dist-info/licenses/LICENSES/README.md",
        ):
            _require_suffix(names, suffix)
        (metadata_name,) = (name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
        if "Name: eegfmlens\n" not in metadata:
            raise RuntimeError("Wheel metadata name mismatch")
        if f"Version: {version}\n" not in metadata:
            raise RuntimeError("Wheel metadata version mismatch")

    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        if any(member.issym() or member.islnk() for member in members):
            raise RuntimeError("Source archive must not contain links")
        names = tuple(member.name for member in members)
        _audit_names(names)
        for suffix in (
            "/README.md",
            "/LICENSE",
            "/LICENSES/README.md",
            "/docs/assets/eegfmlens-hero.png",
            "/docs/restoration-workflow.md",
            "/docs/interpretability.md",
            "/docs/migration-a17.md",
            "/examples/interpretability_methods.py",
            "/examples/cross_model_analysis.py",
            "/examples/concept_erasure.py",
            "/examples/concept_attribution.py",
            "/examples/source_attribution.py",
            "/tools/check_release.py",
        ):
            _require_suffix(names, suffix)


def check(repository, *, tag=None, dist=None):
    repository = repository.resolve()
    project = tomllib.loads((repository / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if tag is not None and tag != f"v{version}":
        raise RuntimeError(f"Release tag must be exactly v{version}, got {tag}")
    changelog = (repository / "CHANGELOG.md").read_text()
    if f"## {version} " not in changelog and f"## {version}—" not in changelog:
        raise RuntimeError(f"CHANGELOG has no release heading for {version}")
    for name in (
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "docs/releasing.md",
        "docs/restoration-workflow.md",
        "docs/interpretability.md",
        "examples/interpretability_methods.py",
        "examples/cross_model_analysis.py",
        "examples/concept_erasure.py",
        "examples/concept_attribution.py",
        "examples/source_attribution.py",
        "docs/migration-a17.md",
        "LICENSES/README.md",
    ):
        if not (repository / name).is_file():
            raise RuntimeError(f"Release source is missing {name}")
    tracked = tuple(path.relative_to(repository).as_posix() for path in repository.rglob("*"))
    _audit_names(tracked)
    if dist is not None:
        audit_distributions(dist.resolve(), version)
    print(f"Release contract passed for eegfmlens {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tag")
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()
    check(args.repository, tag=args.tag, dist=args.dist)
