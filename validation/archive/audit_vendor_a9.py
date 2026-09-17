"""Compare bundled model sources/licenses to pinned local Git objects; inspect release archives."""

import argparse
import ast
import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

REVISIONS = {
    "CBraMod": "b9e961003214326972c567eff390e75b0287e32a",
    "LaBraM": "c431221e6cfd23dbfa9950e0180682fb322b0548",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def blob(repositories, name, path):
    return subprocess.check_output(
        ["git", "-C", str(repositories / name), "show", f"{REVISIONS[name]}:{path}"]
    )


def literal_assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal assignment {name}")


def run(repository, repositories, wheel, sdist, output):
    vendor = repository / "src/eeglens/_vendor"
    rows = []
    for name, source, local in (
        ("CBraMod", "LICENSE", "cbramod/LICENSE"),
        ("CBraMod", "models/criss_cross_transformer.py", "cbramod/criss_cross_transformer.py"),
        ("LaBraM", "LICENSE", "LABRAM_LICENSE"),
        ("LaBraM", "modeling_finetune.py", "labram.py"),
    ):
        original = blob(repositories, name, source)
        current = (vendor / local).read_bytes()
        if original != current:
            raise ValueError(f"Unexpected vendored byte difference: {local}")
        rows.append(
            dict(
                project=name,
                revision=REVISIONS[name],
                upstream_path=source,
                local_path=local,
                upstream_sha256=digest(original),
                bundled_sha256=digest(current),
                comparison="exact bytes",
            )
        )
    original = blob(repositories, "CBraMod", "models/cbramod.py")
    tree = ast.parse(original)
    main_test = ast.dump(ast.parse("if __name__ == '__main__': pass").body[0].test)
    mains = [n for n in tree.body if isinstance(n, ast.If) and ast.dump(n.test) == main_test]
    if len(mains) != 1:
        raise ValueError("Unexpected CBraMod demonstration structure")
    tree.body.remove(mains[0])
    imports = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module == "models.criss_cross_transformer"
    ]
    if len(imports) != 1:
        raise ValueError("Unexpected CBraMod transformer import")
    imports[0].module, imports[0].level = "criss_cross_transformer", 1
    current = (vendor / "cbramod/model.py").read_bytes()
    if ast.dump(tree) != ast.dump(ast.parse(current)):
        raise ValueError(
            "CBraMod has changes beyond relative import and removal of __main__ demonstration"
        )
    rows.append(
        dict(
            project="CBraMod",
            revision=REVISIONS["CBraMod"],
            upstream_path="models/cbramod.py",
            local_path="cbramod/model.py",
            upstream_sha256=digest(original),
            bundled_sha256=digest(current),
            comparison="AST exact after two explicitly allowed packaging changes",
        )
    )
    original = blob(repositories, "LaBraM", "utils.py")
    current = (vendor / "labram_channels.py").read_bytes()
    if tuple(literal_assignment(original, "standard_1020")) != literal_assignment(
        current, "CHANNELS"
    ):
        raise ValueError("LaBraM channel order differs from pinned upstream")
    rows.append(
        dict(
            project="LaBraM",
            revision=REVISIONS["LaBraM"],
            upstream_path="utils.py:standard_1020",
            local_path="labram_channels.py",
            upstream_sha256=digest(original),
            bundled_sha256=digest(current),
            comparison="literal channel order exact; upstream module never executed",
        )
    )
    pytorch_license = (vendor / "PYTORCH_LICENSE").read_bytes()
    pytorch_license_sha = "47a26beb94e3f6b333a3677fc85d546f1fdfd2f0b3686c26d2fb5b10e0134165"
    if digest(pytorch_license) != pytorch_license_sha:
        raise ValueError("PyTorch notice differs from recorded v2.6.0 upstream license")
    rows.append(
        dict(
            project="PyTorch",
            revision="v2.6.0",
            upstream_path="https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/LICENSE",
            local_path="PYTORCH_LICENSE",
            upstream_sha256=pytorch_license_sha,
            bundled_sha256=digest(pytorch_license),
            comparison="Exact hash of recorded upstream license; not a complete derivation audit",
        )
    )
    acknowledged_licenses = [
        (
            "beitv2",
            "ca43e4cd19445a536f133bf2bc25b573b2f0c7c5",
            "https://raw.githubusercontent.com/microsoft/unilm/ca43e4cd19445a536f133bf2bc25b573b2f0c7c5/LICENSE",
            "BEIT2_LICENSE",
            "904dc4d8749877f1dba1cda48200d2462dccbeb7c134d5e4ef6fa75e0198c8fe",
        ),
        (
            "deit",
            "7e160fe43f0252d17191b71cbb5826254114ea5b",
            "https://raw.githubusercontent.com/facebookresearch/deit/7e160fe43f0252d17191b71cbb5826254114ea5b/LICENSE",
            "DEIT_LICENSE",
            "b208c52b00a2df2c8f4c3298c34407bf2bfe409968213372251c91fcc737a1a5",
        ),
        (
            "dino",
            "7c446df5b9f45747937fb0d72314eb9f7b66930a",
            "https://raw.githubusercontent.com/facebookresearch/dino/7c446df5b9f45747937fb0d72314eb9f7b66930a/LICENSE",
            "DINO_LICENSE",
            "50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a",
        ),
        (
            "timm",
            "7096b52a613eefb4f6d8107366611c8983478b19",
            "https://raw.githubusercontent.com/huggingface/pytorch-image-models/7096b52a613eefb4f6d8107366611c8983478b19/LICENSE",
            "TIMM_LICENSE",
            "71b111620fa32c17a80ccd6db2da759d31a591e497d8b5f382174f59d1b59d49",
        ),
    ]
    for project, revision, url, local, expected in acknowledged_licenses:
        current = (vendor / local).read_bytes()
        if digest(current) != expected:
            raise ValueError(f"Acknowledged upstream notice changed: {local}")
        rows.append(
            dict(
                project=project,
                revision=revision,
                upstream_path=url,
                local_path=local,
                upstream_sha256=expected,
                bundled_sha256=digest(current),
                comparison="Recorded reference license; not a proven derivation revision",
            )
        )
    with zipfile.ZipFile(wheel) as archive:
        for row in rows:
            name = "eeglens/_vendor/" + row["local_path"]
            if digest(archive.read(name)) != row["bundled_sha256"]:
                raise ValueError(f"Wheel differs from audited vendor source: {name}")
        for filename in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            names = [n for n in archive.namelist() if n.endswith("/licenses/" + filename)]
            if len(names) != 1 or archive.read(names[0]) != (repository / filename).read_bytes():
                raise ValueError(f"Missing or stale wheel license/notice: {filename}")
        wheel_members = archive.namelist()
    with tarfile.open(sdist) as archive:
        sdist_members = archive.getnames()
        for filename in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            members = [
                m
                for m in archive.getmembers()
                if len(Path(m.name).parts) == 2 and Path(m.name).name == filename
            ]
            if (
                len(members) != 1
                or archive.extractfile(members[0]).read() != (repository / filename).read_bytes()
            ):
                raise ValueError(f"Missing or stale sdist license/notice: {filename}")
        for row in rows:
            suffix = "/src/eeglens/_vendor/" + row["local_path"]
            members = [m for m in archive.getmembers() if m.name.endswith(suffix)]
            if (
                len(members) != 1
                or digest(archive.extractfile(members[0]).read()) != row["bundled_sha256"]
            ):
                raise ValueError(f"Missing or stale sdist vendor file: {suffix}")
    for name in (*wheel_members, *sdist_members):
        path = Path(name)
        if (
            path.suffix
            in {
                ".edf",
                ".pt",
                ".pth",
                ".ckpt",
                ".safetensors",
                ".pyc",
                ".token",
                ".password",
                ".api_key",
            }
            or any(p in {".git", ".local", "__pycache__", "repos"} for p in path.parts)
            or path.name.startswith(".env")
        ):
            raise ValueError(f"Unexpected private/data/checkpoint member: {name}")
    report = dict(
        vendor_files=rows,
        wheel_sha256=digest(wheel.read_bytes()),
        sdist_sha256=digest(sdist.read_bytes()),
        wheel_members=len(wheel_members),
        sdist_members=len(sdist_members),
        runner_sha256=digest(Path(__file__).read_bytes()),
        scope="Pinned-source equivalence and bundled notice inventory; external checkpoints/datasets and transitive upstream provenance require their own terms. No remote access or upstream module execution.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(rows)} vendor source/license checks and both release archives passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--upstream-repositories", type=Path, required=True)
    p.add_argument("--wheel", type=Path, required=True)
    p.add_argument("--sdist", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.repository, a.upstream_repositories, a.wheel, a.sdist, a.output)
