"""Load explicitly supplied, trusted upstream checkouts for examples and tests.

Not part of the installed library. No downloads or source modifications.
"""

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

REVISIONS = {
    "biot": "d138e32634e52ae9fa6ec98ac9c4087b14ca869a",
    "cbramod": "b9e961003214326972c567eff390e75b0287e32a",
    "csbrain": "185aee55b24d0410a830df8dd08d03f675616998",
    "diver": "fae4d7c5a58f2f795ce767939ad191d9c7ba49b8",
    "eegpt": "a0e0a8fad729e2ecf4eedb3a81548a6e6d48a705",
    "labram": "c431221e6cfd23dbfa9950e0180682fb322b0548",
    "steegformer": "542ee17918c3c2c36ba1d4ea02bedff5eb149370",
}


def verified_source(name, source):
    """Require the catalog's exact upstream commit and clean tracked Python."""

    source = Path(source).resolve()
    if name not in REVISIONS:
        raise ValueError(f"No pinned source revision for {name}")
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != REVISIONS[name]:
        raise ValueError(f"Expected {name} revision {REVISIONS[name]}, got {actual}")
    subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", "*.py"], check=True
    )
    return source


def native_module(name, source):
    source = verified_source(name, source)
    if name == "cbramod":
        existing = sys.modules.get("models")
        if existing is not None and (
            not getattr(existing, "__file__", None)
            or source not in Path(existing.__file__).resolve().parents
        ):
            raise RuntimeError(
                "Another upstream 'models' package is loaded; use a separate process"
            )
        sys.path.insert(0, str(source))
        try:
            return importlib.import_module("models.cbramod")
        finally:
            sys.path.remove(str(source))
    spec = importlib.util.spec_from_file_location(
        "eeglens_example_labram", source / "modeling_finetune.py"
    )
    existing = sys.modules.get(spec.name)
    if existing is not None:
        if Path(existing.__file__).resolve() != source / "modeling_finetune.py":
            raise RuntimeError("Another LaBraM checkout is loaded; use a separate process")
        return existing
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # Required by the upstream timm registry.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module
