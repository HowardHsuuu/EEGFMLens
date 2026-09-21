"""Versioned local run bundles: JSON metadata plus weights-only tensor storage."""

import json
import os
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path

import torch

from .errors import ValidationError
from .loading import sha256_file
from .types import Activation, RunResult


def save_run(run: RunResult, directory) -> Path:
    """Save a new directory; refuse to overwrite existing results.

    Tensors move to CPU for storage. No model code, callbacks or arbitrary Python
    objects are pickled. Tuple/list subclasses are stored as plain containers;
    their class and namedtuple field names are not restored. Other custom native
    output objects must be converted by caller.
    """
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".eeglens-", dir=destination.parent))
    tensors: dict[str, torch.Tensor] = {}

    def pack(value):
        if isinstance(value, torch.Tensor):
            key = f"tensor_{len(tensors)}"
            tensors[key] = value.detach().cpu().clone()
            return {"kind": "tensor", "key": key}
        if isinstance(value, (list, tuple)):
            kind = "tuple" if isinstance(value, tuple) else "list"
            return {"kind": kind, "items": [pack(v) for v in value]}
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            return {"kind": "dict", "items": {k: pack(v) for k, v in value.items()}}
        if value is None or isinstance(value, (str, int, float, bool)):
            return {"kind": "scalar", "value": value}
        raise ValidationError(f"Unsupported output for export: {type(value).__name__}")

    try:
        caches = {}
        for name, activation in run.cache.items():
            item = {
                f.name: getattr(activation, f.name)
                for f in fields(activation)
                if f.name != "tensor"
            }
            item["tensor"] = pack(activation.tensor)
            caches[name] = item
        document = {
            "schema_version": 1,
            "model_id": run.model_id,
            "run_id": run.run_id,
            "calls": run.calls,
            "metadata": run.metadata,
            "output": pack(run.output),
            "cache": caches,
        }
        torch.save(tensors, temporary / "tensors.pt")
        document["tensor_sha256"] = sha256_file(temporary / "tensors.pt")
        (temporary / "run.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def load_run(directory) -> RunResult:
    """Read a v1 run bundle on CPU, checking tensor-file integrity first."""
    directory = Path(directory)
    document = json.loads((directory / "run.json").read_text())
    if document.get("schema_version") != 1:
        raise ValidationError("Unsupported run schema")
    if sha256_file(directory / "tensors.pt") != document["tensor_sha256"]:
        raise ValidationError("Run tensor checksum mismatch")
    tensors = torch.load(directory / "tensors.pt", map_location="cpu", weights_only=True)

    def unpack(node):
        kind = node["kind"]
        if kind == "tensor":
            return tensors[node["key"]]
        if kind == "scalar":
            return node["value"]
        if kind == "dict":
            return {k: unpack(v) for k, v in node["items"].items()}
        if kind in {"tuple", "list"}:
            values = [unpack(v) for v in node["items"]]
            return tuple(values) if kind == "tuple" else values
        raise ValidationError(f"Unknown tensor-tree kind: {kind}")

    cache = {}
    for name, item in document["cache"].items():
        item["tensor"] = unpack(item["tensor"])
        for key in ("trial_ids", "channels", "native_shape"):
            item[key] = tuple(item[key])
        cache[name] = Activation(**item)
    return RunResult(
        unpack(document["output"]),
        cache,
        document["calls"],
        document["model_id"],
        document["run_id"],
        document["metadata"],
    )
