"""Explicit local checkpoint loading. No implicit network access or partial loads."""

import hashlib
import inspect
from pathlib import Path

import torch
from torch import nn

from .adapters.cbramod import CBraModAdapter
from .adapters.labram import LaBraMAdapter
from .errors import ValidationError
from .model import EEGLens

CBRAMOD_REVISION = "b9e961003214326972c567eff390e75b0287e32a"
LABRAM_REVISION = "c431221e6cfd23dbfa9950e0180682fb322b0548"
LABRAM_BASE_SHA256 = "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c"


def _factory_identity(factory):
    if not callable(factory):
        raise ValidationError("model_factory must be callable")
    try:
        source = inspect.getsourcefile(factory)
    except TypeError:
        source = None
    return {
        "module": getattr(factory, "__module__", type(factory).__module__),
        "name": getattr(factory, "__qualname__", type(factory).__qualname__),
        "source_file_sha256": sha256_file(source) if source and Path(source).is_file() else None,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path, expected_sha256):
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValidationError("Checkpoint SHA256 mismatch")
    return digest


def load_cbramod(
    path,
    *,
    model_factory,
    output="features",
    device="cpu",
    expected_sha256=None,
    source_revision=None,
):
    """Load the official 12-block CBraMod state dict, strictly.

    ``features`` explicitly removes the reconstruction projection *after* strict
    loading; ``reconstruction`` retains it. Neither attaches a task classifier.
    """
    implementation = _factory_identity(model_factory)

    if output not in {"features", "reconstruction"}:
        raise ValidationError("output must be features or reconstruction")
    digest = _identity(path, expected_sha256)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = model_factory()
    if not isinstance(model, nn.Module):
        raise ValidationError("model_factory must return a torch.nn.Module")
    model.load_state_dict(state, strict=True)
    if output == "features":
        model.proj_out = nn.Identity()
    model.to(device).eval()
    model.requires_grad_(False)
    lens = EEGLens(model, CBraModAdapter(model), model_id=f"cbramod:{digest}:{output}")
    lens.manifest = {
        "architecture": "CBraMod",
        "source_revision": source_revision,
        "validated_source_revision": CBRAMOD_REVISION,
        "implementation": implementation,
        "checkpoint_sha256": digest,
        "output": output,
        "missing_keys": [],
        "unexpected_keys": [],
        "dtype": "float32",
    }
    return lens


def load_labram(
    path,
    *,
    model_factory,
    output="patch_tokens",
    device="cpu",
    expected_sha256=None,
    source_revision=None,
):
    """Load LaBraM base's pretrained student encoder with its learned final norm.

    Unlike a downstream training script, this does not initialize a new fc_norm
    or classifier. The native non-mean-pooling encoder path retains student.norm.
    Legacy metadata allowlisting is restricted to the known official file hash.
    """
    implementation = _factory_identity(model_factory)
    if output not in {"patch_tokens", "all_tokens", "pooled"}:
        raise ValidationError("output must be patch_tokens, all_tokens or pooled")

    digest = _identity(path, expected_sha256)
    if digest == LABRAM_BASE_SHA256:
        import argparse

        import numpy as np
        from numpy.core.multiarray import scalar

        allowed = [
            (scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
            type(np.dtype("float64")),
            argparse.Namespace,
        ]
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    raw = checkpoint.get("model", checkpoint)
    discarded = []
    if any(k.startswith("student.") for k in raw):
        state = {}
        allowed_outer = {
            "logit_scale",
            "lm_head.weight",
            "lm_head.bias",
            "projection_head.0.weight",
            "projection_head.0.bias",
        }
        allowed_student = {"mask_token", "lm_head.weight", "lm_head.bias"}
        for key, value in raw.items():
            if key.startswith("student."):
                name = key[len("student.") :]
                if name in allowed_student:
                    discarded.append(key)
                else:
                    state[name] = value
            elif key in allowed_outer:
                discarded.append(key)
            else:
                raise ValidationError(f"Unknown non-encoder checkpoint key: {key}")
    else:
        state = raw
    model = model_factory(num_classes=0, init_values=0.1, use_mean_pooling=False)
    if not isinstance(model, nn.Module):
        raise ValidationError("model_factory must return a torch.nn.Module")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    lens = EEGLens(
        model,
        LaBraMAdapter(model, output=output),
        model_id=f"labram:{digest}:{output}:pretrained_norm",
    )
    lens.manifest = {
        "architecture": "LaBraM-base",
        "source_revision": source_revision,
        "validated_source_revision": LABRAM_REVISION,
        "implementation": implementation,
        "checkpoint_sha256": digest,
        "output": output,
        "normalization": "pretrained student.norm; no new fc_norm",
        "discarded_pretraining_keys": sorted(discarded),
        "missing_keys": [],
        "unexpected_keys": [],
        "dtype": "float32",
    }
    return lens
