"""Strict local loading for LaBraM's task receiver and contextual tokenizer."""

import re
from collections.abc import Callable, Mapping
from typing import Any

import torch

from .adapters.labram_pretraining import LaBraMPretrainingAdapter
from .adapters.labram_tokenizer import LaBraMTokenizerAdapter
from .errors import ValidationError
from .loading import LABRAM_BASE_SHA256, _factory_identity, _identity
from .model import EEGLens
from .provenance import tensor_digest

LABRAM_TOKENIZER_SHA256 = "d8f14b4232f06a5c37b6386ab021c270dd4bfa12bfecdd60372dc8ac7b711101"


def _read_state(path, digest):
    # Only these byte-identical official checkpoints receive legacy metadata
    # allowlisting. No arbitrary pickle fallback or weights_only=False path.
    allowed: list[Callable[..., Any] | tuple[Callable[..., Any], str]] = []
    if digest in {LABRAM_BASE_SHA256, LABRAM_TOKENIZER_SHA256}:
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
    if not isinstance(checkpoint, Mapping):
        raise ValidationError("Checkpoint must contain a tensor state mapping")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValidationError("Checkpoint state must map names to tensors")
    return dict(state)


def _load(path, model_factory, expected_sha256, source_revision, device, kind, omit_scale):
    implementation = _factory_identity(model_factory)
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch("[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValidationError("An explicit lowercase checkpoint SHA256 is required")
    digest = _identity(path, expected_sha256)
    state = _read_state(path, digest)
    model = model_factory()
    if not isinstance(model, torch.nn.Module):
        raise ValidationError("model_factory must return a torch.nn.Module")
    omitted = []
    if omit_scale:
        if (
            kind != "pretraining"
            or digest != LABRAM_BASE_SHA256
            or "logit_scale" not in state
            or "logit_scale" in model.state_dict()
            or state["logit_scale"].ndim != 0
        ):
            raise ValidationError(
                "Unused logit_scale exception requires the exact official checkpoint"
            )
        value = state.pop("logit_scale")
        omitted.append(
            {
                "key": "logit_scale",
                "tensor_sha256": tensor_digest(value),
                "shape": list(value.shape),
                "reason": "explicit caller declaration: unused by supplied native forward",
                "applied_to_logits": False,
            }
        )
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    adapter = (
        LaBraMPretrainingAdapter(model) if kind == "pretraining" else LaBraMTokenizerAdapter(model)
    )
    lens = EEGLens(model, adapter, model_id=f"labram:{kind}:{digest}")
    lens.manifest.update(
        {
            "architecture": f"LaBraM-{kind}",
            "checkpoint_sha256": digest,
            "source_revision": source_revision,
            "implementation": implementation,
            "strict_loaded_tensor_count": len(state),
            "omitted_checkpoint_keys": omitted,
            "missing_keys": [],
            "unexpected_keys": [],
            "training_run_attested": False,
            "source_revision_independently_verified": False,
            "adapter": adapter.metadata(),
        }
    )
    return lens


def load_labram_pretraining(
    path,
    *,
    model_factory,
    expected_sha256,
    source_revision=None,
    device="cpu",
    acknowledge_unused_logit_scale=False,
):
    """Load the complete native wrapper, including both stored prediction heads.

    The zero-argument factory must construct the complete pretraining architecture.
    No head or normalization is replaced. A known extra scalar may be omitted only
    with an explicit declaration that it is unused by the supplied native forward,
    and only for the exact official checkpoint. The manifest records this exception;
    source/native-forward equivalence still requires an independent admission.
    """
    return _load(
        path,
        model_factory,
        expected_sha256,
        source_revision,
        device,
        "pretraining",
        acknowledge_unused_logit_scale,
    )


def load_labram_tokenizer(
    path,
    *,
    model_factory,
    expected_sha256,
    source_revision=None,
    device="cpu",
):
    """Strictly load all VQ-NSP state, including decoder and initialized codebook.

    The zero-argument factory supplies the exact native architecture. No key is
    discarded, no codebook fitted, and no remote download is performed by this
    loader. Persistent evaluation usage counts are preserved by the adapter.
    """
    return _load(path, model_factory, expected_sha256, source_revision, device, "tokenizer", False)
