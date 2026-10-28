"""Stable hashes for recording tensor contents without retaining inputs."""

import hashlib
import json

import torch


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    # uint8 supports bfloat16 and does not require NumPy to understand the dtype.
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + raw).hexdigest()
