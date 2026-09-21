"""Adapter contracts; adapters own semantic mapping, not research hypotheses."""

import torch
from torch import nn

from ..errors import UnsupportedSiteError, ValidationError
from ..types import ActivationSite, SignalBatch


class Adapter:
    """Explicit output hooks on an existing native model.

    Custom adapters may override validate/forward. Site names are never inferred
    as proof of support. Each site targets one declared invocation; the total
    number of module calls must match expected_calls (one by default).
    """

    def __init__(self, sites: list[ActivationSite]):
        for s in sites:
            if not isinstance(s, ActivationSite):
                raise ValidationError("Sites must be ActivationSite objects")
            if not isinstance(s.name, str) or not s.name:
                raise ValidationError("Site names must be nonempty strings")
            # An empty module path deliberately hooks the root module.
            if not isinstance(s.module_path, str):
                raise ValidationError("Site module paths must be strings")
            if not isinstance(s.layout, str) or not s.layout:
                raise ValidationError("Site layouts must be nonempty strings")
            if type(s.writable) is not bool:
                raise ValidationError("Site writable must be a boolean")
            if s.tensor_index is not None and (type(s.tensor_index) not in (int, str)):
                raise ValidationError("Site tensor_index must be an integer or string key")
            if (
                type(s.call_index) is not int
                or type(s.expected_calls) is not int
                or not 0 <= s.call_index < s.expected_calls
            ):
                raise ValidationError("Invalid site call_index/expected_calls")
        self.sites = {s.name: s for s in sites}
        if len(self.sites) != len(sites):
            raise ValidationError("Duplicate site names")

    def require(self, name: str) -> ActivationSite:
        try:
            return self.sites[name]
        except KeyError as e:
            raise UnsupportedSiteError(
                f"Unsupported site {name!r}; available: {list(self.sites)}"
            ) from e

    def validate(self, batch: SignalBatch) -> None:
        pass

    def metadata(self):
        return {}

    def expose(self, tensor, site, batch):
        return expose(tensor, site, batch)

    def restore(self, tensor, site, native_shape):
        return restore(tensor, site, native_shape)

    def forward(self, model: nn.Module, batch: SignalBatch, **kwargs):
        return model(batch.data, **kwargs)


def expose(tensor: torch.Tensor, site: ActivationSite, batch: SignalBatch) -> torch.Tensor:
    """Convert folded native axes to a batch-first semantic view."""
    b, c, p, _ = batch.data.shape
    if site.layout == "spatial":
        if tensor.ndim != 3 or tensor.shape[:2] != (b * p, c):
            raise ValidationError(f"Unexpected spatial shape at {site.name}: {tensor.shape}")
        return tensor.reshape(b, p, c, -1).permute(0, 2, 1, 3)
    if site.layout == "temporal":
        if tensor.ndim != 3 or tensor.shape[:2] != (b * c, p):
            raise ValidationError(f"Unexpected temporal shape at {site.name}: {tensor.shape}")
        return tensor.reshape(b, c, p, -1)
    if site.layout == "bcpd" and (tensor.ndim != 4 or tensor.shape[:3] != (b, c, p)):
        raise ValidationError(f"Unexpected sensor/patch shape at {site.name}: {tensor.shape}")
    if site.layout in {"tokens", "patch_tokens"} and (
        tensor.ndim != 3 or tensor.shape[:2] != (b, int(site.layout == "tokens") + c * p)
    ):
        raise ValidationError(f"Unexpected CLS/token shape at {site.name}: {tensor.shape}")
    if site.layout not in {"bcpd", "tokens", "patch_tokens", "batch"}:
        raise ValidationError(f"Unknown layout: {site.layout}")
    if tensor.shape[0] != b:
        raise ValidationError("First exposed axis must match batch")
    return tensor


def restore(tensor: torch.Tensor, site: ActivationSite, native_shape: tuple) -> torch.Tensor:
    if site.layout == "spatial":
        return tensor.permute(0, 2, 1, 3).reshape(native_shape)
    return tensor.reshape(native_shape)
