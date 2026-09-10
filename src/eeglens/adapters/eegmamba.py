"""Experimental EEGMamba native hooks; requires upstream Mamba/Triton runtime.

Not checkpoint-validated on the local Mac and intentionally not exported from
the package root. The encoder reverses token order after every block, so block
caches deliberately do not advertise physical sensor/patch selection.
"""

from ..errors import ValidationError
from ..types import ActivationSite
from .base import Adapter


class EEGMambaAdapter(Adapter):
    def __init__(self, model, *, channels):
        self.channels = tuple(channels)
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValidationError("Declare unique ordered EEGMamba input channels")
        sites = [ActivationSite("embedding.output", "patch_embedding", "bcpd")]
        for i in range(len(model.encoder.layers)):
            for j, branch in enumerate(("hidden", "residual")):
                sites.append(
                    ActivationSite(
                        f"blocks.{i}.{branch}", f"encoder.layers.{i}", "batch", tensor_index=j
                    )
                )
        sites.extend(
            [
                ActivationSite("encoder.output", "encoder", "batch"),
                ActivationSite("projection.output", "proj_out", "bcpd"),
            ]
        )
        super().__init__(sites)

    def validate(self, batch):
        if batch.channels != self.channels or batch.sampling_rate != 200:
            raise ValidationError("EEGMamba input channels/order or sampling rate mismatch")
        if batch.data.shape[-1] != 200 or batch.stride != 200:
            raise ValidationError("EEGMamba requires contiguous 200-sample patches")

    def forward(self, model, batch, **kwargs):
        if kwargs:
            raise ValidationError("EEGMamba adapter exposes unmasked native forward only")
        return model(batch.data)
