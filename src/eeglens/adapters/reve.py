"""Experimental native REVE adapter; official gated checkpoint not yet validated."""

import torch

from ..errors import ValidationError
from ..provenance import tensor_digest
from ..types import ActivationSite
from .continuous import _ContinuousAdapter


class REVEAdapter(_ContinuousAdapter):
    """Supply coordinates in the native REVE convention and the checkpoint rate.

    Native forward performs its own overlapping unfold on continuous input.
    SignalBatch patches are storage chunks, not REVE token windows. Sites use
    batch layout until that separate coordinate mapping has been validated.
    """

    def __init__(self, model, *, channels, positions, sampling_rate):
        if (
            not isinstance(positions, torch.Tensor)
            or positions.shape != (len(channels), 3)
            or not positions.is_floating_point()
            or not torch.isfinite(positions).all()
        ):
            raise ValidationError("REVE positions must be finite floating [channels, 3]")
        if sampling_rate <= 0:
            raise ValidationError("Declare a positive checkpoint sampling rate")
        sites = [ActivationSite("embedding.output", "to_patch_embedding", "batch")]
        for i in range(len(model.transformer.layers)):
            for j, branch in enumerate(("attention", "feedforward")):
                sites.append(
                    ActivationSite(
                        f"blocks.{i}.{branch}.output", f"transformer.layers.{i}.{j}", "batch"
                    )
                )
        sites.append(ActivationSite("encoder.output", "transformer", "batch"))
        super().__init__(sites, channels, sampling_rate)
        self.positions = positions.detach().clone()
        self.position_hash = tensor_digest(self.positions)
        self.patch_size = model.patch_size
        self.overlap = model.overlap_size
        if not 0 <= self.overlap < self.patch_size:
            raise ValidationError("Invalid REVE patch overlap")

    def validate(self, batch):
        super().validate(batch)
        if batch.data.shape[2] * batch.data.shape[3] < self.patch_size:
            raise ValidationError("Input is shorter than REVE patch size")
        if tensor_digest(self.positions) != self.position_hash:
            raise ValidationError("REVE positions changed; construct a fresh adapter")

    def forward(self, model, batch, **kwargs):
        if kwargs:
            raise ValidationError("REVE adapter exposes native forward with return_output=False")
        if (model.patch_size, model.overlap_size) != (self.patch_size, self.overlap):
            raise ValidationError("REVE patch configuration changed after adapter creation")
        pos = self.positions.to(batch.data).unsqueeze(0).expand(batch.data.shape[0], -1, -1)
        return model(batch.data.flatten(2), pos, return_output=False)

    def metadata(self):
        return {
            "position_sha256": self.position_hash,
            "native_patch_size": self.patch_size,
            "native_patch_overlap": self.overlap,
        }
