"""Native CBraMod block and branch outputs; no reconstructed attention."""

import torch

from ..errors import ValidationError
from ..types import ActivationSite, SignalBatch
from .base import Adapter


class CBraModAdapter(Adapter):
    def __init__(self, model):
        sites = [ActivationSite("embedding.output", "patch_embedding")]
        for i in range(len(model.encoder.layers)):
            prefix = f"encoder.layers.{i}"
            sites.extend(
                [
                    ActivationSite(f"blocks.{i}.output", prefix),
                    ActivationSite(
                        f"blocks.{i}.spatial.output", prefix + ".self_attn_s", "spatial", 0
                    ),
                    ActivationSite(
                        f"blocks.{i}.temporal.output", prefix + ".self_attn_t", "temporal", 0
                    ),
                    ActivationSite(f"blocks.{i}.mlp.output", prefix + ".dropout2"),
                ]
            )
        super().__init__(sites)

    def validate(self, batch: SignalBatch) -> None:
        if batch.data.shape[-1] != 200 or batch.sampling_rate != 200:
            raise ValidationError("CBraMod requires 200-sample patches at 200 Hz")

    def forward(self, model, batch, **kwargs):
        if not kwargs:
            return model(batch.data)
        if set(kwargs) != {"mask"}:
            raise ValidationError("CBraMod accepts only an explicit native mask")
        try:
            mask = torch.as_tensor(kwargs["mask"], device=batch.data.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValidationError("Invalid CBraMod mask") from error
        if mask.dtype != torch.bool or mask.shape != batch.data.shape[:-1]:
            raise ValidationError("CBraMod mask must be boolean B,C,P in input coordinates")
        return model(batch.data, mask=mask)
