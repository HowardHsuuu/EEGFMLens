"""Native CBraMod block and branch outputs; no reconstructed attention."""

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
        if kwargs:
            raise ValidationError(
                "CBraMod adapter does not accept execution kwargs; supply model-ready data"
            )
        return model(batch.data)
