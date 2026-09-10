"""CSBrain's native region-sorted encoder with original sensor ordering exposed."""

import torch

from ..errors import ValidationError
from ..types import ActivationSite
from .base import Adapter


class CSBrainAdapter(Adapter):
    def __init__(self, model, *, channels):
        self.channels = tuple(channels)
        self.order = tuple(model.sorted_indices)
        if sorted(self.order) != list(range(len(self.channels))):
            raise ValidationError("CSBrain sorted_indices must permute the declared input sensors")
        self.inverse = tuple(sorted(range(len(self.order)), key=self.order.__getitem__))
        sites = [ActivationSite("embedding.output", "patch_embedding")]
        sites += [
            ActivationSite(f"blocks.{i}.output", f"encoder.layers.{i}")
            for i in range(model.encoder.num_layers)
        ]
        sites += [ActivationSite("projection.output", "proj_out")]
        super().__init__(sites)

    def validate(self, batch):
        if (
            batch.channels != self.channels
            or batch.sampling_rate != 200
            or batch.data.shape[-1] != 200
        ):
            raise ValidationError(
                "CSBrain requires configured sensor order and 200-sample patches at 200 Hz"
            )

    def forward(self, model, batch, **kwargs):
        if kwargs or tuple(model.sorted_indices) != self.order:
            raise ValidationError(
                "CSBrain adapter requires unchanged channel ordering and unmasked inference"
            )
        return model(batch.data)

    def expose(self, tensor, site, batch):
        tensor = super().expose(tensor, site, batch)
        return tensor.index_select(1, torch.tensor(self.inverse, device=tensor.device))

    def restore(self, tensor, site, native_shape):
        return tensor.index_select(1, torch.tensor(self.order, device=tensor.device)).reshape(
            native_shape
        )

    def metadata(self):
        return {
            "native_sensor_permutation": self.order,
            "cache_sensor_order": "SignalBatch.channels",
        }
