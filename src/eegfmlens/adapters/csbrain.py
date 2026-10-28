"""CSBrain's native region-sorted encoder with original sensor ordering exposed."""

import torch

from ..errors import ValidationError
from ..types import ActivationSite
from .base import Adapter


class CSBrainAdapter(Adapter):
    def __init__(self, model, *, channels, projection_order="native_sorted"):
        if projection_order not in {"native_sorted", "input"}:
            raise ValidationError("CSBrain projection_order must be native_sorted or input")
        self.projection_order = projection_order
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
        if set(kwargs) - {"mask"} or tuple(model.sorted_indices) != self.order:
            raise ValidationError(
                "CSBrain adapter requires unchanged channel ordering and only an optional mask"
            )
        if "mask" not in kwargs:
            return model(batch.data)
        mask = torch.as_tensor(kwargs["mask"], device=batch.data.device)
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(batch.data.shape[:-1]):
            raise ValidationError("CSBrain mask must be boolean [batch, sensor, patch]")
        # Native forward sorts x but leaves its mask untouched. Public mask
        # coordinates are SignalBatch.channels, so apply the same permutation.
        native_mask = mask.index_select(1, torch.tensor(self.order, device=mask.device))
        return model(batch.data, mask=native_mask)

    def expose(self, tensor, site, batch):
        tensor = super().expose(tensor, site, batch)
        if site.name == "projection.output" and self.projection_order == "input":
            return tensor
        return tensor.index_select(1, torch.tensor(self.inverse, device=tensor.device))

    def restore(self, tensor, site, native_shape):
        if site.name == "projection.output" and self.projection_order == "input":
            return tensor.reshape(native_shape)
        return tensor.index_select(1, torch.tensor(self.order, device=tensor.device)).reshape(
            native_shape
        )

    def metadata(self):
        return {
            "native_sensor_permutation": self.order,
            "cache_sensor_order": "SignalBatch.channels",
            "projection_order": self.projection_order,
            "projection_semantics": "Caller must establish trained output target order; encoder order alone is insufficient",
        }
