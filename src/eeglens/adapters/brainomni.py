"""Official BrainOmni encode path with fixed sensor geometry."""

import math

import torch

from ..errors import ValidationError
from ..provenance import tensor_digest
from ..types import ActivationSite
from .base import Adapter


class BrainOmniAdapter(Adapter):
    """Connect native ``encode`` with an explicit ordered sensor configuration.

    ``positions`` is [C,6] in the upstream position/direction convention;
    ``sensor_types`` is [C] using the upstream sensor vocabulary. Neither is
    inferred from channel names. Latent sensor units are not input electrodes,
    so all declared sites use conservative batch geometry.
    """

    def __init__(self, model, *, channels, positions, sensor_types):
        window = getattr(model, "window_length", None)
        overlap = getattr(model, "overlap_ratio", None)
        if (
            type(window) is not int
            or window <= 0
            or getattr(model.tokenizer, "window_length", None) != window
            or type(overlap) not in (int, float)
            or not math.isfinite(overlap)
            or not 0 <= overlap < 1
            or int(window * (1 - overlap)) < 1
        ):
            raise ValidationError(
                "BrainOmni requires matching positive native windows and a valid overlap"
            )
        self._window_config = (window, overlap, model.tokenizer.window_length)
        self.channels = tuple(channels)
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValidationError("Declare unique ordered sensor names")
        if (
            positions.shape != (len(channels), 6)
            or not positions.is_floating_point()
            or not torch.isfinite(positions).all()
        ):
            raise ValidationError("BrainOmni positions must be finite floating [sensor,6]")
        if sensor_types.shape != (len(channels),) or sensor_types.dtype != torch.long:
            raise ValidationError("BrainOmni sensor_types must be int64 [sensor]")
        n_types = model.tokenizer.sensor_embed.sensor_embedding_layer.num_embeddings
        if (sensor_types < 0).any() or (sensor_types >= n_types).any():
            raise ValidationError("Sensor type index is outside the native embedding vocabulary")
        self.positions = positions.detach().clone()
        self.sensor_types = sensor_types.detach().clone()
        self._geometry = (tensor_digest(self.positions), tensor_digest(self.sensor_types))
        sites = [ActivationSite("projection.output", "projection", "batch")]
        # Upstream encode intentionally omits the last block; pretraining forward differs.
        sites += [
            ActivationSite(f"blocks.{i}.output", f"blocks.{i}", "batch")
            for i in range(len(model.blocks) - 1)
        ]
        super().__init__(sites)

    def validate(self, batch):
        if batch.channels != self.channels or batch.sampling_rate != 256:
            raise ValidationError("BrainOmni sensor order or sampling rate mismatch")
        if batch.stride != batch.data.shape[-1]:
            raise ValidationError("BrainOmni requires contiguous input patches")
        if self._geometry != (tensor_digest(self.positions), tensor_digest(self.sensor_types)):
            raise ValidationError("Sensor geometry changed; construct a fresh adapter")

    def forward(self, model, batch, **kwargs):
        if kwargs:
            raise ValidationError("BrainOmni adapter exposes encode without execution kwargs")
        if (
            model.window_length,
            model.overlap_ratio,
            model.tokenizer.window_length,
        ) != self._window_config:
            raise ValidationError(
                "BrainOmni window/overlap configuration changed; construct a fresh adapter"
            )
        b = batch.data.shape[0]
        pos = self.positions.to(batch.data).unsqueeze(0).expand(b, -1, -1)
        kinds = self.sensor_types.to(batch.data.device).unsqueeze(0).expand(b, -1)
        return model.encode(batch.data.flatten(2), pos, kinds)

    def metadata(self):
        return {
            "native_method": "encode",
            "native_window_samples": self._window_config[0],
            "native_overlap_ratio": self._window_config[1],
            "native_stride_samples": int(self._window_config[0] * (1 - self._window_config[1])),
            "position_sha256": self._geometry[0],
            "sensor_types_sha256": self._geometry[1],
        }
