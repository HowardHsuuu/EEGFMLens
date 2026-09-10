"""DIVER-1 native unmasked EEG backbone with declared electrode coordinates."""

import torch

from ..errors import ValidationError
from ..provenance import tensor_digest
from ..types import ActivationSite
from .base import Adapter


class DIVERAdapter(Adapter):
    """Use the 500 Hz EEG path; coordinates follow the author's xyz_id convention.

    The transformer prepends both a special channel and a special time position.
    Its internal caches use batch layout. Embedding and original-position output
    retain physical channel/patch axes. This does not validate iEEG preprocessing.
    """

    def __init__(self, model, *, channels, positions):
        self.channels = tuple(channels)
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValidationError("Declare unique ordered DIVER channels")
        if (
            not isinstance(positions, torch.Tensor)
            or positions.shape != (len(channels), 3)
            or not positions.is_floating_point()
            or not torch.isfinite(positions).all()
        ):
            raise ValidationError("DIVER positions must be finite floating [channels, 3]")
        if model.patcher.patch_len != 500 or model.patcher.stride != 500:
            raise ValidationError("This adapter requires DIVER's 500-sample EEG configuration")
        self.positions = positions.detach().clone()
        self.position_hash = tensor_digest(self.positions)
        sites = [ActivationSite("embedding.output", "embedding", "bcpd")]
        sites += [
            ActivationSite(f"blocks.{i}.output", f"encoder.encoder.layers.{i}", "batch")
            for i in range(len(model.encoder.encoder.layers))
        ]
        sites += [
            ActivationSite("encoder.output", "encoder", "batch"),
            ActivationSite("features.output", "head", "bcpd"),
        ]
        super().__init__(sites)

    def validate(self, batch):
        if batch.channels != self.channels or batch.sampling_rate != 500:
            raise ValidationError("DIVER requires declared channel order and 500 Hz EEG")
        if batch.data.shape[-1] != 500 or batch.stride != 500:
            raise ValidationError("DIVER requires contiguous 500-sample patches")
        if tensor_digest(self.positions) != self.position_hash:
            raise ValidationError("DIVER positions changed; construct a fresh adapter")

    def data_info(self, batch):
        return [
            {"xyz_id": self.positions.to(batch.data), "modality": "EEG", "coord_subtype": None}
            for _ in batch.trial_ids
        ]

    def forward(self, model, batch, **kwargs):
        if kwargs:
            raise ValidationError("DIVER adapter exposes the native unmasked EEG feature path")
        manager = model.token_manager
        previous_flag = manager.RAN_PREPEND
        previous_params = getattr(manager, "PREPEND_PARAMS", None)
        try:
            return model(batch.data, data_info_list=self.data_info(batch), use_mask=False)["y"]
        except BaseException:
            # Upstream changes Python bookkeeping before executing the encoder.
            # A failed hook must not leave the next call stuck in PREPEND state.
            manager.RAN_PREPEND = previous_flag
            if previous_params is None:
                if hasattr(manager, "PREPEND_PARAMS"):
                    del manager.PREPEND_PARAMS
            else:
                manager.PREPEND_PARAMS = previous_params
            raise

    def metadata(self):
        return {
            "position_sha256": self.position_hash,
            "modality": "EEG",
            "native_output": "y (original-position encoder features)",
            "rng_note": "Upstream SDPA applies dropout in eval; use paired RNG for comparisons",
        }
