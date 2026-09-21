"""DIVER-1 native EEG features or masked time-domain reconstruction."""

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

    def __init__(self, model, *, channels, positions, output="features"):
        if not isinstance(output, str) or output not in {"features", "reconstruction"}:
            raise ValidationError("DIVER output must be features or reconstruction")
        self.output = output
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
        sites.append(ActivationSite("encoder.output", "encoder", "batch"))
        if output == "features":
            sites.append(ActivationSite("features.output", "head", "bcpd"))
        else:
            try:
                model.get_submodule("heads.org.heads.time_head")
                model.mask_generator
            except (AttributeError, KeyError) as error:
                raise ValidationError(
                    "DIVER reconstruction requires the native time head and mask generator"
                ) from error
            sites.append(
                ActivationSite("reconstruction.output", "heads.org.heads.time_head", "bcpd")
            )
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
        mask = None
        if self.output == "features":
            if kwargs:
                raise ValidationError("DIVER feature output exposes the native unmasked path")
        else:
            if set(kwargs) != {"mask"}:
                raise ValidationError("DIVER reconstruction requires an explicit boolean mask")
            try:
                mask = torch.as_tensor(kwargs["mask"], device=batch.data.device)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValidationError("Invalid DIVER mask") from error
            if mask.dtype != torch.bool or mask.shape != batch.data.shape[:-1]:
                raise ValidationError("DIVER mask must be boolean B,C,P in input coordinates")
        manager = model.token_manager
        previous_flag = manager.RAN_PREPEND
        previous_params = getattr(manager, "PREPEND_PARAMS", None)
        handle = None
        try:
            if mask is not None:
                # Preserve native RNG consumption, replacing only the generated mask.
                handle = model.mask_generator.register_forward_hook(
                    lambda module, inputs, generated: mask.to(generated)
                )
            result = model(
                batch.data, data_info_list=self.data_info(batch), use_mask=mask is not None
            )
            if self.output == "features":
                return result["y"]
            return result["y_org"]["time_head_output"]
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
        finally:
            if handle is not None:
                handle.remove()

    def metadata(self):
        return {
            "position_sha256": self.position_hash,
            "modality": "EEG",
            "native_output": (
                "y (original-position encoder features)"
                if self.output == "features"
                else "y_org.time_head_output (native waveform patches)"
            ),
            "mask_contract": "explicit boolean B,C,P"
            if self.output == "reconstruction"
            else "unmasked",
            "rng_note": "Upstream SDPA applies dropout in eval; use paired RNG for comparisons",
        }
