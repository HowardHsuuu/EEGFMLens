"""Native EEGPT encoder: window-folded attention with trailing summary tokens."""

import torch

from ..errors import ValidationError
from ..types import ActivationSite
from .base import Adapter


class EEGPTAdapter(Adapter):
    """Connect the official EEGTransformer encoder, not its downstream classifier.

    Exposed activations retain window/token/feature axes under layout ``batch``.
    Sensor/patch selections are deliberately unavailable: summary tokens do not
    represent electrodes. No token is silently dropped or relabeled as CLS.
    """

    def __init__(self, model):
        self.patch_size = model.patch_embed.patch_size
        if model.patch_embed.patch_stride not in (None, self.patch_size):
            raise ValidationError("Overlapping EEGPT tokenization is not supported by this adapter")
        self.num_patches = tuple(model.num_patches)
        self.summary_tokens = model.embed_num
        sites = [ActivationSite("embedding.output", "patch_embed", "batch")]
        sites += [
            ActivationSite(f"blocks.{i}.output", f"blocks.{i}", "batch")
            for i in range(len(model.blocks))
        ]
        sites += [ActivationSite("norm.output", "norm", "batch")]
        super().__init__(sites)

    def validate(self, batch):
        if batch.sampling_rate != 256 or batch.data.shape[-1] != self.patch_size:
            raise ValidationError("EEGPT requires 256 Hz and patches matching its patch embedder")
        if batch.stride != self.patch_size or batch.data.shape[1:3] != self.num_patches:
            raise ValidationError(
                "EEGPT requires contiguous patches with configured channel/window counts"
            )

    def forward(self, model, batch, **kwargs):
        if kwargs:
            raise ValidationError("EEGPT adapter exposes unmasked encoder inference only")
        try:
            channel_ids = model.prepare_chan_ids(batch.channels)
        except (AssertionError, KeyError, ValueError) as exc:
            raise ValidationError(
                "EEGPT channel name is absent from the native vocabulary"
            ) from exc
        if (
            not isinstance(channel_ids, torch.Tensor)
            or channel_ids.dtype != torch.long
            or channel_ids.shape != (1, len(batch.channels))
            or channel_ids.unique().numel() != len(batch.channels)
        ):
            raise ValidationError("EEGPT channel lookup must return unique int64 IDs per sensor")
        return model(
            batch.data.flatten(2),
            chan_ids=channel_ids.to(batch.data.device),
        )

    def expose(self, tensor, site, batch):
        b, c, p, _ = batch.data.shape
        if site.name == "embedding.output":
            if tensor.ndim != 4 or tensor.shape[:3] != (b, p, c):
                raise ValidationError("Unexpected EEGPT embedding geometry")
            return tensor
        tokens = self.summary_tokens if site.name == "norm.output" else c + self.summary_tokens
        if tensor.ndim != 3 or tensor.shape[:2] != (b * p, tokens):
            raise ValidationError("Unexpected EEGPT window-folded geometry")
        return tensor.reshape(b, p, tokens, tensor.shape[-1])
