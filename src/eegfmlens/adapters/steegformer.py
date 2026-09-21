"""Official ST-EEGFormer downstream encoder, preserving time-major token order."""

import torch

from ..errors import ValidationError
from ..types import ActivationSite
from .continuous import _ContinuousAdapter


class STEEGFormerAdapter(_ContinuousAdapter):
    """Call native forward_features, without a freshly initialized classifier.

    Tokens are time-major in upstream code. They are exposed conservatively in
    batch layout, so they cannot be mistaken for LaBraM's channel-major tokens.
    """

    def __init__(self, model, *, channels, channel_mapping):
        if model.global_pool:
            raise ValidationError("Use the pretrained norm/CLS path (global_pool=False)")
        try:
            indices = tuple(channel_mapping[c] for c in channels)
        except KeyError as exc:
            raise ValidationError("Channel absent from the official ST-EEGFormer mapping") from exc
        size = model.enc_channel_emd.channel_transformation.num_embeddings
        if any(type(i) is not int or not 0 <= i < size for i in indices):
            raise ValidationError("Channel index is outside the pretrained embedding table")
        self.channel_indices = indices
        self.patch_size = model.patch_embed.p
        self.max_patches = model.enc_temporal_emd.pe.shape[1] - 1
        sites = [ActivationSite("embedding.output", "patch_embed", "batch")]
        sites += [
            ActivationSite(f"blocks.{i}.output", f"blocks.{i}", "batch")
            for i in range(len(model.blocks))
        ]
        sites.append(ActivationSite("norm.output", "norm", "batch"))
        super().__init__(sites, channels, 128)

    def validate(self, batch):
        super().validate(batch)
        n = batch.data.shape[2] * batch.data.shape[3]
        if n % self.patch_size or not 1 <= n // self.patch_size <= self.max_patches:
            raise ValidationError(
                "Input must contain complete ST-EEGFormer patches within time vocabulary"
            )

    def forward(self, model, batch, **kwargs):
        if kwargs:
            raise ValidationError("ST-EEGFormer adapter exposes native forward_features only")
        if model.global_pool or model.patch_embed.p != self.patch_size:
            raise ValidationError(
                "ST-EEGFormer execution configuration changed after adapter creation"
            )
        indices = torch.tensor(self.channel_indices, device=batch.data.device).expand(
            batch.data.shape[0], -1
        )
        return model.forward_features(batch.data.flatten(2), indices)

    def metadata(self):
        return {
            "native_method": "forward_features",
            "token_order": "time-major",
            "channel_indices": self.channel_indices,
            "native_patch_size": self.patch_size,
        }
