"""LaBraM native encoder sites, preserving CLS and channel-major token order."""

from ..errors import ValidationError
from ..types import ActivationSite, SignalBatch
from .base import Adapter
from .labram_channels import CHANNELS


class LaBraMAdapter(Adapter):
    def __init__(self, model, *, output="patch_tokens"):
        if output not in {"patch_tokens", "all_tokens", "pooled"}:
            raise ValidationError("output must be patch_tokens, all_tokens or pooled")
        self.output = output
        self.max_patches = model.time_embed.shape[1]
        self.max_channel_index = model.pos_embed.shape[1] - 1
        sites = [ActivationSite("embedding.output", "patch_embed", "patch_tokens")]
        for i in range(len(model.blocks)):
            prefix = f"blocks.{i}"
            sites.extend(
                [
                    ActivationSite(prefix + ".output", prefix, "tokens"),
                    ActivationSite(prefix + ".attention.output", prefix + ".attn", "tokens"),
                    ActivationSite(prefix + ".mlp.output", prefix + ".mlp", "tokens"),
                ]
            )
        super().__init__(sites)

    def channel_indices(self, batch):
        try:
            indices = [CHANNELS.index(c) + 1 for c in batch.channels]
        except ValueError as e:
            raise ValidationError(
                "Use exact LaBraM channel names; aliases require explicit mapping"
            ) from e
        if max(indices) > self.max_channel_index:
            raise ValidationError("Channel embedding index exceeds checkpoint capacity")
        return [0, *indices]

    def validate(self, batch: SignalBatch):
        if batch.data.shape[-1] != 200 or batch.sampling_rate != 200:
            raise ValidationError("LaBraM requires 200-sample patches at 200 Hz")
        if batch.data.shape[2] > self.max_patches:
            raise ValidationError("Too many patches for the native time embedding")
        self.channel_indices(batch)

    def forward(self, model, batch, **kwargs):
        if kwargs:
            raise ValidationError("LaBraM execution options are set on the adapter, not per run")
        return model.forward_features(
            batch.data,
            input_chans=self.channel_indices(batch),
            return_patch_tokens=self.output == "patch_tokens",
            return_all_tokens=self.output == "all_tokens",
        )
