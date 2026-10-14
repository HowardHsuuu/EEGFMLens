"""Official ST-EEGFormer downstream encoder, preserving time-major token order."""

import torch

from ..errors import ValidationError
from ..types import ActivationSite
from .base import Adapter
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


class STEEGFormerMAEAdapter(Adapter):
    """Execute the native masked-autoencoder decoder and expose physical tokens.

    The upstream MAE uses time-major tokens. Only full-length decoder sites
    are reordered into EEGLens's channel-major ``tokens`` layout. Its encoder
    contains visible tokens only and remains in conservative ``batch`` layout.
    The caller supplies the pretrained MAE model and its channel vocabulary.
    """

    def __init__(self, model, *, channels, channel_mapping):
        try:
            indices = tuple(channel_mapping[channel] for channel in channels)
        except KeyError as error:
            raise ValidationError("Channel absent from the ST-EEGFormer MAE mapping") from error
        if not channels or len(set(channels)) != len(channels):
            raise ValidationError("Declare unique ordered ST-EEGFormer MAE channels")
        try:
            vocabulary_size = model.enc_channel_emd.channel_transformation.num_embeddings
            patch_size = model.patch_embed.p
            max_patches = min(
                model.enc_temporal_emd.pe.shape[1], model.dec_temporal_emd.pe.shape[1]
            ) - 1
            decoder_depth = len(model.decoder_blocks)
            encoder_depth = len(model.blocks)
            model.forward_demo
            model.decoder_pred
        except AttributeError as error:
            raise ValidationError("Expected the native ST-EEGFormer MAE model") from error
        if any(type(index) is not int or not 0 <= index < vocabulary_size for index in indices):
            raise ValidationError("Channel index is outside the MAE embedding table")
        if type(patch_size) is not int or patch_size <= 0 or max_patches <= 0:
            raise ValidationError("Invalid native MAE patch or temporal vocabulary")
        if not encoder_depth or not decoder_depth:
            raise ValidationError("Native MAE needs encoder and decoder blocks")
        self.channels = tuple(channels)
        self.channel_indices = indices
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.decoder_depth = decoder_depth
        sites = [ActivationSite("embedding.output", "patch_embed", "batch")]
        sites += [
            ActivationSite(f"encoder.{index}.output", f"blocks.{index}", "batch")
            for index in range(encoder_depth)
        ]
        for index in range(decoder_depth):
            prefix = f"decoder_blocks.{index}"
            sites.extend(
                (
                    ActivationSite(f"decoder.{index}.attention", prefix + ".attn.proj", "tokens"),
                    ActivationSite(f"decoder.{index}.native_attention", prefix + ".attn", "tokens"),
                    ActivationSite(f"decoder.{index}.mlp", prefix + ".mlp.fc2", "tokens"),
                    ActivationSite(f"decoder.{index}.output", prefix, "tokens"),
                )
            )
        sites.append(ActivationSite("projection.output", "decoder_pred", "tokens"))
        super().__init__(sites)

    def validate(self, batch):
        if batch.channels != self.channels or batch.sampling_rate != 128:
            raise ValidationError("ST-EEGFormer MAE requires declared channels at 128 Hz")
        if batch.data.shape[-1] != self.patch_size:
            raise ValidationError("ST-EEGFormer MAE patch width changed")
        if not 1 <= batch.data.shape[2] <= self.max_patches:
            raise ValidationError("Input exceeds the MAE temporal vocabulary")

    def expose(self, tensor, site, batch):
        if site.layout != "tokens":
            return super().expose(tensor, site, batch)
        batch_size, channels, patches, _ = batch.data.shape
        if tensor.ndim != 3 or tensor.shape[:2] != (batch_size, 1 + channels * patches):
            raise ValidationError(f"Unexpected full decoder token shape at {site.name}")
        values = tensor[:, 1:].reshape(batch_size, patches, channels, -1)
        physical = values.transpose(1, 2).reshape(batch_size, channels * patches, -1)
        return super().expose(torch.cat((tensor[:, :1], physical), dim=1), site, batch)

    def restore(self, tensor, site, native_shape):
        if site.layout != "tokens":
            return super().restore(tensor, site, native_shape)
        batch_size, token_count, width = native_shape
        channels = len(self.channels)
        if tensor.shape != native_shape or (token_count - 1) % channels:
            raise ValidationError(f"Invalid physical decoder token shape at {site.name}")
        patches = (token_count - 1) // channels
        values = tensor[:, 1:].reshape(batch_size, channels, patches, width)
        native = values.transpose(1, 2).reshape(batch_size, channels * patches, width)
        return torch.cat((tensor[:, :1], native), dim=1)

    def forward(self, model, batch, **kwargs):
        if set(kwargs) != {"mask"}:
            raise ValidationError("ST-EEGFormer MAE requires an explicit boolean mask")
        try:
            mask = torch.as_tensor(kwargs["mask"], device=batch.data.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValidationError("Invalid ST-EEGFormer MAE mask") from error
        if mask.dtype != torch.bool or mask.shape != batch.data.shape[:-1]:
            raise ValidationError("MAE mask must be boolean B,C,P in input coordinates")
        if (
            model.patch_embed.p != self.patch_size
            or len(model.decoder_blocks) != self.decoder_depth
        ):
            raise ValidationError("ST-EEGFormer MAE execution configuration changed")
        flat = mask.transpose(1, 2).flatten(1)
        visible = (~flat).sum(1)
        if not torch.all(visible == visible[0]) or not 0 < visible[0] < flat.shape[1]:
            raise ValidationError("MAE trials need equal nonzero visible and masked counts")
        shuffle = torch.argsort(flat.int(), dim=1, stable=True)
        keep = shuffle[:, :int(visible[0])]
        restore = torch.argsort(shuffle, dim=1)
        indices = torch.tensor(self.channel_indices, device=batch.data.device).expand(
            batch.data.shape[0], -1
        )
        native = model.forward_demo(
            batch.data.flatten(2), indices, ids_keep=keep,
            mask=flat.float(), ids_restore=restore,
        )
        prediction = native[1]
        batch_size, channels, patches, width = batch.data.shape
        if prediction.shape != (batch_size, channels * patches, width):
            raise ValidationError("Native MAE reconstruction shape changed")
        return prediction.reshape(batch_size, patches, channels, width).transpose(1, 2).contiguous()

    def metadata(self):
        return {
            "native_method": "forward_demo",
            "native_output": "masked-autoencoder reconstruction",
            "native_token_order": "time-major",
            "exposed_decoder_order": "channel-major with leading CLS",
            "channel_indices": self.channel_indices,
            "native_patch_size": self.patch_size,
            "mask_contract": "explicit boolean B,C,P",
        }
