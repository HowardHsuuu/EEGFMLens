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


class EEGPTMaskedReconstructionAdapter(Adapter):
    """Run the native encoder, predictor and reconstructor with explicit masks.

    The predictor's combined states are indexed by physical time patch, then
    by learned summary token. Summary tokens have no sensor identity, so the
    package permits patch selection there but deliberately rejects sensors.
    """

    def __init__(self, model):
        self.num_patches = tuple(model.encoder.num_patches)
        self.patch_size = model.encoder.patch_embed.patch_size
        self.summary_tokens = model.encoder.embed_num
        if model.encoder.patch_embed.patch_stride not in (None, self.patch_size):
            raise ValidationError("Overlapping EEGPT tokenization is unsupported")
        super().__init__([
            ActivationSite("encoder.output", "encoder", "batch"),
            ActivationSite("predictor.combined", "predictor", "bpfd", tensor_index=1),
            ActivationSite("reconstruction.output", "reconstructor", "batch"),
        ])

    def validate(self, batch):
        if (
            batch.sampling_rate != 256
            or batch.data.shape[1:3] != self.num_patches
            or batch.data.shape[-1] != self.patch_size
            or batch.stride != self.patch_size
        ):
            raise ValidationError("EEGPT masked reconstruction requires native 256 Hz geometry")

    def forward(self, model, batch, *, mask_x, mask_y):
        self.validate(batch)
        c, p = self.num_patches
        mx = torch.as_tensor(mask_x, dtype=torch.long, device=batch.data.device)
        my = torch.as_tensor(mask_y, dtype=torch.long, device=batch.data.device)
        if (
            mx.ndim != 2 or mx.shape[0] != p - 1 or mx.shape[1] != c - 1
            or my.ndim != 1 or my.numel() == 0
            or mx.numel() != mx.unique().numel()
            or my.numel() != my.unique().numel()
            or (mx < 0).any() or (mx >= c * p).any()
            or (my < 0).any() or (my >= c * p).any()
            or torch.isin(my, mx.flatten()).any()
        ):
            raise ValidationError("EEGPT masks must be unique, disjoint native physical tokens")
        observed_times = mx[:, 0] // c
        if (
            not torch.equal(mx // c, observed_times[:, None].expand_as(mx))
            or not torch.all(observed_times[1:] > observed_times[:-1])
            or not torch.all(mx[:, 1:] > mx[:, :-1])
        ):
            raise ValidationError("EEGPT visible tokens must form ordered physical time rows")
        try:
            channel_ids = model.encoder.prepare_chan_ids(batch.channels)
        except (AssertionError, KeyError, ValueError) as exc:
            raise ValidationError("EEGPT channel name is absent from the native vocabulary") from exc
        if (
            not isinstance(channel_ids, torch.Tensor)
            or channel_ids.dtype != torch.long
            or channel_ids.shape != (1, c)
            or channel_ids.unique().numel() != c
        ):
            raise ValidationError("EEGPT native channel lookup differs")
        channel_ids = channel_ids.to(batch.data.device)
        encoded = model.encoder(batch.data.flatten(2), channel_ids, mask_x=mx)
        _, combined = model.predictor(encoded, mask_x=mx)
        return model.reconstructor(combined, channel_ids, mask_y=my)

    def expose(self, tensor, site, batch):
        if site.name == "predictor.combined" and (
            tensor.ndim != 4
            or tensor.shape[:3] != (batch.data.shape[0], self.num_patches[1], self.summary_tokens)
        ):
            raise ValidationError("Unexpected EEGPT predictor time/summary geometry")
        return super().expose(tensor, site, batch)

    def metadata(self):
        return {
            "predictor_combined_layout": "batch/time_patch/summary_token/feature",
            "summary_tokens_are_not_sensors": True,
            "reconstruction_target": "native per-patch normalized waveform",
        }
