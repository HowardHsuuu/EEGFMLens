"""Native Braindecode SignalJEPA encoder with its configured channel embeddings."""

import torch

from ..errors import ValidationError
from ..types import ActivationSite
from .continuous import _ContinuousAdapter


class SignalJEPAAdapter(_ContinuousAdapter):
    """Observe local features and contextual blocks, excluding the unused decoder.

    Convolutional output time points do not coincide with SignalBatch patches;
    caches therefore use batch layout rather than claiming patch selections.
    """

    def __init__(self, model):
        channels = tuple(info["ch_name"] for info in model.chs_info)
        sites = [ActivationSite("local.output", "feature_encoder", "batch")]
        sites.extend(
            ActivationSite(f"blocks.{i}.output", f"transformer.encoder.layers.{i}", "batch")
            for i in range(len(model.transformer.encoder.layers))
        )
        sites.append(ActivationSite("encoder.output", "transformer.encoder", "batch"))
        super().__init__(sites, channels, model.sfreq)
        self.channel_indices = model.pos_encoder.default_ch_idxs.detach().cpu().clone()
        self.conv_spec = tuple(tuple(s) for s in model.feature_encoder.conv_layers_spec)

    def validate(self, batch):
        super().validate(batch)
        n = batch.data.shape[2] * batch.data.shape[3]
        for _, width, stride in self.conv_spec:
            if n < width:
                raise ValidationError(
                    "Input is shorter than SignalJEPA convolution receptive field"
                )
            n = (n - width) // stride + 1

    def forward(self, model, batch, **kwargs):
        if not torch.equal(model.pos_encoder.default_ch_idxs.cpu(), self.channel_indices):
            raise ValidationError(
                "SignalJEPA channel embedding mapping changed after adapter creation"
            )
        return super().forward(model, batch, **kwargs)

    def metadata(self):
        return {
            "channel_indices": self.channel_indices.tolist(),
            "native_path": "SignalJEPA.forward (local features plus contextual encoder)",
        }
