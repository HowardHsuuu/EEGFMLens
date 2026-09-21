"""NeuroRVQ's four native frequency branches with invocation-specific hooks."""

import torch

from ..errors import ValidationError
from ..types import ActivationSite
from .base import Adapter


class NeuroRVQAdapter(Adapter):
    def __init__(self, model, *, channel_vocabulary):
        self.vocabulary = tuple(str(c).lower() for c in channel_vocabulary)
        if (
            len(set(self.vocabulary)) != len(self.vocabulary)
            or len(self.vocabulary) + 1 != model.pos_embed.shape[0]
        ):
            raise ValidationError("NeuroRVQ vocabulary must match pretrained spatial embeddings")
        self.max_patches = model.time_embed.shape[0]
        sites = []
        for branch in range(4):
            for i in range(len(model.blocks)):
                sites.append(
                    ActivationSite(
                        f"branches.{branch}.blocks.{i}.output",
                        f"blocks.{i}",
                        "tokens",
                        call_index=branch,
                        expected_calls=4,
                    )
                )
            sites.append(
                ActivationSite(
                    f"branches.{branch}.output",
                    "norm",
                    "tokens",
                    call_index=branch,
                    expected_calls=4,
                )
            )
        super().__init__(sites)

    def validate(self, batch):
        if len({c.lower() for c in batch.channels}) != len(batch.channels):
            raise ValidationError("NeuroRVQ channel names must map to distinct native electrodes")
        if batch.sampling_rate != 200 or batch.data.shape[-1] != 200 or batch.stride != 200:
            raise ValidationError("NeuroRVQ requires contiguous 200-sample patches at 200 Hz")
        if batch.data.shape[2] > self.max_patches or any(
            c.lower() not in self.vocabulary for c in batch.channels
        ):
            raise ValidationError("Unsupported temporal extent or channel vocabulary")

    def indices(self, batch):
        b, c, p, _ = batch.data.shape
        time = (
            torch.arange(self.max_patches - p, self.max_patches, device=batch.data.device)
            .repeat(c)
            .expand(b, -1)
        )
        space = (
            torch.tensor(
                [self.vocabulary.index(n.lower()) for n in batch.channels], device=batch.data.device
            )
            .repeat_interleave(p)
            .expand(b, -1)
        )
        return time, space

    def forward(self, model, batch, **kwargs):
        if kwargs:
            raise ValidationError("NeuroRVQ adapter uses the unmasked four-branch encoder path")
        time, space = self.indices(batch)
        outputs = model(batch.data, time, space, return_patch_tokens=True)
        return torch.stack(outputs[:4], dim=2)

    def metadata(self):
        return {
            "channel_vocabulary": self.vocabulary,
            "time_alignment": "right",
            "branches": 4,
            "output": "stacked branch patch features",
        }
