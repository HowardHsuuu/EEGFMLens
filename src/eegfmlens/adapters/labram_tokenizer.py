"""Native VQ-NSP codes with persistent evaluation state preserved."""

from dataclasses import replace
from typing import cast

import torch

from ..errors import ValidationError
from ..types import ActivationSite
from .labram import LaBraMAdapter


class LaBraMTokenizerAdapter(LaBraMAdapter):
    """Execute native ``get_codebook_indices`` without accumulating usage counts.

    The published quantizer updates ``quantize.cluster_size`` even in eval.
    That statistic does not enter the code assignment. A temporary clone receives
    the native update; the original buffer object is restored even on exceptions.
    No codebook weights, forward implementation or generic state guard is changed.
    The model must already have an initialized codebook and be in eval mode.

    Codes are contextual outputs, not independent labels of individual patches.
    Callers must explicitly choose and preserve the input used to generate labels.
    """

    def __init__(self, model):
        self._require_ready(model)
        super().__init__(model.encoder, output="patch_tokens")
        self.sites = {
            name: replace(site, module_path="encoder." + site.module_path)
            for name, site in self.sites.items()
        }
        self.sites["code_projection.output"] = ActivationSite(
            "code_projection.output", "encode_task_layer", "patch_tokens"
        )

    @staticmethod
    def _require_ready(model):
        quantizer = getattr(model, "quantize", None)
        embedding = getattr(quantizer, "embedding", None)
        initialized = getattr(embedding, "initted", None)
        if (
            not callable(getattr(model, "get_codebook_indices", None))
            or not hasattr(model, "encoder")
            or not isinstance(initialized, torch.Tensor)
            or initialized.numel() != 1
            or initialized.item() != 1
        ):
            raise ValidationError("Native VQ-NSP requires an initialized codebook")
        if any(module.training for module in model.modules()):
            raise ValidationError("Native VQ-NSP must be entirely in eval mode")
        counter = cast(torch.nn.Module, quantizer)._buffers.get("cluster_size")
        if (
            not isinstance(counter, torch.Tensor)
            or counter.shape != (getattr(quantizer, "num_tokens", None),)
            or not counter.is_floating_point()
        ):
            raise ValidationError("Native VQ-NSP requires its registered usage buffer")

    def forward(self, model, batch, **kwargs):
        if kwargs:
            raise ValidationError("Native VQ-NSP has no per-run execution options")
        self._require_ready(model)
        channels = self.channel_indices(batch)
        quantizer = model.quantize
        original = quantizer.cluster_size
        try:
            quantizer.cluster_size = original.clone()
            return model.get_codebook_indices(batch.data, input_chans=channels)
        finally:
            quantizer.cluster_size = original

    def metadata(self):
        return {
            "output": "native_contextual_code_indices",
            "token_order": "SignalBatch channel-major sensor/patch",
            "label_input": "supplied SignalBatch, not chosen by adapter",
            "usage_counter": "native update on temporary clone, original buffer restored",
            "codebook_initialization": "must already be initialized; no fitting",
        }
