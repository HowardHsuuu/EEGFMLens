"""One declared LaBraM pretraining mask arm, preserving the native prediction head."""

from dataclasses import replace

import torch

from ..errors import ValidationError
from ..types import ActivationSite
from .labram import LaBraMAdapter


class LaBraMPretrainingAdapter(LaBraMAdapter):
    """Return all patch logits from the native wrapper's outer ``lm_head``.

    ``mask`` is required, boolean [batch, sensor, patch], in SignalBatch order.
    This runs one mask arm. Call again with its complement to reproduce the
    wrapper's symmetric training arm. There is no new model or fitted readout.

    Native masking happens AFTER TemporalConv. Its GroupNorm may already mix
    information from masked raw samples into other patch embeddings. This API
    preserves that behavior; it does not claim raw-waveform target hiding.
    Any explicit raw occlusion belongs in a separately identified SignalBatch.
    """

    def __init__(self, model):
        if not hasattr(model, "student") or not isinstance(
            getattr(model, "lm_head", None), torch.nn.Module
        ):
            raise ValidationError(
                "LaBraM pretraining requires the native student and outer lm_head"
            )
        super().__init__(model.student, output="patch_tokens")
        self.sites = {
            name: replace(site, module_path="student." + site.module_path)
            for name, site in self.sites.items()
        }
        self.sites.update(
            {
                "normalization.output": ActivationSite(
                    "normalization.output", "student.norm", "tokens"
                ),
                "prediction.output": ActivationSite("prediction.output", "lm_head", "patch_tokens"),
            }
        )

    def forward(self, model, batch, **kwargs):
        if set(kwargs) != {"mask"}:
            raise ValidationError("LaBraM pretraining requires only an explicit physical mask")
        mask = torch.as_tensor(kwargs["mask"], device=batch.data.device)
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(batch.data.shape[:-1]):
            raise ValidationError("LaBraM mask must be boolean [batch, sensor, patch]")
        features = model.student(
            batch.data,
            input_chans=self.channel_indices(batch),
            bool_masked_pos=mask.flatten(1),
            return_all_patch_tokens=True,
        )
        # Native NeuralTransformerForMEM.forward uses this outer head, not the
        # separately stored student.lm_head. Keep CLS out of prediction output.
        return model.lm_head(features[:, 1:])

    def metadata(self):
        return {
            "output": "all_patch_code_logits",
            "prediction_head": "outer lm_head",
            "mask_coordinates": "SignalBatch channel-major sensor/patch",
            "native_mask_stage": "after TemporalConv embedding",
            "raw_target_hiding_guaranteed": False,
            "symmetric_arm": "separate execution with complementary mask",
            "tokenizer_and_labels": "not supplied or fitted by this adapter",
        }
