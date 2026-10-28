"""Composition of the original BENDR encoder and contextualizer forwards."""

from ..errors import ValidationError
from ..types import ActivationSite
from .continuous import _ContinuousAdapter


class BENDRAdapter(_ContinuousAdapter):
    """Accept nn.Sequential(native_encoder, native_contextualizer).

    The start token and downsampled time axis are retained. No electrode/time
    selection is inferred from contextual feature positions.
    """

    def __init__(self, model, *, channels):
        try:
            if len(model) != 2:
                raise ValidationError("Expected exactly a BENDR encoder and contextualizer")
            encoder, context = model[0], model[1]
        except (TypeError, IndexError, KeyError) as exc:
            raise ValidationError("Expected an ordered BENDR encoder/contextualizer pair") from exc
        if (
            not hasattr(encoder, "in_features")
            or not hasattr(encoder, "encoder_h")
            or not hasattr(context, "in_features")
            or not hasattr(context, "transformer_layers")
        ):
            raise ValidationError(
                "Expected native BENDR encoder/contextualizer components in order"
            )
        if len(channels) != encoder.in_features:
            raise ValidationError(
                "Expected BENDR encoder/contextualizer and matching input channels"
            )
        if encoder.encoder_h != context.in_features:
            raise ValidationError("BENDR encoder and contextualizer feature dimensions differ")
        sites = [ActivationSite("encoder.output", "0", "batch")]
        sites.append(ActivationSite("context.input", "1.input_conditioning", "batch"))
        sites.extend(
            ActivationSite(f"context.blocks.{i}.output", f"1.transformer_layers.{i}", "batch")
            for i in range(len(context.transformer_layers))
        )
        sites.append(ActivationSite("context.output", "1.output_layer", "batch"))
        super().__init__(sites, channels, 256)

    @staticmethod
    def _sequence_first(site):
        return site.module_path == "1.input_conditioning" or site.module_path.startswith(
            "1.transformer_layers."
        )

    def expose(self, tensor, site, batch):
        if self._sequence_first(site):
            if tensor.ndim != 3:
                raise ValidationError("Expected sequence-first BENDR contextual activation")
            tensor = tensor.transpose(0, 1)
        else:
            if tensor.ndim != 3:
                raise ValidationError("Expected BENDR convolution output [batch, features, time]")
            tensor = tensor.transpose(1, 2)
        return super().expose(tensor, site, batch)

    def restore(self, tensor, site, native_shape):
        if site.module_path == "1.input_conditioning":
            # Native conditioning ends with B,D,T -> T,B,D. Preserve that
            # memory layout as well as values; kernels can depend on strides.
            return tensor.transpose(1, 2).contiguous().permute(2, 0, 1)
        if self._sequence_first(site):
            return tensor.transpose(0, 1).contiguous().reshape(native_shape)
        return tensor.transpose(1, 2).contiguous().reshape(native_shape)
