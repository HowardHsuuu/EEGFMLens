"""Explicit continuous-input contracts for externally loaded BIOT/BENDR encoders."""

from ..errors import ValidationError
from ..types import ActivationSite
from .base import Adapter


class _ContinuousAdapter(Adapter):
    def __init__(self, sites, channels, sampling_rate):
        super().__init__(sites)
        self.channels = tuple(channels)
        self.sampling_rate = sampling_rate
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValidationError("Declare unique ordered input channel names")

    def validate(self, batch):
        if batch.channels != self.channels or batch.sampling_rate != self.sampling_rate:
            raise ValidationError(
                "Input channels/order or sampling rate differ from adapter contract"
            )
        if batch.stride != batch.data.shape[-1]:
            raise ValidationError("Continuous adapters require consecutive nonoverlapping patches")

    def forward(self, model, batch, **kwargs):
        if kwargs:
            raise ValidationError("Continuous encoder adapter does not accept execution kwargs")
        return model(batch.data.flatten(2))


class BIOTAdapter(_ContinuousAdapter):
    """Official BIOTEncoder with an explicit checkpoint channel vocabulary.

    STFT token spacing differs from SignalBatch patches, so sites use ``batch``
    layout. Shared per-channel embedding modules are not registered as sites.
    ``channels`` must describe the checkpoint's complete ordered vocabulary.
    """

    def __init__(self, model, *, channels):
        if len(channels) != model.channel_tokens.num_embeddings:
            raise ValidationError("Channel vocabulary must match BIOT checkpoint embeddings")
        sites = [ActivationSite("transformer.output", "transformer", "batch")]
        for i in range(len(model.transformer.layers.layers)):
            for j, branch in enumerate(["attention", "feedforward"]):
                sites.append(
                    ActivationSite(
                        f"blocks.{i}.{branch}.output", f"transformer.layers.layers.{i}.{j}", "batch"
                    )
                )
        super().__init__(sites, channels, 200)
        self.n_fft = model.n_fft

    def validate(self, batch):
        super().validate(batch)
        if batch.data.shape[2] * batch.data.shape[3] < self.n_fft:
            raise ValidationError("Input is shorter than BIOT STFT window")


class BENDREncoderAdapter(_ContinuousAdapter):
    """Official ConvEncoderBENDR only, excluding BENDRContextualizer.

    Declare the actual ordered model input, including any auxiliary scale input.
    This adapter does not implement DN3 channel mapping or normalization.
    """

    def __init__(self, model, *, channels):
        if len(channels) != model.in_features:
            raise ValidationError("Input channel vocabulary must match BENDR encoder")
        sites = [
            ActivationSite(f"encoder.{name}.output", f"encoder.{name}", "batch")
            for name, _ in model.encoder.named_children()
        ]
        super().__init__(sites, channels, 256)

    def expose(self, tensor, site, batch):
        if tensor.ndim != 3:
            raise ValidationError("Expected BENDR convolution output [batch, features, time]")
        return super().expose(tensor.transpose(1, 2), site, batch)

    def restore(self, tensor, site, native_shape):
        return tensor.transpose(1, 2).contiguous().reshape(native_shape)
