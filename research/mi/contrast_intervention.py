"""Research-only minimum-norm edit of the time-averaged C4-C3 feature contrast."""

from dataclasses import replace

from eeglens import Selection, SubspaceAblation
from eeglens.errors import ValidationError


def contrast_donor(activation, batch, basis, center_difference):
    b, c, p, _ = batch.data.shape
    if activation.layout == "tokens":
        physical = activation.tensor[:, 1:].reshape(b, c, p, -1)
    elif activation.layout == "bcpd":
        physical = activation.tensor
    else:
        raise ValidationError("Contrast edit requires verified bcpd or CLS-token geometry")
    c3, c4 = batch.channels.index("C3"), batch.channels.index("C4")
    contrast = physical[:, c4].mean(1) - physical[:, c3].mean(1)
    # Reuse the public subspace validation and arithmetic on a batch-first view.
    residual = SubspaceAblation(activation.site, basis, center_difference, Selection()).apply(
        contrast, batch, "batch", activation.model_id
    )
    removed = contrast - residual
    values = activation.tensor.clone()
    edited = values[:, 1:].reshape(b, c, p, -1) if activation.layout == "tokens" else values
    edited[:, c3] += removed[:, None] / 2
    edited[:, c4] -= removed[:, None] / 2
    return replace(activation, tensor=values)
