"""Raw paired effects; no clipped or fabricated recovery scores."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedEffect:
    clean: float
    recipient: float
    patched: float
    delta: float
    recovery: float | None
    reason: str | None


def paired_effect(clean: float, recipient: float, patched: float, *, epsilon=1e-8):
    """For a prespecified higher-is-better scalar metric. Aggregate separately."""
    if epsilon <= 0 or not all(math.isfinite(x) for x in (clean, recipient, patched, epsilon)):
        raise ValueError("Expected finite metrics and a positive epsilon")
    delta = patched - recipient
    denominator = clean - recipient
    if denominator <= epsilon:
        return PairedEffect(clean, recipient, patched, delta, None, "no_meaningful_degradation")
    return PairedEffect(clean, recipient, patched, delta, delta / denominator, None)
