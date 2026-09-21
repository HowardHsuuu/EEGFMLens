"""Descriptive diagnostics for asking what cached representations organize."""

from dataclasses import dataclass

import torch

from .errors import ValidationError


def _embedding_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or not embeddings.is_floating_point()
        or embeddings.shape[0] < 2
        or embeddings.shape[1] < 1
        or not torch.isfinite(embeddings).all()
    ):
        raise ValidationError("Embeddings must be a finite floating [trial, feature] tensor")
    return embeddings


def _groups(values: tuple[str, ...], rows: int, name: str) -> tuple[str, ...]:
    if (
        not isinstance(values, tuple)
        or len(values) != rows
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise ValidationError(f"{name} must have one nonempty string per trial")
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class GroupVariance:
    """Exact between/within decomposition around the global representation mean."""

    total_sum_squares: float
    between_group_sum_squares: float
    within_group_sum_squares: float
    between_group_fraction: float
    group_count: int


def group_variance_decomposition(
    embeddings: torch.Tensor, group_ids: tuple[str, ...]
) -> GroupVariance:
    """Measure how much representation variance is associated with group means.

    This descriptive ANOVA identity is commonly useful for subject-identity
    audits.  It does not establish decodability, independence, or causality.
    """

    embeddings = _embedding_matrix(embeddings)
    groups = _groups(group_ids, embeddings.shape[0], "group_ids")
    if len(groups) < 2:
        raise ValidationError("Variance decomposition requires at least two groups")
    grand_mean = embeddings.mean(0)
    total = (embeddings - grand_mean).square().sum()
    if float(total) <= torch.finfo(total.dtype).eps:
        raise ValidationError("Variance decomposition is undefined for constant embeddings")
    between = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    within = torch.zeros_like(between)
    for group in groups:
        indices = torch.tensor(
            [value == group for value in group_ids],
            dtype=torch.bool,
            device=embeddings.device,
        )
        subset = embeddings[indices]
        group_mean = subset.mean(0)
        between += subset.shape[0] * (group_mean - grand_mean).square().sum()
        within += (subset - group_mean).square().sum()
    return GroupVariance(
        float(total),
        float(between),
        float(within),
        float(between / total),
        len(groups),
    )


@dataclass(frozen=True)
class ContrastConsistency:
    """Agreement of a binary condition contrast across groups."""

    group_ids: tuple[str, ...]
    contrasts: torch.Tensor
    mean_pairwise_cosine: float
    contrast_signal_to_noise: float


def within_group_contrast_consistency(
    embeddings: torch.Tensor,
    group_ids: tuple[str, ...],
    labels: torch.Tensor,
) -> ContrastConsistency:
    """Compare condition-1 minus condition-0 directions across groups.

    Every retained group must contain both binary labels.  The reported SNR is
    ``||mean contrast|| / RMS(||contrast_i - mean contrast||)``; it is descriptive
    and should be paired with subject-level uncertainty in a scientific study.
    """

    embeddings = _embedding_matrix(embeddings)
    groups = _groups(group_ids, embeddings.shape[0], "group_ids")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or labels.shape[0] != embeddings.shape[0]
    ):
        raise ValidationError("Labels must be a one-dimensional tensor with one value per trial")
    if labels.device != embeddings.device or not torch.isfinite(labels).all():
        raise ValidationError("Labels must be finite and on the embedding device")
    unique = torch.unique(labels)
    if unique.numel() != 2:
        raise ValidationError("Contrast consistency requires exactly two label values")
    low, high = unique[0], unique[1]
    contrasts, retained = [], []
    for group in groups:
        members = torch.tensor(
            [value == group for value in group_ids],
            dtype=torch.bool,
            device=embeddings.device,
        )
        if not bool((members & (labels == low)).any()) or not bool(
            (members & (labels == high)).any()
        ):
            raise ValidationError("Every group must contain both contrast labels")
        contrasts.append(
            embeddings[members & (labels == high)].mean(0)
            - embeddings[members & (labels == low)].mean(0)
        )
        retained.append(group)
    matrix = torch.stack(contrasts)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if bool((norms <= torch.finfo(norms.dtype).eps).any()):
        raise ValidationError("A group has a zero condition contrast")
    normalized = matrix / norms[:, None]
    similarities = normalized @ normalized.T
    pair_mask = torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
    if not bool(pair_mask.any()):
        raise ValidationError("Contrast consistency requires at least two groups")
    mean = matrix.mean(0)
    noise = torch.sqrt((matrix - mean).square().sum(1).mean())
    snr = torch.linalg.vector_norm(mean) / noise.clamp_min(torch.finfo(noise.dtype).eps)
    return ContrastConsistency(
        tuple(retained), matrix, float(similarities[pair_mask].mean()), float(snr)
    )
