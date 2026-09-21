"""Leakage-explicit comparison of representations across EEG models."""

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch

from .errors import ValidationError
from .probes import Pooling, activation_matrix
from .types import Activation

RSAMetric = Literal["correlation", "cosine", "euclidean"]


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[0] < 2
        or value.shape[1] < 1
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValidationError(f"{name} must be a finite floating [trial, feature] tensor")
    return value


def _paired(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left = _matrix(left, "left")
    right = _matrix(right, "right")
    if left.shape[0] != right.shape[0]:
        raise ValidationError("Representations must contain the same paired trials")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValidationError("Representations must share device and dtype")
    return left, right


def center_within_groups(features: torch.Tensor, groups: Sequence[str]) -> torch.Tensor:
    """Subtract each declared group's mean without dropping or reordering trials."""

    features = _matrix(features, "features")
    if (
        isinstance(groups, (str, bytes))
        or len(groups) != features.shape[0]
        or any(not isinstance(group, str) or not group for group in groups)
    ):
        raise ValidationError("Groups must provide one nonempty string per trial")
    result = torch.empty_like(features)
    for group in dict.fromkeys(groups):
        indices = torch.tensor(
            [index for index, value in enumerate(groups) if value == group],
            device=features.device,
        )
        selected = features.index_select(0, indices)
        result.index_copy_(0, indices, selected - selected.mean(dim=0, keepdim=True))
    return result


def linear_cka(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    groups: Sequence[str] | None = None,
) -> float:
    """Compute linear centered-kernel alignment over paired trials.

    Supplying ``groups`` first removes each group mean from both representations.
    For EEG-FM comparisons this can separate shared within-subject geometry from
    similarity driven only by subject identity.
    """

    left, right = _paired(left, right)
    if groups is not None:
        left = center_within_groups(left, groups)
        right = center_within_groups(right, groups)
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    left_gram = left @ left.T
    right_gram = right @ right.T
    left_norm = torch.linalg.vector_norm(left_gram)
    right_norm = torch.linalg.vector_norm(right_gram)
    denominator = left_norm * right_norm
    if float(denominator) <= torch.finfo(denominator.dtype).eps:
        raise ValidationError("CKA is undefined for a constant centered representation")
    value = (left_gram * right_gram).sum() / denominator
    tolerance = 64 * torch.finfo(value.dtype).eps
    if float(value) < -tolerance or float(value) > 1 + tolerance:
        raise ValidationError("Numerical CKA result lies outside [0, 1]")
    return float(value.clamp(0, 1))


def rsa_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    metric: RSAMetric = "correlation",
    groups: Sequence[str] | None = None,
) -> float:
    """Pearson correlation between upper triangles of two trial-similarity matrices."""

    left, right = _paired(left, right)
    if left.shape[0] < 3:
        raise ValidationError("RSA requires at least three paired trials")
    if groups is not None:
        left = center_within_groups(left, groups)
        right = center_within_groups(right, groups)

    def similarities(values: torch.Tensor) -> torch.Tensor:
        if metric == "correlation":
            values = values - values.mean(dim=1, keepdim=True)
        if metric in {"correlation", "cosine"}:
            norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
            if bool((norms <= torch.finfo(norms.dtype).eps).any()):
                raise ValidationError(f"RSA {metric} similarity is undefined for a zero row")
            return (values / norms) @ (values / norms).T
        if metric == "euclidean":
            return -torch.cdist(values, values)
        raise ValidationError("RSA metric must be correlation, cosine or euclidean")

    rows, columns = torch.triu_indices(left.shape[0], left.shape[0], offset=1, device=left.device)
    left_values = similarities(left)[rows, columns]
    right_values = similarities(right)[rows, columns]
    left_values = left_values - left_values.mean()
    right_values = right_values - right_values.mean()
    denominator = torch.linalg.vector_norm(left_values) * torch.linalg.vector_norm(right_values)
    if float(denominator) <= torch.finfo(denominator.dtype).eps:
        raise ValidationError("RSA is undefined for a constant similarity structure")
    return float((left_values * right_values).sum() / denominator)


@dataclass(frozen=True)
class CrossModelSimilarity:
    """Layer-by-layer similarities over one exactly matched trial cohort."""

    left_sites: tuple[str, ...]
    right_sites: tuple[str, ...]
    values: torch.Tensor
    method: str
    trial_ids: tuple[str, ...]
    pooling: Pooling
    group_centered: bool
    left_model_id: str
    right_model_id: str


def _validate_activation_collection(
    activations: Mapping[str, Activation], name: str
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    if not activations:
        raise ValidationError(f"{name} activations cannot be empty")
    sites = tuple(activations)
    if any(not isinstance(site, str) or not site for site in sites):
        raise ValidationError(f"{name} activation names must be nonempty strings")
    first = next(iter(activations.values()))
    if not isinstance(first, Activation):
        raise ValidationError("Cross-model comparison requires Activation records")
    trial_ids, model_id = first.trial_ids, first.model_id
    if (
        len(trial_ids) != first.tensor.shape[0]
        or len(set(trial_ids)) != len(trial_ids)
        or any(not isinstance(trial, str) or not trial for trial in trial_ids)
    ):
        raise ValidationError(f"{name} trial IDs must be unique and match the batch axis")
    for site, activation in activations.items():
        if not isinstance(activation, Activation):
            raise ValidationError("Cross-model comparison requires Activation records")
        if activation.trial_ids != trial_ids:
            raise ValidationError(f"{name} site {site!r} has a different trial order")
        if activation.model_id != model_id:
            raise ValidationError(f"{name} activations mix model identities")
        if activation.site != site:
            raise ValidationError(f"{name} mapping key {site!r} differs from its activation site")
        if activation.tensor.shape[0] != len(trial_ids):
            raise ValidationError(f"{name} site {site!r} has a mismatched batch axis")
    return sites, trial_ids, model_id


def cross_model_similarity(
    left: Mapping[str, Activation],
    right: Mapping[str, Activation],
    *,
    method: Literal["linear_cka", "rsa"] = "linear_cka",
    pooling: Pooling = "mean",
    groups: Sequence[str] | None = None,
    rsa_metric: RSAMetric = "correlation",
) -> CrossModelSimilarity:
    """Compare every left/right site after exact trial-ID alignment.

    The left collection defines output trial order. The right collection may be
    reordered but must contain exactly the same unique trial IDs. No examples are
    silently intersected or discarded.
    """

    left_sites, trial_ids, left_model_id = _validate_activation_collection(left, "left")
    right_sites, right_trial_ids, right_model_id = _validate_activation_collection(right, "right")
    if set(trial_ids) != set(right_trial_ids):
        raise ValidationError("Cross-model activations must contain exactly the same trials")
    order = torch.tensor(
        [right_trial_ids.index(trial) for trial in trial_ids],
        device=next(iter(right.values())).tensor.device,
    )
    left_matrices = [activation_matrix(left[site], pooling=pooling) for site in left_sites]
    right_matrices = [
        activation_matrix(right[site], pooling=pooling).index_select(0, order)
        for site in right_sites
    ]
    reference = left_matrices[0]
    if any(
        matrix.device != reference.device or matrix.dtype != reference.dtype
        for matrix in [
            *left_matrices,
            *right_matrices,
        ]
    ):
        raise ValidationError("All compared activations must share device and dtype")
    values = reference.new_empty((len(left_sites), len(right_sites)))
    for left_index, left_matrix in enumerate(left_matrices):
        for right_index, right_matrix in enumerate(right_matrices):
            if method == "linear_cka":
                value = linear_cka(left_matrix, right_matrix, groups=groups)
            elif method == "rsa":
                value = rsa_correlation(
                    left_matrix,
                    right_matrix,
                    metric=rsa_metric,
                    groups=groups,
                )
            else:
                raise ValidationError("Similarity method must be linear_cka or rsa")
            values[left_index, right_index] = value
    return CrossModelSimilarity(
        left_sites,
        right_sites,
        values,
        method if method == "linear_cka" else f"rsa:{rsa_metric}",
        trial_ids,
        pooling,
        groups is not None,
        left_model_id,
        right_model_id,
    )
