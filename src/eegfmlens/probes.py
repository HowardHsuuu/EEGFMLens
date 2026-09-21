"""Leakage-explicit linear probes and representation subspaces."""

from dataclasses import dataclass
from typing import Literal, Mapping

import torch

from .errors import ValidationError
from .types import Activation

Pooling = Literal["mean", "flatten", "cls"]


def categorical_targets(
    labels: tuple[str, ...], *, dtype=torch.float32, device=None
) -> torch.Tensor:
    """Deterministically one-hot encode string labels in first-seen order."""

    if (
        not isinstance(labels, tuple)
        or not labels
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise ValidationError("Labels must be a nonempty tuple of nonempty strings")
    classes = tuple(dict.fromkeys(labels))
    return torch.tensor(
        [[float(label == category) for category in classes] for label in labels],
        dtype=dtype,
        device=device,
    )


def activation_matrix(
    activation: Activation | torch.Tensor, *, pooling: Pooling = "mean"
) -> torch.Tensor:
    """Convert a cached activation to an explicit ``[trial, feature]`` matrix.

    Mean pooling retains the final feature axis and averages every intervening
    position.  Flattening treats all non-batch coordinates as predictors.  CLS
    pooling is accepted only for cached ``tokens`` layouts.
    """

    tensor = activation.tensor if isinstance(activation, Activation) else activation
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or not tensor.is_floating_point():
        raise ValidationError("Activation must be a floating tensor with batch and feature axes")
    if pooling == "flatten":
        matrix = tensor.reshape(tensor.shape[0], -1)
    elif pooling == "mean":
        matrix = tensor.reshape(tensor.shape[0], -1, tensor.shape[-1]).mean(dim=1)
    elif pooling == "cls":
        if (
            not isinstance(activation, Activation)
            or activation.layout != "tokens"
            or tensor.ndim != 3
        ):
            raise ValidationError(
                "CLS pooling requires a cached [batch, token, feature] tokens site"
            )
        matrix = tensor[:, 0]
    else:
        raise ValidationError("Pooling must be mean, flatten or cls")
    if not torch.isfinite(matrix).all():
        raise ValidationError("Activation matrix contains non-finite values")
    return matrix


@dataclass(frozen=True)
class RidgeProbe:
    """A fitted multi-output linear ridge readout."""

    weight: torch.Tensor
    intercept: torch.Tensor
    target_was_vector: bool
    alpha: float

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        _matrix(features, "features", columns=self.weight.shape[0])
        if features.device != self.weight.device or features.dtype != self.weight.dtype:
            raise ValidationError("Probe features must match fitted device and dtype")
        prediction = features @ self.weight + self.intercept
        return prediction[:, 0] if self.target_was_vector else prediction


@dataclass(frozen=True)
class ProbeScore:
    """Per-target and uniformly averaged coefficient of determination."""

    per_target_r2: torch.Tensor
    mean_r2: float


@dataclass(frozen=True)
class LayerProbeResult:
    site: str
    feature_count: int
    train: ProbeScore
    test: ProbeScore


def _matrix(value: torch.Tensor, name: str, *, columns: int | None = None) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or not value.is_floating_point()
        or value.shape[0] < 1
        or value.shape[1] < 1
        or not torch.isfinite(value).all()
    ):
        raise ValidationError(f"{name} must be a finite floating [sample, feature] tensor")
    if columns is not None and value.shape[1] != columns:
        raise ValidationError(f"{name} feature count differs from fitted probe")
    return value


def fit_ridge_probe(
    features: torch.Tensor, targets: torch.Tensor, *, alpha: float = 1.0
) -> RidgeProbe:
    """Fit a centered linear probe from training rows only.

    Callers split trials or subjects before calling this function.  The dual
    solution is used when there are more features than training examples.
    """

    features = _matrix(features, "features")
    vector = isinstance(targets, torch.Tensor) and targets.ndim == 1
    if vector:
        targets = targets[:, None]
    targets = _matrix(targets, "targets")
    if targets.shape[0] != features.shape[0]:
        raise ValidationError("Probe features and targets must have the same number of rows")
    alpha_tensor = torch.tensor(alpha, dtype=torch.float64)
    if not bool(torch.isfinite(alpha_tensor)) or alpha <= 0:
        raise ValidationError("Ridge alpha must be finite and positive")
    if targets.device != features.device or targets.dtype != features.dtype:
        raise ValidationError("Probe features and targets must share device and dtype")
    x_mean, y_mean = features.mean(0), targets.mean(0)
    x, y = features - x_mean, targets - y_mean
    samples, dimensions = x.shape
    if dimensions <= samples:
        gram = x.T @ x
        weight = torch.linalg.solve(
            gram + alpha * torch.eye(dimensions, device=x.device, dtype=x.dtype), x.T @ y
        )
    else:
        gram = x @ x.T
        weight = x.T @ torch.linalg.solve(
            gram + alpha * torch.eye(samples, device=x.device, dtype=x.dtype), y
        )
    return RidgeProbe(weight, y_mean - x_mean @ weight, vector, float(alpha))


def r2_score(targets: torch.Tensor, predictions: torch.Tensor) -> ProbeScore:
    """Score held-out rows; constant targets are rejected rather than hidden."""

    if targets.ndim == 1:
        targets = targets[:, None]
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    targets = _matrix(targets, "targets")
    predictions = _matrix(predictions, "predictions")
    if targets.shape != predictions.shape:
        raise ValidationError("Targets and predictions must have identical shape")
    total = (targets - targets.mean(0)).square().sum(0)
    if bool((total <= torch.finfo(total.dtype).eps).any()):
        raise ValidationError("R2 is undefined for a constant target")
    residual = (targets - predictions).square().sum(0)
    scores = 1 - residual / total
    return ProbeScore(scores, float(scores.mean()))


def _split_indices(mask: torch.Tensor, rows: int, name: str) -> torch.Tensor:
    if (
        not isinstance(mask, torch.Tensor)
        or mask.ndim != 1
        or mask.shape[0] != rows
        or mask.dtype != torch.bool
    ):
        raise ValidationError(f"{name} must be a boolean mask with one value per trial")
    indices = mask.nonzero(as_tuple=False).flatten()
    if indices.numel() < 2:
        raise ValidationError(f"{name} must select at least two trials")
    return indices


def layerwise_ridge_probe(
    activations: Mapping[str, Activation | torch.Tensor],
    targets: torch.Tensor,
    *,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    pooling: Pooling = "mean",
    alpha: float = 1.0,
) -> tuple[LayerProbeResult, ...]:
    """Fit the same declared split independently at every activation site."""

    if not activations:
        raise ValidationError("At least one activation site is required")
    first = next(iter(activations.values()))
    rows = first.tensor.shape[0] if isinstance(first, Activation) else first.shape[0]
    train = _split_indices(train_mask, rows, "train_mask")
    test = _split_indices(test_mask, rows, "test_mask")
    if bool((train_mask & test_mask).any()):
        raise ValidationError("Train and test masks must be disjoint")
    if targets.shape[0] != rows:
        raise ValidationError("Targets must contain one row per trial")
    results = []
    for site, activation in activations.items():
        matrix = activation_matrix(activation, pooling=pooling)
        if matrix.shape[0] != rows:
            raise ValidationError("Every activation site must contain the same trials")
        probe = fit_ridge_probe(matrix[train], targets[train], alpha=alpha)
        results.append(
            LayerProbeResult(
                site,
                matrix.shape[1],
                r2_score(targets[train], probe.predict(matrix[train])),
                r2_score(targets[test], probe.predict(matrix[test])),
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class CrossCovarianceSubspace:
    """Euclidean feature directions covarying with training-set concepts.

    This is the simplified cross-covariance/SVD construction used by BrainPEC;
    it is not covariance-whitened LEACE.
    """

    basis: torch.Tensor
    center: torch.Tensor
    singular_values: torch.Tensor
    concept_center: torch.Tensor


def fit_cross_covariance_subspace(
    features: torch.Tensor,
    concepts: torch.Tensor,
    *,
    rank: int | None = None,
    tolerance: float | None = None,
) -> CrossCovarianceSubspace:
    """Fit concept-associated orthonormal directions on training rows only."""

    features = _matrix(features, "features")
    if concepts.ndim == 1:
        concepts = concepts[:, None]
    concepts = _matrix(concepts, "concepts")
    if features.shape[0] != concepts.shape[0]:
        raise ValidationError("Features and concepts must have the same number of rows")
    if concepts.device != features.device or concepts.dtype != features.dtype:
        raise ValidationError("Features and concepts must share device and dtype")
    x_center, y_center = features.mean(0), concepts.mean(0)
    cross_covariance = (features - x_center).T @ (concepts - y_center) / features.shape[0]
    u, singular_values, _ = torch.linalg.svd(cross_covariance, full_matrices=False)
    threshold = (
        max(cross_covariance.shape)
        * torch.finfo(singular_values.dtype).eps
        * float(singular_values.max())
        if tolerance is None
        else tolerance
    )
    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise ValidationError("Subspace tolerance must be nonnegative")
    available = int((singular_values > threshold).sum())
    selected = available if rank is None else rank
    if type(selected) is not int or selected < 1 or selected > available:
        raise ValidationError("Requested subspace rank exceeds nonzero cross-covariance rank")
    return CrossCovarianceSubspace(u[:, :selected], x_center, singular_values[:selected], y_center)
