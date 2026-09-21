"""Held-out concept directions and directional-sensitivity tests."""

import math
from dataclasses import dataclass
from typing import Literal

import torch

from .attribution import AttributionResult
from .errors import ValidationError
from .probes import fit_ridge_probe
from .provenance import tensor_digest

ConceptSampleUnit = Literal["trial", "position"]


def _feature_matrix(features: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(features, torch.Tensor)
        or not features.is_floating_point()
        or features.ndim != 2
        or min(features.shape) < 1
        or not torch.isfinite(features).all()
    ):
        raise ValidationError("Concept features must be a finite floating [sample, feature] tensor")
    return features


def _binary_labels(labels: torch.Tensor, rows: int, device: torch.device) -> torch.Tensor:
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or labels.shape[0] != rows
        or labels.device != device
    ):
        raise ValidationError(
            "Concept labels must have one value per feature row on the same device"
        )
    if labels.dtype == torch.bool:
        return labels
    if labels.is_floating_point() and not torch.isfinite(labels).all():
        raise ValidationError("Concept labels contain non-finite values")
    if (
        labels.dtype == torch.bool
        or labels.is_floating_point()
        or labels.dtype
        in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
    ):
        if bool(((labels != 0) & (labels != 1)).any()):
            raise ValidationError("Concept labels must contain only zero and one")
        return labels.bool()
    raise ValidationError("Concept labels must be boolean or numeric zeros and ones")


def _split_mask(mask: torch.Tensor, rows: int, device: torch.device, name: str) -> torch.Tensor:
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 1
        or mask.shape[0] != rows
        or mask.device != device
    ):
        raise ValidationError(f"{name} must be a same-device boolean mask with one value per row")
    if int(mask.sum()) < 2:
        raise ValidationError(f"{name} must select at least two rows")
    return mask


@dataclass(frozen=True)
class ConceptDirection:
    """Unit ridge-classifier normal pointing toward a binary concept."""

    direction: torch.Tensor
    intercept: torch.Tensor
    alpha: float
    fit_rows: int
    validation_rows: int
    positive_fit_rows: int
    validation_balanced_accuracy: float

    def __post_init__(self):
        if (
            not isinstance(self.direction, torch.Tensor)
            or not isinstance(self.intercept, torch.Tensor)
            or not self.direction.is_floating_point()
            or not self.intercept.is_floating_point()
            or self.direction.ndim != 1
            or self.direction.numel() < 1
            or self.intercept.shape != ()
            or self.direction.device != self.intercept.device
            or self.direction.dtype != self.intercept.dtype
            or not torch.isfinite(self.direction).all()
            or not torch.isfinite(self.intercept)
        ):
            raise ValidationError("Concept-direction parameters are invalid")
        if not torch.allclose(
            torch.linalg.vector_norm(self.direction),
            self.direction.new_tensor(1.0),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValidationError("Concept direction must have unit norm")
        if type(self.alpha) not in {int, float} or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValidationError("Concept-direction alpha must be finite and positive")
        if (
            type(self.fit_rows) is not int
            or self.fit_rows < 2
            or type(self.validation_rows) is not int
            or self.validation_rows < 2
            or type(self.positive_fit_rows) is not int
            or not 0 < self.positive_fit_rows < self.fit_rows
        ):
            raise ValidationError("Concept-direction row counts are invalid")
        if not math.isfinite(self.validation_balanced_accuracy) or not (
            0 <= self.validation_balanced_accuracy <= 1
        ):
            raise ValidationError("Concept-direction validation score is invalid")

    @property
    def feature_count(self) -> int:
        return self.direction.numel()

    def decision_function(self, features: torch.Tensor) -> torch.Tensor:
        features = _feature_matrix(features)
        if features.shape[1] != self.feature_count:
            raise ValidationError("Concept features differ from the fitted feature dimension")
        if features.device != self.direction.device or features.dtype != self.direction.dtype:
            raise ValidationError("Concept features must match the fitted device and dtype")
        return features @ self.direction + self.intercept

    def provenance(self) -> dict[str, str | int | float]:
        return {
            "alpha": float(self.alpha),
            "fit_rows": self.fit_rows,
            "validation_rows": self.validation_rows,
            "positive_fit_rows": self.positive_fit_rows,
            "validation_balanced_accuracy": self.validation_balanced_accuracy,
            "direction_sha256": tensor_digest(self.direction),
            "intercept_sha256": tensor_digest(self.intercept),
        }


def fit_concept_direction(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    alpha: float = 1.0,
) -> ConceptDirection:
    """Fit a leakage-explicit ridge CAV for a binary concept.

    The caller owns subject/group splitting. Labels equal to one define the
    positive direction. The held-out balanced accuracy diagnoses whether the
    fitted direction distinguishes the supplied concept examples.
    """

    features = _feature_matrix(features)
    labels = _binary_labels(labels, features.shape[0], features.device)
    train = _split_mask(train_mask, features.shape[0], features.device, "train_mask")
    test = _split_mask(test_mask, features.shape[0], features.device, "test_mask")
    if bool((train & test).any()):
        raise ValidationError("Concept train and test masks must be disjoint")
    for mask, name in ((train, "train_mask"), (test, "test_mask")):
        selected = labels[mask]
        if not bool(selected.any()) or bool(selected.all()):
            raise ValidationError(f"{name} must contain both concept classes")

    targets = labels.to(features.dtype).mul(2).sub(1)
    probe = fit_ridge_probe(features[train], targets[train], alpha=alpha)
    normal = probe.weight[:, 0]
    norm = torch.linalg.vector_norm(normal)
    if float(norm) <= torch.finfo(normal.dtype).eps:
        raise ValidationError("Concept classifier has no nonzero separating direction")
    direction = normal / norm
    intercept = probe.intercept[0] / norm
    predictions = features[test] @ direction + intercept > 0
    truth = labels[test]
    positive_recall = predictions[truth].float().mean()
    negative_recall = (~predictions[~truth]).float().mean()
    balanced_accuracy = float((positive_recall + negative_recall) / 2)
    return ConceptDirection(
        direction,
        intercept,
        float(alpha),
        int(train.sum()),
        int(test.sum()),
        int(labels[train].sum()),
        balanced_accuracy,
    )


@dataclass(frozen=True)
class TCAVScore:
    """Directional sensitivities and their classic positive-sign fraction."""

    site: str
    sample_unit: ConceptSampleUnit
    score: float
    mean_sensitivity: float
    positive_count: int
    sample_count: int
    sensitivities: torch.Tensor
    direction_sha256: str


def _evaluation_mask(
    mask: torch.Tensor | None,
    rows: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(rows, dtype=torch.bool, device=device)
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 1
        or mask.shape[0] != rows
        or mask.device != device
        or not bool(mask.any())
    ):
        raise ValidationError("evaluation_mask must select trials on the attribution device")
    return mask


def tcav_score(
    attribution: AttributionResult,
    site: str,
    concept: ConceptDirection,
    *,
    evaluation_mask: torch.Tensor | None = None,
    sample_unit: ConceptSampleUnit = "trial",
) -> TCAVScore:
    """Measure objective sensitivity along a fitted concept direction.

    Classic TCAV is the fraction of selected evaluation samples whose
    directional derivative is positive. A trial sample averages derivatives
    over all site positions; a position sample treats every non-feature
    coordinate as one example. Raw sensitivities are always returned.
    """

    if not isinstance(attribution, AttributionResult) or attribution.method != "gradient":
        raise ValidationError("TCAV requires an AttributionResult computed with method='gradient'")
    if site not in attribution.site_multipliers:
        raise ValidationError(f"TCAV site was not attributed: {site}")
    if not isinstance(concept, ConceptDirection):
        raise ValidationError("TCAV requires a fitted ConceptDirection")
    gradients = attribution.site_multipliers[site]
    if gradients.ndim < 2 or gradients.shape[-1] != concept.feature_count:
        raise ValidationError("TCAV concept direction differs from the site feature axis")
    if gradients.device != concept.direction.device or gradients.dtype != concept.direction.dtype:
        raise ValidationError("TCAV direction must match the site-gradient device and dtype")
    mask = _evaluation_mask(evaluation_mask, gradients.shape[0], gradients.device)
    directional = (gradients * concept.direction).sum(dim=-1)[mask]
    if sample_unit == "trial":
        sensitivities = directional.reshape(directional.shape[0], -1).mean(dim=1)
    elif sample_unit == "position":
        sensitivities = directional.reshape(-1)
    else:
        raise ValidationError("TCAV sample_unit must be trial or position")
    if not torch.isfinite(sensitivities).all():
        raise ValidationError("TCAV produced non-finite directional sensitivities")
    positive = int((sensitivities > 0).sum())
    count = sensitivities.numel()
    return TCAVScore(
        site,
        sample_unit,
        positive / count,
        float(sensitivities.mean()),
        positive,
        count,
        sensitivities.detach().clone(),
        tensor_digest(concept.direction),
    )


@dataclass(frozen=True)
class TCAVPermutationResult:
    """Classic TCAV score compared with same-split random-label CAVs."""

    concept: ConceptDirection
    observed: TCAVScore
    null_scores: torch.Tensor
    p_value: float
    permutations: int
    seed: int


def _permute_within_splits(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    permuted = labels.clone()
    for mask in (train_mask, test_mask):
        indices = mask.nonzero(as_tuple=False).flatten()
        order = torch.randperm(indices.numel(), generator=generator).to(indices.device)
        permuted[indices] = labels[indices[order]]
    return permuted


def tcav_permutation_test(
    attribution: AttributionResult,
    site: str,
    fit_features: torch.Tensor,
    fit_labels: torch.Tensor,
    *,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    evaluation_mask: torch.Tensor | None = None,
    sample_unit: ConceptSampleUnit = "trial",
    alpha: float = 1.0,
    permutations: int = 50,
    seed: int = 0,
) -> TCAVPermutationResult:
    """Fit a ridge CAV and compare classic TCAV with random-label CAVs.

    Labels are permuted independently within the declared train and test
    splits, preserving their class counts. The finite-sample one-tailed
    p-value includes the observed statistic in the denominator.
    """

    if type(permutations) is not int or permutations < 1:
        raise ValidationError("TCAV permutations must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValidationError("TCAV seed must be an integer in [0, 2^63)")
    concept = fit_concept_direction(
        fit_features,
        fit_labels,
        train_mask=train_mask,
        test_mask=test_mask,
        alpha=alpha,
    )
    observed = tcav_score(
        attribution,
        site,
        concept,
        evaluation_mask=evaluation_mask,
        sample_unit=sample_unit,
    )
    labels = _binary_labels(fit_labels, fit_features.shape[0], fit_features.device)
    generator = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(permutations):
        permuted = _permute_within_splits(labels, train_mask, test_mask, generator)
        random_concept = fit_concept_direction(
            fit_features,
            permuted,
            train_mask=train_mask,
            test_mask=test_mask,
            alpha=alpha,
        )
        null.append(
            tcav_score(
                attribution,
                site,
                random_concept,
                evaluation_mask=evaluation_mask,
                sample_unit=sample_unit,
            ).score
        )
    null_scores = fit_features.new_tensor(null)
    p_value = (1 + int((null_scores >= observed.score).sum())) / (permutations + 1)
    return TCAVPermutationResult(concept, observed, null_scores, p_value, permutations, seed)
