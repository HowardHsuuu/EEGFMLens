"""Clean-room sparse-autoencoder primitives for cached EEG-FM activations."""

from dataclasses import dataclass

import torch
from torch import nn

from .errors import ValidationError
from .interventions import AxisSelection, Selection
from .provenance import tensor_digest


@dataclass(frozen=True)
class SAEOutput:
    reconstruction: torch.Tensor
    codes: torch.Tensor


class TopKSAE(nn.Module):
    """A ReLU Top-K sparse autoencoder with unit-norm decoder directions.

    The module accepts tensors whose final axis is ``input_dim``.  Its decoder
    rows are feature directions in activation space.  Training data collection,
    train/test separation and interpretation of learned features remain explicit
    caller responsibilities.
    """

    def __init__(self, input_dim: int, n_features: int, k: int):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (input_dim, n_features, k)):
            raise ValidationError("SAE dimensions and k must be positive integers")
        if k > n_features:
            raise ValidationError("SAE k cannot exceed n_features")
        self.input_dim = input_dim
        self.n_features = n_features
        self.k = k
        self.encoder_weight = nn.Parameter(torch.empty(input_dim, n_features))
        self.encoder_bias = nn.Parameter(torch.zeros(n_features))
        self.decoder_weight = nn.Parameter(torch.empty(n_features, input_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.decoder_weight)
        with torch.no_grad():
            self.decoder_weight.copy_(torch.nn.functional.normalize(self.decoder_weight, dim=1))
            self.encoder_weight.copy_(self.decoder_weight.T)

    def _validate(self, activations: torch.Tensor):
        if (
            not isinstance(activations, torch.Tensor)
            or not activations.is_floating_point()
            or activations.ndim < 2
            or activations.shape[-1] != self.input_dim
            or not torch.isfinite(activations).all()
        ):
            raise ValidationError(
                f"SAE activations must be finite floating tensors ending in {self.input_dim}"
            )

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        self._validate(activations)
        preactivations = torch.relu(
            (activations - self.decoder_bias) @ self.encoder_weight + self.encoder_bias
        )
        values, indices = torch.topk(preactivations, self.k, dim=-1, sorted=False)
        return torch.zeros_like(preactivations).scatter(-1, indices, values)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(codes, torch.Tensor)
            or not codes.is_floating_point()
            or codes.ndim < 2
            or codes.shape[-1] != self.n_features
            or not torch.isfinite(codes).all()
        ):
            raise ValidationError(
                f"SAE codes must be finite floating tensors ending in {self.n_features}"
            )
        return codes @ self.decoder_weight + self.decoder_bias

    def forward(self, activations: torch.Tensor) -> SAEOutput:
        codes = self.encode(activations)
        return SAEOutput(self.decode(codes), codes)

    @torch.no_grad()
    def normalize_decoder_(self):
        """Project decoder rows to unit norm after an optimizer update."""

        norms = torch.linalg.vector_norm(self.decoder_weight, dim=1, keepdim=True)
        if bool((norms <= torch.finfo(norms.dtype).eps).any()):
            raise ValidationError("Cannot normalize a zero SAE decoder direction")
        self.decoder_weight.div_(norms)


@dataclass(frozen=True)
class SAEMetrics:
    mean_squared_error: float
    explained_variance: float
    mean_l0: float
    active_feature_fraction: float


def sae_metrics(sae: TopKSAE, activations: torch.Tensor) -> SAEMetrics:
    """Measure reconstruction and feature use without changing the SAE."""

    with torch.no_grad():
        sae._validate(activations)
        flattened = activations.reshape(-1, sae.input_dim)
        output = sae(flattened)
        residual = flattened - output.reconstruction
        mse = residual.square().mean()
        centered = flattened - flattened.mean(dim=0, keepdim=True)
        variance = centered.square().mean()
        if float(variance) <= torch.finfo(variance.dtype).eps:
            raise ValidationError("Explained variance is undefined for constant activations")
        active = output.codes > 0
        return SAEMetrics(
            float(mse),
            float(1 - residual.square().mean() / variance),
            float(active.sum(1).float().mean()),
            float(active.any(0).float().mean()),
        )


@dataclass(frozen=True)
class SAETrainingConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    seed: int = 0

    def __post_init__(self):
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValidationError("SAE epochs must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValidationError("SAE batch size must be a positive integer")
        values = torch.tensor([self.learning_rate], dtype=torch.float64)
        if not bool(torch.isfinite(values).all()) or self.learning_rate <= 0:
            raise ValidationError("SAE learning rate must be finite and positive")
        if type(self.seed) is not int:
            raise ValidationError("SAE seed must be an integer")


@dataclass(frozen=True)
class SAETrainingResult:
    losses: tuple[float, ...]
    final_metrics: SAEMetrics


def train_sae(
    sae: TopKSAE,
    activations: torch.Tensor,
    *,
    config: SAETrainingConfig = SAETrainingConfig(),
) -> SAETrainingResult:
    """Train a Top-K SAE on an already selected training activation matrix.

    This compact reference loop minimizes reconstruction MSE.  It intentionally
    does not choose layers, create dataset splits, or reinitialize dead features.
    """

    if not isinstance(sae, TopKSAE):
        raise ValidationError("train_sae expects a TopKSAE")
    sae._validate(activations)
    if activations.ndim != 2:
        raise ValidationError("SAE training activations must have shape [sample, feature]")
    config.__post_init__()
    if (
        next(sae.parameters()).device != activations.device
        or next(sae.parameters()).dtype != activations.dtype
    ):
        raise ValidationError("SAE parameters and training activations must share device and dtype")
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    losses = []
    was_training = sae.training
    sae.train()
    try:
        for _ in range(config.epochs):
            order = torch.randperm(activations.shape[0], generator=generator)
            total = 0.0
            for start in range(0, activations.shape[0], config.batch_size):
                indices = order[start : start + config.batch_size].to(activations.device)
                batch = activations[indices]
                output = sae(batch)
                loss = (output.reconstruction - batch).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                sae.normalize_decoder_()
                total += float(loss.detach()) * batch.shape[0]
            losses.append(total / activations.shape[0])
    finally:
        sae.train(was_training)
    return SAETrainingResult(tuple(losses), sae_metrics(sae, activations))


def _sae_parameter_hashes(sae: TopKSAE) -> tuple[tuple[str, str], ...]:
    return tuple((name, tensor_digest(value)) for name, value in sae.state_dict().items())


def _sae_activation_matrix(sae: TopKSAE, activations: torch.Tensor, name: str) -> torch.Tensor:
    sae._validate(activations)
    if activations.ndim != 2:
        raise ValidationError(f"{name} must have shape [sample, feature]")
    parameter = next(sae.parameters())
    if parameter.device != activations.device or parameter.dtype != activations.dtype:
        raise ValidationError(f"{name} must match the SAE device and dtype")
    return activations


def sae_feature_firing_rates(sae: TopKSAE, activations: torch.Tensor) -> torch.Tensor:
    """Return the fraction of declared activation rows firing each SAE feature."""

    if not isinstance(sae, TopKSAE):
        raise ValidationError("sae_feature_firing_rates expects a TopKSAE")
    activations = _sae_activation_matrix(sae, activations, "SAE firing-rate activations")
    with torch.no_grad():
        return (sae.encode(activations) > 0).to(activations.dtype).mean(dim=0)


def sae_feature_alignment(sae: TopKSAE, direction: torch.Tensor) -> torch.Tensor:
    """Cosine-align every decoder direction with one activation-space direction."""

    if not isinstance(sae, TopKSAE):
        raise ValidationError("sae_feature_alignment expects a TopKSAE")
    parameter = next(sae.parameters())
    if (
        not isinstance(direction, torch.Tensor)
        or not direction.is_floating_point()
        or direction.shape != (sae.input_dim,)
        or direction.device != parameter.device
        or direction.dtype != parameter.dtype
        or not torch.isfinite(direction).all()
    ):
        raise ValidationError("SAE concept direction must match the input feature axis")
    direction_norm = torch.linalg.vector_norm(direction)
    decoder_norms = torch.linalg.vector_norm(sae.decoder_weight, dim=1)
    epsilon = torch.finfo(direction.dtype).eps
    if float(direction_norm) <= epsilon or bool((decoder_norms <= epsilon).any()):
        raise ValidationError("SAE feature alignment is undefined for a zero direction")
    return (sae.decoder_weight @ direction) / (decoder_norms * direction_norm)


@dataclass(frozen=True)
class SAEConceptProfile:
    """Descriptive firing-rate enrichment and decoder/CAV alignment."""

    positive_firing_rate: torch.Tensor
    negative_firing_rate: torch.Tensor
    firing_rate_difference: torch.Tensor
    decoder_alignment: torch.Tensor
    positive_rows: int
    negative_rows: int


def sae_concept_profile(
    sae: TopKSAE,
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    direction: torch.Tensor,
) -> SAEConceptProfile:
    """Describe which SAE features fire for and align with a concept.

    Rows are the caller's declared sampling units. This function reports no
    p-values because independence and exchangeability depend on the study's
    subject/window design.
    """

    positive_activations = _sae_activation_matrix(
        sae, positive_activations, "Positive concept activations"
    )
    negative_activations = _sae_activation_matrix(
        sae, negative_activations, "Negative concept activations"
    )
    positive = sae_feature_firing_rates(sae, positive_activations)
    negative = sae_feature_firing_rates(sae, negative_activations)
    return SAEConceptProfile(
        positive,
        negative,
        positive - negative,
        sae_feature_alignment(sae, direction),
        positive_activations.shape[0],
        negative_activations.shape[0],
    )


@dataclass(frozen=True)
class SAECodeReference:
    """Mean SAE code fitted on a declared target/reference cohort."""

    values: torch.Tensor
    fit_rows: int
    sae_parameter_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if (
            not isinstance(self.values, torch.Tensor)
            or not self.values.is_floating_point()
            or self.values.ndim != 1
            or self.values.numel() < 1
            or not torch.isfinite(self.values).all()
        ):
            raise ValidationError("SAE code reference must be a finite floating vector")
        if type(self.fit_rows) is not int or self.fit_rows < 1:
            raise ValidationError("SAE code reference fit_rows must be positive")
        if (
            not isinstance(self.sae_parameter_hashes, tuple)
            or not self.sae_parameter_hashes
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)
                or len(item[1]) != 64
                for item in self.sae_parameter_hashes
            )
        ):
            raise ValidationError("SAE code reference parameter hashes are invalid")


def fit_sae_code_reference(sae: TopKSAE, activations: torch.Tensor) -> SAECodeReference:
    """Fit a reusable mean code on target/reference activation rows."""

    if not isinstance(sae, TopKSAE):
        raise ValidationError("fit_sae_code_reference expects a TopKSAE")
    activations = _sae_activation_matrix(sae, activations, "SAE reference activations")
    with torch.no_grad():
        values = sae.encode(activations).mean(dim=0)
    return SAECodeReference(values, activations.shape[0], _sae_parameter_hashes(sae))


def _validate_features(sae: TopKSAE, features: tuple[int, ...]):
    if (
        not isinstance(features, tuple)
        or not features
        or any(type(index) is not int or index < 0 or index >= sae.n_features for index in features)
        or len(set(features)) != len(features)
    ):
        raise ValidationError("SAE features must be unique in-range integer indices")


def _sae_provenance(sae: TopKSAE) -> dict[str, str | int]:
    return {
        "input_dim": sae.input_dim,
        "n_features": sae.n_features,
        "k": sae.k,
        **{f"{name}_sha256": tensor_digest(value) for name, value in sae.state_dict().items()},
    }


@dataclass(frozen=True)
class SAEFeatureAblation:
    """Subtract selected SAE decoder contributions while preserving SAE residual."""

    site: str
    sae: TopKSAE
    features: tuple[int, ...]
    selection: Selection | AxisSelection = Selection()

    def apply(self, current, batch, layout, model_id):
        _validate_features(self.sae, self.features)
        self.sae._validate(current)
        if (
            next(self.sae.parameters()).device != current.device
            or next(self.sae.parameters()).dtype != current.dtype
        ):
            raise ValidationError("SAE parameters must match activation device and dtype")
        codes = self.sae.encode(current)
        contribution = codes[..., self.features] @ self.sae.decoder_weight[list(self.features)]
        edited = current - contribution
        return torch.where(self.selection.mask(current, batch, layout), edited, current)

    def provenance(self):
        return {"features": self.features, "sae": _sae_provenance(self.sae)}


@dataclass(frozen=True)
class SAEFeatureSteering:
    """Add fixed multiples of unit-norm SAE decoder directions."""

    site: str
    sae: TopKSAE
    features: tuple[int, ...]
    coefficients: tuple[float, ...]
    selection: Selection | AxisSelection = Selection()

    def apply(self, current, batch, layout, model_id):
        _validate_features(self.sae, self.features)
        if len(self.coefficients) != len(self.features) or any(
            not bool(torch.isfinite(torch.tensor(value))) for value in self.coefficients
        ):
            raise ValidationError("Steering requires one finite coefficient per SAE feature")
        self.sae._validate(current)
        if (
            next(self.sae.parameters()).device != current.device
            or next(self.sae.parameters()).dtype != current.dtype
        ):
            raise ValidationError("SAE parameters must match activation device and dtype")
        coefficients = current.new_tensor(self.coefficients)
        direction = coefficients @ self.sae.decoder_weight[list(self.features)]
        edited = current + direction
        return torch.where(self.selection.mask(current, batch, layout), edited, current)

    def provenance(self):
        return {
            "features": self.features,
            "coefficients": self.coefficients,
            "sae": _sae_provenance(self.sae),
        }


@dataclass(frozen=True)
class SAEFeatureClamping:
    """Replace selected SAE codes by a fixed target-cohort centroid.

    Only the selected decoder contributions change. The native activation's
    SAE reconstruction residual is preserved exactly.
    """

    site: str
    sae: TopKSAE
    features: tuple[int, ...]
    reference: SAECodeReference
    selection: Selection | AxisSelection = Selection()

    def apply(self, current, batch, layout, model_id):
        _validate_features(self.sae, self.features)
        if not isinstance(self.reference, SAECodeReference):
            raise ValidationError("SAE feature clamping requires a fitted SAECodeReference")
        self.sae._validate(current)
        parameter = next(self.sae.parameters())
        if parameter.device != current.device or parameter.dtype != current.dtype:
            raise ValidationError("SAE parameters must match activation device and dtype")
        if (
            self.reference.values.shape != (self.sae.n_features,)
            or self.reference.values.device != current.device
            or self.reference.values.dtype != current.dtype
        ):
            raise ValidationError("SAE code reference differs from the intervention SAE")
        if self.reference.sae_parameter_hashes != _sae_parameter_hashes(self.sae):
            raise ValidationError("SAE parameters changed after fitting the code reference")
        if isinstance(self.selection, AxisSelection) and self.selection.axis == current.ndim - 1:
            raise ValidationError(
                "SAE feature clamping cannot select only part of the decoded feature axis"
            )
        indices = list(self.features)
        codes = self.sae.encode(current)
        difference = self.reference.values[indices] - codes[..., indices]
        edited = current + difference @ self.sae.decoder_weight[indices]
        return torch.where(self.selection.mask(current, batch, layout), edited, current)

    def provenance(self):
        return {
            "features": self.features,
            "reference_fit_rows": self.reference.fit_rows,
            "reference_sha256": tensor_digest(self.reference.values),
            "sae": _sae_provenance(self.sae),
        }
