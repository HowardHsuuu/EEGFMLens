"""EEGFMLens: explicit observation and intervention on native EEG models."""

from .adapters.base import Adapter
from .adapters.bendr import BENDRAdapter
from .adapters.brainomni import BrainOmniAdapter
from .adapters.cbramod import CBraModAdapter
from .adapters.continuous import BENDREncoderAdapter, BIOTAdapter
from .adapters.csbrain import CSBrainAdapter
from .adapters.diver import DIVERAdapter
from .adapters.eegpt import EEGPTAdapter
from .adapters.generic import GenericAdapter, ModuleInfo, inspect_modules
from .adapters.labram import LaBraMAdapter
from .adapters.neurorvq import NeuroRVQAdapter
from .adapters.signaljepa import SignalJEPAAdapter
from .adapters.steegformer import STEEGFormerAdapter
from .attribution import (
    AttributionMethod,
    AttributionResult,
    SpectralAttribution,
    attribute,
    channel_attribution,
    patch_attribution,
    spectral_attribution,
    spectral_band_attribution,
    temporal_attribution,
)
from .catalog import IntegrationSpec, ModelSpec, connect, model_info, supported_models
from .circuits import PathTraceResult, path_patch
from .diagnostics import (
    ContrastConsistency,
    GroupVariance,
    group_variance_decomposition,
    within_group_contrast_consistency,
)
from .interventions import Ablation, AxisSelection, Replacement, Selection, SubspaceAblation
from .io import load_run, save_run
from .loading import load_cbramod, load_labram
from .metrics import PairedEffect, paired_effect
from .model import EEGLens, SiteCapability
from .perturbation import (
    BandTarget,
    InputTarget,
    PerturbationCurve,
    attribution_cosine_consistency,
    occlusion_curve,
    spectral_perturbation_curve,
)
from .probes import (
    CrossCovarianceSubspace,
    LayerProbeResult,
    ProbeScore,
    RidgeProbe,
    activation_matrix,
    categorical_targets,
    fit_cross_covariance_subspace,
    fit_ridge_probe,
    layerwise_ridge_probe,
    r2_score,
)
from .sae import (
    SAEFeatureAblation,
    SAEFeatureSteering,
    SAEMetrics,
    SAEOutput,
    SAETrainingConfig,
    SAETrainingResult,
    TopKSAE,
    sae_metrics,
    train_sae,
)
from .spectral import (
    CANONICAL_BANDS,
    FrequencyBand,
    PowerSpectrum,
    band_power,
    patch_frequency_band,
    power_spectrum,
    scale_frequency_band,
)
from .sweep import MatchedReplacement, SweepResult, SweepTarget, patch_grid, patching_sweep
from .types import Activation, ActivationSite, RunResult, SignalBatch, SignalTransform
from .workflows import restoration_sweep

__all__ = [
    "EEGLens",
    "SiteCapability",
    "IntegrationSpec",
    "ModelSpec",
    "connect",
    "model_info",
    "supported_models",
    "NeuroRVQAdapter",
    "SignalJEPAAdapter",
    "STEEGFormerAdapter",
    "DIVERAdapter",
    "BrainOmniAdapter",
    "CSBrainAdapter",
    "GenericAdapter",
    "EEGPTAdapter",
    "BIOTAdapter",
    "BENDREncoderAdapter",
    "BENDRAdapter",
    "ModuleInfo",
    "inspect_modules",
    "MatchedReplacement",
    "SweepResult",
    "SweepTarget",
    "patch_grid",
    "patching_sweep",
    "Adapter",
    "CBraModAdapter",
    "SignalBatch",
    "ActivationSite",
    "Activation",
    "RunResult",
    "Replacement",
    "Ablation",
    "Selection",
    "AxisSelection",
    "PairedEffect",
    "paired_effect",
    "LaBraMAdapter",
    "load_cbramod",
    "load_labram",
    "SubspaceAblation",
    "save_run",
    "load_run",
    "restoration_sweep",
    "SignalTransform",
    "FrequencyBand",
    "PowerSpectrum",
    "CANONICAL_BANDS",
    "power_spectrum",
    "band_power",
    "scale_frequency_band",
    "patch_frequency_band",
    "RidgeProbe",
    "ProbeScore",
    "LayerProbeResult",
    "CrossCovarianceSubspace",
    "activation_matrix",
    "categorical_targets",
    "fit_ridge_probe",
    "r2_score",
    "layerwise_ridge_probe",
    "fit_cross_covariance_subspace",
    "GroupVariance",
    "ContrastConsistency",
    "group_variance_decomposition",
    "within_group_contrast_consistency",
    "TopKSAE",
    "SAEOutput",
    "SAEMetrics",
    "SAETrainingConfig",
    "SAETrainingResult",
    "sae_metrics",
    "train_sae",
    "SAEFeatureAblation",
    "SAEFeatureSteering",
    "PathTraceResult",
    "path_patch",
    "AttributionResult",
    "AttributionMethod",
    "SpectralAttribution",
    "attribute",
    "patch_attribution",
    "channel_attribution",
    "temporal_attribution",
    "spectral_attribution",
    "spectral_band_attribution",
    "InputTarget",
    "BandTarget",
    "PerturbationCurve",
    "occlusion_curve",
    "spectral_perturbation_curve",
    "attribution_cosine_consistency",
]
