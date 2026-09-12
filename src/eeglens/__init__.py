"""EEGLens: explicit observation and intervention on native EEG models."""

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
from .interventions import Ablation, AxisSelection, Replacement, Selection, SubspaceAblation
from .io import load_run, save_run
from .loading import load_cbramod, load_labram
from .metrics import PairedEffect, paired_effect
from .model import EEGLens
from .sweep import MatchedReplacement, SweepResult, SweepTarget, patch_grid, patching_sweep
from .types import Activation, ActivationSite, RunResult, SignalBatch

__all__ = [
    "EEGLens",
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
]
