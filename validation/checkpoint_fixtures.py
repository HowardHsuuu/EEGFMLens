"""Checkpoint and input fixtures for native integration validation.

The recipe identifier is retained for compatibility with recorded evidence.
This module does not fit readouts or perform scientific analysis.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from eeglens import CSBrainAdapter, EEGLens, load_cbramod, load_labram

CHANNELS = (
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "FZ",
    "CZ",
    "PZ",
)
RECIPE = "mi-shared-v1:original-reference:19ch:polyphase160to200:4s:uV/100:no-bandpass"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(name, root, random_seed=None):
    if name in {"cbramod", "labram"}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        from native_sources import native_module

        module = native_module(name, root / "upstreams" / name)
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if name == "cbramod":
            from eeglens import CBraModAdapter

            CBraMod = module.CBraMod

            model = CBraMod()
            model.proj_out = torch.nn.Identity()
            model.eval().requires_grad_(False)
            return EEGLens(model, CBraModAdapter(model)), None
        if name == "labram":
            from eeglens import LaBraMAdapter

            labram_base_patch200_200 = module.labram_base_patch200_200

            model = labram_base_patch200_200(num_classes=0, init_values=0.1, use_mean_pooling=False)
            model.eval().requires_grad_(False)
            return EEGLens(model, LaBraMAdapter(model)), None
    if name == "cbramod":
        path = root / "eeglens_build_evidence/cbramod.pth"
        return load_cbramod(path, model_factory=module.CBraMod), path
    if name == "labram":
        path = root / "eeglens_build_evidence/labram-base.pth"
        return load_labram(path, model_factory=module.labram_base_patch200_200), path
    sys.path.insert(0, str(root / "eeglens_model_validation/repos/CSBrain"))
    from models.CSBrain import CSBrain

    regions = [0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1]
    order = [0, 10, 2, 16, 3, 11, 1, 6, 18, 7, 12, 14, 15, 13, 8, 9, 4, 17, 5]
    model = CSBrain(brain_regions=regions, sorted_indices=order)
    path = root / "eeglens_model_validation/expansion/CSBrain.pth"
    if random_seed is None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not all(k.startswith("module.") for k in state):
            raise ValueError("Unexpected CSBrain checkpoint naming")
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    model.eval().requires_grad_(False)
    return EEGLens(
        model, CSBrainAdapter(model, channels=CHANNELS)
    ), path if random_seed is None else None


def model_input(volts, channels):
    indices = [channels.index(c) for c in CHANNELS]
    if volts.ndim != 3 or volts.shape[-1] != 640:
        raise ValueError("Expected four-second raw 160 Hz epochs")
    signal = resample_poly(volts[:, indices], 5, 4, axis=-1)
    return torch.from_numpy((signal * 1e4).astype(np.float32)).reshape(-1, 19, 4, 200)
