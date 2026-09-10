"""Three-model shared-trial feature extraction; explicit experimental input recipe."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from eeglens import CSBrainAdapter, EEGLens, SignalBatch, load_cbramod, load_labram

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
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if name == "cbramod":
            from eeglens import CBraModAdapter
            from eeglens._vendor.cbramod.model import CBraMod

            model = CBraMod()
            model.proj_out = torch.nn.Identity()
            model.eval().requires_grad_(False)
            return EEGLens(model, CBraModAdapter(model)), None
        if name == "labram":
            from eeglens import LaBraMAdapter
            from eeglens._vendor.labram import labram_base_patch200_200

            model = labram_base_patch200_200(num_classes=0, init_values=0.1, use_mean_pooling=False)
            model.eval().requires_grad_(False)
            return EEGLens(model, LaBraMAdapter(model)), None
    if name == "cbramod":
        path = root / "eeglens_build_evidence/cbramod.pth"
        return load_cbramod(path), path
    if name == "labram":
        path = root / "eeglens_build_evidence/labram-base.pth"
        return load_labram(path), path
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


def spatial_features(activation):
    x = activation.tensor
    if activation.layout == "tokens":
        x = x[:, 1:].reshape(x.shape[0], 19, 4, x.shape[-1])
    if x.shape[1:3] != (19, 4):
        raise ValueError("Unexpected physical feature geometry")
    return x.mean(dim=2).numpy()  # retain electrodes; average time only


def run(prepared, name, root, output, random_seed=None):
    torch.set_num_threads(2)
    torch.manual_seed(6281)
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest["artifact_sha256"] != sha(prepared / "trials.npz"):
        raise ValueError("Prepared artifact hash mismatch")
    if manifest["data_unit"] != "volts" or manifest["sampling_rate"] != 160:
        raise ValueError("Unexpected prepared units/rate")
    arrays = np.load(prepared / "trials.npz", allow_pickle=False)
    values = model_input(arrays["volts"], list(arrays["channels"]))
    lens, checkpoint = build(name, root, random_seed=random_seed)
    sites = ("blocks.5.output", "blocks.11.output")
    features = {site: [] for site in sites}
    maximum_error = 0.0
    for start in range(0, len(values), 4):
        batch = SignalBatch(
            values[start : start + 4],
            tuple(arrays["trial_ids"][start : start + 4]),
            CHANNELS,
            200,
            RECIPE,
        )
        clean = lens.run_with_cache(batch, sites=sites)
        with torch.no_grad():
            native = lens.adapter.forward(lens.model, batch)
        torch.testing.assert_close(clean.output, native, rtol=0, atol=0)
        maximum_error = max(maximum_error, float((native - clean.output).abs().max()))
        for site in sites:
            features[site].append(spatial_features(clean.cache[site]))
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / f"{name}.npz"
    np.savez_compressed(
        artifact,
        middle=np.concatenate(features[sites[0]]),
        final=np.concatenate(features[sites[1]]),
        trial_ids=arrays["trial_ids"],
        labels=arrays["labels"],
        subjects=arrays["subjects"],
        descriptors=arrays["descriptors"],
        channels=np.array(CHANNELS),
    )
    report = dict(
        model=name,
        trials=len(values),
        checkpoint_sha256=sha(checkpoint) if checkpoint is not None else None,
        initialization="pretrained" if random_seed is None else "random",
        random_seed=random_seed,
        prepared_manifest_sha256=sha(prepared / "manifest.json"),
        runner_sha256=sha(__file__),
        artifact_sha256=sha(artifact),
        input_recipe=RECIPE,
        sites=sites,
        pooling="time mean, physical electrode order retained; CLS omitted",
        native_wrapper_max_error=maximum_error,
        scope="Shared experimental preprocessing; not a replication of every native training pipeline",
        partition=manifest.get("partition", "development"),
        torch_version=torch.__version__,
    )
    (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(name, len(values), "trials extracted with native parity", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--model", choices=("cbramod", "labram", "csbrain"), required=True)
    p.add_argument("--root", type=Path, default=Path("research"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--random-seed", type=int, help="Use constructor initialization without loading weights"
    )
    a = p.parse_args()
    run(a.prepared, a.model, a.root, a.output, random_seed=a.random_seed)
