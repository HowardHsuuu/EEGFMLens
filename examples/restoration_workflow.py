"""Run the flagship activation-restoration workflow and emit an auditable report.

Offline smoke test:
  python examples/restoration_workflow.py --demo --output /tmp/eeglens-restoration

Real EEG:
  python examples/restoration_workflow.py --model cbramod --upstream /path/to/CBraMod \
    --checkpoint /path/to/cbramod.pth --edf /path/to/S001R04.edf \
    --output /path/to/new-output-directory
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from eeglens import (
    ActivationSite,
    Adapter,
    EEGLens,
    Selection,
    SignalBatch,
    SweepTarget,
    load_cbramod,
    load_labram,
    restoration_sweep,
)
from eeglens.loading import sha256_file

DEFAULT_SITES = (
    "embedding.output",
    "blocks.0.output",
    "blocks.5.output",
    "blocks.11.output",
)


class DemoEncoder(nn.Module):
    """Small deterministic model used only to exercise the complete workflow."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Identity()
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])

    def forward(self, data):
        hidden = self.embedding(data)
        hidden = self.blocks[0](hidden * 2)
        return self.blocks[1](hidden + hidden.mean(dim=(1, 2), keepdim=True))


def demo_inputs():
    torch.manual_seed(20260921)
    data = torch.randn(2, 3, 4, 8)
    batch = SignalBatch(
        data,
        ("demo-a", "demo-b"),
        ("C3", "CZ", "C4"),
        200,
        "offline-restoration-demo:v1",
    )
    model = DemoEncoder().eval()
    sites = [
        ActivationSite("embedding.output", "embedding"),
        ActivationSite("blocks.0.output", "blocks.0"),
        ActivationSite("blocks.1.output", "blocks.1"),
    ]
    lens = EEGLens(model, Adapter(sites), model_id="offline-demo")
    return lens, batch


def load_real_lens(model_name, checkpoint, upstream):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from native_sources import REVISIONS, native_module

    module = native_module(model_name, upstream)
    if model_name == "cbramod":
        return load_cbramod(
            checkpoint,
            model_factory=module.CBraMod,
            source_revision=REVISIONS[model_name],
        )
    return load_labram(
        checkpoint,
        model_factory=module.labram_base_patch200_200,
        source_revision=REVISIONS[model_name],
    )


def target_grid(batch, channels):
    unknown = tuple(channel for channel in channels if channel not in batch.channels)
    if unknown:
        raise ValueError(f"Unknown target channels: {unknown}")
    return tuple(
        SweepTarget(
            f"{channel}:patch{patch}",
            Selection(sensors=(channel,), patches=(patch,)),
        )
        for channel in channels
        for patch in range(batch.data.shape[2])
    )


def render_heatmap(result, sites, targets, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = {
        (row["site"], row["target"]): row["mean_clean_error_reduction"]
        for row in result.summary()
        if row["kind"] == "event"
    }
    matrix = np.full((len(sites), len(targets)), np.nan)
    for i, site in enumerate(sites):
        for j, target in enumerate(targets):
            value = rows.get((site, target.name))
            if value is not None:
                matrix[i, j] = value
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(finite.max()) if finite.size else 0.0, 1e-8)
    figure, axis = plt.subplots(figsize=(max(8, 0.55 * len(targets)), max(3.5, 0.55 * len(sites))))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_xticks(
        range(len(targets)), [target.name for target in targets], rotation=60, ha="right"
    )
    axis.set_yticks(range(len(sites)), sites)
    axis.set_xlabel("Patched sensor/time coordinate")
    axis.set_ylabel("Activation site")
    axis.set_title("Mean clean-output error reduction")
    figure.colorbar(image, ax=axis, label="Restoration score")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(args):
    torch.set_num_threads(2)
    if args.demo:
        lens, clean = demo_inputs()
        sites = tuple(args.sites or ("embedding.output", "blocks.0.output", "blocks.1.output"))
        target_channels = tuple(clean.channels)
        corrupt_channel, corrupt_patch = "C3", 1
        model_name = "offline-demo"
    else:
        from eegmmidb import prepare_eegmmidb

        missing = [
            name for name in ("upstream", "checkpoint", "edf") if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(f"Real workflow requires: {', '.join(missing)}")
        clean = prepare_eegmmidb(args.edf)
        lens = load_real_lens(args.model, args.checkpoint, args.upstream)
        sites = tuple(args.sites or DEFAULT_SITES)
        target_channels = (
            tuple(clean.channels)
            if args.target_channels == "all"
            else tuple(
                part.strip().upper() for part in args.target_channels.split(",") if part.strip()
            )
        )
        corrupt_channel, corrupt_patch = args.corrupt_channel.upper(), args.corrupt_patch
        model_name = args.model

    if corrupt_channel not in clean.channels:
        raise ValueError(f"Unknown corruption channel: {corrupt_channel}")
    if not 0 <= corrupt_patch < clean.data.shape[2]:
        raise ValueError("Corruption patch is outside the input")
    declared = {capability.name: capability for capability in lens.capabilities()}
    if any(site not in declared for site in sites):
        raise ValueError(f"Unknown sites; available: {tuple(declared)}")
    if any("sensor" not in declared[site].selectors for site in sites):
        raise ValueError("Flagship workflow requires physical sensor/patch sites")

    recipient_data = clean.data.clone()
    recipient_data[:, clean.channels.index(corrupt_channel), corrupt_patch] = 0
    recipient = replace(clean, data=recipient_data)
    targets = target_grid(clean, target_channels)
    result = restoration_sweep(
        lens,
        clean,
        recipient,
        sites=sites,
        targets=targets,
        random_controls=not args.no_random_controls,
        seed=args.seed,
        recovery_site=sites[-1],
    )

    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".eeglens-workflow-", dir=destination.parent))
    try:
        sweep_path = temporary / "sweep.json"
        figure_path = temporary / "restoration_heatmap.png"
        result.save(sweep_path)
        render_heatmap(result, sites, targets, figure_path)
        event_rows = [row for row in result.rows if row["kind"] == "event"]
        report = {
            "schema": "eeglens.restoration_workflow.v1",
            "mode": "demo" if args.demo else "real-eeg",
            "model": model_name,
            "manifest": lens.manifest,
            "preprocessing": clean.preprocessing_id,
            "input_shape": list(clean.data.shape),
            "corruption": {"channel": corrupt_channel, "patch": corrupt_patch, "value": 0},
            "sites": list(sites),
            "targets": [target.name for target in targets],
            "capabilities": [asdict(item) for item in lens.capabilities()],
            "event_rows": len(event_rows),
            "valid_event_rows": sum(row["valid"] for row in event_rows),
            "excluded_event_rows": sum(not row["valid"] for row in event_rows),
            "random_controls": not args.no_random_controls,
            "seed": args.seed,
            "artifacts": {
                "sweep.json": sha256_file(sweep_path),
                "restoration_heatmap.png": sha256_file(figure_path),
            },
            "inputs": {
                "edf_sha256": None if args.demo else sha256_file(args.edf),
                "checkpoint_sha256": None if args.demo else sha256_file(args.checkpoint),
                "example_sha256": sha256_file(__file__),
            },
            "scope": "activation-restoration diagnostic; no task accuracy or physiological mechanism claim",
        }
        (temporary / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Restoration workflow artifacts: {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Run the offline deterministic demo")
    mode.add_argument("--model", choices=("cbramod", "labram"), help="Official model family")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--edf", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--sites", nargs="+", help="Physical activation sites")
    parser.add_argument("--target-channels", default="C3,CZ,C4", help="Comma list or 'all'")
    parser.add_argument("--corrupt-channel", default="C3")
    parser.add_argument("--corrupt-patch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--no-random-controls", action="store_true")
    run(parser.parse_args())
