"""Frozen feature extraction and local throughput measurement, no head fitting."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from eeglens import CBraModAdapter, EEGLens, LaBraMAdapter, SignalBatch, load_cbramod, load_labram


def extract(model, checkpoint, prepared, output, limit=None, random_weights=False):
    torch.set_num_threads(2)
    torch.manual_seed(4311)
    manifest = json.loads((prepared / "manifest.json").read_text())
    rows = manifest["rows"]
    inputs = np.load(prepared / "inputs.npz")["x"]
    if random_weights:
        if model == "cbramod":
            from eeglens.models import CBraMod

            native = CBraMod()
            native.proj_out = torch.nn.Identity()
            adapter = CBraModAdapter(native)
        else:
            from eeglens._vendor.labram import labram_base_patch200_200

            native = labram_base_patch200_200(
                num_classes=0, init_values=0.1, use_mean_pooling=False
            )
            adapter = LaBraMAdapter(native)
        lens = EEGLens(native.eval(), adapter, model_id=f"{model}:random:4311")
        lens.manifest.update(initialization="random", seed=4311)
    else:
        lens = (load_cbramod if model == "cbramod" else load_labram)(checkpoint)
    filename = model + ("-random" if random_weights else "")
    sites = ["blocks.5.output", "blocks.8.output"]
    collected = {name: [] for name in ["output", *sites]}
    indices = []
    start = time.monotonic()
    for subject in range(1, 9):
        subject_ids = [i for i, row in enumerate(rows) if row["subject"] == subject]
        if limit is not None:
            subject_ids = subject_ids[:limit]
        for offset in range(0, len(subject_ids), 8):
            ids = subject_ids[offset : offset + 8]
            current = [rows[i] for i in ids]
            batch = SignalBatch(
                torch.from_numpy(inputs[ids]),
                tuple(f"s{r['subject']}:w{r['window']}" for r in current),
                (current[0]["channel"],),
                200,
                manifest["recipe"],
            )
            result = lens.run_with_cache(batch, sites=sites)
            native = result.output
            pooled = native.mean((1, 2)) if native.ndim == 4 else native.mean(1)
            collected["output"].append(pooled.cpu().numpy())
            for site in sites:
                activation = result.cache[site].tensor
                pooled = (
                    activation.mean((1, 2)) if model == "cbramod" else activation[:, 1:].mean(1)
                )
                collected[site].append(pooled.cpu().numpy())
            indices.extend(ids)
        print(f"{model}: subject {subject}, {len(indices)} windows", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"{filename}.npz",
        indices=np.asarray(indices),
        **{key: np.concatenate(value) for key, value in collected.items()},
    )
    record = dict(
        model=lens.manifest,
        windows=len(indices),
        elapsed_seconds=time.monotonic() - start,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        platform="macOS CPU",
        torch=str(torch.__version__),
        scope="frozen pooled features; no outcome-based selection",
    )
    (output / f"{filename}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cbramod", "labram"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--random-weights", action="store_true")
    a = p.parse_args()
    extract(a.model, a.checkpoint, a.prepared, a.output, a.limit, a.random_weights)
