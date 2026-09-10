"""Fixed held-out heads, patch-resolved responses and native donor restoration."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from eeglens import (
    CBraModAdapter,
    EEGLens,
    LaBraMAdapter,
    Selection,
    SignalBatch,
    SweepTarget,
    load_cbramod,
    load_labram,
    patching_sweep,
)

SITES = ["embedding.output", *[f"blocks.{i}.output" for i in [2, 5, 8, 11]]]


def model_loader(name, checkpoint, random):
    if not random:
        return (load_cbramod if name == "cbramod" else load_labram)(checkpoint)
    torch.manual_seed(4311)
    if name == "cbramod":
        from eeglens.models import CBraMod

        native = CBraMod()
        native.proj_out = torch.nn.Identity()
        adapter = CBraModAdapter(native)
    else:
        from eeglens._vendor.labram import labram_base_patch200_200

        native = labram_base_patch200_200(num_classes=0, init_values=0.1, use_mean_pooling=False)
        adapter = LaBraMAdapter(native)
    return EEGLens(native.eval(), adapter, model_id=f"{name}:random:4311")


def pooled(output):
    return output.mean((1, 2)) if output.ndim == 4 else output.mean(1)


def evaluate(name, checkpoint, previous, prepared, output, random=False):
    torch.set_num_threads(2)
    label = name + ("-random" if random else "")
    doc = json.loads((prepared / "cases.json").read_text())
    original_manifest = json.loads((previous / "prepared/manifest.json").read_text())
    rows = original_manifest["rows"]
    originals = np.load(previous / "prepared/inputs.npz")["x"]
    signals = np.load(prepared / "signals.npz")["signals"]
    readouts = np.load(previous / "results/readouts.npz")
    features = np.load(previous / f"features/{label}.npz")["output"]
    subjects = np.array([r["subject"] for r in rows])
    lens = model_loader(name, checkpoint, random)
    responses, restorations, audit_rows = [], [], []
    for case in doc["cases"]:
        row = case["row"]
        subject = case["subject"]
        validation = [subject % 8 + 1, (subject + 1) % 8 + 1]
        train = ~np.isin(subjects, [subject, *validation])
        weight = readouts[f"{label}_s{subject}_weight"]
        intercept = readouts[f"{label}_s{subject}_intercept"]
        scale = float((features[train] @ weight).std())
        batch = SignalBatch(
            torch.from_numpy(originals[row : row + 1]),
            (f"s{subject}:w{case['window']}:r{case['replicate']}",),
            (case["channel"],),
            200,
            doc["recipe"],
        )

        def score(native_output, current_batch):
            values = pooled(native_output).numpy()
            if torch.equal(current_batch.data, batch.data):
                np.testing.assert_allclose(values[0], features[row], atol=1e-5, rtol=1e-5)
            return torch.from_numpy(values @ weight + intercept)

        def region(name):
            start = case[name + "_start"]
            return Selection(patches=tuple(range(start, start + 3)))

        target = SweepTarget("spindle", region("event"), region("off"), region("random"))
        for vi, variant in enumerate(doc["variants"]):
            recipient = replace(batch, data=torch.from_numpy(signals[case["case"], vi : vi + 1]))
            observe_only = random or variant == "circular_shift"
            result = patching_sweep(
                lens,
                batch,
                recipient,
                score,
                sites=SITES,
                targets=() if observe_only else (target,),
                recovery_site=None if observe_only else "blocks.11.output",
            )
            baseline = result.baselines[0]
            for site in SITES:
                responses.append(
                    dict(
                        **case,
                        model=label,
                        variant=variant,
                        site=site,
                        clean_margin=baseline["clean_score"],
                        margin=baseline["recipient_score"],
                        margin_scale=scale,
                        patch_relative_change=baseline["patch_changes"][site][0],
                    )
                )
            for r in result.rows:
                if r["kind"] == "identity":
                    continue
                restorations.append(
                    dict(
                        **case,
                        model=label,
                        variant=variant,
                        site=r["site"],
                        position="off" if r["kind"] == "off_event" else r["kind"],
                        status=r["status"],
                        clean_margin=r["clean_score"],
                        corrupt_margin=r["recipient_score"],
                        restored_margin=r["patched_score"],
                        margin_scale=scale,
                        activation_delta_norm=r["actual_delta_norm"],
                        target_norm=r["target_norm"],
                        multiplier=r["multiplier"],
                    )
                )
            # Keep richer public-API validity/provenance records alongside legacy-compatible tables.
            audit_rows.extend(dict(**r, case=case["case"], variant=variant) for r in result.rows)
        if case["case"] % 15 == 0:
            print(label, "case", case["case"], "/", len(doc["cases"]), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{label}-responses.json").write_text(json.dumps(responses, allow_nan=False) + "\n")
    (output / f"{label}-restorations.json").write_text(
        json.dumps(restorations, allow_nan=False) + "\n"
    )
    (output / f"{label}-sweep-audit.json").write_text(
        json.dumps(audit_rows, allow_nan=False) + "\n"
    )
    (output / f"{label}-model.json").write_text(json.dumps(lens.manifest, indent=2) + "\n")
    print(
        label,
        "completed",
        len(responses),
        "response rows and",
        len(restorations),
        "restoration rows",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cbramod", "labram"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--random", action="store_true")
    for key in ["previous", "prepared", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    evaluate(a.model, a.checkpoint, a.previous, a.prepared, a.output, a.random)
