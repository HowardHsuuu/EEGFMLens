"""Map fitted contrast erasure to two-sensor native hooks on real training EEG."""

import argparse
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from extract import CHANNELS, RECIPE, build, model_input, sha, spatial_features
from probes import load_features

from eeglens import Replacement, Selection, SignalBatch, SubspaceAblation
from eeglens._vendor.labram_channels import CHANNELS as STANDARD_1020


def run(name, root, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    folder = root / "eeglens_mi/iterative-recovery-v1" / name
    report = json.loads((folder / "summary.json").read_text())
    assert sha(folder / "bases.npz") == report["artifacts"]["bases.npz"]
    basis = np.load(folder / "bases.npz")["fold_0_concept"]
    features = root / "eeglens_mi/features-v1/train" / f"{name}.npz"
    assert sha(features) == report["feature_sha256"]
    data = load_features(features, "train")
    fitting = np.isin(data["subjects"], report["folds"][0]["fitting_subjects"])
    centers = {
        c: data["middle"][fitting, list(data["channels"]).index(c)].mean(0).astype(np.float32)
        for c in ("C3", "C4")
    }
    prepared = root / "eeglens_mi/prepared-v1/train"
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert (
        manifest["partition"] == "train"
        and sha(prepared / "trials.npz") == manifest["artifact_sha256"]
    )
    raw = np.load(prepared / "trials.npz")
    indices = np.flatnonzero(np.isin(raw["subjects"], report["folds"][0]["evaluation_subjects"]))[
        :2
    ]
    values = model_input(raw["volts"][indices], list(raw["channels"]))
    batch = SignalBatch(values, tuple(raw["trial_ids"][indices]), CHANNELS, 200, RECIPE)
    lens, checkpoint = build(name, root)
    feature_meta = json.loads(features.with_suffix(".json").read_text())
    assert sha(checkpoint) == feature_meta["checkpoint_sha256"]
    site = "blocks.5.output"
    clean = lens.run_with_cache(batch, sites=(site,))
    activation = clean.cache[site]
    physical = spatial_features(activation)
    contrast = physical[:, CHANNELS.index("C4")] - physical[:, CHANNELS.index("C3")]
    center_difference = centers["C4"] - centers["C3"]
    order = [0, 10, 2, 16, 3, 11, 1, 6, 18, 7, 12, 14, 15, 13, 8, 9, 4, 17, 5]
    records = []
    for rank in (2, 4, 8, 16, 32):
        q = torch.tensor(basis[:, :rank], dtype=torch.float32)
        edited = activation.tensor
        for c in ("C3", "C4"):
            edited = SubspaceAblation(
                site, q, torch.tensor(centers[c]), Selection(sensors=(c,))
            ).apply(edited, batch, activation.layout, lens.model_id)
        donor = replace(activation, tensor=edited)
        actual = lens.run_with_interventions(
            batch, interventions=[Replacement(site, donor)], sites=(site,)
        )
        pooled = spatial_features(actual.cache[site])
        observed = pooled[:, CHANNELS.index("C4")] - pooled[:, CHANNELS.index("C3")]
        expected = contrast - ((contrast - center_difference) @ q.numpy()) @ q.numpy().T
        np.testing.assert_allclose(observed, expected, atol=5e-5, rtol=5e-5)
        untouched = [i for i, c in enumerate(CHANNELS) if c not in ("C3", "C4")]
        np.testing.assert_array_equal(pooled[:, untouched], physical[:, untouched])

        def hook(module, args, original):
            updated = original
            for c in ("C3", "C4"):
                projected = updated - ((updated - torch.tensor(centers[c])) @ q) @ q.T
                mask = torch.zeros_like(updated, dtype=torch.bool)
                if name == "labram":
                    start = 1 + CHANNELS.index(c) * 4
                    mask[:, start : start + 4] = True
                else:
                    index = (
                        order.index(CHANNELS.index(c)) if name == "csbrain" else CHANNELS.index(c)
                    )
                    mask[:, index] = True
                updated = torch.where(mask, projected, updated)
            return updated

        handle = lens.model.get_submodule(
            lens.adapter.require(site).module_path
        ).register_forward_hook(hook)
        try:
            with torch.no_grad():
                if name == "labram":
                    native = lens.model.forward_features(
                        batch.data,
                        input_chans=[0, *[STANDARD_1020.index(c) + 1 for c in CHANNELS]],
                        return_patch_tokens=True,
                        return_all_tokens=False,
                    )
                else:
                    native = lens.model(batch.data)
        finally:
            handle.remove()
        torch.testing.assert_close(actual.output, native, atol=0, rtol=0)
        records.append(
            dict(
                rank=rank,
                native_output_exact=True,
                untouched_sensor_means_exact=True,
                contrast_max_absolute_error=float(np.max(np.abs(observed - expected))),
            )
        )
    torch.testing.assert_close(
        lens.run_with_cache(batch, sites=()).output, clean.output, atol=0, rtol=0
    )
    assert all(not m._forward_hooks for m in lens.model.modules())
    result = dict(
        model=name,
        records=records,
        trial_ids=list(batch.trial_ids),
        checkpoint_sha256=sha(checkpoint),
        runner_sha256=sha(__file__),
        builder_sha256=sha(inspect.getfile(build)),
        basis_sha256=sha(folder / "bases.npz"),
        feature_sha256=sha(features),
        prepared_sha256=sha(prepared / "trials.npz"),
        scope="Two real training trials from outer fold 0; native output exact, pooled contrast float32 tolerance 5e-5; per-sensor fitting means required. No task inference.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(name, records, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cbramod", "labram", "csbrain"], required=True)
    parser.add_argument("--root", type=Path, default=Path("research"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.root, args.output)
