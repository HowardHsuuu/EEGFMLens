"""Fold-wise native task effects of contrast-only versus sensor erasure."""

import argparse
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from contrast_intervention import contrast_donor
from extract import CHANNELS, RECIPE, build, model_input, sha, spatial_features
from probes import fit_ridge, load_features

from eeglens import Replacement, Selection, SignalBatch, SubspaceAblation


def donor_for(method, activation, batch, q, centers):
    if method == "contrast":
        return contrast_donor(activation, batch, q, centers["C4"] - centers["C3"])
    values = activation.tensor
    for channel in ("C3", "C4"):
        values = SubspaceAblation(
            activation.site, q, centers[channel], Selection(sensors=(channel,))
        ).apply(values, batch, activation.layout, activation.model_id)
    return replace(activation, tensor=values)


def run(name, fold, root, protocol, output):
    if output.exists():
        raise FileExistsError(output)
    policy = json.loads(protocol.read_text())
    if (
        name not in policy["models"]
        or policy["ranks"] != [2, 4, 8, 16, 32]
        or policy["random_seeds"] != [31, 71, 113]
        or policy["methods"] != ["contrast", "sensor"]
    ):
        raise ValueError("Unsupported protocol")
    torch.set_num_threads(2)
    folder = root / "eeglens_mi/iterative-recovery-v1" / name
    history = json.loads((folder / "summary.json").read_text())
    assert sha(folder / "bases.npz") == history["artifacts"]["bases.npz"]
    bases = np.load(folder / "bases.npz")
    roles = history["folds"][fold]
    features = root / "eeglens_mi/features-v1/train" / f"{name}.npz"
    assert sha(features) == history["feature_sha256"]
    data = load_features(features, "train")
    fit, val, test = [
        np.isin(data["subjects"], roles[k])
        for k in ("fitting_subjects", "validation_subjects", "evaluation_subjects")
    ]
    assert not (fit & val).any() and not (fit & test).any() and not (val & test).any()
    y = np.column_stack((data["labels"] * 2 - 1, data["descriptors"]))
    final = data["final"].reshape(len(y), -1)
    readout = fit_ridge(final[fit], y[fit], final[val], y[val])
    centers = {
        c: torch.tensor(
            data["middle"][fit, list(data["channels"]).index(c)].mean(0), dtype=torch.float32
        )
        for c in ("C3", "C4")
    }
    prepared = root / "eeglens_mi/prepared-v1/train"
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert (
        manifest["partition"] == "train"
        and sha(prepared / "trials.npz") == manifest["artifact_sha256"]
    )
    raw = np.load(prepared / "trials.npz")
    indices = np.flatnonzero(np.isin(raw["subjects"], roles["evaluation_subjects"]))
    lookup = {t: i for i, t in enumerate(data["trial_ids"])}
    feature_indices = np.array([lookup[t] for t in raw["trial_ids"][indices]])
    assert set(raw["subjects"][indices]) == set(roles["evaluation_subjects"])
    values = model_input(raw["volts"][indices], list(raw["channels"]))
    lens, checkpoint = build(name, root)
    feature_meta = json.loads(features.with_suffix(".json").read_text())
    assert sha(checkpoint) == feature_meta["checkpoint_sha256"]
    conditions = [("clean", None, 0, None, False)]
    for method in policy["methods"]:
        for rank in policy["ranks"]:
            conditions.append((f"{method}-concept-{rank}", method, rank, "concept", False))
            for seed in policy["random_seeds"]:
                for matched in (False, True):
                    conditions.append(
                        (
                            f"{method}-{'matched' if matched else 'random'}-{seed}-{rank}",
                            method,
                            rank,
                            f"random_{seed}",
                            matched,
                        )
                    )
    all_predictions, all_norms, all_gains = [], [], []
    site = "blocks.5.output"
    for start in range(0, len(values), 4):
        batch = SignalBatch(
            values[start : start + 4],
            tuple(raw["trial_ids"][indices][start : start + 4]),
            CHANNELS,
            200,
            RECIPE,
        )
        clean = lens.run_with_cache(batch, sites=(site, "blocks.11.output"))
        activation = clean.cache[site]
        clean_final = spatial_features(clean.cache["blocks.11.output"]).reshape(
            len(batch.trial_ids), -1
        )
        np.testing.assert_allclose(
            clean_final, final[feature_indices[start : start + 4]], atol=2e-5, rtol=2e-5
        )
        predictions, norms, gains = [], [], []
        target_norms = {}
        for label, method, rank, kind, matched in conditions:
            if method is None:
                current = clean
                norm = gain = torch.zeros(len(batch.trial_ids))
            else:
                q = torch.tensor(bases[f"fold_{fold}_{kind}"][:, :rank], dtype=torch.float32)
                donor = donor_for(method, activation, batch, q, centers)
                delta = donor.tensor - activation.tensor
                norm = delta.flatten(1).norm(dim=1)
                gain = torch.ones_like(norm)
                if kind == "concept":
                    target_norms[method, rank] = norm
                if matched:
                    if (norm <= 1e-12).any():
                        raise ValueError("Zero random norm; refuse undefined matched control")
                    gain = target_norms[method, rank] / norm
                    donor = replace(
                        donor,
                        tensor=activation.tensor
                        + delta * gain.reshape(-1, *([1] * (delta.ndim - 1))),
                    )
                    norm = (donor.tensor - activation.tensor).flatten(1).norm(dim=1)
                    torch.testing.assert_close(
                        norm, target_norms[method, rank], atol=1e-5, rtol=1e-5
                    )
                current = lens.run_with_interventions(
                    batch, interventions=[Replacement(site, donor)], sites=("blocks.11.output",)
                )
            z = spatial_features(current.cache["blocks.11.output"]).reshape(
                len(batch.trial_ids), -1
            )
            predictions.append(z @ readout["coef"] + readout["intercept"])
            norms.append(norm.numpy())
            gains.append(gain.numpy())
        all_predictions.append(np.stack(predictions, axis=1))
        all_norms.append(np.stack(norms, axis=1))
        all_gains.append(np.stack(gains, axis=1))
        if start % 40 == 0:
            print(name, fold, min(start + 4, len(values)), "/", len(values), flush=True)
    assert all(not m._forward_hooks for m in lens.model.modules())
    output.mkdir(parents=True)
    artifact = output / "responses.npz"
    np.savez_compressed(
        artifact,
        predictions=np.concatenate(all_predictions),
        norms=np.concatenate(all_norms),
        gains=np.concatenate(all_gains),
        conditions=np.array([c[0] for c in conditions]),
        trial_ids=raw["trial_ids"][indices],
        subjects=raw["subjects"][indices],
        labels=raw["labels"][indices],
        descriptors=raw["descriptors"][indices],
        training_target_variance=y[fit].var(0),
        readout_coef=readout["coef"],
        readout_intercept=readout["intercept"],
    )
    record = dict(
        model=name,
        fold=fold,
        roles=roles,
        readout_alpha=readout["alpha"].tolist(),
        trials=len(values),
        conditions=len(conditions),
        artifact_sha256=sha(artifact),
        protocol_sha256=sha(protocol),
        runner_sha256=sha(__file__),
        helper_sha256=sha(inspect.getfile(contrast_donor)),
        fitter_sha256=sha(inspect.getfile(fit_ridge)),
        builder_sha256=sha(inspect.getfile(build)),
        checkpoint_sha256=sha(checkpoint),
        feature_sha256=sha(features),
        basis_sha256=sha(folder / "bases.npz"),
        scope=policy["scope"],
    )
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(name, fold, "complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cbramod", "labram", "csbrain"], required=True)
    p.add_argument("--fold", type=int, choices=[0, 1, 2], required=True)
    p.add_argument("--root", type=Path, default=Path("research"))
    p.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("contrast_task_protocol.json")
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.model, a.fold, a.root, a.protocol, a.output)
