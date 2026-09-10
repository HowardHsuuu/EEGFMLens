"""Middle-layer interventions with frozen readouts and paired random controls."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from extract import CHANNELS, RECIPE, build, model_input, sha, spatial_features

from eeglens import Replacement, Selection, SignalBatch, SubspaceAblation


def make_donor(activation, batch, basis, center, dose, target_norm=None):
    selection = Selection(sensors=("C3", "C4"))
    erased = SubspaceAblation(activation.site, basis, center, selection).apply(
        activation.tensor, batch, activation.layout, activation.model_id
    )
    delta = erased - activation.tensor
    norms = delta.flatten(1).norm(dim=1)
    multiplier = torch.full_like(norms, dose)
    if target_norm is not None:
        if (norms <= 1e-12).any():
            raise ValueError("Random projection has zero norm; cannot energy-match")
        multiplier = dose * target_norm / norms
    change = delta * multiplier.reshape(-1, *([1] * (delta.ndim - 1)))
    values = activation.tensor + change
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite intervention")
    return replace(activation, tensor=values), change.flatten(1).norm(dim=1), multiplier


def run(prepared, probes, name, root, output):
    source = json.loads((prepared / "manifest.json").read_text())
    if source.get("partition") != "validation":
        raise ValueError("This pilot runner only accepts the validation partition")
    probe_meta = json.loads((probes / "manifest.json").read_text())
    if probe_meta["model"] != name or probe_meta["artifact_sha256"] != sha(probes / "probes.npz"):
        raise ValueError("Probe identity/hash mismatch")
    if source["artifact_sha256"] != sha(prepared / "trials.npz"):
        raise ValueError("Prepared hash mismatch")
    data = np.load(prepared / "trials.npz", allow_pickle=False)
    fitted = np.load(probes / "probes.npz", allow_pickle=False)
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
    lens, checkpoint = build(name, root)
    if sha(checkpoint) != probe_meta["checkpoint_sha256"]:
        raise ValueError("Probe checkpoint mismatch")
    values = model_input(data["volts"], list(data["channels"]))
    basis = torch.tensor(fitted["basis"], dtype=torch.float32)
    center = torch.tensor(fitted["center"], dtype=torch.float32)
    random_bases = []
    for seed in (31, 71, 113):
        generator = torch.Generator().manual_seed(seed)
        random_bases.append(torch.linalg.qr(torch.randn(*basis.shape, generator=generator)).Q)
    conditions = [("clean", 0.0, None, False)]
    for dose in (0.5, 1.0):
        conditions.append((f"concept-{dose}", dose, basis, False))
        for seed, q in zip((31, 71, 113), random_bases):
            conditions.extend(
                [
                    (f"random-{seed}-{dose}", dose, q, False),
                    (f"matched-{seed}-{dose}", dose, q, True),
                ]
            )
    predictions, norms, multipliers = [], [], []
    for start in range(0, len(values), 4):
        batch = SignalBatch(
            values[start : start + 4],
            tuple(data["trial_ids"][start : start + 4]),
            CHANNELS,
            200,
            RECIPE,
        )
        clean = lens.run_with_cache(batch, sites=("blocks.5.output", "blocks.11.output"))
        activation = clean.cache["blocks.5.output"]
        _, target_norm, _ = make_donor(activation, batch, basis, center, 1.0)
        row_predictions, row_norms, row_multipliers = [], [], []
        for label, dose, q, matched in conditions:
            if q is None:
                result = clean
                norm = gain = torch.zeros(len(batch.trial_ids))
            else:
                donor, norm, gain = make_donor(
                    activation, batch, q, center, dose, target_norm if matched else None
                )
                result = lens.run_with_interventions(
                    batch,
                    interventions=[Replacement(activation.site, donor)],
                    sites=("blocks.11.output",),
                )
            final = spatial_features(result.cache["blocks.11.output"]).reshape(
                len(batch.trial_ids), -1
            )
            row_predictions.append(final @ fitted["readout_coef"] + fitted["readout_intercept"])
            row_norms.append(norm.numpy())
            row_multipliers.append(gain.numpy())
        predictions.append(np.stack(row_predictions, axis=1))
        norms.append(np.stack(row_norms, axis=1))
        multipliers.append(np.stack(row_multipliers, axis=1))
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "responses.npz"
    np.savez_compressed(
        artifact,
        predictions=np.concatenate(predictions),
        norms=np.concatenate(norms),
        multipliers=np.concatenate(multipliers),
        conditions=np.array([c[0] for c in conditions]),
        trial_ids=data["trial_ids"],
        subjects=data["subjects"],
        labels=data["labels"],
        descriptors=data["descriptors"],
    )
    report = dict(
        model=name,
        partition="validation",
        runner_sha256=sha(__file__),
        artifact_sha256=sha(artifact),
        probes_manifest_sha256=sha(probes / "manifest.json"),
        prepared_manifest_sha256=sha(prepared / "manifest.json"),
        controls="Three rank-two random bases; separate per-trial norm-matched controls",
        note="Large norm-matching multipliers are retained for explicit diagnostics, not silently capped",
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(name, len(values), "validation trial responses saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--model", choices=("cbramod", "labram", "csbrain"), required=True)
    p.add_argument("--root", type=Path, default=Path("research"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.prepared, a.probes, a.model, a.root, a.output)
