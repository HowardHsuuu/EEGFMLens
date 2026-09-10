"""Native intermediate correction with frozen training-derived parameters."""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from eeglens import Replacement, Selection, SignalBatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase"))
from evaluate import SITES, model_loader, pooled


@dataclass
class Correction:
    site: str
    selection: Selection
    bias: torch.Tensor
    matrix: torch.Tensor | None = None
    reference_bias: torch.Tensor | None = None
    reference_matrix: torch.Tensor | None = None
    effects: list = field(default_factory=list)

    def apply(self, current, batch, layout, model_id):
        mask = self.selection.mask(current, batch, layout)
        candidate = (
            current + self.bias if self.matrix is None else current @ self.matrix + self.bias
        )
        delta = torch.where(mask, candidate - current, 0)
        norm = torch.linalg.vector_norm(delta.flatten(1), dim=1)
        target = norm
        multiplier = torch.ones_like(norm)
        if self.reference_matrix is not None:
            reference = torch.where(
                mask, current @ self.reference_matrix + self.reference_bias - current, 0
            )
            target = torch.linalg.vector_norm(reference.flatten(1), dim=1)
            multiplier = target / norm.clamp_min(1e-12)
        missing = (norm < 1e-12) & (target > 1e-8)
        multiplier = multiplier.masked_fill(missing, 0)
        changed = current + delta * multiplier.reshape((-1,) + (1,) * (current.ndim - 1))
        actual = torch.linalg.vector_norm((changed - current).flatten(1), dim=1)
        original = torch.linalg.vector_norm(current.flatten(1), dim=1)
        self.effects = [
            dict(
                actual_norm=float(a),
                target_norm=float(t),
                multiplier=float(m),
                relative_edit=float(a / o.clamp_min(1e-8)),
                valid=not bool(f) and bool(torch.isclose(a, t, rtol=1e-4, atol=1e-6)),
            )
            for a, t, m, o, f in zip(actual, target, multiplier, original, missing)
        ]
        assert torch.isfinite(changed).all()
        return changed


def run(model, checkpoint, previous, features, maps_dir, output):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / f"{model}.json").exists()
    manifest = json.loads((previous / "prepared/manifest.json").read_text())
    rows = manifest["rows"]
    subjects = np.array([r["subject"] for r in rows])
    x = np.load(previous / "prepared/inputs.npz")["x"]
    readouts = np.load(previous / "results/readouts.npz")
    tokens = np.load(features / f"{model}.npz")["features"]
    maps = np.load(maps_dir / "maps.npz")
    maps_meta = json.loads((maps_dir / "maps.json").read_text())
    assert (
        hashlib.sha256((features / f"{model}.npz").read_bytes()).hexdigest()
        == maps_meta["feature_sha256"][model]
    )
    lens = model_loader(model, checkpoint, False)
    selection = Selection(patches=tuple(range(15)))
    records = []
    for subject in range(1, 9):
        ids_all = np.flatnonzero(subjects == subject)
        w = readouts[f"{model}_s{subject}_weight"]
        b = readouts[f"{model}_s{subject}_intercept"]
        for gi, gain in [(1, 0.75), (2, 1.0), (3, 1.5)]:
            for start in range(0, len(ids_all), 8):
                ids = ids_all[start : start + 8]
                batch = SignalBatch(
                    torch.from_numpy((x[ids] * gain).astype(np.float32)),
                    tuple(f"s{subject}:w{rows[i]['window']}" for i in ids),
                    (rows[ids[0]]["channel"],),
                    200,
                    manifest["recipe"] + ":gain-calibration-counterfactual",
                )
                corrupt = lens.run_with_cache(batch, sites=SITES)
                np.testing.assert_allclose(
                    pooled(corrupt.output).numpy(), tokens[gi, ids].mean(1), rtol=1e-5, atol=1e-5
                )
                donor_batch = SignalBatch(
                    torch.from_numpy(x[ids]),
                    batch.trial_ids,
                    batch.channels,
                    200,
                    batch.preprocessing_id,
                )
                donor = lens.run_with_cache(donor_batch, sites=SITES)
                clean_scores = pooled(donor.output).numpy() @ w + b
                corrupt_scores = pooled(corrupt.output).numpy() @ w + b
                for site in SITES:
                    key = f"{model}:s{subject}:{site}"
                    identity = lens.run_with_interventions(
                        batch, interventions=[Replacement(site, corrupt.cache[site])]
                    )
                    torch.testing.assert_close(
                        identity.output, corrupt.output, rtol=1e-5, atol=1e-6
                    )
                    oracle = lens.run_with_interventions(
                        batch, interventions=[Replacement(site, donor.cache[site], selection)]
                    )
                    oracle_scores = pooled(oracle.output).numpy() @ w + b
                    for kind in ["mean", "random_mean", "affine", "permuted_affine"]:
                        bias = torch.from_numpy(maps[key + ":" + kind + ":bias"])
                        matrix = (
                            torch.from_numpy(maps[key + ":" + kind + ":matrix"])
                            if "affine" in kind
                            else None
                        )
                        edit = Correction(site, selection, bias, matrix)
                        if kind == "permuted_affine":
                            edit.reference_matrix = torch.from_numpy(maps[key + ":affine:matrix"])
                            edit.reference_bias = torch.from_numpy(maps[key + ":affine:bias"])
                        result = lens.run_with_interventions(batch, interventions=[edit])
                        scores = pooled(result.output).numpy() @ w + b
                        for j, index in enumerate(ids):
                            records.append(
                                dict(
                                    subject=subject,
                                    row=int(index),
                                    gain=gain,
                                    site=site,
                                    kind=kind,
                                    n2=rows[index]["n2"],
                                    clean_margin=float(clean_scores[j]),
                                    corrupt_margin=float(corrupt_scores[j]),
                                    oracle_margin=float(oracle_scores[j]),
                                    margin=float(scores[j]) if edit.effects[j]["valid"] else None,
                                    **edit.effects[j],
                                )
                            )
                full = lens.run_with_interventions(
                    batch, interventions=[Replacement(SITES[-1], donor.cache[SITES[-1]])]
                )
                torch.testing.assert_close(full.output, donor.output, rtol=1e-5, atol=1e-6)
                after = lens.run_with_cache(batch, sites=[])
                torch.testing.assert_close(after.output, corrupt.output, rtol=1e-5, atol=1e-6)
            print(model, "subject", subject, "gain", gain, "complete", flush=True)
    doc = dict(
        records=records,
        model=lens.manifest,
        maps_sha256=hashlib.sha256((maps_dir / "maps.npz").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / f"{model}.json").write_text(json.dumps(doc, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cbramod", "labram"], required=True)
    for key in ["checkpoint", "previous", "features", "maps", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    run(a.model, a.checkpoint, a.previous, a.features, a.maps, a.output)
