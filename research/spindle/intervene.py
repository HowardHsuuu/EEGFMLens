"""Native internal interventions with per-window strength-matched controls."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from baselines import parameters, splits
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeglens import Replacement, Selection, SignalBatch, load_cbramod, load_labram


def unit(x):
    norm = np.linalg.norm(x)
    if norm < 1e-10 or not np.isfinite(norm):
        raise ValueError("Degenerate direction")
    return x / norm


def slow_wave_score(inputs):
    sos = butter(3, [0.5, 2], btype="bandpass", fs=200, output="sos")
    output = []
    for x in inputs[:, 0].reshape(len(inputs), -1) * 100:
        x = sosfiltfilt(sos, x)
        negative = x < 0
        starts = np.flatnonzero(np.diff(negative.astype(int), prepend=0) == 1)
        ends = np.flatnonzero(np.diff(negative.astype(int), append=0) == -1) + 1
        output.append(
            sum(
                50 <= b - a <= 200 and a >= 200 and b <= 2800 and x[a:b].min() < -40
                for a, b in zip(starts, ends)
            )
        )
    return np.asarray(output)


@dataclass
class MatchedErasure:
    site: str
    basis: torch.Tensor
    center: torch.Tensor
    reference: torch.Tensor
    selection: Selection = Selection(patches=tuple(range(15)))
    last_norm: torch.Tensor | None = None
    last_reference_norm: torch.Tensor | None = None

    def apply(self, current, batch, layout, model_id):
        mask = self.selection.mask(current, batch, layout)
        centered = current - self.center
        candidate = ((centered @ self.basis) @ self.basis.T) * mask
        reference = ((centered @ self.reference) @ self.reference.T) * mask
        dimensions = tuple(range(1, current.ndim))
        actual_norm = torch.linalg.vector_norm(candidate, dim=dimensions, keepdim=True)
        target_norm = torch.linalg.vector_norm(reference, dim=dimensions, keepdim=True)
        if torch.any((actual_norm < 1e-12) & (target_norm > 1e-8)):
            raise ValueError("Cannot match zero control perturbation to nonzero target")
        delta = candidate * (target_norm / actual_norm.clamp_min(1e-12))
        self.last_norm = torch.linalg.vector_norm(delta, dim=dimensions).detach()
        self.last_reference_norm = target_norm.flatten().detach()
        return current - delta


def run(model, checkpoint, prepared, features, results):
    torch.set_num_threads(2)
    manifest = json.loads((prepared / "manifest.json").read_text())
    rows = manifest["rows"]
    inputs = np.load(prepared / "inputs.npz")["x"]
    subjects = np.array([r["subject"] for r in rows])
    stage = np.array([r["n2"] for r in rows])
    concept = np.array([r["spindle"] for r in rows])
    spectral = np.array([r["spectral"] for r in rows])
    slow = slow_wave_score(inputs)
    feature = np.load(features / f"{model}.npz")
    readouts = np.load(results / "readouts.npz")
    baselines = json.loads((results / "baselines.json").read_text())
    lens = (load_cbramod if model == "cbramod" else load_labram)(checkpoint)
    output, fitted = [], {}
    for held in range(1, 9):
        train, validation, test = splits(subjects, held)
        eligible = train & (stage == 1) & (concept >= 0)
        selected = next(
            r
            for r in baselines
            if r["model"] == model and r["held_subject"] == held and r["task"] == "spindle_given_n2"
        )
        site = selected["site"]
        h = feature[site].astype(float)
        center = h[eligible].mean(0)
        raw = unit(readouts[f"{model}_s{held}_concept_weight"])
        standard = StandardScaler().fit(spectral[eligible])
        z = standard.transform(spectral)
        nuisance = Ridge(alpha=10).fit(z[eligible], h[eligible])
        residual = h - nuisance.predict(z)
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=selected["C"], class_weight="balanced", max_iter=3000),
        )
        probe.fit(residual[eligible], concept[eligible])
        residual_weight, residual_intercept = parameters(probe)
        span, singular, _ = np.linalg.svd(nuisance.coef_, full_matrices=False)
        span = span[:, singular > singular.max() * 1e-6]
        adjusted = unit(residual_weight - span @ (span.T @ residual_weight))
        sigma = unit(Ridge(alpha=10).fit(h[eligible] - center, spectral[eligible, 3]).coef_)
        slow_direction = unit(Ridge(alpha=10).fit(h[eligible] - center, slow[eligible]).coef_)
        rng = np.random.default_rng(4311 + 31 * held)
        shuffled = probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=selected["C"], class_weight="balanced", max_iter=3000),
        )
        shuffled.fit(h[eligible], rng.permutation(concept[eligible]))
        shuffled_weight, _ = parameters(shuffled)
        directions = dict(
            spindle=raw,
            adjusted_spindle=adjusted,
            sigma=sigma,
            slow_wave=slow_direction,
            shuffled=unit(shuffled_weight),
        )
        directions.update(
            {
                f"random_{i}": unit(
                    np.random.default_rng(4311 + 31 * held + i).normal(size=len(raw))
                )
                for i in range(10)
            }
        )
        fitted[f"s{held}_center"] = center
        fitted[f"s{held}_nuisance_coef"] = nuisance.coef_
        fitted[f"s{held}_nuisance_intercept"] = nuisance.intercept_
        fitted[f"s{held}_spectral_mean"] = standard.mean_
        fitted[f"s{held}_spectral_scale"] = standard.scale_
        fitted[f"s{held}_residual_probe_weight"] = residual_weight
        fitted[f"s{held}_residual_probe_intercept"] = np.array(residual_intercept)
        for name, q in directions.items():
            fitted[f"s{held}_{name}"] = q
        stage_weight = readouts[f"{model}_s{held}_weight"]
        intercept = readouts[f"{model}_s{held}_intercept"]
        ids = np.flatnonzero(test)
        for offset in range(0, len(ids), 8):
            current_ids = ids[offset : offset + 8]
            batch = SignalBatch(
                torch.from_numpy(inputs[current_ids]),
                tuple(f"s{held}:w{rows[i]['window']}" for i in current_ids),
                (rows[current_ids[0]]["channel"],),
                200,
                manifest["recipe"],
            )
            clean = lens.run_with_cache(batch, sites=[site])
            pooled = clean.output.mean((1, 2)) if model == "cbramod" else clean.output.mean(1)
            np.testing.assert_allclose(
                pooled.numpy(), feature["output"][current_ids], rtol=1e-5, atol=1e-5
            )
            baseline_margin = pooled.numpy() @ stage_weight + intercept
            identity = lens.run_with_interventions(
                batch, interventions=[Replacement(site, clean.cache[site])]
            )
            torch.testing.assert_close(identity.output, clean.output, atol=1e-6, rtol=1e-5)
            for name, q in directions.items():
                edit = MatchedErasure(
                    site,
                    torch.tensor(q[:, None], dtype=torch.float32),
                    torch.tensor(center, dtype=torch.float32),
                    torch.tensor(raw[:, None], dtype=torch.float32),
                )
                changed = lens.run_with_interventions(batch, interventions=[edit])
                pooled = (
                    changed.output.mean((1, 2)) if model == "cbramod" else changed.output.mean(1)
                )
                margin = pooled.numpy() @ stage_weight + intercept
                torch.testing.assert_close(
                    edit.last_norm, edit.last_reference_norm, atol=1e-5, rtol=1e-5
                )
                for j, row_id in enumerate(current_ids):
                    output.append(
                        dict(
                            row=int(row_id),
                            subject=held,
                            site=site,
                            control=name,
                            baseline=float(baseline_margin[j]),
                            changed=float(margin[j]),
                            norm=float(edit.last_norm[j]),
                            target_norm=float(edit.last_reference_norm[j]),
                            residual_concept_score=float(
                                residual[row_id] @ residual_weight + residual_intercept
                            ),
                            adjusted_direction_score=float(h[row_id] @ adjusted),
                            slow_wave_count=int(slow[row_id]),
                        )
                    )
        print(model, "finished held subject", held, flush=True)
    (results / f"{model}-interventions.json").write_text(json.dumps(output, allow_nan=False) + "\n")
    np.savez_compressed(results / f"{model}-directions.npz", **fitted)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cbramod", "labram"], required=True)
    p.add_argument("--checkpoint", required=True)
    for name in ["prepared", "features", "results"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.model, a.checkpoint, a.prepared, a.features, a.results)
