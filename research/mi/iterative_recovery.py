"""Fixed-rank exploratory recovery curves; no original validation/test data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from probes import fit_ridge, load_features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def r2(y, prediction):
    denominator = ((y - y.mean(0)) ** 2).sum(0)
    if np.any(denominator <= 1e-12):
        raise ValueError("Degenerate target variance")
    return (1 - ((y - prediction) ** 2).sum(0) / denominator).tolist()


def directions(x, y, max_rank, alpha):
    """Fit-only directions, projected out of the accumulated erased span."""
    q = np.empty((x.shape[1], 0))
    for _ in range(max_rank // 2):
        residual = x - (x @ q) @ q.T
        fit = fit_ridge(residual, y, residual, y, alphas=(alpha,))
        coef = fit["coef"] - q @ (q.T @ fit["coef"])
        u, singular, _ = np.linalg.svd(coef, full_matrices=False)
        if singular[-1] <= max(singular[0] * 1e-8, 1e-12):
            raise ValueError("Concept directions exhausted before planned rank")
        q = np.column_stack((q, u[:, :2]))
        q = np.linalg.qr(q)[0]
        np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10, rtol=0)
    return q


def run(features, split, protocol, output):
    if output.exists():
        raise FileExistsError(output)
    policy = json.loads(protocol.read_text())
    partition = json.loads(split.read_text())
    data = load_features(features, "train")
    subjects = np.unique(data["subjects"])
    if set(subjects) != set(partition["train"]) or len(subjects) != 18:
        raise ValueError("Expected exactly original 18 training subjects")
    channels = list(data["channels"])
    x = (data["middle"][:, channels.index("C4")] - data["middle"][:, channels.index("C3")]).astype(
        np.float64
    )
    y = data["descriptors"].astype(np.float64)
    if y.shape[1] != 4:
        raise ValueError("Expected four physiological descriptors")
    ranks = policy["ranks"]
    if (
        ranks != sorted(set(ranks))
        or ranks[0] != 0
        or ranks[-1] >= x.shape[1]
        or any(r % 2 for r in ranks)
    ):
        raise ValueError("Invalid fixed ranks")
    rng = np.random.default_rng(policy["seed"])
    folds = np.array_split(rng.permutation(subjects), policy["outer_folds"])
    predictions, records, bases = {}, [], {}
    visited = np.zeros(len(y), dtype=int)
    for fold, evaluation in enumerate(folds):
        remaining = rng.permutation(np.setdiff1d(subjects, evaluation))
        validation = remaining[: policy["inner_validation_subjects"]]
        fitting = remaining[policy["inner_validation_subjects"] :]
        fit, val, test = [np.isin(data["subjects"], s) for s in (fitting, validation, evaluation)]
        assert not (fit & val).any() and not (fit & test).any() and not (val & test).any()
        candidates = {
            "concept": directions(x[fit], y[fit, :2], ranks[-1], policy["direction_alpha"])
        }
        for seed in policy["random_seeds"]:
            candidates[f"random_{seed}"] = np.linalg.qr(
                np.random.default_rng(seed).normal(size=(x.shape[1], ranks[-1]))
            )[0]
        center = x[fit].mean(0)
        centered = x - center
        energy = (centered[test] ** 2).sum()
        fold_rows = []
        for kind, all_q in candidates.items():
            bases[f"fold_{fold}_{kind}"] = all_q
            for rank in ranks:
                if rank == 0 and kind != "concept":
                    continue
                key = "clean" if rank == 0 else f"{kind}_rank{rank}"
                q = all_q[:, :rank]
                delta = (centered @ q) @ q.T
                residual = x - delta
                fitted = fit_ridge(
                    residual[fit], y[fit], residual[val], y[val], alphas=policy["readout_alphas"]
                )
                prediction = residual[test] @ fitted["coef"] + fitted["intercept"]
                predictions.setdefault(key, np.full_like(y, np.nan))[test] = prediction
                fold_rows.append(
                    dict(
                        condition=key,
                        rank=rank,
                        alpha=fitted["alpha"].tolist(),
                        removed_energy_fraction=float((delta[test] ** 2).sum() / energy),
                    )
                )
        records.append(
            dict(
                fold=fold,
                fitting_subjects=fitting.tolist(),
                validation_subjects=validation.tolist(),
                evaluation_subjects=evaluation.tolist(),
                conditions=fold_rows,
            )
        )
        visited[test] += 1
    assert (visited == 1).all() and all(np.isfinite(p).all() for p in predictions.values())
    report = dict(
        model=features.stem,
        folds=records,
        pooled_r2={k: r2(y, p) for k, p in predictions.items()},
        subject_r2=[
            dict(
                subject=int(s),
                scores={
                    k: r2(y[data["subjects"] == s], p[data["subjects"] == s])
                    for k, p in predictions.items()
                },
            )
            for s in subjects
        ],
        scope=policy["scope"],
        feature_sha256=sha(features),
        split_sha256=sha(split),
        protocol_sha256=sha(protocol),
        runner_sha256=sha(__file__),
        fitter_sha256=sha(Path(__file__).with_name("probes.py")),
    )
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "predictions.npz",
        **predictions,
        targets=y,
        subjects=data["subjects"],
        trial_ids=data["trial_ids"],
    )
    np.savez_compressed(output / "bases.npz", **bases)
    report["artifacts"] = {n: sha(output / n) for n in ("predictions.npz", "bases.npz")}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        features.stem,
        {
            k: v[:2]
            for k, v in report["pooled_r2"].items()
            if k == "clean" or k.startswith("concept")
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "split", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.features, args.split, args.protocol, args.output)
