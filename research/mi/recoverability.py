"""Training-subject cross-fitting of physiological concept stability and recoverability."""

import argparse
import json
from pathlib import Path

import numpy as np
from extract import sha
from probes import fit_ridge, load_features


def predict(fitted, x):
    return x @ fitted["coef"] + fitted["intercept"]


def scores(target, prediction):
    denominator = ((target - target.mean(0)) ** 2).sum(0)
    if (denominator <= 1e-12).any():
        raise ValueError("Degenerate target variance")
    return (1 - ((prediction - target) ** 2).sum(0) / denominator).tolist()


def run(features, split, protocol, output):
    policy = json.loads(protocol.read_text())
    partition = json.loads(split.read_text())
    data = load_features(features, "train")
    subjects = np.unique(data["subjects"])
    if set(subjects) != set(partition["train"]) or len(subjects) != 18:
        raise ValueError("Expected only the 18 original training subjects")
    rng = np.random.default_rng(policy["seed"])
    folds = np.array_split(rng.permutation(subjects), policy["outer_folds"])
    channels = list(data["channels"])
    x = (data["middle"][:, channels.index("C4")] - data["middle"][:, channels.index("C3")]).astype(
        np.float64
    )
    y = data["descriptors"][:, :2].astype(np.float64)
    names = [
        "clean",
        "frozen_erased",
        "recovered",
        *[f"random_{s}" for s in policy["random_seeds"]],
    ]
    predictions = {name: np.full_like(y, np.nan) for name in names}
    records = []
    visited = np.zeros(len(y), dtype=int)
    for index, test_subjects in enumerate(folds):
        remaining = rng.permutation(np.setdiff1d(subjects, test_subjects))
        val_subjects = remaining[: policy["inner_validation_subjects"]]
        fit_subjects = remaining[policy["inner_validation_subjects"] :]
        assert len(fit_subjects) == policy["inner_fit_subjects"]
        fit, val, test = [
            np.isin(data["subjects"], s) for s in (fit_subjects, val_subjects, test_subjects)
        ]
        assert not (fit & val).any() and not (fit & test).any() and not (val & test).any()
        fitted = fit_ridge(x[fit], y[fit], x[val], y[val], alphas=policy["regularization"])
        u, singular, _ = np.linalg.svd(fitted["coef"], full_matrices=False)
        if singular[-1] <= singular[0] * 1e-8:
            raise ValueError("Degenerate concept coefficient span")
        q = u[:, :2]
        erased = x - (x @ q) @ q.T
        # Both sensors use the same center, which cancels in C4-C3 contrast.
        np.testing.assert_allclose(erased @ fitted["coef"], 0, atol=1e-10, rtol=0)
        recovery = fit_ridge(
            erased[fit], y[fit], erased[val], y[val], alphas=policy["regularization"]
        )
        predictions["clean"][test] = predict(fitted, x[test])
        predictions["frozen_erased"][test] = predict(fitted, erased[test])
        predictions["recovered"][test] = predict(recovery, erased[test])
        for seed in policy["random_seeds"]:
            random_q = np.linalg.qr(np.random.default_rng(seed).normal(size=(x.shape[1], 2)))[0]
            residual = x - (x @ random_q) @ random_q.T
            random_fit = fit_ridge(
                residual[fit], y[fit], residual[val], y[val], alphas=policy["regularization"]
            )
            predictions[f"random_{seed}"][test] = predict(random_fit, residual[test])
        visited[test] += 1
        records.append(
            dict(
                fold=index,
                fit_subjects=fit_subjects.tolist(),
                validation_subjects=val_subjects.tolist(),
                evaluation_subjects=test_subjects.tolist(),
                clean_alpha=fitted["alpha"].tolist(),
                recovered_alpha=recovery["alpha"].tolist(),
                scores={name: scores(y[test], p[test]) for name, p in predictions.items()},
            )
        )
    assert (visited == 1).all() and all(np.isfinite(p).all() for p in predictions.values())
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "predictions.npz"
    np.savez_compressed(
        artifact, **predictions, targets=y, subjects=data["subjects"], trial_ids=data["trial_ids"]
    )
    report = dict(
        model=features.stem,
        folds=records,
        pooled_out_of_fold_r2={name: scores(y, p) for name, p in predictions.items()},
        subject_r2=[
            dict(
                subject=int(s),
                scores={
                    name: scores(y[data["subjects"] == s], p[data["subjects"] == s])
                    for name, p in predictions.items()
                },
            )
            for s in subjects
        ],
        artifact_sha256=sha(artifact),
        feature_sha256=sha(features),
        split_sha256=sha(split),
        protocol_sha256=sha(protocol),
        runner_sha256=sha(__file__),
        fitter_sha256=sha(Path(__file__).with_name("probes.py")),
        scope=policy["interpretation"],
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(features.stem, report["pooled_out_of_fold_r2"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.features, a.split, a.protocol, a.output)
