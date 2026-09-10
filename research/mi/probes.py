"""Training-only ridge transforms; validation selects regularization, never test data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def fit_ridge(x, y, validation_x, validation_y, alphas=(0.001, 0.01, 0.1, 1.0, 10.0)):
    x, y, vx, vy = [np.asarray(v, dtype=np.float64) for v in (x, y, validation_x, validation_y)]
    if any(v.ndim != 2 or not np.isfinite(v).all() for v in (x, y, vx, vy)):
        raise ValueError("Expected finite two-dimensional regression arrays")
    if (
        len(x) != len(y)
        or len(vx) != len(vy)
        or x.shape[1] != vx.shape[1]
        or y.shape[1] != vy.shape[1]
    ):
        raise ValueError("Regression shape mismatch")
    if min(len(x), len(vx)) < 2 or not alphas or any(a <= 0 for a in alphas):
        raise ValueError("Need nonempty train/validation samples and positive regularization")
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    target_mean = y.mean(0)
    z = (x - mean) / scale
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    projected = u.T @ (y - target_mean)
    candidates, errors = [], []
    for alpha in alphas:
        standardized = vt.T @ ((singular / (singular**2 + len(x) * alpha))[:, None] * projected)
        coef = standardized / scale[:, None]
        intercept = target_mean - mean @ coef
        prediction = vx @ coef + intercept
        candidates.append((coef, intercept))
        errors.append(np.mean((prediction - vy) ** 2, axis=0))
    selected = np.argmin(errors, axis=0)
    coef = np.column_stack([candidates[a][0][:, j] for j, a in enumerate(selected)])
    intercept = np.array([candidates[a][1][j] for j, a in enumerate(selected)])
    return dict(
        coef=coef,
        intercept=intercept,
        mean=mean,
        scale=scale,
        alpha=np.asarray(alphas)[selected],
        validation_mse=np.asarray(errors),
    )


def load_features(path, expected_partition):
    with path.with_suffix(".json").open() as source:
        manifest = json.load(source)
    if manifest["partition"] != expected_partition:
        raise ValueError(f"Expected {expected_partition} feature artifact")
    if manifest["artifact_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Feature artifact hash mismatch")
    return np.load(path, allow_pickle=False)


def run(train_path, validation_path, output):
    train_meta = json.loads(train_path.with_suffix(".json").read_text())
    validation_meta = json.loads(validation_path.with_suffix(".json").read_text())
    for key in ("model", "checkpoint_sha256", "input_recipe", "sites", "pooling"):
        if train_meta[key] != validation_meta[key]:
            raise ValueError(f"Train/validation feature contract mismatch: {key}")
    for key in ("initialization", "random_seed"):
        if train_meta.get(key) != validation_meta.get(key):
            raise ValueError(f"Train/validation initialization mismatch: {key}")
    train, validation = (
        load_features(train_path, "train"),
        load_features(validation_path, "validation"),
    )
    if set(train["subjects"]) & set(validation["subjects"]):
        raise ValueError("Train/validation subject leakage")
    if not np.array_equal(train["channels"], validation["channels"]):
        raise ValueError("Feature channel mismatch")
    channels = list(train["channels"])
    # Frozen final-layer readouts: task margin, mu, beta, occipital alpha, global RMS.
    y = np.column_stack([train["labels"] * 2 - 1, train["descriptors"]])
    vy = np.column_stack([validation["labels"] * 2 - 1, validation["descriptors"]])
    readout = fit_ridge(
        train["final"].reshape(len(y), -1), y, validation["final"].reshape(len(vy), -1), vy
    )
    c3, c4 = channels.index("C3"), channels.index("C4")
    contrast = train["middle"][:, c4] - train["middle"][:, c3]
    vcontrast = validation["middle"][:, c4] - validation["middle"][:, c3]
    concept = fit_ridge(
        contrast, train["descriptors"][:, :2], vcontrast, validation["descriptors"][:, :2]
    )
    # Convert fitted native-coordinate coefficient columns to an orthonormal span.
    u, singular, _ = np.linalg.svd(concept["coef"], full_matrices=False)
    if singular[-1] <= singular[0] * 1e-8:
        raise ValueError("Two concept directions are not independently identifiable")
    basis = u[:, :2]
    center = train["middle"][:, [c3, c4]].mean(axis=(0, 1))
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "probes.npz"
    np.savez_compressed(
        artifact,
        **{f"readout_{k}": v for k, v in readout.items()},
        **{f"concept_{k}": v for k, v in concept.items()},
        basis=basis,
        center=center,
    )
    report = dict(
        model=train_meta["model"],
        checkpoint_sha256=train_meta["checkpoint_sha256"],
        initialization=train_meta.get("initialization", "pretrained"),
        random_seed=train_meta.get("random_seed"),
        train_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rank=2,
        intervention_sensors=["C3", "C4"],
        direction="Middle-layer C4-C3 feature contrast predicting mu/beta asymmetries",
        readout_targets=[
            "task_margin",
            "mu_asymmetry",
            "beta_asymmetry",
            "occipital_alpha",
            "global_rms",
        ],
        scope="Train-fitted coefficients; validation-selected alpha; no test evaluation",
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Frozen readouts and rank-two concept span saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.train, a.validation, a.output)
