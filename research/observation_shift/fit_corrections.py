"""Fit shared, training-only internal maps; no held-out labels or clean donor."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

SITES = ["embedding.output", *[f"blocks.{i}.output" for i in [2, 5, 8, 11]]]


def ridge_candidates(x, y):
    xm, xs = x.mean(0), x.std(0).clip(1e-8)
    ym, ys = y.mean(0), y.std(0).clip(1e-8)
    xx = (x - xm) / xs
    yy = (y - ym) / ys
    gram = xx.T @ xx / len(xx)
    cross = xx.T @ yy / len(xx)
    result = []
    for alpha in [0.01, 0.1, 1.0]:
        coef = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), cross)
        matrix = coef * ys[None, :] / xs[:, None]
        bias = ym - xm @ matrix
        result.append((alpha, matrix, bias, ys))
    return result


def fit(features, previous, output):
    output.mkdir(parents=True, exist_ok=False)
    subjects = np.array(
        [
            r["subject"]
            for r in json.loads((previous / "prepared/manifest.json").read_text())["rows"]
        ]
    )
    maps, records = {}, []
    with threadpool_limits(limits=2):
        for model in ["cbramod", "labram"]:
            layers = np.load(features / f"{model}.npz")["layers"]
            for held in range(1, 9):
                validation = [held % 8 + 1, (held + 1) % 8 + 1]
                train = np.flatnonzero(~np.isin(subjects, [held, *validation]))
                assert not np.any(subjects[train] == held)
                for si, site in enumerate(SITES):
                    x = (
                        np.concatenate([layers[g, train, si] for g in [0, 4]])
                        .reshape(-1, 200)
                        .astype(float)
                    )
                    target = layers[2, train, si].astype(float)
                    y = np.concatenate([target, target]).reshape(-1, 200)
                    key = f"{model}:s{held}:{site}"
                    mean = (y - x).mean(0)
                    rng = np.random.default_rng(9137 + held * 100 + si)
                    direction = rng.normal(size=200)
                    direction *= np.linalg.norm(mean) / np.linalg.norm(direction)
                    maps[key + ":mean:bias"] = mean.astype(np.float32)
                    maps[key + ":random_mean:bias"] = direction.astype(np.float32)
                    permutation = rng.permutation(len(train))
                    shuffled = np.concatenate([target[permutation], target[permutation]]).reshape(
                        -1, 200
                    )
                    for kind, yy in [("affine", y), ("permuted_affine", shuffled)]:
                        candidates = []
                        for alpha, matrix, bias, ys in ridge_candidates(x, yy):
                            errors = []
                            for val in validation:
                                mask = subjects == val
                                for g in [0, 4]:
                                    prediction = layers[g, mask, si].astype(float) @ matrix + bias
                                    errors.append(
                                        np.mean(((prediction - layers[2, mask, si]) / ys) ** 2)
                                    )
                            candidates.append((float(np.mean(errors)), alpha, matrix, bias))
                        error, alpha, matrix, bias = min(candidates, key=lambda a: a[0])
                        maps[key + ":" + kind + ":matrix"] = matrix.astype(np.float32)
                        maps[key + ":" + kind + ":bias"] = bias.astype(np.float32)
                        records.append(
                            dict(
                                model=model,
                                held=held,
                                site=site,
                                kind=kind,
                                alpha=alpha,
                                validation_mse=error,
                                train_rows=train.tolist(),
                                validation_subjects=validation,
                            )
                        )
                print(model, "fit held subject", held, flush=True)
    np.savez_compressed(output / "maps.npz", **maps)
    metadata = dict(
        records=records,
        sites=SITES,
        protocol_sha256=hashlib.sha256(
            Path(__file__).with_name("PROTOCOL.md").read_bytes()
        ).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        feature_sha256={
            m: hashlib.sha256((features / f"{m}.npz").read_bytes()).hexdigest()
            for m in ["cbramod", "labram"]
        },
    )
    (output / "maps.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["features", "previous", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    fit(a.features, a.previous, a.output)
