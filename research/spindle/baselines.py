"""Subject-disjoint frozen readouts and conditional spindle probes."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def splits(subjects, held):
    validation = [held % 8 + 1, (held + 1) % 8 + 1]
    train = ~np.isin(subjects, [held, *validation])
    val = np.isin(subjects, validation)
    test = subjects == held
    assert not (train & val).any() and not (train & test).any() and not (val & test).any()
    assert train.sum() + val.sum() + test.sum() == len(subjects)
    return train, val, test


def fit(x, y, train, val, criterion="balanced_accuracy"):
    candidates = []
    assert len(np.unique(y[train])) == len(np.unique(y[val])) == 2
    for c in [0.1, 1.0, 10.0]:
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c, class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=4311
            ),
        )
        estimator.fit(x[train], y[train])
        score = estimator.decision_function(x[val])
        metric = (
            roc_auc_score(y[val], score)
            if criterion == "auc"
            else balanced_accuracy_score(y[val], score >= 0)
        )
        candidates.append((metric, c, estimator))
    return max(candidates, key=lambda candidate: candidate[0])


def parameters(estimator):
    scale = estimator[0]
    linear = estimator[1]
    weight = linear.coef_[0] / scale.scale_
    intercept = linear.intercept_[0] - scale.mean_ @ weight
    return weight, float(intercept)


def metrics(y, score):
    return dict(
        n=len(y),
        positive=int(y.sum()),
        balanced_accuracy=float(balanced_accuracy_score(y, score >= 0)),
        auc=float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
    )


def run(prepared, features, output):
    manifest = json.loads((prepared / "manifest.json").read_text())
    rows = manifest["rows"]
    subjects = np.array([r["subject"] for r in rows])
    stage = np.array([r["n2"] for r in rows])
    concept = np.array([r["spindle"] for r in rows])
    reviewed_n2 = (concept >= 0) & (stage == 1)
    spectral = np.array([r["spectral"] for r in rows])
    output.mkdir(parents=True, exist_ok=True)
    results, all_predictions, coefficients = [], {}, {}
    for name in ["spectral", "cbramod", "labram", "cbramod-random", "labram-random"]:
        bundle = None if name == "spectral" else np.load(features / f"{name}.npz")
        if bundle is not None:
            assert np.array_equal(bundle["indices"], np.arange(len(rows)))
        x = spectral if bundle is None else bundle["output"]
        oof = np.full(len(rows), np.nan)
        for held in range(1, 9):
            train, val, test = splits(subjects, held)
            validation_score, c, estimator = fit(x, stage, train, val)
            score = estimator.decision_function(x[test])
            oof[test] = score
            weight, intercept = parameters(estimator)
            coefficients[f"{name}_s{held}_weight"] = weight
            coefficients[f"{name}_s{held}_intercept"] = np.array(intercept)
            results.append(
                dict(
                    model=name,
                    held_subject=held,
                    task="n2",
                    C=c,
                    validation_score=float(validation_score),
                    **metrics(stage[test], score),
                )
            )
            if name in ["cbramod", "labram", "spectral"]:
                eligible_train, eligible_val, eligible_test = [
                    mask & reviewed_n2 for mask in (train, val, test)
                ]
                candidates = []
                sites = (
                    ["spectral"] if name == "spectral" else ["blocks.5.output", "blocks.8.output"]
                )
                for site in sites:
                    h = spectral if name == "spectral" else bundle[site]
                    score_val, cp, probe = fit(h, concept, eligible_train, eligible_val, "auc")
                    candidates.append((score_val, site, cp, probe, h))
                selected = max(candidates, key=lambda candidate: candidate[0])
                score_val, site, cp, probe, h = selected
                scores = probe.decision_function(h[eligible_test])
                weight, intercept = parameters(probe)
                key = f"{name}_s{held}_concept"
                coefficients[key + "_weight"] = weight
                coefficients[key + "_intercept"] = np.array(intercept)
                coefficients[key + "_center"] = h[eligible_train].mean(0)
                results.append(
                    dict(
                        model=name,
                        held_subject=held,
                        task="spindle_given_n2",
                        site=site,
                        C=cp,
                        validation_score=float(score_val),
                        **metrics(concept[eligible_test], scores),
                    )
                )
        assert np.isfinite(oof).all()
        all_predictions[name] = oof
        print(
            name,
            "mean subject balanced accuracy",
            np.mean(
                [
                    r["balanced_accuracy"]
                    for r in results
                    if r["model"] == name and r["task"] == "n2"
                ]
            ),
            flush=True,
        )
    (output / "baselines.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "readouts.npz", **coefficients)
    np.savez_compressed(output / "predictions.npz", **all_predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "features", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.prepared, args.features, args.output)
