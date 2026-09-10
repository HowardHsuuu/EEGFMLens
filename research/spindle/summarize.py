"""Subject-level outcomes, matching diagnostics and exploratory bootstrap intervals."""

import argparse
import json
from pathlib import Path

import numpy as np
from baselines import splits
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def interval(values):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all() and len(values) > 0
    rng = np.random.default_rng(4311)
    means = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return dict(
        mean=float(values.mean()),
        low=float(np.quantile(means, 0.025)),
        high=float(np.quantile(means, 0.975)),
        subjects=len(values),
    )


def run(prepared, features, results):
    rows = json.loads((prepared / "manifest.json").read_text())["rows"]
    subjects = np.array([r["subject"] for r in rows])
    stage = np.array([r["n2"] for r in rows])
    concept = np.array([r["spindle"] for r in rows])
    spectral = np.array([r["spectral"] for r in rows])
    baseline = json.loads((results / "baselines.json").read_text())
    readouts = np.load(results / "readouts.npz")
    summary = {
        "baseline": {},
        "concept": {},
        "matching": [],
        "interventions": {},
        "paired_control_differences": {},
    }
    for model in ["spectral", "cbramod", "labram", "cbramod-random", "labram-random"]:
        selected = [r for r in baseline if r["model"] == model and r["task"] == "n2"]
        summary["baseline"][model] = {
            m: interval([r[m] for r in selected]) for m in ["balanced_accuracy", "auc"]
        }
        selected = [r for r in baseline if r["model"] == model and r["task"] == "spindle_given_n2"]
        if selected:
            summary["concept"][model] = interval([r["auc"] for r in selected])
    summary["baseline"]["majority"] = {
        metric: interval([0.5] * 8) for metric in ["balanced_accuracy", "auc"]
    }
    assert all(len(np.unique(stage[subjects == held])) == 2 for held in range(1, 9))
    summary["pretrained_minus_random"] = {}
    for model in ["cbramod", "labram"]:

        def subject_metrics(name):
            return np.array(
                [
                    r["balanced_accuracy"]
                    for r in sorted(baseline, key=lambda r: r["held_subject"])
                    if r["model"] == name and r["task"] == "n2"
                ]
            )

        summary["pretrained_minus_random"][model] = interval(
            subject_metrics(model) - subject_metrics(model + "-random")
        )
    pairs = {}
    for held in range(1, 9):
        train, val, test = splits(subjects, held)
        eligible = train & (stage == 1) & (concept >= 0)
        z = StandardScaler().fit(spectral[eligible]).transform(spectral)
        positive = np.flatnonzero(test & (stage == 1) & (concept == 1))
        negative = np.flatnonzero(test & (stage == 1) & (concept == 0))
        cost = cdist(z[positive], z[negative])
        ip, ine = linear_sum_assignment(cost)
        keep = cost[ip, ine] <= np.sqrt(7)
        pos, neg = positive[ip[keep]], negative[ine[keep]]
        pairs[held] = (pos, neg)
        summary["matching"].append(
            dict(
                subject=held,
                pairs=len(pos),
                positive_available=len(positive),
                negative_available=len(negative),
                mean_distance=float(cost[ip[keep], ine[keep]].mean()) if len(pos) else None,
                mean_standardized_covariate_difference=(z[pos] - z[neg]).mean(0).tolist()
                if len(pos)
                else None,
            )
        )
    raw_effects = {}
    for model in ["cbramod", "labram"]:
        data = json.loads((results / f"{model}-interventions.json").read_text())
        feature = np.load(features / f"{model}.npz")["output"]
        controls = sorted(set(r["control"] for r in data))
        scores, original, norms, relative_norms = {}, {}, {}, {}
        for control in controls:
            selected = [r for r in data if r["control"] == control]
            assert sorted(r["row"] for r in selected) == list(range(len(rows)))
            scores[control] = np.array(
                [r["changed"] for r in sorted(selected, key=lambda r: r["row"])]
            )
            original[control] = np.array(
                [r["baseline"] for r in sorted(selected, key=lambda r: r["row"])]
            )
            norms[control] = max(abs(r["norm"] - r["target_norm"]) for r in selected)
            relative_norms[control] = max(
                abs(r["norm"] - r["target_norm"]) / max(r["target_norm"], 1e-8) for r in selected
            )
        clean = original["spindle"]
        assert all(np.array_equal(clean, v) for v in original.values())
        scores["random_mean"] = np.mean([scores[f"random_{i}"] for i in range(10)], axis=0)
        model_results = {}
        for control, changed in scores.items():
            per_subject = []
            for held in range(1, 9):
                train, val, test = splits(subjects, held)
                scale = (feature[train] @ readouts[f"{model}_s{held}_weight"]).std()
                delta = (changed - clean) / scale
                pos, neg = pairs[held]
                per_subject.append(
                    dict(
                        subject=held,
                        balanced_accuracy_drop=float(
                            balanced_accuracy_score(stage[test], clean[test] >= 0)
                            - (
                                np.mean(
                                    [
                                        balanced_accuracy_score(
                                            stage[test], scores[f"random_{i}"][test] >= 0
                                        )
                                        for i in range(10)
                                    ]
                                )
                                if control == "random_mean"
                                else balanced_accuracy_score(stage[test], changed[test] >= 0)
                            )
                        ),
                        all_margin_change=float(delta[test].mean()),
                        matched_selectivity=float((delta[pos] - delta[neg]).mean())
                        if len(pos)
                        else None,
                    )
                )
            raw_effects[(model, control)] = np.array(
                [
                    r["matched_selectivity"]
                    for r in per_subject
                    if r["matched_selectivity"] is not None
                ]
            )
            model_results[control] = dict(
                selectivity=interval(raw_effects[(model, control)]),
                accuracy_drop=interval([r["balanced_accuracy_drop"] for r in per_subject]),
                per_subject=per_subject,
                maximum_norm_error=norms.get(control),
                maximum_relative_norm_error=relative_norms.get(control),
            )
        summary["interventions"][model] = model_results
        summary["paired_control_differences"][model] = {
            control: interval(raw_effects[(model, "spindle")] - raw_effects[(model, control)])
            for control in ["random_mean", "sigma", "slow_wave", "shuffled", "adjusted_spindle"]
        }
        selected = sorted([r for r in data if r["control"] == "spindle"], key=lambda r: r["row"])
        for key in ["residual_concept_score", "adjusted_direction_score"]:
            score = np.array([r[key] for r in selected])
            aucs = []
            for held in range(1, 9):
                take = (subjects == held) & (stage == 1) & (concept >= 0)
                aucs.append(roc_auc_score(concept[take], score[take]))
            summary["concept"][model + "_" + key] = interval(aucs)
        # Independent second-rater sensitivity; no thresholds or sites reselected.
        second_aucs = []
        for held in range(1, 7):
            chosen = next(
                r
                for r in baseline
                if r["model"] == model
                and r["held_subject"] == held
                and r["task"] == "spindle_given_n2"
            )
            h = np.load(features / f"{model}.npz")[chosen["site"]]
            ids = [
                i
                for i, r in enumerate(rows)
                if r["subject"] == held
                and r["n2"] == 1
                and r["expert2_seconds"] is not None
                and (r["expert2_seconds"] == 0 or r["expert2_seconds"] >= 0.5)
            ]
            labels = np.array([rows[i]["expert2_seconds"] >= 0.5 for i in ids])
            if len(np.unique(labels)) == 2:
                second_aucs.append(
                    roc_auc_score(labels, h[ids] @ readouts[f"{model}_s{held}_concept_weight"])
                )
        summary["concept"][model + "_expert2"] = interval(second_aucs)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "baseline": summary["baseline"],
                "concept": summary["concept"],
                "matching_pairs": [r["pairs"] for r in summary["matching"]],
                "spindle_vs_controls": summary["paired_control_differences"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "features", "results"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.prepared, a.features, a.results)
