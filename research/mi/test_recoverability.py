"""A label-leakage regression check for the exploratory outer-subject evaluation."""

import json
from pathlib import Path

import numpy as np
from extract import sha
from recoverability import run


def test_outer_labels_cannot_change_outer_predictions(tmp_path):
    source = Path(__file__).parent
    split = source / "split-v1.json"
    protocol = source / "recovery_protocol.json"
    subjects = np.repeat(json.loads(split.read_text())["train"], 8)
    rng = np.random.default_rng(453)
    middle = rng.normal(size=(len(subjects), 2, 200))
    contrast = middle[:, 1] - middle[:, 0]
    target = contrast[:, :2] + rng.normal(scale=0.2, size=(len(subjects), 2))
    features = tmp_path / "fixture.npz"

    def save(y):
        np.savez_compressed(
            features,
            subjects=subjects,
            middle=middle,
            descriptors=y,
            channels=np.array(["C3", "C4"]),
            trial_ids=np.array([f"t{i}" for i in range(len(subjects))]),
        )
        features.with_suffix(".json").write_text(
            json.dumps(dict(partition="train", artifact_sha256=sha(features)))
        )

    save(target)
    run(features, split, protocol, tmp_path / "before")
    report = json.loads((tmp_path / "before/summary.json").read_text())
    held_out = report["folds"][0]["evaluation_subjects"]
    mask = np.isin(subjects, held_out)
    altered = target.copy()
    altered[mask] = altered[mask] * 5 + 100
    save(altered)
    run(features, split, protocol, tmp_path / "after")
    before = np.load(tmp_path / "before/predictions.npz")
    after = np.load(tmp_path / "after/predictions.npz")
    for name in ("clean", "frozen_erased", "recovered", "random_31", "random_71", "random_113"):
        np.testing.assert_array_equal(before[name][mask], after[name][mask])
    # Each fold must cover its six subjects, with all three roles pairwise disjoint.
    evaluated = []
    for fold in report["folds"]:
        fit, val, test = [
            set(fold[k]) for k in ("fit_subjects", "validation_subjects", "evaluation_subjects")
        ]
        assert (len(fit), len(val), len(test)) == (9, 3, 6)
        assert not (fit & val or fit & test or val & test)
        evaluated.extend(test)
    assert sorted(evaluated) == sorted(set(subjects))
