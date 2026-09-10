"""Evaluation labels cannot influence iterative directions or predictions."""

import json
from pathlib import Path

import numpy as np
from iterative_recovery import run, sha


def test_evaluation_labels_do_not_change_directions_or_predictions(tmp_path):
    source = Path(__file__).parent
    split = source / "split-v1.json"
    protocol = source / "iterative_recovery_protocol.json"
    subjects = np.repeat(json.loads(split.read_text())["train"], 8)
    rng = np.random.default_rng(64)
    middle = rng.normal(size=(len(subjects), 2, 40))
    y = (middle[:, 1] - middle[:, 0])[:, :4] + rng.normal(size=(len(subjects), 4))
    path = tmp_path / "fixture.npz"

    def save(target):
        np.savez_compressed(
            path,
            middle=middle,
            descriptors=target,
            subjects=subjects,
            channels=np.array(["C3", "C4"]),
            trial_ids=np.array([f"t{i}" for i in range(len(y))]),
        )
        path.with_suffix(".json").write_text(
            json.dumps(dict(partition="train", artifact_sha256=sha(path)))
        )

    save(y)
    run(path, split, protocol, tmp_path / "before")
    report = json.loads((tmp_path / "before/summary.json").read_text())
    heldout = np.isin(subjects, report["folds"][0]["evaluation_subjects"])
    changed = y.copy()
    changed[heldout] = changed[heldout] * 5 + 100
    save(changed)
    run(path, split, protocol, tmp_path / "after")
    with (
        np.load(tmp_path / "before/predictions.npz") as a,
        np.load(tmp_path / "after/predictions.npz") as b,
    ):
        for key in report["pooled_r2"]:
            np.testing.assert_array_equal(a[key][heldout], b[key][heldout])
    with np.load(tmp_path / "before/bases.npz") as a, np.load(tmp_path / "after/bases.npz") as b:
        np.testing.assert_array_equal(a["fold_0_concept"], b["fold_0_concept"])
    evaluated = []
    for fold in report["folds"]:
        fit, val, test = [
            set(fold[k]) for k in ("fitting_subjects", "validation_subjects", "evaluation_subjects")
        ]
        assert (len(fit), len(val), len(test)) == (9, 3, 6)
        assert not (fit & val or fit & test or val & test)
        evaluated.extend(test)
    assert sorted(evaluated) == sorted(set(subjects))
