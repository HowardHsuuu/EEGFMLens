"""Known subject-level task/loss effects and corrupt-control rejection."""

import json

import numpy as np
import pytest
from summarize_contrast_task import run, sha


def test_known_effects_and_mismatched_norm_rejected(tmp_path):
    conditions = ["clean"]
    for method in ("contrast", "sensor"):
        for rank in (2, 4, 8, 16, 32):
            conditions.append(f"{method}-concept-{rank}")
            for seed in (31, 71, 113):
                conditions.extend(
                    [f"{method}-random-{seed}-{rank}", f"{method}-matched-{seed}-{rank}"]
                )
    for fold in range(3):
        folder = tmp_path / f"fold{fold}"
        folder.mkdir()
        evaluation = np.arange(fold * 6 + 1, fold * 6 + 7)
        remaining = np.setdiff1d(np.arange(1, 19), evaluation)
        subjects = np.repeat(evaluation, 45)
        labels = np.tile(np.arange(45) % 2, 6)
        prediction = np.zeros((270, 71, 5))
        prediction[:, :, 0] = (labels * 2 - 1)[:, None]
        for i, c in enumerate(conditions):
            if "-concept-" in c:
                prediction[:, i, 0] *= -1
                prediction[:, i, 1:3] = 1
        np.savez_compressed(
            folder / "responses.npz",
            predictions=prediction,
            descriptors=np.zeros((270, 4)),
            subjects=subjects,
            trial_ids=np.array([f"{fold}-{i}" for i in range(270)]),
            labels=labels,
            conditions=np.array(conditions),
            norms=np.ones((270, 71)),
            gains=np.ones((270, 71)),
            training_target_variance=np.ones(5),
        )
        (folder / "manifest.json").write_text(
            json.dumps(
                dict(
                    fold=fold,
                    model="fixture",
                    checkpoint_sha256="checkpoint",
                    feature_sha256="features",
                    basis_sha256="basis",
                    runner_sha256="runner",
                    helper_sha256="helper",
                    fitter_sha256="fitter",
                    builder_sha256="builder",
                    protocol_sha256="fixture",
                    artifact_sha256=sha(folder / "responses.npz"),
                    roles=dict(
                        fitting_subjects=remaining[:9].tolist(),
                        validation_subjects=remaining[9:].tolist(),
                        evaluation_subjects=evaluation.tolist(),
                    ),
                )
            )
        )
    run(tmp_path, tmp_path / "summary.json")
    result = json.loads((tmp_path / "summary.json").read_text())
    for c in result["concept_minus_matched_random"]:
        assert c["means"] == dict(
            accuracy_change=-1.0, target_loss_change=1.0, off_target_loss_change=0.0
        )
    meta_path = tmp_path / "fold1/manifest.json"
    original = meta_path.read_text()
    for field, value, message in (
        ("model", "different-model", "Mixed model"),
        ("runner_sha256", "different-code", "Mixed model"),
    ):
        changed = json.loads(original)
        changed[field] = value
        meta_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=message):
            run(tmp_path, tmp_path / f"bad-{field}.json")
        assert not (tmp_path / f"bad-{field}.json").exists()
    changed = json.loads(original)
    changed["roles"]["validation_subjects"][0] = changed["roles"]["fitting_subjects"][0]
    meta_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="9/3/6"):
        run(tmp_path, tmp_path / "bad-roles.json")
    assert not (tmp_path / "bad-roles.json").exists()
    meta_path.write_text(original)
    p = tmp_path / "fold0/responses.npz"
    with np.load(p) as x:
        data = {k: x[k] for k in x.files}
    data["norms"][:, conditions.index("contrast-matched-31-2")] = 2
    np.savez_compressed(p, **data)
    m = tmp_path / "fold0/manifest.json"
    record = json.loads(m.read_text())
    record["artifact_sha256"] = sha(p)
    m.write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        run(tmp_path, tmp_path / "corrupt-summary.json")
    assert not (tmp_path / "corrupt-summary.json").exists()
