from dataclasses import replace

import pytest
import torch
from torch import nn

from eeglens import (
    ActivationSite,
    Adapter,
    EEGLens,
    MatchedReplacement,
    Selection,
    SignalBatch,
    SweepTarget,
    patching_sweep,
)


class Analytic(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Identity()
        self.late = nn.Identity()

    def forward(self, x):
        x = self.early(x)
        x = self.late(x * 2)
        return 3 * x[:, 0, 1, 0] - x[:, 1, 2, 0]


def fixture():
    model = Analytic().eval()
    lens = EEGLens(
        model, Adapter([ActivationSite("early", "early"), ActivationSite("late", "late")])
    )
    clean = SignalBatch(
        torch.tensor(
            [
                [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
                [[[7.0], [8.0], [9.0]], [[10.0], [11.0], [12.0]]],
            ]
        ),
        ("a", "b"),
        ("C3", "C4"),
        200,
        "known",
    )
    recipient = replace(clean, data=torch.zeros_like(clean.data))
    return lens, clean, recipient


def score(output, batch):
    return output


def test_exact_grid_localizes_known_causal_coordinates(tmp_path):
    lens, clean, recipient = fixture()
    result = patching_sweep(
        lens, clean, recipient, score, sites=["early", "late"], random_controls=False
    )
    for r in result.rows:
        if r["kind"] == "identity":
            assert r["patched_score"] == 0
            continue
        i = clean.trial_ids.index(r["trial_id"])
        expected = (
            6 * float(clean.data[i, 0, 1, 0])
            if r["target"] == "C3:patch1"
            else -2 * float(clean.data[i, 1, 2, 0])
            if r["target"] == "C4:patch2"
            else 0
        )
        assert r["delta"] == expected
        assert r["actual_delta_norm"] > 0
    result.save(tmp_path / "sweep.json")
    with pytest.raises(FileExistsError):
        result.save(tmp_path / "sweep.json")
    assert not lens.model.early._forward_hooks and not lens.model.late._forward_hooks


def test_batch_partition_order_donor_pairing_and_seed_invariance():
    lens, clean, recipient = fixture()
    target = SweepTarget(
        "causal",
        Selection(sensors=("C3",), patches=(1,)),
        off_event=Selection(sensors=("C4",), patches=(0,)),
    )
    options = dict(score=score, sites=["early", "late"], targets=[target], seed=91)
    original = patching_sweep(lens, clean, recipient, **options)
    reversed_clean = replace(clean, data=clean.data.flip(0), trial_ids=clean.trial_ids[::-1])
    paired = patching_sweep(lens, reversed_clean, recipient, **options)
    assert original.rows == paired.rows
    parts = []
    for i in [1, 0]:
        c = replace(clean, data=clean.data[i : i + 1], trial_ids=(clean.trial_ids[i],))
        r = replace(recipient, data=recipient.data[i : i + 1], trial_ids=(recipient.trial_ids[i],))
        parts.extend(patching_sweep(lens, c, r, **options).rows)

    def key(r):
        return (r["trial_id"], r["site"], str(r["target"]), r["kind"])

    assert sorted(original.rows, key=key) == sorted(parts, key=key)
    for r in original.rows:
        if r["kind"] == "off_event":
            assert (
                r["actual_delta_norm"] == pytest.approx(2 if r["trial_id"] == "a" else 8, rel=1e-6)
                if r["site"] == "early"
                else r["actual_delta_norm"]
                == pytest.approx(4 if r["trial_id"] == "a" else 16, rel=1e-6)
            )


def test_direct_matching_is_per_trial_and_flags_unavailable():
    lens, clean, recipient = fixture()
    donor = lens.run_with_cache(clean, sites=["early"]).cache["early"]
    target = Selection(sensors=("C3",), patches=(1,))
    control = Selection(sensors=("C4",), patches=(0,))
    edit = MatchedReplacement("early", donor, control, target)
    result = lens.run_with_interventions(recipient, interventions=[edit])
    assert [d["multiplier"] for d in edit.diagnostics] == pytest.approx([0.5, 0.8])
    torch.testing.assert_close(result.output, torch.zeros(2))
    values = donor.tensor.clone()
    values[0, 1, 0] = 0
    values[1, 1, 0] = 0.01
    edit = MatchedReplacement("early", replace(donor, tensor=values), control, target)
    lens.run_with_interventions(recipient, interventions=[edit])
    assert edit.diagnostics[0]["status"] == "unmatchable_zero_control"
    assert edit.diagnostics[0]["multiplier"] is None
    assert edit.diagnostics[1]["excessive_multiplier"]
    assert edit.diagnostics[1]["actual_delta_norm"] == pytest.approx(8)


def test_missing_random_and_score_contract_cleanup():
    lens, clean, recipient = fixture()
    result = patching_sweep(
        lens, clean, recipient, score, sites=["early"], targets=[SweepTarget("all", Selection())]
    )
    assert sum(r["status"] == "no_disjoint_random_location" for r in result.rows) == 2
    with pytest.raises(Exception, match="Score must return"):
        patching_sweep(lens, clean, recipient, lambda out, b: out.mean(), sites=["early"])
    assert not lens.model.early._forward_hooks


def test_matched_restore_support_norm_and_missing_control():
    from dataclasses import replace

    import pytest
    import torch

    from eeglens import Replacement, Selection, SignalBatch
    from eeglens.errors import ValidationError
    from eeglens.types import Activation

    batch = SignalBatch(torch.zeros(1, 1, 6, 200), ("a",), ("C3",), 200, "test")
    current = torch.zeros(1, 7, 4)
    values = torch.ones_like(current)
    values[:, 1:4] *= 2
    donor = Activation(
        values,
        "site",
        "model",
        batch.trial_ids,
        batch.channels,
        batch.preprocessing_id,
        200,
        200,
        200,
        batch.unit,
        "tokens",
        tuple(values.shape),
    )
    event = Selection(patches=(0, 1, 2))
    control = Selection(patches=(3, 4, 5))
    edit = MatchedReplacement("site", donor, control, event)
    out = edit.apply(current, batch, "tokens", "model")
    torch.testing.assert_close(out[:, :4], current[:, :4])
    torch.testing.assert_close(out[:, 4:], torch.full_like(out[:, 4:], 2))
    assert edit.diagnostics[0]["multiplier"] == 2
    actual = MatchedReplacement("site", donor, event, event).apply(
        current, batch, "tokens", "model"
    )
    expected = Replacement("site", donor, event).apply(current, batch, "tokens", "model")
    torch.testing.assert_close(actual, expected)
    values[:, 4:] = 0
    missing = MatchedReplacement("site", donor, control, event)
    missing.apply(current, batch, "tokens", "model")
    assert missing.diagnostics[0]["status"] == "unmatchable_zero_control"
    with pytest.raises(ValidationError):
        MatchedReplacement("site", replace(donor, preprocessing_id="wrong"), event, event).apply(
            current, batch, "tokens", "model"
        )


def test_invalid_controls_rejected_and_invalid_rows_not_averaged():
    from eeglens.errors import ValidationError

    lens, clean, recipient = fixture()
    with pytest.raises(ValidationError, match="disjoint"):
        patching_sweep(
            lens,
            clean,
            recipient,
            score,
            sites=["early"],
            targets=[SweepTarget("overlap", Selection(patches=(0,)), Selection(patches=(0,)))],
        )
    result = patching_sweep(
        lens,
        clean,
        clean,
        score,
        sites=["early"],
        targets=[SweepTarget("zero", Selection(patches=(0,)))],
    )
    assert all(
        not r["valid"] and r["patched_score"] is None
        for r in result.rows
        if r["kind"] != "identity"
    )
    assert all(r["mean_clean_error_reduction"] is None for r in result.summary())
    good = patching_sweep(lens, clean, recipient, score, sites=["early"], random_controls=False)
    summary = good.summary(groups={"a": "subject1", "b": "subject1"})
    assert all(r["groups"] == 1 and r["valid_rows"] == 2 for r in summary)


def test_direct_diagnostics_survive_run_serialization(tmp_path):
    from eeglens import load_run, save_run

    lens, clean, recipient = fixture()
    donor = lens.run_with_cache(clean, sites=["early"]).cache["early"]
    edit = MatchedReplacement(
        "early",
        donor,
        Selection(sensors=("C4",), patches=(0,)),
        Selection(sensors=("C3",), patches=(1,)),
    )
    result = lens.run_with_interventions(recipient, interventions=[edit])
    save_run(result, tmp_path / "run")
    loaded = load_run(tmp_path / "run")
    assert loaded.metadata["interventions"][0]["diagnostics"] == edit.diagnostics


def test_multichannel_token_grid_preserves_cls_coordinates():
    class Tokens(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = nn.Identity()

        def forward(self, x):
            patches = x.flatten(1, 2)
            cls = torch.full_like(patches[:, :1], 999)
            tokens = self.tokens(torch.cat([cls, patches], dim=1))
            return 6 * tokens[:, 2, 0] - 2 * tokens[:, 6, 0]

    _, clean, recipient = fixture()
    lens = EEGLens(Tokens().eval(), Adapter([ActivationSite("tokens", "tokens", layout="tokens")]))
    result = patching_sweep(lens, clean, recipient, score, random_controls=False)
    for r in result.rows:
        if r["kind"] == "identity":
            continue
        i = clean.trial_ids.index(r["trial_id"])
        expected = {
            "C3:patch1": 6 * float(clean.data[i, 0, 1, 0]),
            "C4:patch2": -2 * float(clean.data[i, 1, 2, 0]),
        }.get(r["target"], 0)
        assert r["delta"] == expected
    for b in result.baselines:
        assert b["patch_changes"]["tokens"] == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
