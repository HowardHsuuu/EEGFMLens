"""Subject-level paired estimates; all specified layers, including failed controls."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def estimate(rows, value):
    # First mean repetitions within each window, then windows within subject.
    windows = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None and np.isfinite(v):
            windows[(r["subject"], r["row"])].append(float(v))
    subjects = defaultdict(list)
    for (s, _), v in windows.items():
        subjects[s].append(np.mean(v))
    means = {str(s): float(np.mean(v)) for s, v in sorted(subjects.items())}
    x = np.array(list(means.values()))
    if not len(x):
        return dict(mean=None, ci95=None, subjects={}, windows=0)
    rng = np.random.default_rng(8721)
    boots = rng.choice(x, (10000, len(x)), replace=True).mean(1)
    return dict(
        mean=float(x.mean()),
        ci95=np.quantile(boots, [0.025, 0.975]).tolist(),
        subjects=means,
        windows=len(windows),
    )


def grouped(rows, keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def improvement(r):
    return (
        abs(r["corrupt_margin"] - r["clean_margin"]) - abs(r["restored_margin"] - r["clean_margin"])
    ) / r["margin_scale"]


def summarize(work):
    prepared, results = work / "prepared", work / "results"
    cases = json.loads((prepared / "cases.json").read_text())
    qc = json.loads((prepared / "quality.json").read_text())
    clean = {r["case"]: r for r in qc if r["variant"] == "clean"}
    quality = []
    for (variant,), rs in grouped(qc, ["variant"]).items():
        metrics = {
            k: estimate(rs, lambda r, k=k: r.get(k))
            for k in [
                "global_fft_error",
                "local_fft_error",
                "relative_input_delta",
                "sigma_event_fraction",
                "sigma_envelope_cv",
                "distribution_wasserstein_sd",
                "peak_sd",
                "join_jump_sd",
            ]
        }
        metrics["sigma_concentration_change"] = estimate(
            rs, lambda r: r["sigma_event_fraction"] - clean[r["case"]]["sigma_event_fraction"]
        )
        metrics["fraction_concentration_reduced"] = estimate(
            rs,
            lambda r: float(r["sigma_event_fraction"] < clean[r["case"]]["sigma_event_fraction"]),
        )
        quality.append(
            dict(
                variant=variant,
                metrics=metrics,
                max_global_fft_error=max(r["global_fft_error"] for r in rs),
                max_local_fft_error=max(r.get("local_fft_error", 0) for r in rs),
            )
        )
    responses = []
    input_contrasts = []
    restorations = []
    comparisons = []
    for label in ["cbramod", "labram", "cbramod-random", "labram-random"]:
        rr = json.loads((results / f"{label}-responses.json").read_text())
        assert len(rr) == len(cases["cases"]) * 25
        assert len({(r["case"], r["variant"], r["site"]) for r in rr}) == len(rr)
        for (variant, site), rs in grouped(rr, ["variant", "site"]).items():
            metrics = {
                "absolute_margin_change": estimate(
                    rs, lambda r: abs(r["margin"] - r["clean_margin"]) / r["margin_scale"]
                ),
                "signed_margin_change": estimate(
                    rs, lambda r: (r["margin"] - r["clean_margin"]) / r["margin_scale"]
                ),
                "prediction_flip": estimate(
                    rs, lambda r: float((r["margin"] >= 0) != (r["clean_margin"] >= 0))
                ),
                "clean_n2_positive": estimate(rs, lambda r: float(r["clean_margin"] >= 0)),
                "surrogate_n2_positive": estimate(rs, lambda r: float(r["margin"] >= 0)),
            }
            for position in ["event", "off", "random"]:
                metrics[position + "_patch_change"] = estimate(
                    rs,
                    lambda r, p=position: np.mean(
                        r["patch_relative_change"][r[p + "_start"] : r[p + "_start"] + 3]
                    ),
                )
            metrics["other_patch_change"] = estimate(
                rs,
                lambda r: np.mean(
                    [
                        v
                        for i, v in enumerate(r["patch_relative_change"])
                        if not r["event_start"] <= i < r["event_start"] + 3
                    ]
                ),
            )
            responses.append(dict(model=label, variant=variant, site=site, metrics=metrics))
        for (site, _), rs in grouped(rr, ["site", "case"]).items():
            by_variant = {r["variant"]: r for r in rs}
            event, off = by_variant["local_event"], by_variant["local_off"]
            input_contrasts.append(
                dict(
                    model=label,
                    site=site,
                    subject=event["subject"],
                    row=event["row"],
                    difference=(
                        abs(event["margin"] - event["clean_margin"])
                        - abs(off["margin"] - off["clean_margin"])
                    )
                    / event["margin_scale"],
                )
            )
        patches = json.loads((results / f"{label}-restorations.json").read_text())
        if label.endswith("-random"):
            assert not patches
            continue
        assert len(patches) == len(cases["cases"]) * 60
        assert len({(r["case"], r["variant"], r["site"], r["position"]) for r in patches}) == len(
            patches
        )
        for (variant, site, position), rs in grouped(
            patches, ["variant", "site", "position"]
        ).items():
            matched = [r for r in rs if r["status"] == "matched"]
            norms = np.array([r["activation_delta_norm"] for r in matched])
            targets = np.array([r["target_norm"] for r in matched])
            np.testing.assert_allclose(norms, targets, rtol=1e-4, atol=1e-6)
            multipliers = [r["multiplier"] for r in matched]
            restorations.append(
                dict(
                    model=label,
                    variant=variant,
                    site=site,
                    position=position,
                    matched=len(matched),
                    unavailable=len(rs) - len(matched),
                    zero_target=sum(r["target_norm"] < 1e-8 for r in matched),
                    multiplier_quantiles=np.quantile(multipliers, [0, 0.5, 0.95, 1]).tolist()
                    if multipliers
                    else [],
                    improvement=estimate(matched, improvement),
                    signed_delta=estimate(
                        matched,
                        lambda r: (r["restored_margin"] - r["corrupt_margin"]) / r["margin_scale"],
                    ),
                )
            )
        pairs = grouped(patches, ["variant", "site", "case"])
        contrasts = []
        for (variant, site, _), rs in pairs.items():
            positions = {r["position"]: r for r in rs}
            for control in ["off", "random"]:
                e, c = positions["event"], positions[control]
                if e["status"] != "matched" or c["status"] != "matched" or e["target_norm"] < 1e-8:
                    continue
                contrasts.append(
                    dict(
                        subject=e["subject"],
                        row=e["row"],
                        variant=variant,
                        site=site,
                        control=control,
                        difference=improvement(e) - improvement(c),
                        multiplier=c["multiplier"],
                    )
                )
        for (variant, site, control), rs in grouped(
            contrasts, ["variant", "site", "control"]
        ).items():
            comparisons.append(
                dict(
                    model=label,
                    variant=variant,
                    site=site,
                    control=control,
                    event_advantage=estimate(rs, lambda r: r["difference"]),
                    # Descriptive sensitivity only; this threshold was not used for primary selection.
                    scale_at_most_two=estimate(
                        [r for r in rs if r["multiplier"] <= 2], lambda r: r["difference"]
                    ),
                )
            )
    doc = dict(
        cases=len(cases["cases"]),
        windows=len({c["row"] for c in cases["cases"]}),
        excluded_windows=len(cases["exclusions"]),
        quality=quality,
        responses=responses,
        restorations=restorations,
        comparisons=comparisons,
        local_input_contrasts=[
            dict(
                model=model,
                site=site,
                event_minus_off_absolute_margin=estimate(rs, lambda r: r["difference"]),
            )
            for (model, site), rs in grouped(input_contrasts, ["model", "site"]).items()
        ],
    )
    (results / "summary.json").write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print("Summarized", doc["windows"], "windows and all four encoders", flush=True)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    summarize(p.parse_args().work)
