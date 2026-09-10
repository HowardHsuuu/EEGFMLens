"""Render descriptive subject estimates without choosing a best layer."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SITES = ["embedding.output", *[f"blocks.{i}.output" for i in [2, 5, 8, 11]]]
VARIANTS = ["global_phase", "global_sigma_phase", "local_event", "local_off", "circular_shift"]


def report(work):
    results = work / "results"
    doc = json.loads((results / "summary.json").read_text())
    prepared = work / "prepared"
    cases = json.loads((prepared / "cases.json").read_text())
    # First selected case, fixed independently of model results.
    c = cases["cases"][0]
    signals = np.load(prepared / "signals.npz")["signals"][0, :, 0].reshape(5, -1)
    # Undo the known shift to recover the original without another external path.
    clean = np.roll(signals[4], -1000)
    from signals import sigma_envelope

    fig, axs = plt.subplots(4, 1, figsize=(12, 8), layout="constrained", sharex=True)
    time = np.arange(3000) / 200
    for ax, x, name in zip(
        axs,
        [clean, signals[0], signals[1], signals[2]],
        ["Original", "Global phase", "Sigma-only phase", "Local event phase"],
    ):
        ax.plot(time, x * 100, lw=0.5, label="EEG")
        ax.plot(time, sigma_envelope(x) * 100, lw=1, label="Sigma envelope")
        ax.axvspan(c["event_start"], c["event_start"] + 3, color="orange", alpha=0.15)
        ax.set(ylabel="microvolts", title=name)
    axs[0].legend(loc="upper right")
    axs[-1].set_xlabel("Seconds; shading marks selected 3-second event ROI")
    fig.savefig(results / "signals.png", dpi=170)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for model, color in [
        ("cbramod", "tab:blue"),
        ("labram", "tab:orange"),
        ("cbramod-random", "tab:green"),
        ("labram-random", "tab:red"),
    ]:
        rs = [
            next(
                r
                for r in doc["responses"]
                if r["model"] == model and r["variant"] == v and r["site"] == SITES[-1]
            )
            for v in VARIANTS
        ]
        y = np.array([r["metrics"]["absolute_margin_change"]["mean"] for r in rs])
        axs[0, 0].plot(range(5), y, "o-", label=model, color=color)
    axs[0, 0].set(
        xticks=range(5),
        xticklabels=["Global", "Sigma only", "Local event", "Local off", "Shift"],
        ylabel="Absolute N2 margin change / train SD",
        title="Fixed held-out heads; equal weight per subject",
    )
    axs[0, 0].legend(fontsize=8)
    for ax, model in zip(axs[1], ["cbramod", "labram"]):
        for control in ["off", "random"]:
            rs = [
                next(
                    r
                    for r in doc["comparisons"]
                    if r["model"] == model
                    and r["variant"] == "global_sigma_phase"
                    and r["site"] == s
                    and r["control"] == control
                )
                for s in SITES
            ]
            y = np.array([r["event_advantage"]["mean"] for r in rs])
            ci = np.array([r["event_advantage"]["ci95"] for r in rs]).T
            ax.errorbar(
                range(5),
                y,
                yerr=np.maximum(0, np.vstack([y - ci[0], ci[1] - y])),
                fmt="o-",
                capsize=3,
                label="event minus " + control,
            )
        ax.axhline(0, color="gray", lw=1)
        ax.set(
            xticks=range(5),
            xticklabels=["Embed", "B2", "B5", "B8", "B11"],
            ylabel="Clean-margin error reduction advantage / SD",
            title=model + ": sigma-only restoration",
        )
        ax.legend(fontsize=8)
    for v in VARIANTS:
        q = next(r for r in doc["quality"] if r["variant"] == v)
        axs[0, 1].scatter(
            q["metrics"]["distribution_wasserstein_sd"]["mean"],
            q["metrics"]["sigma_concentration_change"]["mean"],
            label=v,
        )
    axs[0, 1].axhline(0, color="gray", lw=1)
    axs[0, 1].set(
        xlabel="Sample-distribution distance / signal SD",
        ylabel="Change in sigma energy fraction inside annotations",
        title="Surrogate quality; shift relocates intact morphology",
    )
    axs[0, 1].legend(fontsize=8)
    fig.savefig(results / "overview.png", dpi=170)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, model in enumerate(["cbramod", "labram"]):
        records = json.loads((results / f"{model}-responses.json").read_text())
        for col, variant in enumerate(["global_sigma_phase", "local_event"]):
            matrix = []
            for site in SITES:
                subjects = {}
                for r in records:
                    if r["site"] == site and r["variant"] == variant:
                        # Align selected event ROI start at column 3, averaging only existing patches.
                        z = np.full(17, np.nan)
                        for i, v in enumerate(r["patch_relative_change"]):
                            j = i - r["event_start"] + 3
                            if 0 <= j < 17:
                                z[j] = v
                        subjects.setdefault(r["subject"], []).append(z)

                def available_mean(values):
                    values = np.asarray(values)
                    counts = np.isfinite(values).sum(axis=0)
                    return np.divide(
                        np.nansum(values, axis=0),
                        counts,
                        out=np.full(counts.shape, np.nan),
                        where=counts > 0,
                    )

                matrix.append(available_mean([available_mean(v) for v in subjects.values()]))
            im = axs[row, col].imshow(
                matrix, aspect="auto", extent=(-3.5, 13.5, 4.5, -0.5), cmap="viridis"
            )
            axs[row, col].axvline(-0.5, color="white", ls="--")
            axs[row, col].axvline(2.5, color="white", ls="--")
            axs[row, col].set(
                yticks=range(5),
                yticklabels=["Embed", "B2", "B5", "B8", "B11"],
                xlabel="Patch offset from selected event ROI start",
                title=model + " / " + variant,
            )
            fig.colorbar(im, ax=axs[row, col], label="Relative activation change")
    fig.savefig(results / "patches.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    report(p.parse_args().work)
