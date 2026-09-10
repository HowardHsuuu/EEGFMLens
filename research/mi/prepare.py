"""Prepare shared raw-voltage trials and prespecified EEG descriptors, without model scaling."""

import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
from inspect_recording import inspect_recording
from scipy.signal import welch

FEATURES = (
    "mu_logpower_C4_minus_C3",
    "beta_logpower_C4_minus_C3",
    "occipital_alpha_logpower",
    "global_log_rms",
)


def descriptors(volts, channels, fs=160):
    """CAR spectral descriptors; log powers use V²/Hz and RMS uses V.

    These are concurrent power asymmetries, not baseline-relative ERD estimates.
    The original-reference input array is never modified.
    """
    car = volts - volts.mean(axis=1, keepdims=True)
    frequencies, psd = welch(car, fs=fs, nperseg=fs * 2, noverlap=fs, axis=-1)

    def power(lo, hi):
        band = (frequencies >= lo) & (frequencies < hi)
        return np.log(np.maximum(psd[..., band].mean(axis=-1), np.finfo(float).tiny))

    c3, c4 = channels.index("C3"), channels.index("C4")
    mu, beta = power(8, 13), power(13, 30)
    occipital = mu[:, [channels.index("O1"), channels.index("O2")]].mean(axis=1)
    rms = np.log(np.maximum(np.sqrt(np.mean(car**2, axis=(1, 2))), np.finfo(float).tiny))
    return np.column_stack([mu[:, c4] - mu[:, c3], beta[:, c4] - beta[:, c3], occipital, rms])


def prepare(paths, output):
    trials, records, sources, dropped = [], [], [], []
    channels = None
    seen = set()
    for path in sorted(paths):
        inventory = inspect_recording(path)
        if channels is None:
            channels = inventory["channels"]
        if inventory["channels"] != channels:
            raise ValueError("Input channel order differs; explicitly reconcile before preparing")
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        values = raw.get_data()  # MNE EEG SI volts; no resampling/filtering/scaling here.
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite EEG recording")
        sources.append({k: inventory[k] for k in ("source", "sha256", "subject", "run")})
        for event in inventory["events"]:
            if event["trial_id"] in seen:
                raise ValueError("Duplicate trial ID / repeated recording")
            seen.add(event["trial_id"])
            onset = event["onset_seconds"] * 160
            start = int(round(onset))
            if abs(start - onset) > 1e-5:
                raise ValueError("Event onset is not aligned to a native EEG sample")
            if not event["complete_four_seconds"]:
                dropped.append(dict(**event, reason="less than four seconds of task/data"))
                continue
            epoch = values[:, start : start + 640]
            if epoch.shape != (64, 640):
                raise ValueError("Truncated epoch")
            trials.append(epoch.copy())
            records.append(
                dict(
                    **event,
                    subject=inventory["subject"],
                    run=inventory["run"],
                    native_start_sample=start,
                    label=int(event["imagery"] == "right_fist"),
                )
            )
    if not trials:
        raise ValueError("No complete imagery trials")
    volts = np.stack(trials)
    concepts = descriptors(volts, channels)
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "trials.npz"
    np.savez_compressed(
        artifact,
        volts=volts,
        descriptors=concepts,
        labels=np.array([r["label"] for r in records]),
        subjects=np.array([r["subject"] for r in records]),
        trial_ids=np.array([r["trial_id"] for r in records]),
        channels=np.array(channels),
    )
    manifest = dict(
        sources=sources,
        records=records,
        dropped=dropped,
        sampling_rate=160,
        data_unit="volts",
        data_reference="as recorded; no rereference applied to saved EEG",
        epoch_seconds=[0, 4],
        descriptor_names=FEATURES,
        descriptor_recipe="64-channel CAR; Welch Hann 2s/1s overlap; mean PSD [8,13), [13,30) Hz; natural log",
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        mne_version=mne.__version__,
        scope="Data preparation only. No model-specific preprocessing or fitted parameters. No split assigned.",
        caveats=[
            "Concurrent asymmetry is not ERD",
            "Cue direction covaries with imagery label",
            "Pretraining subject exposure has not been excluded",
        ],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(len(records), "trials prepared;", len(set(r["subject"] for r in records)), "subjects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.edf, args.output)
