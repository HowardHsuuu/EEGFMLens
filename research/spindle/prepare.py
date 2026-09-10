"""Audit and prepare DREAMS without treating unreviewed time as negatives."""

import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
from scipy.signal import welch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(events, start, end):
    return float(
        np.maximum(
            0, np.minimum(events[:, 0] + events[:, 1], end) - np.maximum(events[:, 0], start)
        ).sum()
    )


def voltage_correction(reader_values, text_microvolts):
    """Resolve DREAMS uppercase UV against the independent physical-value export.

    Some MNE versions normalize the display label but do not apply the UV -> V
    scale. Compare samples, not the normalized label or plausible amplitudes.
    """
    target = text_microvolts[: len(reader_values)]
    if len(target) != len(reader_values):
        raise ValueError("Text signal shorter than EDF")
    if np.allclose(reader_values * 1e6, target, atol=5.1e-5, rtol=1e-7):
        return 1.0
    if np.allclose(reader_values, target, atol=5.1e-5, rtol=1e-7):
        return 1e-6
    raise ValueError("EDF and physical-value text export do not align")


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=True)
    values, records, subjects = [], [], []
    for subject in range(1, 9):
        edf = source / f"excerpt{subject}.edf"
        annotation = source / f"Visual_scoring1_excerpt{subject}.txt"
        channel = annotation.read_text().splitlines()[0].split("/", 1)[1].rstrip("]")
        raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
        raw.pick([channel]).load_data()
        text_signal = np.loadtxt(source / f"excerpt{subject}.txt", skiprows=1)
        correction = voltage_correction(raw.get_data()[0], text_signal)
        raw.apply_function(lambda x: x * correction, picks=[channel], verbose="ERROR")
        np.testing.assert_allclose(
            raw.get_data(units="uV")[0], text_signal[: raw.n_times], atol=5.1e-5, rtol=1e-7
        )
        original_fs = float(raw.info["sfreq"])
        duration = raw.n_times / original_fs
        stages = np.loadtxt(source / f"Hypnogram_excerpt{subject}.txt", skiprows=1).astype(int)
        assert duration == 1800 and len(stages) == 360
        assert set(stages) <= {0, 1, 2, 3, 4, 5, -1}
        events = np.loadtxt(annotation, skiprows=1, ndmin=2)
        assert events.shape[1] == 2 and (events[:, 1] > 0).all()
        second = source / f"Visual_scoring2_excerpt{subject}.txt"
        events2 = np.loadtxt(second, skiprows=1, ndmin=2) if second.exists() else None
        raw.filter(0.5, 20, verbose="ERROR").resample(200, verbose="ERROR")
        data = raw.get_data(units="uV")[0]
        if not 0.01 < np.median(abs(data)) < 500 or np.max(abs(data)) > 10000:
            raise ValueError("Implausible microvolt scale; inspect source units")
        kept = 0
        for window in range(120):
            start, end = window * 15, (window + 1) * 15
            labels = stages[window * 3 : (window + 1) * 3]
            if len(set(labels)) != 1 or labels[0] < 0:
                continue  # no majority-vote relabeling of stage transitions
            signal = data[start * 200 : end * 200]
            frequencies, power = welch(signal, fs=200, nperseg=400)
            bands = [(0.5, 4), (4, 8), (8, 11), (11, 16), (16, 20)]
            features = [
                float(
                    np.log(
                        np.trapezoid(
                            power[(frequencies >= a) & (frequencies < b)],
                            frequencies[(frequencies >= a) & (frequencies < b)],
                        )
                        + 1e-12
                    )
                )
                for a, b in bands
            ]
            features += [float(np.log(np.std(signal) + 1e-9)), float(np.log(np.ptp(signal) + 1e-9))]
            covered = end <= 990  # conservative common boundary before documented 1000 s cutoff
            amount = overlap(events, start, end) if covered else None
            label = -1 if not covered or 0 < amount < 0.5 else int(amount >= 0.5)
            amount2 = overlap(events2, start, end) if covered and events2 is not None else None
            records.append(
                dict(
                    subject=subject,
                    window=window,
                    start=start,
                    stage=int(labels[0]),
                    n2=int(labels[0] == 2),
                    spindle=label,
                    spindle_seconds=amount,
                    expert2_seconds=amount2,
                    reviewed=covered,
                    channel=channel.split("-")[0].upper(),
                    reference=channel.split("-")[1],
                    spectral=features,
                )
            )
            values.append((signal / 100).astype(np.float32).reshape(1, 15, 200))
            kept += 1
        subjects.append(
            dict(
                subject=subject,
                source_fs=original_fs,
                reader_to_volts_factor=correction,
                signal_text_sha256=digest(source / f"excerpt{subject}.txt"),
                filtered_abs_uV_quantiles=np.quantile(abs(data), [0.5, 0.95, 0.99, 1]).tolist(),
                channel=channel,
                duration=duration,
                stage_step_seconds=duration / len(stages),
                kept_windows=kept,
                events_expert1=len(events),
                events_expert2=len(events2) if events2 is not None else None,
                edf_sha256=digest(edf),
                hypnogram_sha256=digest(source / f"Hypnogram_excerpt{subject}.txt"),
                annotation_sha256=digest(annotation),
            )
        )
    np.savez_compressed(output / "inputs.npz", x=np.stack(values))
    document = dict(
        recipe="dreams-v2:UV-text-verified:central-A1:0.5-20Hz:200Hz:uV/100:15s:unanimous-stage:expert1-before990s",
        annotation_policy="spindle=1: >=0.5s overlap; 0: no overlap in reviewed time; -1: unknown/boundary",
        subjects=subjects,
        rows=records,
    )
    (output / "manifest.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    for subject in subjects:
        rows = [r for r in records if r["subject"] == subject["subject"]]
        n2 = [r for r in rows if r["n2"] == 1 and r["spindle"] >= 0]
        print(
            json.dumps(
                dict(
                    subject=subject["subject"],
                    windows=len(rows),
                    n2=sum(r["n2"] for r in rows),
                    reviewed_n2=len(n2),
                    spindle_positive_n2=sum(r["spindle"] for r in n2),
                )
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
