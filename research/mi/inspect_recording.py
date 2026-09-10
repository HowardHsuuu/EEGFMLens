"""Audit a local unilateral-imagery EEGMMIDB EDF without fitting any models."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import mne


def inspect_recording(path):
    match = re.fullmatch(r"S(\d{3})R(\d{2})\.edf", path.name)
    if match is None or int(match[2]) not in (4, 8, 12):
        raise ValueError("Expected EEGMMIDB unilateral imagery run 04, 08, or 12")
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    if raw.info["sfreq"] != 160 or len(raw.ch_names) != 64:
        raise ValueError("Unexpected sampling rate or channel count; audit before use")
    channels = [name.rstrip(".").upper() for name in raw.ch_names]
    if len(set(channels)) != 64 or not {"C3", "C4"} <= set(channels):
        raise ValueError("Invalid channel vocabulary")
    events = []
    for index, annotation in enumerate(raw.annotations):
        label = annotation["description"]
        if label not in ("T0", "T1", "T2"):
            raise ValueError(f"Unexpected annotation: {label}")
        if label == "T0":
            continue
        start = float(annotation["onset"])
        duration = float(annotation["duration"])
        # Inventory only: short events are recorded, never silently padded.
        events.append(
            dict(
                trial_id=f"S{match[1]}R{match[2]}:annotation-{index}",
                onset_seconds=start,
                duration_seconds=duration,
                imagery="left_fist" if label == "T1" else "right_fist",
                complete_four_seconds=bool(duration >= 4 and start + 4 <= raw.n_times / 160),
            )
        )
    return dict(
        dataset="EEGMMIDB 1.0.0",
        source=f"https://physionet.org/files/eegmmidb/1.0.0/S{match[1]}/{path.name}",
        license="Open Data Commons Attribution License v1.0",
        citation="Schalk (2009), doi:10.13026/C28G6P",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        subject=int(match[1]),
        run=int(match[2]),
        sampling_rate=160,
        samples=int(raw.n_times),
        channels=channels,
        mne_version=mne.__version__,
        mne_eeg_array_unit="volts",
        events=events,
        scope="Recording inventory only; no preprocessing, split or scientific result.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_recording(args.edf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(len(result["events"]), "imagery events inventoried")
