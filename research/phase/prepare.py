"""Generate a blinded, audited paired-signal experiment from corrected inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from signals import make_variants, off_regions, quality, regions

VARIANTS = ["global_phase", "global_sigma_phase", "local_event", "local_off", "circular_shift"]


def prepare(previous, events_path, output):
    manifest = json.loads((previous / "prepared/manifest.json").read_text())
    assert "UV-text-verified" in manifest["recipe"], "Corrected voltage inputs required"
    source_hashes = json.loads((previous / "artifact_hashes.json").read_text())
    for name in ["prepared/inputs.npz", "prepared/manifest.json", "results/readouts.npz"]:
        assert hashlib.sha256((previous / name).read_bytes()).hexdigest() == source_hashes[name]
    data = np.load(previous / "prepared/inputs.npz")["x"]
    events = {}
    for subject in manifest["subjects"]:
        path = events_path / f"Visual_scoring1_excerpt{subject['subject']}.txt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == subject["annotation_sha256"]
        events[subject["subject"]] = np.loadtxt(path, skiprows=1, ndmin=2)
    cases, exclusions, quality_rows, signals = [], [], [], []
    for row, r in enumerate(manifest["rows"]):
        if r["n2"] != 1 or r["spindle"] != 1:
            continue
        start, relative = regions(events[r["subject"]], r["start"])
        if start is None:
            exclusions.append(dict(row=row, reason="no contained event ROI"))
            continue
        options = off_regions(relative, start)
        if not options:
            exclusions.append(dict(row=row, reason="no guarded off-event ROI"))
            continue
        x = data[row].reshape(-1)

        def rms(k):
            return np.sqrt(np.mean(x[k * 200 : (k + 3) * 200].astype(float) ** 2))

        off = min(options, key=lambda k: abs(rms(k) - rms(start)))
        for replicate in range(3):
            seed = 8721 + row * 10 + replicate * 100000
            variants = make_variants(x, start, off, seed)
            random_options = [k for k in range(1, 12) if k + 3 <= start or start + 3 <= k]
            random_position = int(np.random.default_rng(seed + 7).choice(random_options))
            case = dict(
                case=len(cases),
                row=row,
                subject=r["subject"],
                replicate=replicate,
                seed=seed,
                event_start=start,
                off_start=off,
                random_start=random_position,
                channel=r["channel"],
                window=r["window"],
            )
            cases.append(case)
            signals.append(np.stack([variants[v].reshape(1, 15, 200) for v in VARIANTS]))
            quality_rows.append(dict(**case, variant="clean", **quality(x, x, relative)))
            for name, y in variants.items():
                edit_start = (
                    start if name == "local_event" else off if name == "local_off" else None
                )
                quality_rows.append(
                    dict(**case, variant=name, **quality(x, y, relative, edit_start))
                )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "signals.npz", signals=np.stack(signals))
    document = dict(
        recipe=manifest["recipe"],
        variants=VARIANTS,
        cases=cases,
        exclusions=exclusions,
        parent_inputs_sha256=source_hashes["prepared/inputs.npz"],
        parent_readouts_sha256=source_hashes["results/readouts.npz"],
    )
    (output / "cases.json").write_text(json.dumps(document, indent=2) + "\n")
    (output / "quality.json").write_text(json.dumps(quality_rows, allow_nan=False) + "\n")
    print(
        "Prepared",
        len(cases),
        "replicate-cases,",
        len(cases) // 3,
        "windows; excluded",
        len(exclusions),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["previous", "events", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.previous, a.events, a.output)
