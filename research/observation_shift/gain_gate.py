"""Fixed-head, paired gain gate before representation-repair experiments."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from eeglens import SignalBatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase"))
from evaluate import SITES, model_loader

GAINS = (0.5, 0.75, 1.0, 1.5, 2.0)


def token_values(activation):
    x = activation.tensor
    if x.ndim == 4:
        return x[:, 0]
    return x[:, 1:] if activation.layout == "tokens" else x


def run(model, checkpoint, previous, output, random=False):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    label = model + ("-random" if random else "")
    assert not (output / f"{label}.npz").exists(), "Use a fresh output"
    manifest = json.loads((previous / "prepared/manifest.json").read_text())
    assert "UV-text-verified" in manifest["recipe"]
    hashes = json.loads((previous / "artifact_hashes.json").read_text())
    for name in [
        "prepared/inputs.npz",
        "prepared/manifest.json",
        "results/readouts.npz",
        f"features/{label}.npz",
    ]:
        assert hashlib.sha256((previous / name).read_bytes()).hexdigest() == hashes[name]
    x = np.load(previous / "prepared/inputs.npz")["x"]
    rows = manifest["rows"]
    subjects = np.array([r["subject"] for r in rows])
    y = np.array([r["n2"] for r in rows])
    original = np.load(previous / f"features/{label}.npz")["output"]
    readouts = np.load(previous / "results/readouts.npz")
    lens = model_loader(model, checkpoint, random)
    features = np.empty((len(GAINS), len(x), 15, 200), dtype=np.float32)
    layers = np.empty((len(GAINS), len(x), len(SITES), 15, 200), dtype=np.float32)
    inverse_errors = []
    normalized_errors = []
    for gi, gain in enumerate(GAINS):
        changed = (x * gain).astype(np.float32)
        inverse_errors.append(float(np.linalg.norm(changed / gain - x) / np.linalg.norm(x)))

        def standardize(a):
            return (a - a.mean(axis=(2, 3), keepdims=True)) / a.std(
                axis=(2, 3), keepdims=True
            ).clip(1e-8)

        normalized_errors.append(
            float(
                np.linalg.norm(standardize(changed) - standardize(x))
                / np.linalg.norm(standardize(x))
            )
        )
        assert inverse_errors[-1] < 1e-6 and normalized_errors[-1] < 1e-6
        for subject in range(1, 9):
            indices = np.flatnonzero(subjects == subject)
            for start in range(0, len(indices), 8):
                ids = indices[start : start + 8]
                batch = SignalBatch(
                    torch.from_numpy(changed[ids]),
                    tuple(f"s{subject}:w{rows[i]['window']}" for i in ids),
                    (rows[ids[0]]["channel"],),
                    200,
                    manifest["recipe"] + ":gain-calibration-counterfactual",
                )
                result = lens.run_with_cache(batch, sites=SITES)
                features[gi, ids] = (
                    result.output[:, 0].numpy()
                    if result.output.ndim == 4
                    else result.output.numpy()
                )
                for si, site in enumerate(SITES):
                    layers[gi, ids, si] = token_values(result.cache[site]).numpy()
        print(label, "gain", gain, "complete", flush=True)
    np.testing.assert_allclose(features[2].mean(1), original, atol=1e-5, rtol=1e-5)
    predictions = []
    for gi, gain in enumerate(GAINS):
        for subject in range(1, 9):
            test = subjects == subject
            train = ~np.isin(subjects, [subject, subject % 8 + 1, (subject + 1) % 8 + 1])
            weight = readouts[f"{label}_s{subject}_weight"]
            intercept = readouts[f"{label}_s{subject}_intercept"]
            clean = original[test] @ weight + intercept
            margins = features[gi, test].mean(1) @ weight + intercept
            scale = float((original[train] @ weight).std())
            diff = layers[gi, test] - layers[2, test]
            distances = np.linalg.norm(diff, axis=-1) / np.maximum(
                np.linalg.norm(layers[2, test], axis=-1), 1e-8
            )
            predictions.append(
                dict(
                    model=label,
                    subject=subject,
                    gain=gain,
                    n=int(test.sum()),
                    balanced_accuracy=float(balanced_accuracy_score(y[test], margins >= 0)),
                    auc=float(roc_auc_score(y[test], margins)),
                    flip_fraction=float(np.mean((clean >= 0) != (margins >= 0))),
                    mean_absolute_margin_change=float(np.mean(abs(margins - clean)) / scale),
                    mean_signed_margin_change=float(np.mean(margins - clean) / scale),
                    layer_patch_relative_changes=distances.mean(0).tolist(),
                )
            )
    np.savez_compressed(
        output / f"{label}.npz", gains=np.array(GAINS), features=features, layers=layers
    )
    record = dict(
        model=lens.manifest,
        gains=GAINS,
        sites=SITES,
        subjects=8,
        windows=len(x),
        inverse_relative_errors=inverse_errors,
        normalized_relative_errors=normalized_errors,
        results=predictions,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        imported_model_loader_sha256=hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "phase/evaluate.py").read_bytes()
        ).hexdigest(),
        protocol_sha256=hashlib.sha256(
            Path(__file__).with_name("PROTOCOL.md").read_bytes()
        ).hexdigest(),
        parent_artifact_manifest_sha256=hashlib.sha256(
            (previous / "artifact_hashes.json").read_bytes()
        ).hexdigest(),
    )
    (output / f"{label}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(label, "saved complete gain gate", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cbramod", "labram"], required=True)
    p.add_argument("--random", action="store_true")
    for name in ["checkpoint", "previous", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.model, a.checkpoint, a.previous, a.output, a.random)
