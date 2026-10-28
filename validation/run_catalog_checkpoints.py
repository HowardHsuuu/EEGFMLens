"""Repeat checkpoint integration on the remaining declared native components.

All assets are local and supplied explicitly. A manifest binds source/config/
weight bytes before any model import; reports contain hashes, not local paths.
"""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import torch
from checkpoint_components import build, native_call
from run_checkpoint_checks import difference, digest, hook_inventory

import eegfmlens
from eegfmlens import Ablation, EEGLens, Replacement, SignalBatch

MODELS = (
    "eegpt",
    "biot",
    "biot-prest",
    "bendr",
    "bendr-context",
    "brainomni",
    "csbrain",
    "neurorvq",
    "signaljepa",
    "diver",
    "steegformer",
)
SCOPES = {
    "eegpt": "Converted pretrained target encoder; not masked encoder/predictor/reconstructor.",
    "biot": "Six-dataset 18-channel encoder from retained mirror.",
    "biot-prest": "PREST 16-channel encoder.",
    "bendr": "Pretrained convolutional encoder; no DN3 preprocessing or classifier.",
    "bendr-context": "Pretrained convolutional encoder and contextualizer; no classifier.",
    "brainomni": "Tiny encode with synthetic geometry; native eval dropout uses paired RNG.",
    "csbrain": "Pretrained unmasked forward with projection and region-sort mapping.",
    "neurorvq": "Four raw backbone branches; absent downstream norms replaced with identity, pretraining heads excluded.",
    "signaljepa": "Full-channel local/contextual encoder; decoder loaded but not executed. Unmodified modules imported without package initialization.",
    "diver": "EEG backbone y output, synthetic xyz and paired native dropout RNG; not study iEEG reconstruction.",
    "steegformer": "Small forward_features; decoder and unused parent image-position parameter excluded.",
}


def verify_assets(args):
    manifest = json.loads(args.manifest.read_text())
    entry = manifest[args.model]
    for key, expected in entry["files"].items():
        base, relative = key.split(":", 1)
        root = args.braindecode_source if base == "braindecode" else args.asset_root
        if root is None or base not in {"assets", "braindecode"}:
            raise ValueError("Missing or unrecognized asset root")
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Asset digest mismatch: {key}")
    return entry


def run(args):
    start = time.perf_counter()
    if args.report.exists():
        raise FileExistsError("Refusing to overwrite an existing report")
    assets = verify_assets(args)
    torch.set_num_threads(2)
    torch.manual_seed(4311)
    model, adapter, shape, channels, fs = build(
        args.asset_root, args.model, args.braindecode_source
    )
    lens = EEGLens(model, adapter)
    initial = hook_inventory(model)
    records = []

    def execute(function, *a, **kw):
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(9417)
            return function(*a, **kw)

    for b in (1, 2):
        torch.manual_seed(20261028 + b)
        batch = SignalBatch(
            torch.randn(b, *shape),
            tuple(f"synthetic-{i}" for i in range(b)),
            tuple(channels),
            fs,
            "synthetic-no-preprocessing",
        )
        def native():
            return native_call(model, adapter, args.model, batch)
        baseline = execute(lens.run_with_cache, batch)
        baseline_error = difference(baseline.output, execute(native))
        for site in lens.sites():
            identity = execute(
                lens.run_with_interventions,
                batch,
                interventions=[Replacement(site.name, baseline.cache[site.name])],
            )
            identity_error = difference(identity.output, baseline.output)
            edited = execute(
                lens.run_with_interventions,
                batch,
                interventions=[Ablation(site.name)],
                sites=(site.name,),
            )
            assert torch.count_nonzero(edited.cache[site.name].tensor) == 0
            calls = []

            def independent_hook(_module, _input, output):
                index = len(calls)
                calls.append(1)
                if index != site.call_index:
                    return None
                if site.tensor_index is None:
                    return torch.zeros_like(output)
                result = list(output) if isinstance(output, (tuple, list)) else dict(output)
                result[site.tensor_index] = torch.zeros_like(output[site.tensor_index])
                return tuple(result) if isinstance(output, tuple) else result

            handle = model.get_submodule(site.module_path).register_forward_hook(independent_hook)
            try:
                oracle = execute(native)
            finally:
                handle.remove()
            assert len(calls) == site.expected_calls
            oracle_error = difference(edited.output, oracle)
            effect = float((edited.output - baseline.output).abs().max())
            # Retain zero effects too: a declared site need not influence this endpoint.
            assert hook_inventory(model) == initial
            cleanup_error = difference(
                execute(lens.run_with_cache, batch, sites=[]).output, baseline.output
            )
            records.append(
                {
                    "batch": b,
                    "site": site.name,
                    "module": site.module_path,
                    "call_index": site.call_index,
                    "expected_calls": site.expected_calls,
                    "observed_calls": len(calls),
                    "baseline_max_abs_error": baseline_error,
                    "identity_max_abs_error": identity_error,
                    "independent_edit_max_abs_error": oracle_error,
                    "cleanup_max_abs_error": cleanup_error,
                    "effect_max_abs": effect,
                    "selected_zero_exact": True,
                    "hooks_restored": True,
                }
            )
    package = Path(eegfmlens.__file__).parent
    dependencies = {}
    for name in (
        "torch",
        "numpy",
        "timm",
        "einops",
        "safetensors",
        "mup",
        "braindecode",
        "linear-attention-transformer",
        "vector-quantize-pytorch",
        "mne",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {
        "schema": "eegfmlens-executable-checkpoint-v1",
        "status": "pass",
        "model": args.model,
        "component": SCOPES[args.model],
        "assets": assets,
        "runner_sha256": digest(__file__),
        "builder_sha256": digest(Path(__file__).with_name("checkpoint_components.py")),
        "helper_sha256": digest(Path(__file__).with_name("run_checkpoint_checks.py")),
        "runtime_sha256": {
            p.relative_to(package).as_posix(): digest(p) for p in sorted(package.rglob("*.py"))
        },
        "environment": {
            "python": platform.python_version(),
            "dependencies": dependencies,
            "device": "cpu",
            "dtype": "float32",
            "threads": 2,
        },
        "input": {
            "shape": ["B", *shape],
            "batches": [1, 2],
            "channels": list(channels),
            "sampling_rate": fs,
            "seed_rule": "20261028 + batch size",
            "paired_execution_seed": 9417,
        },
        "tolerance": {"atol": 1e-6, "rtol": 1e-5, "selected_zero": "exact"},
        "oracle": "Direct native forward and invocation-specific whole-state zero hook; no adapter forward, expose, restore or selector in the oracle. Declared geometry metadata is shared.",
        "scope": "Synthetic core integration only; whole-state ablation does not validate every partial-coordinate or feature edit. Historical coordinate checks, current fixtures and study admissions retain separate scope.",
        "component_adjustments": getattr(model, "_validation_component", {}),
        "records": records,
        "elapsed_seconds": time.perf_counter() - start,
    }
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "model": args.model,
                "status": "pass",
                "records": len(records),
                "seconds": report["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--braindecode-source", type=Path)
    p.add_argument("--report", type=Path, required=True)
    run(p.parse_args())
