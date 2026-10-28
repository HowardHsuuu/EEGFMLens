"""Execute bounded checkpoint checks on current CBraMod/LaBraM adapters.

Supply trusted local weights and pinned upstream source explicitly. No downloads,
participant data, optimization or task-quality evaluation. Run one model/process.
"""

import argparse
import ast
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

import eegfmlens
from eegfmlens import Ablation, Replacement, Selection, SignalBatch, load_cbramod, load_labram

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from native_sources import REVISIONS, native_module  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def difference(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    return float((a - b).abs().max())


def hook_inventory(model):
    return [(id(m), tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in model.modules()]


def raw_zero(raw, model_name, site_name, size):
    """Native C4/patch-0 indices; independent of adapter expose/restore/Selection."""
    value = raw.clone()
    if model_name == "labram":
        # Three channels, two patches, channel-major tokens; blocks prepend CLS.
        value[:, 4 + (site_name != "embedding.output"), :] = 0
    elif site_name == "waveform.code":
        value[:, :, 4, :] = 0
    elif ".spatial." in site_name:
        value[torch.arange(size) * 2, 2, :] = 0  # [B*patch, channel, feature]
    elif ".temporal." in site_name:
        value[torch.arange(size) * 3 + 2, 0, :] = 0  # [B*channel, patch, feature]
    else:
        value[:, 2, 0, :] = 0
    return value


def run(args):
    start = time.perf_counter()
    if args.report.exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint report")
    module = native_module(args.model, args.source)
    torch.set_num_threads(2)
    torch.manual_seed(20261028)
    if args.model == "cbramod":
        lens = load_cbramod(
            args.checkpoint, model_factory=module.CBraMod, output=args.component,
            source_revision=REVISIONS[args.model], expected_sha256=args.checkpoint_sha256,
        )
        channels = ("C3", "CZ", "C4")
        channel_indices = None
    else:
        if args.component != "features":
            raise ValueError("LaBraM runner supports only the pretrained student feature path")
        lens = load_labram(
            args.checkpoint, model_factory=module.labram_base_patch200_200,
            source_revision=REVISIONS[args.model], expected_sha256=args.checkpoint_sha256,
        )
        channels = ("C3", "CZ", "C4")
        # Read the upstream vocabulary without importing its training dependencies.
        tree = ast.parse((args.source / "utils.py").read_text())
        vocab = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "standard_1020" for t in n.targets))
        channel_indices = [0] + [vocab.index(c) + 1 for c in channels]
    model = lens.model
    initial_hooks = hook_inventory(model)
    records = []
    for size in (1, 2):
        torch.manual_seed(20261028 + size)
        batch = SignalBatch(torch.randn(size, 3, 2, 200), tuple(f"synthetic-{i}" for i in range(size)),
                            channels, 200, "synthetic-no-preprocessing")
        mask = torch.zeros(size, 3, 2, dtype=torch.bool)
        mask[:, 0, 1] = True
        options = {"mask": mask.tolist()} if args.component == "reconstruction" else {}

        def native():
            if args.model == "labram":
                return model.forward_features(batch.data, input_chans=channel_indices,
                                              return_patch_tokens=True, return_all_tokens=False)
            return model(batch.data, mask=mask) if options else model(batch.data)

        with torch.no_grad():
            baseline = lens.run_with_cache(batch, **options)
            baseline_error = difference(baseline.output, native())
            assert all(n == 1 for n in baseline.calls.values())
            for name, activation in baseline.cache.items():
                site = lens.adapter.require(name)
                identity = lens.run_with_interventions(
                    batch, interventions=[Replacement(name, activation)], **options)
                identity_error = difference(identity.output, baseline.output)
                observed = lens.run_with_interventions(
                    batch, interventions=[Ablation(name, Selection(sensors=("C4",), patches=(0,)))],
                    sites=(name,), **options)
                # Independently construct expected exposed coordinates, including unchanged entries.
                expected = activation.tensor.clone()
                if args.model == "labram":
                    expected[:, 4 + (name != "embedding.output"), :] = 0
                else:
                    expected[:, 2, 0, :] = 0
                assert torch.equal(expected, observed.cache[name].tensor)
                calls = []

                def independent_hook(_module, _inputs, output):
                    raw = output[0] if isinstance(output, tuple) else output
                    edited = raw_zero(raw, args.model, name, size)
                    calls.append(1)
                    return (edited, *output[1:]) if isinstance(output, tuple) else edited

                handle = model.get_submodule(site.module_path).register_forward_hook(independent_hook)
                try:
                    oracle = native()
                finally:
                    handle.remove()
                assert len(calls) == 1
                oracle_error = difference(observed.output, oracle)
                effect = float((observed.output - baseline.output).abs().max())
                assert effect > 0, name
                assert hook_inventory(model) == initial_hooks
                cleanup_error = difference(lens.run_with_cache(batch, sites=[], **options).output,
                                           baseline.output)
                records.append({"batch": size, "site": name, "module": site.module_path,
                                "baseline_max_abs_error": baseline_error,
                                "identity_max_abs_error": identity_error,
                                "independent_edit_max_abs_error": oracle_error,
                                "cleanup_max_abs_error": cleanup_error,
                                "effect_max_abs": effect, "selected_zero_and_untouched_exact": True,
                                "single_call": True, "hooks_restored": True})
    package = Path(eegfmlens.__file__).parent
    report = {
        "schema": "eegfmlens-executable-checkpoint-v1", "status": "pass",
        "model": args.model, "component": args.component,
        "checkpoint_sha256": digest(args.checkpoint), "upstream_revision": REVISIONS[args.model],
        "runner_sha256": digest(__file__),
        "runtime_sha256": {p.relative_to(package).as_posix(): digest(p)
                           for p in sorted(package.rglob("*.py"))},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "cpu", "dtype": "float32", "threads": 2},
        "input": {"shape": ["B", 3, 2, 200], "batches": [1, 2], "channels": channels,
                  "sampling_rate": 200, "seed_rule": "20261028 + batch size",
                  "edit": "zero C4 at patch 0", "mask": "C3 patch 1" if options else "none"},
        "tolerance": {"atol": 1e-6, "rtol": 1e-5, "edit_support": "exact"},
        "oracle": "Direct native forward and independent raw-coordinate zero hook; no adapter selector or layout conversion in the oracle.",
        "scope": "Synthetic integration of declared sites, not task quality, all-input equivalence or the six-model scientific assay.",
        "records": records, "elapsed_seconds": time.perf_counter() - start,
    }
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "pass", "model": args.model, "component": args.component,
                      "site_batch_records": len(records), "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cbramod", "labram"], required=True)
    parser.add_argument("--component", choices=["features", "reconstruction"], default="features")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    run(parser.parse_args())
