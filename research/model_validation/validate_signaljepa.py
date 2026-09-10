"""Strictly load the full SignalJEPA checkpoint and verify native interventions."""

import argparse
import hashlib
import importlib
import inspect
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file
from semantic_checks import check_features, check_rejected_selectors

from eeglens import Ablation, EEGLens, Replacement, SignalBatch
from eeglens.adapters.signaljepa import SignalJEPAAdapter


def run(SignalJEPA, *, import_mode):
    root = Path("research/eeglens_model_validation")
    checkpoint_dir = root / "expansion/braindecode-signal-jepa"
    config = json.loads((checkpoint_dir / "config.json").read_text())
    config.pop("braindecode_version")
    assert config.pop("activation") == "torch.nn.modules.activation.GELU"
    config["activation"] = torch.nn.GELU
    torch.set_num_threads(2)
    torch.manual_seed(4311)
    model = SignalJEPA(**config)
    checkpoint = checkpoint_dir / "model.safetensors"
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    model.eval().requires_grad_(False)
    adapter = SignalJEPAAdapter(model)
    lens = EEGLens(model, adapter)
    records = []
    for b in (1, 2):
        batch = SignalBatch(
            torch.randn(b, len(adapter.channels), 3, 128),
            tuple(f"t{i}" for i in range(b)),
            adapter.channels,
            128,
            "synthetic-signaljepa-v1",
        )
        with torch.no_grad():
            native = model(batch.data.flatten(2))
        clean = lens.run_with_cache(batch)
        torch.testing.assert_close(clean.output, native, rtol=0, atol=0)
        for site in lens.sites():
            restored = lens.run_with_interventions(
                batch, interventions=[Replacement(site.name, clean.cache[site.name])]
            )
            torch.testing.assert_close(restored.output, native, rtol=0, atol=0)
            edited = lens.run_with_interventions(batch, interventions=[Ablation(site.name)])
            h = model.get_submodule(site.module_path).register_forward_hook(
                lambda m, a, o: torch.zeros_like(o)
            )
            try:
                with torch.no_grad():
                    manual = model(batch.data.flatten(2))
            finally:
                h.remove()
            torch.testing.assert_close(edited.output, manual, rtol=0, atol=0)
            assert torch.isfinite(edited.output).all()
            semantic = check_features(
                lens, batch, site, clean, lambda: model(batch.data.flatten(2)), native_axis=-1
            )
            semantic.update(check_rejected_selectors(lens, batch, site))
            records.append(
                dict(
                    batch=b,
                    site=site.name,
                    delta=float((edited.output - native).norm()),
                    **semantic,
                )
            )
        order = list(reversed(range(b)))
        recipient = replace(
            batch, data=batch.data[order] + 0.3, trial_ids=tuple(batch.trial_ids[i] for i in order)
        )
        recovered = lens.run_with_interventions(
            recipient, interventions=[Replacement("encoder.output", clean.cache["encoder.output"])]
        )
        torch.testing.assert_close(recovered.output, native[order], rtol=0, atol=0)
        assert all(not m._forward_hooks for m in model.modules())
    result = dict(
        records=records,
        strict_full_load=True,
        import_mode=import_mode,
        reordered_donor_recovery=True,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        semantic_helper_sha256=hashlib.sha256(
            Path(inspect.getfile(check_features)).read_bytes()
        ).hexdigest(),
        adapter_sha256=hashlib.sha256(
            Path(inspect.getfile(SignalJEPAAdapter)).read_bytes()
        ).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        native_source_sha256=hashlib.sha256(
            Path(inspect.getfile(SignalJEPA)).read_bytes()
        ).hexdigest(),
        torch_version=torch.__version__,
    )
    output = root / "semantic-v1"
    output.mkdir(exist_ok=True)
    (output / "signaljepa.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SignalJEPA", len(records), "conditions passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct-source",
        type=Path,
        help="Load unchanged modules from a braindecode source directory, bypassing package __init__ imports",
    )
    args = parser.parse_args()
    if args.direct_source:
        source = args.direct_source.resolve()
        for name, path in (
            ("braindecode", source),
            ("braindecode.models", source / "models"),
            ("braindecode.modules", source / "modules"),
        ):
            if name in sys.modules:
                raise RuntimeError("Direct-source validation requires a fresh Python process")
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        SignalJEPA = importlib.import_module("braindecode.models.signal_jepa").SignalJEPA
        mode = "unchanged source modules; package __init__ imports bypassed"
    else:
        from braindecode.models import SignalJEPA

        mode = "standard braindecode.models import"
    run(SignalJEPA, import_mode=mode)
