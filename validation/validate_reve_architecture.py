"""Architecture-only REVE checks. NO pretrained weights; not a coverage claim."""

import importlib
import json
import sys
import types
from pathlib import Path

import torch

from eeglens import Ablation, EEGLens, Replacement, SignalBatch
from eeglens.adapters.reve import REVEAdapter


def run():
    root = Path("research/eeglens_model_validation")
    source = root / "repos/REVE/hf/reve-base"
    package = types.ModuleType("native_reve")
    package.__path__ = [str(source)]
    sys.modules["native_reve"] = package
    module = importlib.import_module("native_reve.modeling_reve")
    torch.set_num_threads(2)
    torch.manual_seed(4311)
    model = module.Reve(module.ReveConfig()).eval().requires_grad_(False)
    adapter = REVEAdapter(
        model, channels=("C3", "C4"), positions=torch.zeros(2, 3), sampling_rate=200
    )
    lens = EEGLens(model, adapter)
    records = []
    for b in (1, 2):
        batch = SignalBatch(
            torch.randn(b, 2, 4, 100),
            tuple(f"t{i}" for i in range(b)),
            adapter.channels,
            200,
            "synthetic-reve-architecture-only",
        )
        with torch.no_grad():
            native = model(batch.data.flatten(2), torch.zeros(b, 2, 3))
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
                    manual = model(batch.data.flatten(2), torch.zeros(b, 2, 3))
            finally:
                h.remove()
            torch.testing.assert_close(edited.output, manual, rtol=0, atol=0)
            assert torch.isfinite(edited.output).all()
            records.append(dict(batch=b, site=site.name))
        assert all(not m._forward_hooks for m in model.modules())
    (root / "expansion/results/reve-architecture-only.json").write_text(
        json.dumps(
            dict(
                pretrained=False,
                scope="random initialized official base architecture only; gated weights unavailable",
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print("REVE architecture only:", len(records), "conditions passed; NOT pretrained validation")


if __name__ == "__main__":
    run()
