"""Official CSBrain checkpoint and electrode-permutation intervention checks."""

import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
from semantic_checks import check_features

from eeglens import Ablation, EEGLens, Replacement, Selection, SignalBatch
from eeglens.adapters.csbrain import CSBrainAdapter


def run():
    root = Path("research/eeglens_model_validation")
    sys.path.insert(0, str(root / "repos/CSBrain"))
    from models.CSBrain import CSBrain

    channels = (
        "FP1-REF",
        "FP2-REF",
        "F3-REF",
        "F4-REF",
        "C3-REF",
        "C4-REF",
        "P3-REF",
        "P4-REF",
        "O1-REF",
        "O2-REF",
        "F7-REF",
        "F8-REF",
        "T3-REF",
        "T4-REF",
        "T5-REF",
        "T6-REF",
        "FZ-REF",
        "CZ-REF",
        "PZ-REF",
    )
    regions = [0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1]
    order = [0, 10, 2, 16, 3, 11, 1, 6, 18, 7, 12, 14, 15, 13, 8, 9, 4, 17, 5]
    torch.set_num_threads(2)
    torch.manual_seed(4311)
    model = CSBrain(brain_regions=regions, sorted_indices=order)
    checkpoint = root / "expansion/CSBrain.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    print("checkpoint keys", list(state)[:3], flush=True)
    assert all(k.startswith("module.") for k in state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    adapter = CSBrainAdapter(model, channels=channels)
    lens = EEGLens(model, adapter)
    records = []
    for b in [1, 2]:
        batch = SignalBatch(
            torch.randn(b, 19, 4, 200),
            tuple(f"t{i}" for i in range(b)),
            channels,
            200,
            "synthetic-native-v1",
        )
        with torch.no_grad():
            native = model(batch.data)
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
                    manual = model(batch.data)
            finally:
                h.remove()
            torch.testing.assert_close(edited.output, manual, rtol=0, atol=0)
            assert torch.isfinite(edited.output).all()
            semantic = check_features(lens, batch, site, clean, lambda: model(batch.data))
            selection = Selection(sensors=("C3-REF",), patches=(1,))
            selected = lens.run_with_interventions(
                batch, interventions=[Ablation(site.name, selection)]
            )

            def native_selection(module, args, output):
                result = output.clone()
                result[:, order.index(channels.index("C3-REF")), 1, :] = 0
                return result

            handle = model.get_submodule(site.module_path).register_forward_hook(native_selection)
            try:
                with torch.no_grad():
                    expected = model(batch.data)
            finally:
                handle.remove()
            torch.testing.assert_close(selected.output, expected, rtol=0, atol=0)
            records.append(
                dict(
                    batch=b,
                    site=site.name,
                    cache_shape=list(clean.cache[site.name].tensor.shape),
                    delta=float((edited.output - native).norm()),
                    physical_sensor_patch_verified=True,
                    **semantic,
                )
            )
        selection = Selection(sensors=("C3-REF",), patches=(1,))
        edited = lens.run_with_interventions(
            batch, interventions=[Ablation("projection.output", selection)]
        )
        expected = native.clone()
        expected[:, order.index(channels.index("C3-REF")), 1] = 0
        torch.testing.assert_close(edited.output, expected, rtol=0, atol=0)
        assert all(not m._forward_hooks for m in model.modules())
    out = root / "semantic-v1"
    out.mkdir(exist_ok=True)
    (out / "csbrain.json").write_text(
        json.dumps(
            dict(
                records=records,
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                semantic_helper_sha256=hashlib.sha256(
                    Path(inspect.getfile(check_features)).read_bytes()
                ).hexdigest(),
                adapter_sha256=hashlib.sha256(
                    Path(inspect.getfile(CSBrainAdapter)).read_bytes()
                ).hexdigest(),
                native_source_sha256=hashlib.sha256(
                    Path(inspect.getfile(type(model))).read_bytes()
                ).hexdigest(),
                torch_version=torch.__version__,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                strict_load=True,
                sensor_selection_verified=True,
                adapter_metadata=adapter.metadata(),
            ),
            indent=2,
        )
        + "\n"
    )
    print("CSBrain", len(records), "site/batch checks and physical-channel mapping passed")


if __name__ == "__main__":
    run()
