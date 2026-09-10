"""Checkpoint-backed BrainOmni encode integration test; synthetic sensor geometry."""

import hashlib
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from semantic_checks import check_rejected_selectors

from eeglens import Ablation, EEGLens, Replacement, SignalBatch, SubspaceAblation
from eeglens.adapters.brainomni import BrainOmniAdapter


def paired_rng(fn, *args, **kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9137)
        return fn(*args, **kwargs)


def run():
    root = Path("research/eeglens_model_validation")
    source = root / "repos/BrainOmni"
    sys.path.insert(0, str(source))
    from brainomni.model import BrainOmni

    config = root / "expansion/OpenTSLab-BrainOmni/tiny/model_cfg.json"
    checkpoint = config.with_name("BrainOmni.pt")
    torch.set_num_threads(2)
    torch.manual_seed(4311)
    model = BrainOmni(**json.loads(config.read_text()))
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().requires_grad_(False)
    positions = torch.randn(4, 6)
    kinds = torch.zeros(4, dtype=torch.long)
    adapter = BrainOmniAdapter(
        model, channels=("C3", "C4", "F3", "F4"), positions=positions, sensor_types=kinds
    )
    lens = EEGLens(model, adapter)
    records = []
    for b in [1, 2]:
        batch = SignalBatch(
            torch.randn(b, 4, 3, 512),
            tuple(f"t{i}" for i in range(b)),
            adapter.channels,
            256,
            "synthetic-geometry-v1",
        )
        with torch.no_grad():
            native = paired_rng(
                model.encode,
                batch.data.flatten(2),
                positions.unsqueeze(0).expand(b, -1, -1),
                kinds.unsqueeze(0).expand(b, -1),
            )
        clean = paired_rng(lens.run_with_cache, batch)
        torch.testing.assert_close(clean.output, native, rtol=0, atol=0)
        for site in lens.sites():
            identity = paired_rng(
                lens.run_with_interventions,
                batch,
                interventions=[Replacement(site.name, clean.cache[site.name])],
            )
            torch.testing.assert_close(identity.output, native, rtol=0, atol=0)
            ablated = paired_rng(
                lens.run_with_interventions, batch, interventions=[Ablation(site.name)]
            )
            h = model.get_submodule(site.module_path).register_forward_hook(
                lambda m, a, o: torch.zeros_like(o)
            )
            try:
                with torch.no_grad():
                    manual = paired_rng(adapter.forward, model, batch)
            finally:
                h.remove()
            torch.testing.assert_close(ablated.output, manual, rtol=0, atol=0)
            assert torch.isfinite(ablated.output).all()
            current = clean.cache[site.name].tensor
            features = current.shape[-1]
            basis = torch.eye(features)[:, :2]
            for values in ((0.0, 0.0), (0.25, -0.5)):
                center = torch.zeros(features)
                center[:2] = torch.tensor(values)
                edited = paired_rng(
                    lens.run_with_interventions,
                    batch,
                    interventions=[SubspaceAblation(site.name, basis, center)],
                    sites=(site.name,),
                )

                def native_features(module, args, output):
                    # Native projection and encoder blocks retain features last.
                    result = output.clone()
                    result[..., :2] -= result[..., :2] - center[:2]
                    return result

                handle = model.get_submodule(site.module_path).register_forward_hook(
                    native_features
                )
                try:
                    with torch.no_grad():
                        expected = paired_rng(
                            model.encode,
                            batch.data.flatten(2),
                            positions.unsqueeze(0).expand(b, -1, -1),
                            kinds.unsqueeze(0).expand(b, -1),
                        )
                finally:
                    handle.remove()
                torch.testing.assert_close(edited.output, expected, rtol=0, atol=0)
                observed = edited.cache[site.name].tensor
                torch.testing.assert_close(observed[..., 2:], current[..., 2:], rtol=0, atol=0)
                torch.testing.assert_close(
                    observed[..., :2], center[:2].expand_as(observed[..., :2]), rtol=0, atol=1e-6
                )
            rejection = check_rejected_selectors(lens, batch, site)
            records.append(
                dict(
                    batch=b,
                    site=site.name,
                    cache_shape=list(clean.cache[site.name].tensor.shape),
                    delta=float((ablated.output - native).norm()),
                    native_feature_axis=-1,
                    centered_feature_erasure=True,
                    partial_feature_erasure=True,
                    untouched_features_exact=True,
                    **rejection,
                )
            )
        corrupted = replace(
            batch, data=batch.data.flip(0) * 0.3 + 1, trial_ids=batch.trial_ids[::-1]
        )
        last = lens.sites()[-1].name
        restored = paired_rng(
            lens.run_with_interventions,
            corrupted,
            interventions=[Replacement(last, clean.cache[last])],
        )
        torch.testing.assert_close(restored.output, native.flip(0), rtol=0, atol=0)
        assert all(not m._forward_hooks for m in model.modules())
    output = root / "semantic-v1"
    output.mkdir(exist_ok=True)
    doc = dict(
        records=records,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        semantic_helper_sha256=hashlib.sha256(
            Path(inspect.getfile(check_rejected_selectors)).read_bytes()
        ).hexdigest(),
        adapter_sha256=hashlib.sha256(
            Path(inspect.getfile(BrainOmniAdapter)).read_bytes()
        ).hexdigest(),
        native_source_sha256=hashlib.sha256(
            Path(inspect.getfile(type(model))).read_bytes()
        ).hexdigest(),
        torch_version=torch.__version__,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        adapter_metadata=adapter.metadata(),
        strict_load=True,
        paired_rng_seed=9137,
        native_eval_dropout=True,
        scope="BrainOmni tiny encode with synthetic input/geometry, batch 1/2; not EEG/MEG task performance",
        compatibility="local vq.py import deepspeed.comm replaced by torch.distributed for non-distributed CPU inference",
    )
    (output / "brainomni-tiny.json").write_text(json.dumps(doc, indent=2) + "\n")
    print("BrainOmni tiny:", len(records), "site/batch checks passed")


if __name__ == "__main__":
    run()
