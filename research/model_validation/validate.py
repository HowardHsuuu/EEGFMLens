"""Run native-vs-EEGLens checks using externally supplied model code and weights."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from eeglens import Ablation, EEGLens, Replacement, Selection, SignalBatch, SubspaceAblation
from eeglens.adapters.eegpt import EEGPTAdapter
from eeglens.errors import ValidationError


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, name):
    if name == "eegpt":
        from functools import partial

        from safetensors.torch import load_file

        path = root / "repos/EEGPT/downstream/Modules/models/EEGPT_mcae_finetune.py"
        module = load_module(path, "native_eegpt")
        model = module.EEGTransformer(
            img_size=(4, 1024),
            patch_size=64,
            embed_dim=512,
            embed_num=4,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        )
        checkpoint = root / "checkpoints/eegpt-braindecode.safetensors"
        state = load_file(str(checkpoint))
        assert set(k for k in state if not k.startswith("target_encoder.")) == {"chans_id"}
        state = {
            k.removeprefix("target_encoder."): v
            for k, v in state.items()
            if k.startswith("target_encoder.")
        }
        model.load_state_dict(state, strict=True)
        adapter = EEGPTAdapter(model)
        shape = (2, 4, 16, 64)
        channels = ("C3", "C4", "F3", "F4")
        fs = 256
    elif name in {"biot", "biot-prest"}:
        path = root / "biot.py"
        module = load_module(path, "native_biot")
        n_channels = 16 if name == "biot-prest" else 18
        model = module.BIOTEncoder(n_channels=n_channels)
        checkpoint = (
            root
            / "checkpoints"
            / ("biot-prest.ckpt" if name == "biot-prest" else "biot-six-mirror.ckpt")
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        shape = (2, n_channels, 10, 200)
        fs = 200
        channels = (
            "FP1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "FP2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
            "C3-A2",
            "C4-A1",
        )
        channels = channels[:n_channels]
    else:
        sys.path[:0] = [str(root / "repos/dn3"), str(root / "repos/BENDR")]
        import dn3_ext

        path = root / "repos/BENDR/dn3_ext.py"
        checkpoint = root / "checkpoints/bendr-mirror-encoder.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = dn3_ext.ConvEncoderBENDR(in_features=20, encoder_h=512)
        model.load_state_dict(state, strict=True)
        shape = (2, 20, 4, 256)
        fs = 256
        channels = tuple(f"input-{i}" for i in range(20))
    from eeglens import BENDREncoderAdapter, BIOTAdapter

    if name in {"biot", "biot-prest"}:
        adapter = BIOTAdapter(model, channels=channels)
    elif name == "bendr":
        adapter = BENDREncoderAdapter(model, channels=channels)
    model.eval().requires_grad_(False)
    return model, adapter, shape, channels, fs, path, checkpoint


def run(root, name, output):
    torch.set_num_threads(2)
    torch.manual_seed(4311)
    model, adapter, shape, channels, fs, source, checkpoint = build(root, name)
    lens = EEGLens(model, adapter)
    records = []
    for size in [1, 2]:
        batch = SignalBatch(
            torch.randn(size, *shape[1:]),
            tuple(f"t{i}" for i in range(size)),
            channels,
            fs,
            "synthetic-runtime-validation-v1",
        )
        with torch.no_grad():
            native = adapter.forward(model, batch)
        clean = lens.run_with_cache(batch)
        torch.testing.assert_close(clean.output, native, rtol=0, atol=0)
        for site in lens.sites():
            identity = lens.run_with_interventions(
                batch, interventions=[Replacement(site.name, clean.cache[site.name])]
            )
            torch.testing.assert_close(identity.output, native, rtol=0, atol=0)
            edited = lens.run_with_interventions(batch, interventions=[Ablation(site.name)])
            handle = model.get_submodule(site.module_path).register_forward_hook(
                lambda m, a, o: torch.zeros_like(o)
            )
            try:
                with torch.no_grad():
                    manual = adapter.forward(model, batch)
            finally:
                handle.remove()
            torch.testing.assert_close(edited.output, manual, rtol=0, atol=0)
            assert torch.isfinite(edited.output).all()
            for centered in (False, True):
                features = clean.cache[site.name].tensor.shape[-1]
                basis = torch.eye(features)[:, :2]
                center = torch.zeros(features)
                if centered:
                    center[:2] = torch.tensor([0.25, -0.5])
                erased = lens.run_with_interventions(
                    batch,
                    interventions=[SubspaceAblation(site.name, basis, center)],
                    sites=(site.name,),
                )

                def erase_native_features(m, a, o):
                    result = o.clone()
                    # Independent native axes: BENDR is B,D,T; EEGPT and
                    # BIOT keep features last (including folded EEGPT windows).
                    if name == "bendr":
                        result[:, :2, :] -= result[:, :2, :] - center[:2, None]
                    else:
                        result[..., :2] -= result[..., :2] - center[:2]
                    return result

                handle = model.get_submodule(site.module_path).register_forward_hook(
                    erase_native_features
                )
                try:
                    with torch.no_grad():
                        manual_features = adapter.forward(model, batch)
                finally:
                    handle.remove()
                torch.testing.assert_close(erased.output, manual_features, rtol=0, atol=0)
                observed = erased.cache[site.name].tensor
                torch.testing.assert_close(
                    observed[..., :2],
                    center[:2].expand_as(observed[..., :2]),
                    rtol=0,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    observed[..., 2:],
                    clean.cache[site.name].tensor[..., 2:],
                    rtol=0,
                    atol=0,
                )
            for selection in (Selection(sensors=(channels[0],)), Selection(patches=(0,))):
                try:
                    lens.run_with_interventions(
                        batch, interventions=[Ablation(site.name, selection)]
                    )
                except ValidationError as exc:
                    assert "no sensor/patch selector" in str(exc)
                else:
                    raise AssertionError(f"Unsupported physical selector accepted at {site.name}")
                assert all(not m._forward_hooks for m in model.modules())
            records.append(
                dict(
                    batch_size=size,
                    site=site.name,
                    cache_shape=list(clean.cache[site.name].tensor.shape),
                    native_shape=list(clean.cache[site.name].native_shape),
                    output_change=float((edited.output - native).norm()),
                    native_feature_axis=1 if name == "bendr" else -1,
                    partial_feature_erasure=True,
                    centered_feature_erasure=True,
                    untouched_features_exact=True,
                    unsupported_sensor_patch_rejected=True,
                )
            )
        final_site = {
            "eegpt": "norm.output",
            "biot": "transformer.output",
            "biot-prest": "transformer.output",
        }.get(name, lens.sites()[-1].name)
        corrupted = replace(
            batch, data=batch.data.flip(0) * 0.3 + 1, trial_ids=batch.trial_ids[::-1]
        )
        restored = lens.run_with_interventions(
            corrupted, interventions=[Replacement(final_site, clean.cache[final_site])]
        )
        torch.testing.assert_close(restored.output, native.flip(0), rtol=0, atol=0)
        assert all(not m._forward_hooks for m in model.modules())
        torch.testing.assert_close(lens.run_with_cache(batch).output, native, rtol=0, atol=0)
    doc = dict(
        model=name,
        source_sha256=sha(source),
        checkpoint_sha256=sha(checkpoint),
        checkpoint=str(checkpoint),
        strict_load=True,
        reordered_trial_recovery=True,
        runner_sha256=sha(Path(__file__)),
        runtime_sha256=sha(Path(inspect.getfile(EEGLens))),
        output_shape=list(native.shape),
        torch_version=torch.__version__,
        records=records,
        feature_subspace_verified=True,
        adapter_sha256=sha(Path(inspect.getfile(type(adapter)))),
        scope="Synthetic input runtime validation; not task accuracy, physiological validity or full-model pretraining verification. BENDR covers the pretrained convolutional encoder only.",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(
        name,
        len(records),
        "native observation / identity / manual-ablation checks passed",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", choices=["eegpt", "biot", "biot-prest", "bendr"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.model, a.output)
