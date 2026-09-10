"""Native checkpoint validation for DIVER-1 EEG and ST-EEGFormer-small."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import pickle
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
from semantic_checks import check_features, check_rejected_selectors

from eeglens import (
    Ablation,
    DIVERAdapter,
    EEGLens,
    Replacement,
    Selection,
    SignalBatch,
    STEEGFormerAdapter,
    SubspaceAblation,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root, name):
    if name == "steegformer":
        source = root / "repos/STEEGFormer/easy_start/models_vit_eeg.py"
        spec = importlib.util.spec_from_file_location("native_steegformer", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        checkpoint = root / "expansion/steegformer-small.pth"
        with torch.serialization.safe_globals([argparse.Namespace]):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["model"]
        model = module.vit_small_patch16(num_classes=0, global_pool=False)
        # Parent timm adds image position embeddings unused by native EEG forward.
        # Remove that unused, randomly initialized parameter from the load contract.
        model.register_parameter("pos_embed", None)
        extra = sorted(set(state) - set(model.state_dict()))
        assert all(k.startswith(("decoder_", "dec_")) or k == "mask_token" for k in extra), extra
        model.load_state_dict({k: v for k, v in state.items() if k not in extra}, strict=True)
        mapping_path = root / "repos/STEEGFormer/pretrain/senloc_file/sen_chan_idx.pkl"
        mapping = pickle.loads(mapping_path.read_bytes())["channels_mapping"]
        channels = ("C3", "C4", "F3")
        adapter = STEEGFormerAdapter(model, channels=channels, channel_mapping=mapping)
        shape, fs = (3, 48, 16), 128
        metadata = dict(
            strict_backbone_load=True,
            excluded_pretraining_keys=extra,
            unused_parent_parameter="pos_embed removed; native EEG forward does not use it",
            channel_mapping_sha256=sha(mapping_path),
        )
    else:
        source_root = root / "repos/DIVER-1"
        sys.path.insert(0, str(source_root))
        from models.diver import DIVER
        from utils.mup_utils import apply_mup

        checkpoint = root / "expansion/diver1-eeg.pt"
        # The author's LFS pointer provides an independent expected digest.
        expected = "dbfa48289989475a52719b1bcb868e62a82877120ac8272ca7dab772e407b891"
        assert sha(checkpoint) == expected
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["module"]  # Official release uses the documented DeepSpeed format.
        width = state["token_manager.special_tokens.N_token.param"].shape[-1]
        depth = len(
            {int(k.split(".")[3]) for k in state if k.startswith("encoder.encoder.layers.")}
        )
        model = DIVER(d_model=width, e_layer=depth, mup=True, patch_size=500)
        apply_mup(
            model,
            lambda w, d: DIVER(d_model=w, e_layer=d, mup=True, patch_size=500),
            "EEGLens_DIVER_EEG",
            width,
            depth,
            str(root / "expansion/diver-mup-shapes"),
        )
        model.load_state_dict(state, strict=True)
        channels = ("C3", "C4", "F3")
        positions = torch.tensor([[-0.06, 0.0, 0.06], [0.06, 0.0, 0.06], [-0.04, 0.05, 0.07]])
        adapter = DIVERAdapter(model, channels=channels, positions=positions)
        shape, fs = (3, 5, 500), 500
        source = Path(inspect.getfile(DIVER))
        metadata = dict(
            strict_full_load=True,
            width=width,
            depth=depth,
            mup=True,
            official_lfs_digest_matched=True,
            geometry="synthetic xyz, EEG only",
        )
    model.eval().requires_grad_(False)
    return model, adapter, shape, fs, checkpoint, source, metadata


def run(root, name):
    torch.set_num_threads(2)
    torch.manual_seed(9417)
    model, adapter, shape, fs, checkpoint, source, metadata = build(root, name)
    lens = EEGLens(model, adapter)
    records = []

    def execute(function, *args, **kwargs):
        if name != "diver":
            return function(*args, **kwargs)
        # Native DIVER SDPA passes a nonzero dropout probability even in eval.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9417)
            return function(*args, **kwargs)

    for b in (1, 2):
        batch = SignalBatch(
            torch.randn(b, *shape),
            tuple(f"t{i}" for i in range(b)),
            adapter.channels,
            fs,
            f"synthetic-{name}-v1",
        )
        if name == "steegformer":
            indices = torch.tensor(adapter.channel_indices).expand(b, -1)

            def native_call():
                return model.forward_features(batch.data.flatten(2), indices)
        else:
            info = [
                {"xyz_id": adapter.positions.clone(), "modality": "EEG", "coord_subtype": None}
                for _ in range(b)
            ]

            def native_call():
                return model(batch.data, data_info_list=info, use_mask=False)["y"]

        with torch.no_grad():
            native = execute(native_call)
        clean = execute(lens.run_with_cache, batch)
        torch.testing.assert_close(clean.output, native, rtol=0, atol=0)
        for site in lens.sites():
            identity = execute(
                lens.run_with_interventions,
                batch,
                interventions=[Replacement(site.name, clean.cache[site.name])],
            )
            torch.testing.assert_close(identity.output, native, rtol=0, atol=0, msg=site.name)
            edited = execute(
                lens.run_with_interventions, batch, interventions=[Ablation(site.name)]
            )
            handle = model.get_submodule(site.module_path).register_forward_hook(
                lambda m, a, o: torch.zeros_like(o)
            )
            try:
                with torch.no_grad():
                    manual = execute(native_call)
            finally:
                handle.remove()
            torch.testing.assert_close(edited.output, manual, rtol=0, atol=0, msg=site.name)
            assert torch.isfinite(edited.output).all()
            semantic = {}
            if name == "steegformer":
                semantic = check_features(lens, batch, site, clean, native_call, native_axis=-1)
                semantic.update(check_rejected_selectors(lens, batch, site))
            else:
                current = clean.cache[site.name].tensor
                features = current.shape[-1]
                basis = torch.eye(features)[:, :2]
                for values in ((0.0, 0.0), (0.25, -0.5)):
                    center = torch.zeros(features)
                    center[:2] = torch.tensor(values)
                    erased = execute(
                        lens.run_with_interventions,
                        batch,
                        interventions=[SubspaceAblation(site.name, basis, center)],
                        sites=(site.name,),
                    )

                    def native_features(module, args, output):
                        result = output.clone()
                        result[..., :2] -= result[..., :2] - center[:2]
                        return result

                    handle = model.get_submodule(site.module_path).register_forward_hook(
                        native_features
                    )
                    try:
                        with torch.no_grad():
                            expected = execute(native_call)
                    finally:
                        handle.remove()
                    torch.testing.assert_close(erased.output, expected, rtol=0, atol=0)
                    observed = erased.cache[site.name].tensor
                    torch.testing.assert_close(observed[..., 2:], current[..., 2:], rtol=0, atol=0)
                    torch.testing.assert_close(
                        observed[..., :2],
                        center[:2].expand_as(observed[..., :2]),
                        rtol=0,
                        atol=1e-6,
                    )
                semantic = dict(
                    native_feature_axis=-1,
                    partial_feature_erasure=True,
                    centered_feature_erasure=True,
                    untouched_features_exact=True,
                )
                if site.module_path in {"embedding", "head"}:
                    selected = execute(
                        lens.run_with_interventions,
                        batch,
                        interventions=[
                            Ablation(site.name, Selection(sensors=("C4",), patches=(1,)))
                        ],
                    )

                    def native_selection(module, args, output):
                        result = output.clone()
                        result[:, 1, 1, :] = 0
                        return result

                    handle = model.get_submodule(site.module_path).register_forward_hook(
                        native_selection
                    )
                    try:
                        with torch.no_grad():
                            expected = execute(native_call)
                    finally:
                        handle.remove()
                    torch.testing.assert_close(selected.output, expected, rtol=0, atol=0)
                    semantic["physical_sensor_patch_verified"] = True
                else:
                    semantic.update(check_rejected_selectors(lens, batch, site))
            records.append(
                dict(
                    batch=b,
                    site=site.name,
                    cache_shape=list(clean.cache[site.name].tensor.shape),
                    delta=float((edited.output - native).norm()),
                    **semantic,
                )
            )
        order = list(reversed(range(b)))
        recipient = replace(
            batch, data=batch.data[order] + 0.3, trial_ids=tuple(batch.trial_ids[i] for i in order)
        )
        final = "norm.output" if name == "steegformer" else "features.output"
        restored = execute(
            lens.run_with_interventions,
            recipient,
            interventions=[Replacement(final, clean.cache[final])],
        )
        torch.testing.assert_close(restored.output, native[order], rtol=0, atol=0)
        if name == "diver":
            edited = execute(
                lens.run_with_interventions,
                batch,
                interventions=[Ablation(final, Selection(sensors=("C4",), patches=(1,)))],
            )
            expected = native.clone()
            expected[:, 1, 1] = 0
            torch.testing.assert_close(edited.output, expected, rtol=0, atol=0)

            def abort(m, a, o):
                raise RuntimeError("intentional validation abort")

            handle = model.encoder.register_forward_hook(abort)
            try:
                try:
                    execute(lens.run_with_cache, batch)
                except RuntimeError as exc:
                    assert str(exc) == "intentional validation abort"
                else:
                    raise AssertionError("failure injection did not fire")
            finally:
                handle.remove()
            assert not model.token_manager.RAN_PREPEND
            torch.testing.assert_close(
                execute(lens.run_with_cache, batch).output, native, rtol=0, atol=0
            )
        assert all(not m._forward_hooks for m in model.modules())
    result = dict(
        model=name,
        records=records,
        reordered_donor_recovery=True,
        paired_rng_seed=9417 if name == "diver" else None,
        native_eval_dropout=name == "diver",
        checkpoint_sha256=sha(checkpoint),
        native_source_sha256=sha(source),
        runner_sha256=sha(__file__),
        semantic_helper_sha256=sha(inspect.getfile(check_features)),
        runtime_sha256=sha(inspect.getfile(EEGLens)),
        torch_version=torch.__version__,
        adapter_sha256=sha(inspect.getfile(type(adapter))),
        upstream_revision=subprocess.check_output(
            [
                "git",
                "-C",
                str(root / "repos" / ("DIVER-1" if name == "diver" else "STEEGFormer")),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        load=metadata,
        scope="CPU synthetic input native encoder checks; not task performance or biological validation",
    )
    directory = root / "semantic-v1"
    directory.mkdir(exist_ok=True)
    out = directory / f"{name}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(name, len(records), "native conditions and reordered donor recovery passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research/eeglens_model_validation"))
    parser.add_argument("--model", choices=["diver", "steegformer"], required=True)
    args = parser.parse_args()
    run(args.root, args.model)
