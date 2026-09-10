"""Dense native feature-axis checks for full BENDR and SignalJEPA."""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
from semantic_checks import check_rejected_selectors

from eeglens import BENDRAdapter, EEGLens, SignalBatch, SignalJEPAAdapter, SubspaceAblation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root, name):
    if name == "bendr-context":
        sys.path[:0] = [str(root / "repos/dn3"), str(root / "repos/BENDR")]
        import dn3_ext

        encoder = dn3_ext.ConvEncoderBENDR(in_features=20, encoder_h=512)
        context = dn3_ext.BENDRContextualizer(in_features=512)
        checkpoints = [
            root / "checkpoints/bendr-mirror-encoder.pt",
            root / "expansion/bendr-contextualizer.pt",
        ]
        for model, checkpoint in zip((encoder, context), checkpoints):
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
            )
        model = torch.nn.Sequential(encoder, context).eval().requires_grad_(False)
        adapter = BENDRAdapter(model, channels=tuple(f"input-{i}" for i in range(20)))
        return model, adapter, (20, 4, 256), 256, checkpoints, inspect.getfile(type(context))
    from braindecode.models import SignalJEPA
    from safetensors.torch import load_file

    folder = root / "expansion/braindecode-signal-jepa"
    config = json.loads((folder / "config.json").read_text())
    config.pop("braindecode_version")
    assert config.pop("activation") == "torch.nn.modules.activation.GELU"
    config["activation"] = torch.nn.GELU
    model = SignalJEPA(**config)
    checkpoint = folder / "model.safetensors"
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    model.eval().requires_grad_(False)
    adapter = SignalJEPAAdapter(model)
    return (
        model,
        adapter,
        (len(adapter.channels), 3, 128),
        128,
        [checkpoint],
        inspect.getfile(SignalJEPA),
    )


def run(root, name, output):
    torch.set_num_threads(2)
    torch.manual_seed(6163)
    model, adapter, shape, fs, checkpoints, source = build(root, name)
    lens = EEGLens(model, adapter)
    records = []
    for size in (1, 2):
        batch = SignalBatch(
            torch.randn(size, *shape),
            tuple(f"t{i}" for i in range(size)),
            adapter.channels,
            fs,
            "dense-context-v1",
        )
        clean = lens.run_with_cache(batch)
        with torch.no_grad():
            torch.testing.assert_close(clean.output, model(batch.data.flatten(2)), rtol=0, atol=0)
        for site in lens.sites():
            current = clean.cache[site.name].tensor
            d = current.shape[-1]
            q = torch.linalg.qr(torch.randn(d, 3)).Q
            center = torch.linspace(-0.3, 0.4, d)
            axis = (
                1 if name == "bendr-context" and site.module_path in {"0", "1.output_layer"} else -1
            )
            result = lens.run_with_interventions(
                batch, interventions=[SubspaceAblation(site.name, q, center)], sites=(site.name,)
            )

            def hook(module, args, raw):
                assert raw.shape[axis] == d
                # Sequence-first contextual sites need batch-first GEMM grouping
                # to match the public feature operation's floating-point order.
                sequence_first = name == "bendr-context" and axis == -1
                value = raw.transpose(0, 1) if sequence_first else raw.movedim(axis, -1)
                projected = value - ((value - center) @ q) @ q.T
                projected = (
                    projected.transpose(0, 1) if sequence_first else projected.movedim(-1, axis)
                )
                # Preserve native strides independently of adapter.restore.
                edited = raw.clone()
                edited.copy_(projected)
                return edited

            handle = model.get_submodule(site.module_path).register_forward_hook(hook)
            try:
                with torch.no_grad():
                    expected = model(batch.data.flatten(2))
            finally:
                handle.remove()
            torch.testing.assert_close(result.output, expected, rtol=0, atol=0, msg=site.name)
            residual = ((result.cache[site.name].tensor - center) @ q).abs().max().item()
            scale = max(1.0, (current - center).abs().max().item())
            assert residual <= 1e-5 * scale
            rejected = check_rejected_selectors(lens, batch, site)
            records.append(
                dict(
                    batch=size,
                    site=site.name,
                    rank=3,
                    native_axis=axis,
                    native_output_exact=True,
                    relative_projection_residual=residual / scale,
                    **rejected,
                )
            )
        print(name, size, "dense context batch passed", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(
        json.dumps(
            dict(
                model=name,
                records=records,
                runner_sha256=sha(__file__),
                checkpoint_sha256=[sha(p) for p in checkpoints],
                adapter_sha256=sha(inspect.getfile(type(adapter))),
                native_source_sha256=sha(source),
                rejection_helper_sha256=sha(inspect.getfile(check_rejected_selectors)),
                torch_version=torch.__version__,
                scope="CPU float32 synthetic fixed geometry; all-site centered dense rank-three native intervention parity, native strides preserved, physical selectors rejected.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("research/eeglens_model_validation"))
    p.add_argument("--model", choices=("bendr-context", "signaljepa"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.model, a.output)
