"""Dense feature interventions for EEGPT, BIOT and BENDR encoder checkpoints."""

import argparse
import inspect
import json
from pathlib import Path

import torch
from semantic_checks import check_rejected_selectors
from validate import build, sha

from eeglens import EEGLens, SignalBatch, SubspaceAblation


def run(root, name, output):
    torch.set_num_threads(2)
    torch.manual_seed(8139)
    model, adapter, shape, channels, fs, source, checkpoint = build(root, name)
    lens = EEGLens(model, adapter)
    records = []
    for size in (1, 2):
        batch = SignalBatch(
            torch.randn(size, *shape[1:]),
            tuple(f"d{i}" for i in range(size)),
            channels,
            fs,
            "synthetic-dense-encoder-v1",
        )

        def native():
            if name == "eegpt":
                return model(
                    batch.data.flatten(2),
                    chan_ids=model.prepare_chan_ids(channels).to(batch.data.device),
                )
            return model(batch.data.flatten(2))

        clean = lens.run_with_cache(batch)
        with torch.no_grad():
            torch.testing.assert_close(clean.output, native(), rtol=0, atol=0)
        for site in lens.sites():
            current = clean.cache[site.name].tensor
            features = current.shape[-1]
            q = torch.linalg.qr(torch.randn(features, 3)).Q
            center = torch.linspace(-0.3, 0.4, features)
            axis = 1 if name == "bendr" else -1
            result = lens.run_with_interventions(
                batch, interventions=[SubspaceAblation(site.name, q, center)], sites=(site.name,)
            )

            def hook(module, args, raw):
                assert raw.shape[axis] == features
                value = raw.movedim(axis, -1)
                value = value - ((value - center) @ q) @ q.T
                return value.movedim(-1, axis)

            h = model.get_submodule(site.module_path).register_forward_hook(hook)
            try:
                with torch.no_grad():
                    expected = native()
            finally:
                h.remove()
            torch.testing.assert_close(result.output, expected, rtol=0, atol=0, msg=site.name)
            observed = result.cache[site.name].tensor
            residual = ((observed - center) @ q).abs().max().item()
            # Native parity is exact; this separate residual accounts for float32 projection error.
            scale = max(1.0, (current - center).abs().max().item())
            assert residual <= 1e-5 * scale, (site.name, residual, scale)
            rejected = check_rejected_selectors(lens, batch, site)
            assert all(not m._forward_hooks for m in model.modules())
            records.append(
                dict(
                    batch=size,
                    site=site.name,
                    rank=3,
                    native_feature_axis=axis,
                    native_output_exact=True,
                    relative_projection_residual=residual / scale,
                    **rejected,
                )
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(
        json.dumps(
            dict(
                model=name,
                records=records,
                runner_sha256=sha(Path(__file__)),
                builder_sha256=sha(Path(inspect.getfile(build))),
                rejection_helper_sha256=sha(Path(inspect.getfile(check_rejected_selectors))),
                checkpoint_sha256=sha(checkpoint),
                native_source_sha256=sha(source),
                adapter_sha256=sha(Path(inspect.getfile(type(adapter)))),
                torch_version=torch.__version__,
                scope="Synthetic fixed geometry, CPU float32; all-site dense rank-three centered erasure with exact independent native output parity; physical selectors explicitly rejected.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(name, len(records), "dense conditions passed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("research/eeglens_model_validation"))
    p.add_argument("--model", choices=("eegpt", "biot", "biot-prest", "bendr"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.model, a.output)
