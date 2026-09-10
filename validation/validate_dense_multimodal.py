"""Dense native interventions for BrainOmni, DIVER and ST-EEGFormer."""

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
from semantic_checks import check_rejected_selectors
from validate_diver_steegformer import build as build_external
from validate_diver_steegformer import sha

from eeglens import BrainOmniAdapter, EEGLens, Selection, SignalBatch, SubspaceAblation


def run(root, name, output):
    torch.set_num_threads(2)
    torch.manual_seed(5331)
    if name == "brainomni":
        sys.path.insert(0, str(root / "repos/BrainOmni"))
        from brainomni.model import BrainOmni

        config = root / "expansion/OpenTSLab-BrainOmni/tiny/model_cfg.json"
        checkpoint = config.with_name("BrainOmni.pt")
        model = BrainOmni(**json.loads(config.read_text()))
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        model.eval().requires_grad_(False)
        adapter = BrainOmniAdapter(
            model,
            channels=("C3", "C4", "F3", "F4"),
            positions=torch.randn(4, 6),
            sensor_types=torch.zeros(4, dtype=torch.long),
        )
        shape, fs = (4, 3, 512), 256
        source, metadata = inspect.getfile(BrainOmni), dict(config_sha256=sha(config))
    else:
        model, adapter, shape, fs, checkpoint, source, metadata = build_external(root, name)
    lens = EEGLens(model, adapter)

    def execute(fn, *args, **kwargs):
        if name == "steegformer":
            return fn(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9417)
            return fn(*args, **kwargs)

    records = []
    for size in (1, 2):
        batch = SignalBatch(
            torch.randn(size, *shape),
            tuple(f"t{i}" for i in range(size)),
            adapter.channels,
            fs,
            "dense-multimodal-v1",
        )

        def native():
            if name == "brainomni":
                return model.encode(
                    batch.data.flatten(2),
                    adapter.positions.unsqueeze(0).expand(size, -1, -1),
                    adapter.sensor_types.unsqueeze(0).expand(size, -1),
                )
            if name == "diver":
                info = [
                    dict(xyz_id=adapter.positions.clone(), modality="EEG", coord_subtype=None)
                    for _ in range(size)
                ]
                return model(batch.data, data_info_list=info, use_mask=False)["y"]
            return model.forward_features(
                batch.data.flatten(2), torch.tensor(adapter.channel_indices).expand(size, -1)
            )

        clean = execute(lens.run_with_cache, batch)
        with torch.no_grad():
            torch.testing.assert_close(clean.output, execute(native), rtol=0, atol=0)
        for site in lens.sites():
            current = clean.cache[site.name].tensor
            d = current.shape[-1]
            q = torch.linalg.qr(torch.randn(d, 3)).Q
            center = torch.linspace(-0.3, 0.4, d)
            physical = name == "diver" and site.module_path in {"embedding", "head"}
            for local in (False, True) if physical else (False,):
                selection = Selection(sensors=("C4",), patches=(1,)) if local else Selection()
                result = execute(
                    lens.run_with_interventions,
                    batch,
                    interventions=[SubspaceAblation(site.name, q, center, selection)],
                    sites=(site.name,),
                )

                def hook(module, args, raw):
                    assert raw.shape[-1] == d
                    projected = raw - ((raw - center) @ q) @ q.T
                    if local:
                        edited = raw.clone()
                        edited[:, 1, 1] = projected[:, 1, 1]
                        return edited
                    return projected

                h = model.get_submodule(site.module_path).register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        expected = execute(native)
                finally:
                    h.remove()
                torch.testing.assert_close(result.output, expected, rtol=0, atol=0, msg=site.name)
                if local:
                    mask = torch.ones_like(current, dtype=torch.bool)
                    mask[:, 1, 1] = False
                    torch.testing.assert_close(
                        result.cache[site.name].tensor[mask], current[mask], rtol=0, atol=0
                    )
                assert all(not m._forward_hooks for m in model.modules())
                records.append(
                    dict(batch=size, site=site.name, rank=3, local=local, native_output_exact=True)
                )
            if not physical:
                check_rejected_selectors(lens, batch, site)
        print(name, size, "dense batch passed", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(
        json.dumps(
            dict(
                model=name,
                records=records,
                runner_sha256=sha(__file__),
                builder_sha256=sha(inspect.getfile(build_external)),
                checkpoint_sha256=sha(checkpoint),
                adapter_sha256=sha(inspect.getfile(type(adapter))),
                native_source_sha256=sha(source),
                metadata=metadata,
                torch_version=torch.__version__,
                scope="CPU float32 fixed synthetic geometry, centered dense rank-three erasure; paired CPU RNG for BrainOmni/DIVER eval dropout; DIVER physical sites additionally checked, other sites reject physical selectors.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("research/eeglens_model_validation"))
    p.add_argument("--model", choices=("brainomni", "diver", "steegformer"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.model, a.output)
