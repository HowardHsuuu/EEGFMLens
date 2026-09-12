"""Native DIVER time-head/mask validation, independent of scientific datasets."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from validate_diver_steegformer import build

from eeglens import Ablation, DIVERAdapter, EEGLens, Replacement, Selection, SignalBatch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(function, *args, **kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9417)
        return function(*args, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    model, old, shape, fs, checkpoint, source, load = build(args.root, "diver")
    adapter = DIVERAdapter(
        model, channels=old.channels, positions=old.positions, output="reconstruction"
    )
    lens = EEGLens(model, adapter)
    records = []
    with torch.inference_mode():
        for size in (1, 2):
            torch.manual_seed(523)
            batch = SignalBatch(
                torch.randn(size, *shape),
                tuple(f"t{i}" for i in range(size)),
                adapter.channels,
                fs,
                "synthetic-native-time-head-v1",
            )
            mask = torch.zeros(batch.data.shape[:-1], dtype=torch.bool)
            mask[:, 0, 1] = True
            kwargs = {"mask": mask.tolist()}
            clean = paired(lens.run_with_cache, batch, **kwargs)

            def native():
                model.mask_generator(*mask.shape, device=batch.data.device, x=batch.data)
                x = batch.data.clone()
                x[mask] = model.mask_encoding
                return model(x, data_info_list=adapter.data_info(batch), use_mask=False)["y_org"][
                    "time_head_output"
                ]

            expected = paired(native)
            torch.testing.assert_close(clean.output, expected, rtol=0, atol=0)
            for site in lens.sites():
                own = paired(
                    lens.run_with_interventions,
                    batch,
                    interventions=[Replacement(site.name, clean.cache[site.name])],
                    **kwargs,
                )
                torch.testing.assert_close(own.output, expected, rtol=0, atol=0)
                edited = paired(
                    lens.run_with_interventions,
                    batch,
                    interventions=[Ablation(site.name)],
                    **kwargs,
                )
                h = model.get_submodule(site.module_path).register_forward_hook(
                    lambda m, a, o: torch.zeros_like(o)
                )
                try:
                    oracle = paired(native)
                finally:
                    h.remove()
                torch.testing.assert_close(edited.output, oracle, rtol=0, atol=0)
                records.append(
                    {
                        "batch": size,
                        "site": site.name,
                        "identity_exact": True,
                        "independent_native_zero_exact": True,
                        "output_delta_l2": float((edited.output - expected).double().norm()),
                    }
                )
            local = paired(
                lens.run_with_interventions,
                batch,
                interventions=[
                    Ablation("reconstruction.output", Selection(sensors=("C4",), patches=(1,)))
                ],
                **kwargs,
            )
            oracle = expected.clone()
            oracle[:, 1, 1] = 0
            assert expected[:, 1, 1].abs().sum() > 0
            torch.testing.assert_close(local.output, oracle, rtol=0, atol=0)
            again = paired(lens.run_with_cache, batch, **kwargs)
            torch.testing.assert_close(again.output, expected, rtol=0, atol=0)
            assert not model.mask_generator._forward_hooks
    report = {
        "scope": "Official checkpoint, synthetic3-sensor5-second CPU float32 inputs; paired RNG; not scientific efficacy",
        "checkpoint_sha256": sha(checkpoint),
        "native_source_sha256": sha(source),
        "runner_sha256": sha(__file__),
        "adapter_sha256": sha(Path(__file__).parents[1] / "src/eeglens/adapters/diver.py"),
        "load": load,
        "torch_version": torch.__version__,
        "records": records,
        "physical_output_selection_exact": True,
        "cleanup_exact": True,
        "output": adapter.metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{len(records)} native site/batch conditions passed", flush=True)


if __name__ == "__main__":
    main()
