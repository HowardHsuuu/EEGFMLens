"""Independent dense-subspace native hooks on the three MI study models and real EEG."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from checkpoint_fixtures import CHANNELS, RECIPE, build, model_input, sha

from eeglens import Selection, SignalBatch, SubspaceAblation
from eeglens.adapters.labram_channels import CHANNELS as LABRAM_CHANNELS


def run(name, root, output):
    torch.set_num_threads(2)
    torch.manual_seed(1742)
    prepared = root / "eeglens_mi/development-v1"
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["artifact_sha256"] == sha(prepared / "trials.npz")
    arrays = np.load(prepared / "trials.npz", allow_pickle=False)
    values = model_input(arrays["volts"][:2], list(arrays["channels"]))
    lens, checkpoint = build(name, root)
    # Native CSBrain region permutation, independently specified from upstream setup.
    order = [0, 10, 2, 16, 3, 11, 1, 6, 18, 7, 12, 14, 15, 13, 8, 9, 4, 17, 5]
    records = []
    for size in (1, 2):
        batch = SignalBatch(values[:size], tuple(arrays["trial_ids"][:size]), CHANNELS, 200, RECIPE)

        def native():
            if name == "labram":
                return lens.model.forward_features(
                    batch.data,
                    input_chans=[0, *[LABRAM_CHANNELS.index(c) + 1 for c in CHANNELS]],
                    return_patch_tokens=True,
                    return_all_tokens=False,
                )
            return lens.model(batch.data)

        clean = lens.run_with_cache(batch)
        for site in lens.sites():
            current = clean.cache[site.name].tensor
            d = current.shape[-1]
            q = torch.linalg.qr(torch.randn(d, 3)).Q
            center = torch.linspace(-0.3, 0.4, d)
            for local in (False, True):
                selection = Selection(sensors=("C4",), patches=(1,)) if local else Selection()
                result = lens.run_with_interventions(
                    batch,
                    interventions=[SubspaceAblation(site.name, q, center, selection)],
                    sites=(site.name,),
                )

                def hook(module, args, original):
                    raw = original[0] if isinstance(original, tuple) else original
                    edited = raw - ((raw - center) @ q) @ q.T
                    if local:
                        mask = torch.zeros_like(raw, dtype=torch.bool)
                        if name == "labram":
                            mask[
                                :,
                                CHANNELS.index("C4") * 4
                                + 1
                                + int(site.module_path != "patch_embed"),
                            ] = True
                        elif name == "csbrain":
                            mask[:, order.index(CHANNELS.index("C4")), 1] = True
                        elif site.module_path.endswith("self_attn_s"):
                            mask[torch.arange(size) * 4 + 1, CHANNELS.index("C4")] = True
                        elif site.module_path.endswith("self_attn_t"):
                            mask[torch.arange(size) * 19 + CHANNELS.index("C4"), 1] = True
                        else:
                            mask[:, CHANNELS.index("C4"), 1] = True
                        edited = torch.where(mask, edited, raw)
                    return (edited, *original[1:]) if isinstance(original, tuple) else edited

                h = lens.model.get_submodule(site.module_path).register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        expected = native()
                finally:
                    h.remove()
                torch.testing.assert_close(result.output, expected, rtol=0, atol=0, msg=site.name)
                observed = result.cache[site.name].tensor
                if local:
                    mask = torch.zeros_like(current, dtype=torch.bool)
                    if name == "labram":
                        mask[
                            :, CHANNELS.index("C4") * 4 + 1 + int(site.module_path != "patch_embed")
                        ] = True
                    else:
                        mask[:, CHANNELS.index("C4"), 1] = True
                    torch.testing.assert_close(observed[~mask], current[~mask], rtol=0, atol=0)
                assert all(not m._forward_hooks for m in lens.model.modules())
                records.append(
                    dict(
                        batch=size,
                        site=site.name,
                        local=local,
                        rank=3,
                        native_output_exact=True,
                        untouched_locations_exact=local,
                    )
                )
        print(name, size, "dense real-EEG checks passed", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(
        json.dumps(
            dict(
                model=name,
                records=records,
                runner_sha256=sha(__file__),
                builder_sha256=sha(inspect.getfile(build)),
                checkpoint_sha256=sha(checkpoint),
                adapter_sha256=sha(inspect.getfile(type(lens.adapter))),
                native_source_sha256=sha(inspect.getfile(type(lens.model))),
                prepared_manifest_sha256=sha(prepared / "manifest.json"),
                torch_version=torch.__version__,
                scope="CPU float32; first two development EEG trials, 19 channels, 4 patches; dense rank-three global and C4/patch-1 interventions, all exposed sites.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("cbramod", "labram", "csbrain"), required=True)
    p.add_argument("--root", type=Path, default=Path("research"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.model, a.root, a.output)
