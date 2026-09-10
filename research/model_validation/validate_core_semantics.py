"""Official CBraMod/LaBraM partial-feature and physical-coordinate native checks."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch

from eeglens import Ablation, Selection, SignalBatch, SubspaceAblation, load_cbramod, load_labram
from eeglens._vendor.labram_channels import CHANNELS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(name, checkpoint, output):
    torch.set_num_threads(2)
    torch.manual_seed(5623)
    lens = (load_cbramod if name == "cbramod" else load_labram)(checkpoint)
    model = lens.model
    records = []
    for size in (1, 2):
        batch = SignalBatch(
            torch.randn(size, 3, 2, 200),
            tuple(f"t{i}" for i in range(size)),
            ("C3", "CZ", "C4"),
            200,
            "synthetic-core-semantics-v1",
        )

        def native():
            if name == "cbramod":
                return model(batch.data)
            return model.forward_features(
                batch.data,
                input_chans=[0, *[CHANNELS.index(c) + 1 for c in batch.channels]],
                return_patch_tokens=True,
                return_all_tokens=False,
            )

        clean = lens.run_with_cache(batch)
        with torch.no_grad():
            torch.testing.assert_close(clean.output, native(), rtol=0, atol=0)
        for site in lens.sites():
            current = clean.cache[site.name].tensor
            features = current.shape[-1]
            for mode in ("zero_features", "centered_features", "physical"):
                center = torch.zeros(features)
                if mode == "centered_features":
                    center[:2] = torch.tensor([0.25, -0.5])
                edit = (
                    Ablation(site.name, Selection(sensors=("C4",), patches=(1,)))
                    if mode == "physical"
                    else SubspaceAblation(site.name, torch.eye(features)[:, :2], center)
                )
                result = lens.run_with_interventions(
                    batch, interventions=[edit], sites=(site.name,)
                )

                def hook(module, args, original):
                    # CBraMod native attention returns (tensor, attention_weights).
                    value = (original[0] if isinstance(original, tuple) else original).clone()
                    if mode != "physical":
                        value[..., :2] -= value[..., :2] - center[:2]
                    elif name == "labram":
                        # Channel-major: third channel, second patch; CLS only after embedding.
                        value[:, 5 + int(site.module_path != "patch_embed"), :] = 0
                    elif site.module_path.endswith("self_attn_s"):
                        # Native (B*P,C,D): patch 1, channel 2 in every trial.
                        value[torch.arange(size) * 2 + 1, 2, :] = 0
                    elif site.module_path.endswith("self_attn_t"):
                        # Native (B*C,P,D): channel 2, patch 1 in every trial.
                        value[torch.arange(size) * 3 + 2, 1, :] = 0
                    else:
                        value[:, 2, 1, :] = 0
                    return (value, *original[1:]) if isinstance(original, tuple) else value

                handle = model.get_submodule(site.module_path).register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        expected = native()
                finally:
                    handle.remove()
                torch.testing.assert_close(result.output, expected, rtol=0, atol=0, msg=site.name)
                observed = result.cache[site.name].tensor
                if mode != "physical":
                    torch.testing.assert_close(observed[..., 2:], current[..., 2:], rtol=0, atol=0)
                    torch.testing.assert_close(
                        observed[..., :2],
                        center[:2].expand_as(observed[..., :2]),
                        rtol=0,
                        atol=1e-6,
                    )
                else:
                    expected_cache = current.clone()
                    if name == "labram":
                        expected_cache[:, 5 + int(site.module_path != "patch_embed"), :] = 0
                    else:
                        expected_cache[:, 2, 1, :] = 0
                    torch.testing.assert_close(observed, expected_cache, rtol=0, atol=0)
                assert all(not module._forward_hooks for module in model.modules())
            records.append(
                dict(
                    batch=size,
                    site=site.name,
                    cache_shape=list(current.shape),
                    partial_feature_erasure=True,
                    centered_feature_erasure=True,
                    untouched_features_exact=True,
                    physical_sensor_patch_verified=True,
                )
            )
        torch.testing.assert_close(lens.run_with_cache(batch).output, clean.output, rtol=0, atol=0)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(
        model=name,
        records=records,
        checkpoint_sha256=sha(checkpoint),
        runner_sha256=sha(__file__),
        adapter_sha256=sha(inspect.getfile(type(lens.adapter))),
        native_source_sha256=sha(inspect.getfile(type(model))),
        torch_version=torch.__version__,
        scope="Local native checkpoint CPU float32 synthetic checks; not physiological validation.",
    )
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, len(records), "semantic conditions passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cbramod", "labram"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.checkpoint, args.output)
