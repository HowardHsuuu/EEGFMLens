"""Validate pretrained NeuroRVQ backbone without newly initialized downstream norms."""

import hashlib
import inspect
import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch

from eeglens import (
    Ablation,
    EEGLens,
    NeuroRVQAdapter,
    Replacement,
    Selection,
    SignalBatch,
    SubspaceAblation,
)


def run():
    root = Path("research/eeglens_model_validation")
    sys.path.insert(0, str(root / "repos/NeuroRVQ"))
    from inference.modules.NeuroRVQ_EEG_tokenizer_inference_modules import (
        ch_names_global,
        create_embedding_ix,
    )
    from NeuroRVQ_EEG.NeuroRVQ import NeuroRVQFM

    torch.set_num_threads(2)
    torch.manual_seed(4311)
    checkpoint = root / "expansion/NeuroRVQ_EEG_FM.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = NeuroRVQFM(
        num_classes=0,
        qkv_bias=True,
        qk_norm=partial(torch.nn.LayerNorm, eps=1e-6),
        init_values=1e-5,
        n_global_electrodes=len(ch_names_global),
    )
    # Upstream inference declares fresh downstream norms absent from pretraining.
    # Return raw branch features instead: no untrained affine parameters are used.
    for i in range(1, 5):
        setattr(model, f"fc_norm_{i}", torch.nn.Identity())
    extra = set(state) - set(model.state_dict())
    assert extra == {"mask_token", "norm_pre.weight", "norm_pre.bias"} | {
        f"head_pre_{i}.{p}" for i in range(1, 33) for p in ["weight", "bias"]
    }
    selected = {k: v for k, v in state.items() if k not in extra}
    model.load_state_dict(selected, strict=True)
    model.eval().requires_grad_(False)
    adapter = NeuroRVQAdapter(model, channel_vocabulary=[c.decode() for c in ch_names_global])
    lens = EEGLens(model, adapter)
    records = []
    for b in [1, 2]:
        batch = SignalBatch(
            torch.randn(b, 2, 2, 200),
            tuple(f"t{i}" for i in range(b)),
            ("C3", "C4"),
            200,
            "synthetic-neurorvq-v1",
        )
        t, s = create_embedding_ix(2, 256, [b"c3", b"c4"], ch_names_global)
        ti, si = adapter.indices(batch)
        torch.testing.assert_close(ti, t.expand(b, -1))
        torch.testing.assert_close(si, s.expand(b, -1))
        with torch.no_grad():
            native = torch.stack(
                model(batch.data, t.expand(b, -1), s.expand(b, -1), return_patch_tokens=True)[:4],
                dim=2,
            )
        clean = lens.run_with_cache(batch)
        torch.testing.assert_close(clean.output, native, rtol=0, atol=0)
        for site in lens.sites():
            restored = lens.run_with_interventions(
                batch, interventions=[Replacement(site.name, clean.cache[site.name])]
            )
            torch.testing.assert_close(restored.output, native, rtol=0, atol=0)
            edited = lens.run_with_interventions(batch, interventions=[Ablation(site.name)])
            counter = [0]

            def hook(m, a, o):
                i = counter[0]
                counter[0] += 1
                return torch.zeros_like(o) if i == site.call_index else None

            h = model.get_submodule(site.module_path).register_forward_hook(hook)
            try:
                with torch.no_grad():
                    manual = adapter.forward(model, batch)
            finally:
                h.remove()
            torch.testing.assert_close(edited.output, manual, rtol=0, atol=0)
            others = [i for i in range(4) if i != site.call_index]
            torch.testing.assert_close(
                edited.output[:, :, others], native[:, :, others], rtol=0, atol=0
            )
            assert counter[0] == 4 and torch.isfinite(edited.output).all()
            features = clean.cache[site.name].tensor.shape[-1]
            basis = torch.eye(features)[:, :2]
            for mode in ("zero_features", "centered_features", "physical"):
                center = torch.zeros(features)
                if mode == "centered_features":
                    center[:2] = torch.tensor([0.25, -0.5])
                intervention = (
                    Ablation(site.name, Selection(sensors=("C4",), patches=(1,)))
                    if mode == "physical"
                    else SubspaceAblation(site.name, basis, center)
                )
                result = lens.run_with_interventions(
                    batch, interventions=[intervention], sites=(site.name,)
                )
                calls = [0]

                def local_edit(module, args, output):
                    index = calls[0]
                    calls[0] += 1
                    if index != site.call_index:
                        return None
                    value = output.clone()
                    if mode == "physical":
                        # Native CLS + channel-major sequence: C4/patch1 is 1+2+1.
                        value[:, 4, :] = 0
                    else:
                        value[..., :2] -= value[..., :2] - center[:2]
                    return value

                handle = model.get_submodule(site.module_path).register_forward_hook(local_edit)
                try:
                    with torch.no_grad():
                        expected = torch.stack(
                            model(
                                batch.data,
                                t.expand(b, -1),
                                s.expand(b, -1),
                                return_patch_tokens=True,
                            )[:4],
                            dim=2,
                        )
                finally:
                    handle.remove()
                assert calls[0] == 4
                torch.testing.assert_close(result.output, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    result.output[:, :, others], native[:, :, others], rtol=0, atol=0
                )
                observed = result.cache[site.name].tensor
                original = clean.cache[site.name].tensor
                if mode == "physical":
                    torch.testing.assert_close(observed[:, :4], original[:, :4], rtol=0, atol=0)
                    assert observed[:, 4].count_nonzero() == 0
                else:
                    torch.testing.assert_close(observed[..., 2:], original[..., 2:], rtol=0, atol=0)
                    torch.testing.assert_close(
                        observed[..., :2],
                        center[:2].expand_as(observed[..., :2]),
                        rtol=0,
                        atol=1e-6,
                    )
            records.append(
                dict(
                    batch=b,
                    site=site.name,
                    call_index=site.call_index,
                    delta=float((edited.output - native).norm()),
                    partial_feature_erasure=True,
                    centered_feature_erasure=True,
                    untouched_features_exact=True,
                    physical_sensor_patch_verified=True,
                    other_branches_exact=True,
                )
            )
        order = list(reversed(range(b)))
        corrupted = replace(
            batch,
            data=batch.data[order] + 0.3,
            trial_ids=tuple(batch.trial_ids[i] for i in order),
        )
        recovered = lens.run_with_interventions(
            corrupted,
            interventions=[
                Replacement(f"branches.{i}.output", clean.cache[f"branches.{i}.output"])
                for i in range(4)
            ],
        )
        torch.testing.assert_close(recovered.output, native[order], rtol=0, atol=0)
        assert all(not m._forward_hooks for m in model.modules())
    out = root / "semantic-v1"
    out.mkdir(exist_ok=True)
    (out / "neurorvq.json").write_text(
        json.dumps(
            dict(
                records=records,
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                adapter_sha256=hashlib.sha256(
                    Path(inspect.getfile(NeuroRVQAdapter)).read_bytes()
                ).hexdigest(),
                native_source_sha256=hashlib.sha256(
                    Path(inspect.getfile(type(model))).read_bytes()
                ).hexdigest(),
                torch_version=torch.__version__,
                strict_backbone_load=True,
                reordered_donor_recovery=True,
                excluded_pretraining_keys=sorted(extra),
                downstream_norms="replaced by parameter-free identity; raw branch features",
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    print("NeuroRVQ", len(records), "conditions passed with branch-isolated effects")


if __name__ == "__main__":
    run()
