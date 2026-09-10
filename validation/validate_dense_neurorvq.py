"""Dense NeuroRVQ interventions with independent native invocation isolation."""

import inspect
import json
import sys
from functools import partial
from pathlib import Path

import torch
from validate_diver_steegformer import sha

from eeglens import EEGLens, NeuroRVQAdapter, Selection, SignalBatch, SubspaceAblation


def run():
    root = Path("research/eeglens_model_validation")
    sys.path.insert(0, str(root / "repos/NeuroRVQ"))
    from inference.modules.NeuroRVQ_EEG_tokenizer_inference_modules import (
        ch_names_global,
        create_embedding_ix,
    )
    from NeuroRVQ_EEG.NeuroRVQ import NeuroRVQFM

    torch.set_num_threads(2)
    torch.manual_seed(2941)
    checkpoint = root / "expansion/NeuroRVQ_EEG_FM.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = NeuroRVQFM(
        num_classes=0,
        qkv_bias=True,
        qk_norm=partial(torch.nn.LayerNorm, eps=1e-6),
        init_values=1e-5,
        n_global_electrodes=len(ch_names_global),
    )
    for i in range(1, 5):
        setattr(model, f"fc_norm_{i}", torch.nn.Identity())
    extra = set(state) - set(model.state_dict())
    assert extra == {"mask_token", "norm_pre.weight", "norm_pre.bias"} | {
        f"head_pre_{i}.{p}" for i in range(1, 33) for p in ("weight", "bias")
    }
    model.load_state_dict({k: v for k, v in state.items() if k not in extra}, strict=True)
    model.eval().requires_grad_(False)
    lens = EEGLens(
        model, NeuroRVQAdapter(model, channel_vocabulary=[c.decode() for c in ch_names_global])
    )
    records = []
    for size in (1, 2):
        batch = SignalBatch(
            torch.randn(size, 2, 2, 200),
            tuple(f"t{i}" for i in range(size)),
            ("C3", "C4"),
            200,
            "dense-neurorvq-v1",
        )
        ti, si = create_embedding_ix(2, 256, [b"c3", b"c4"], ch_names_global)

        def native():
            return torch.stack(
                model(
                    batch.data, ti.expand(size, -1), si.expand(size, -1), return_patch_tokens=True
                )[:4],
                dim=2,
            )

        clean = lens.run_with_cache(batch)
        with torch.no_grad():
            torch.testing.assert_close(clean.output, native(), rtol=0, atol=0)
        for site in lens.sites():
            current = clean.cache[site.name].tensor
            q = torch.linalg.qr(torch.randn(current.shape[-1], 3)).Q
            center = torch.linspace(-0.3, 0.4, current.shape[-1])
            for local in (False, True):
                selection = Selection(sensors=("C4",), patches=(1,)) if local else Selection()
                result = lens.run_with_interventions(
                    batch,
                    interventions=[SubspaceAblation(site.name, q, center, selection)],
                    sites=(site.name,),
                )
                calls = [0]

                def hook(module, args, raw):
                    index = calls[0]
                    calls[0] += 1
                    if index != site.call_index:
                        return None
                    projected = raw - ((raw - center) @ q) @ q.T
                    if local:
                        edited = raw.clone()
                        # CLS + channel-major: second sensor, second patch.
                        edited[:, 4] = projected[:, 4]
                        return edited
                    return projected

                h = model.get_submodule(site.module_path).register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        expected = native()
                finally:
                    h.remove()
                assert calls[0] == 4
                torch.testing.assert_close(result.output, expected, rtol=0, atol=0, msg=site.name)
                others = [i for i in range(4) if i != site.call_index]
                torch.testing.assert_close(
                    result.output[:, :, others], clean.output[:, :, others], rtol=0, atol=0
                )
                if local:
                    torch.testing.assert_close(
                        result.cache[site.name].tensor[:, :4], current[:, :4], rtol=0, atol=0
                    )
                assert all(not m._forward_hooks for m in model.modules())
                records.append(
                    dict(
                        batch=size,
                        site=site.name,
                        call_index=site.call_index,
                        local=local,
                        rank=3,
                        native_output_exact=True,
                        other_branches_exact=True,
                    )
                )
        print("neurorvq", size, "dense batch passed", flush=True)
    output = Path("eeglens/validation/results/dense-neurorvq-v1")
    output.mkdir(parents=True, exist_ok=True)
    (output / "neurorvq.json").write_text(
        json.dumps(
            dict(
                model="neurorvq",
                records=records,
                runner_sha256=sha(__file__),
                checkpoint_sha256=sha(checkpoint),
                native_source_sha256=sha(inspect.getfile(NeuroRVQFM)),
                adapter_sha256=sha(inspect.getfile(NeuroRVQAdapter)),
                torch_version=torch.__version__,
                scope="CPU float32 synthetic two-sensor/two-patch geometry; dense centered rank-three global and local erasure at every native branch invocation; other branches exact; no freshly initialized downstream norms.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run()
