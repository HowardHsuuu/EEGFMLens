"""Native NeuroRVQ extent, channel alias and rejection/recovery checks."""

import inspect
import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
from validate_diver_steegformer import sha

from eeglens import EEGLens, NeuroRVQAdapter, SignalBatch
from eeglens.errors import ValidationError


def run():
    root = Path("research/eeglens_model_validation")
    sys.path.insert(0, str(root / "repos/NeuroRVQ"))
    from inference.modules.NeuroRVQ_EEG_tokenizer_inference_modules import (
        ch_names_global,
        create_embedding_ix,
    )
    from NeuroRVQ_EEG.NeuroRVQ import NeuroRVQFM

    torch.set_num_threads(2)
    torch.manual_seed(2764)
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
    adapter = NeuroRVQAdapter(model, channel_vocabulary=[c.decode() for c in ch_names_global])
    lens = EEGLens(model, adapter)
    records = []
    for patches in (1, 8):
        batch = SignalBatch(
            torch.randn(1, 2, patches, 200), ("trial",), ("C3", "C4"), 200, "neurorvq-boundary-v1"
        )
        ti, si = create_embedding_ix(patches, adapter.max_patches, [b"c3", b"c4"], ch_names_global)
        clean = lens.run_with_cache(batch)
        with torch.no_grad():
            native = torch.stack(model(batch.data, ti, si, return_patch_tokens=True)[:4], dim=2)
        torch.testing.assert_close(clean.output, native, rtol=0, atol=0)
        aliases = lens.run_with_cache(replace(batch, channels=("c3", "c4")))
        torch.testing.assert_close(aliases.output, clean.output, rtol=0, atol=0)
        invalid = [
            replace(batch, sampling_rate=201),
            replace(batch, patch_stride_samples=100),
            replace(batch, data=batch.data[..., :-1]),
            replace(batch, channels=("C3", "UNKNOWN")),
            replace(batch, channels=("C3", "c3")),
            replace(batch, data=torch.randn(1, 2, adapter.max_patches + 1, 200)),
        ]
        calls = []
        h = model.register_forward_pre_hook(lambda *args: calls.append(1))
        try:
            for bad in invalid:
                count = len(calls)
                try:
                    lens.run_with_cache(bad)
                except ValidationError:
                    pass
                else:
                    raise AssertionError("Invalid NeuroRVQ input accepted")
                assert len(calls) == count
                assert all(not m._forward_hooks for m in model.modules())
            recovered = lens.run_with_cache(batch)
            torch.testing.assert_close(recovered.output, clean.output, rtol=0, atol=0)
            assert len(calls) == 1
        finally:
            h.remove()
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        records.append(
            dict(
                patches=patches,
                native_output_exact=True,
                valid_case_aliases_exact=True,
                invalid_inputs_rejected_before_native=len(invalid),
                four_branch_recovery_exact=True,
            )
        )
    output = Path("eeglens/research/model_validation/results/neurorvq-boundaries-v1.json")
    output.write_text(
        json.dumps(
            dict(
                model="neurorvq",
                records=records,
                runner_sha256=sha(__file__),
                adapter_sha256=sha(inspect.getfile(NeuroRVQAdapter)),
                checkpoint_sha256=sha(checkpoint),
                native_source_sha256=sha(inspect.getfile(NeuroRVQFM)),
                scope="CPU float32 two-sensor synthetic input, one/eight patches; temporal over-capacity rejected without executing maximum-length input. Raw pretrained branches, no new downstream norms.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("NeuroRVQ", records, flush=True)


if __name__ == "__main__":
    run()
