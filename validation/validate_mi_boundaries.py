"""Native checkpoint input boundaries for the three MI models, without fitting."""

import inspect
import json
from dataclasses import replace
from pathlib import Path

import torch
from checkpoint_fixtures import CHANNELS, build, sha

from eeglens import SignalBatch
from eeglens.errors import ValidationError


def run(name):
    torch.set_num_threads(2)
    torch.manual_seed(9763)
    lens, checkpoint = build(name, Path("research"))
    lengths = (1, lens.adapter.max_patches) if name == "labram" else (1, 5)
    records = []
    for patches in lengths:
        batch = SignalBatch(
            torch.randn(1, 19, patches, 200), ("boundary",), CHANNELS, 200, "synthetic-boundary-v1"
        )
        calls = []
        h = lens.model.register_forward_pre_hook(lambda *args: calls.append(1))
        # LaBraM adapter invokes forward_features directly, bypassing model.__call__.
        first = lens.model.patch_embed if name == "labram" else lens.model
        if name == "labram":
            h.remove()
            h = first.register_forward_pre_hook(lambda *args: calls.append(1))
        try:
            result = lens.run_with_cache(batch, sites=("blocks.0.output",))
            with torch.no_grad():
                expected = lens.adapter.forward(lens.model, batch)
            torch.testing.assert_close(result.output, expected, rtol=0, atol=0)
            invalid = [
                replace(batch, sampling_rate=201),
                replace(batch, data=batch.data[..., :199]),
            ]
            if name == "csbrain":
                invalid.append(replace(batch, channels=CHANNELS[::-1]))
            if name == "labram":
                invalid.append(replace(batch, channels=("unknown", *CHANNELS[1:])))
                invalid.append(
                    replace(batch, data=torch.randn(1, 19, lens.adapter.max_patches + 1, 200))
                )
            for bad in invalid:
                count = len(calls)
                try:
                    lens.run_with_cache(bad, sites=("blocks.0.output",))
                except ValidationError:
                    pass
                else:
                    raise AssertionError("Invalid model input accepted")
                assert len(calls) == count, "Native input computation started before rejection"
            torch.testing.assert_close(
                lens.run_with_cache(batch, sites=()).output, result.output, rtol=0, atol=0
            )
        finally:
            h.remove()
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in lens.model.modules())
        records.append(
            dict(
                patches=patches,
                native_output_exact=True,
                invalid_inputs_rejected_before_native=len(invalid),
                recovery_exact=True,
            )
        )
    out = Path("eeglens/validation/results/input-boundaries-v1")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(
        json.dumps(
            dict(
                model=name,
                records=records,
                checkpoint_sha256=sha(checkpoint),
                runner_sha256=sha(__file__),
                builder_sha256=sha(inspect.getfile(build)),
                adapter_sha256=sha(inspect.getfile(type(lens.adapter))),
                runtime_sha256=sha(inspect.getfile(type(lens))),
                scope="Synthetic 19-channel CPU float32 native parity and pre-forward invalid-input rejection. LaBraM native maximum time vocabulary; CBraMod/CSBrain sampled lengths are not claimed maxima.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(name, records, flush=True)


if __name__ == "__main__":
    for name in ("cbramod", "labram", "csbrain"):
        run(name)
