"""Official EEGPT checkpoint rejection and recovery for native channel lookup."""

import inspect
import json
from dataclasses import replace
from pathlib import Path

import torch
from validate import build, sha

from eeglens import EEGLens, SignalBatch
from eeglens.errors import ValidationError


def run():
    torch.set_num_threads(2)
    torch.manual_seed(1249)
    model, adapter, shape, channels, fs, source, checkpoint = build(
        Path("research/eeglens_model_validation"), "eegpt"
    )
    lens = EEGLens(model, adapter)
    batch = SignalBatch(torch.randn(1, *shape[1:]), ("trial",), channels, fs, "boundary-eegpt-v1")
    clean = lens.run_with_cache(batch)
    calls = []
    handle = model.register_forward_pre_hook(lambda *args: calls.append(1))
    records = []
    try:
        for label, bad in (
            ("unknown-channel", replace(batch, channels=("UNKNOWN", *channels[1:]))),
            ("colliding-alias", replace(batch, channels=("C3", "c3.", *channels[2:]))),
            ("wrong-rate", replace(batch, sampling_rate=200)),
            ("wrong-window-count", replace(batch, data=batch.data[:, :, :-1])),
        ):
            count = len(calls)
            try:
                lens.run_with_cache(bad)
            except ValidationError as exc:
                message = str(exc)
            else:
                raise AssertionError(f"Accepted {label}")
            assert len(calls) == count
            assert all(not m._forward_hooks for m in model.modules())
            recovered = lens.run_with_cache(batch)
            torch.testing.assert_close(recovered.output, clean.output, rtol=0, atol=0)
            records.append(
                dict(case=label, message=message, rejected_before_native=True, recovery_exact=True)
            )
        aliases = replace(batch, channels=tuple(c.lower() + "." for c in channels))
        torch.testing.assert_close(
            lens.run_with_cache(aliases).output, clean.output, rtol=0, atol=0
        )
    finally:
        handle.remove()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    out = Path("eeglens/research/model_validation/results/adapter-errors-v1/eegpt-boundaries.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(
                records=records,
                normalized_aliases_native_parity_exact=True,
                runner_sha256=sha(Path(__file__)),
                builder_sha256=sha(Path(inspect.getfile(build))),
                adapter_sha256=sha(Path(inspect.getfile(type(adapter)))),
                checkpoint_sha256=sha(checkpoint),
                native_source_sha256=sha(source),
                torch_version=torch.__version__,
            ),
            indent=2,
        )
        + "\n"
    )
    print("EEGPT: four pre-forward rejections, exact recovery, native alias parity passed")


if __name__ == "__main__":
    run()
