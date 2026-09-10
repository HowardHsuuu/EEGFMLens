"""Native continuous-input length, segmentation and rejection checks."""

import argparse
import inspect
import json
from dataclasses import replace
from pathlib import Path

import torch
from validate import build, sha
from validate_dense_context import build as build_context

from eeglens import EEGLens, SignalBatch
from eeglens.errors import ValidationError


def run(name):
    root = Path("research/eeglens_model_validation")
    torch.set_num_threads(2)
    torch.manual_seed(6349)
    if name in {"bendr-context", "signaljepa"}:
        model, adapter, shape, fs, checkpoints, source = build_context(root, name)
        channels = adapter.channels
        width = shape[-1]
        builder = build_context
    else:
        model, adapter, shape, channels, fs, source, checkpoint = build(root, name)
        width, checkpoints, builder = shape[-1], [checkpoint], build
    lens = EEGLens(model, adapter)

    def hook_state():
        return {
            key: (tuple(m._forward_hooks.items()), tuple(m._forward_pre_hooks.items()))
            for key, m in model.named_modules()
        }

    original_hooks = hook_state()
    records = []
    for patches in (3, 6) if name == "signaljepa" else (1, 6):
        batch = SignalBatch(
            torch.randn(1, len(channels), patches, width),
            ("trial",),
            channels,
            fs,
            "continuous-boundary-v1",
        )
        clean = lens.run_with_cache(batch)
        with torch.no_grad():
            torch.testing.assert_close(clean.output, model(batch.data.flatten(2)), rtol=0, atol=0)
        # Container partitioning must not change the native continuous signal.
        regrouped = replace(
            batch, data=batch.data.reshape(1, len(channels), patches * 2, width // 2)
        )
        result = lens.run_with_cache(regrouped)
        torch.testing.assert_close(result.output, clean.output, rtol=0, atol=0)
        for site in clean.cache:
            torch.testing.assert_close(
                result.cache[site].tensor, clean.cache[site].tensor, rtol=0, atol=0
            )
        invalid = [
            replace(batch, sampling_rate=fs + 1),
            replace(batch, channels=channels[::-1]),
            replace(batch, patch_stride_samples=width // 2),
        ]
        if name.startswith("biot"):
            invalid.append(replace(batch, data=torch.ones(1, len(channels), 1, adapter.n_fft - 1)))
        minimum = None
        if name == "signaljepa":
            minimum = 1
            for _, kernel, stride in reversed(adapter.conv_spec):
                minimum = (minimum - 1) * stride + kernel
            invalid.append(replace(batch, data=torch.ones(1, len(channels), 1, minimum - 1)))
            boundary = replace(batch, data=torch.randn(1, len(channels), 1, minimum))
            observed = lens.run_with_cache(boundary)
            with torch.no_grad():
                torch.testing.assert_close(
                    observed.output, model(boundary.data.flatten(2)), rtol=0, atol=0
                )
        calls = []
        handle = model.register_forward_pre_hook(lambda *args: calls.append(1))
        try:
            for bad in invalid:
                before = len(calls)
                try:
                    lens.run_with_cache(bad)
                except ValidationError:
                    pass
                else:
                    raise AssertionError("Invalid continuous input accepted")
                assert len(calls) == before
                assert all(not m._forward_hooks for m in model.modules())
            torch.testing.assert_close(
                lens.run_with_cache(batch).output, clean.output, rtol=0, atol=0
            )
        finally:
            handle.remove()
        assert hook_state() == original_hooks
        records.append(
            dict(
                patches=patches,
                samples=patches * width,
                native_output_exact=True,
                container_resegmentation_output_and_all_caches_exact=True,
                invalid_inputs_rejected_before_native=len(invalid),
                recovery_exact=True,
                native_hooks_preserved=True,
                convolution_minimum_samples=minimum,
            )
        )
    output = Path("eeglens/validation/results/continuous-boundaries-v1")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(
        json.dumps(
            dict(
                model=name,
                records=records,
                runner_sha256=sha(Path(__file__)),
                builder_sha256=sha(Path(inspect.getfile(builder))),
                checkpoint_sha256=[sha(p) for p in checkpoints],
                source_sha256=sha(Path(source)),
                adapter_sha256=sha(Path(inspect.getfile(type(adapter)))),
                runtime_sha256=sha(Path(inspect.getfile(type(lens)))),
                scope="Synthetic CPU float32; two sampled lengths, not an exhaustive length domain; same continuous samples repartitioned into container patches must preserve every cache and output.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(name, records, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("biot", "biot-prest", "bendr", "bendr-context", "signaljepa")
    )
    args = parser.parse_args()
    for name in (args.model,) if args.model else ("biot", "biot-prest", "bendr", "bendr-context"):
        run(name)
