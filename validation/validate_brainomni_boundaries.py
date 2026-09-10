"""BrainOmni native padding, geometry/configuration rejection and recovery."""

import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from validate_diver_steegformer import sha

from eeglens import BrainOmniAdapter, EEGLens, SignalBatch
from eeglens.errors import ValidationError


def run():
    root = Path("research/eeglens_model_validation")
    sys.path.insert(0, str(root / "repos/BrainOmni"))
    from brainomni.model import BrainOmni

    torch.set_num_threads(2)
    torch.manual_seed(8871)
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
    lens = EEGLens(model, adapter)

    def execute(fn, *args, **kwargs):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9137)
            return fn(*args, **kwargs)

    def hooks():
        return {
            n: (tuple(m._forward_hooks.items()), tuple(m._forward_pre_hooks.items()))
            for n, m in model.named_modules()
        }

    original_hooks = hooks()
    rows = []
    for samples in (1, 512, 513, 1536):
        batch = SignalBatch(
            torch.randn(1, 4, 1, samples),
            ("trial",),
            adapter.channels,
            256,
            "brainomni-boundary-v1",
        )
        clean = execute(lens.run_with_cache, batch)
        window = model.window_length
        step = int(window * (1 - model.overlap_ratio))
        padded_length = window + max(0, (samples - window + step - 1) // step) * step
        padded = torch.nn.functional.pad(batch.data.flatten(2), (0, padded_length - samples))
        with torch.no_grad():
            expected = execute(
                model.encode,
                padded,
                adapter.positions.unsqueeze(0),
                adapter.sensor_types.unsqueeze(0),
            )
        torch.testing.assert_close(clean.output, expected, rtol=0, atol=0)
        if samples % 2 == 0:
            regrouped = replace(batch, data=batch.data.reshape(1, 4, 2, samples // 2))
            other = execute(lens.run_with_cache, regrouped)
            torch.testing.assert_close(other.output, clean.output, rtol=0, atol=0)
            for site in clean.cache:
                torch.testing.assert_close(
                    other.cache[site].tensor, clean.cache[site].tensor, rtol=0, atol=0
                )
        invalid = [
            replace(batch, sampling_rate=200),
            replace(batch, channels=adapter.channels[::-1]),
            replace(batch, patch_stride_samples=samples + 1),
        ]
        calls = []
        h = model.tokenizer.sensor_embed.register_forward_pre_hook(lambda *args: calls.append(1))

        def reject(value):
            count = len(calls)
            try:
                execute(lens.run_with_cache, value)
            except ValidationError:
                pass
            else:
                raise AssertionError("Invalid BrainOmni input/configuration accepted")
            assert len(calls) == count

        try:
            for value in invalid:
                reject(value)
            for target, attr in (
                (model, "window_length"),
                (model, "overlap_ratio"),
                (model.tokenizer, "window_length"),
            ):
                saved = getattr(target, attr)
                try:
                    setattr(target, attr, saved + 1)
                    reject(batch)
                finally:
                    setattr(target, attr, saved)
            for tensor in (adapter.positions, adapter.sensor_types):
                saved = tensor.clone()
                try:
                    tensor.add_(1)
                    reject(batch)
                finally:
                    tensor.copy_(saved)
            torch.testing.assert_close(
                execute(lens.run_with_cache, batch).output, clean.output, rtol=0, atol=0
            )
            assert len(calls) == 1
        finally:
            h.remove()
        assert hooks() == original_hooks
        rows.append(
            dict(
                samples=samples,
                explicit_padding_samples=padded_length,
                native_padding_parity_exact=True,
                invalid_inputs=3,
                mutable_configurations=5,
                recovery_exact=True,
                resegmentation_all_caches_exact=samples % 2 == 0,
            )
        )
    output = Path("eeglens/validation/results/brainomni-boundaries-v1.json")
    output.write_text(
        json.dumps(
            dict(
                model="brainomni-tiny",
                records=rows,
                runner_sha256=sha(__file__),
                adapter_sha256=sha(inspect.getfile(BrainOmniAdapter)),
                checkpoint_sha256=sha(checkpoint),
                native_source_sha256=sha(inspect.getfile(BrainOmni)),
                tokenizer_source_sha256=sha(inspect.getfile(type(model.tokenizer))),
                scope="CPU float32, synthetic geometry; paired RNG for native eval dropout; 512-sample windows with 0.25 overlap. Single-sample acceptance is native padding behavior, not meaningful EEG evidence.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("BrainOmni", rows, flush=True)


if __name__ == "__main__":
    run()
