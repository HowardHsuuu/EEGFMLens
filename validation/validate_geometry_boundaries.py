"""DIVER/ST-EEGFormer native input and mutable-configuration boundaries."""

import inspect
import json
from dataclasses import replace
from pathlib import Path

import torch
from validate_diver_steegformer import build, sha

from eeglens import EEGLens, SignalBatch
from eeglens.errors import ValidationError


def run(name):
    torch.set_num_threads(2)
    torch.manual_seed(6823)
    model, adapter, shape, fs, checkpoint, source, metadata = build(
        Path("research/eeglens_model_validation"), name
    )
    lens = EEGLens(model, adapter)

    def execute(fn, *args, **kwargs):
        if name != "diver":
            return fn(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9417)
            return fn(*args, **kwargs)

    def hooks():
        return {
            k: (tuple(m._forward_hooks.items()), tuple(m._forward_pre_hooks.items()))
            for k, m in model.named_modules()
        }

    original_hooks = hooks()
    records = []
    for patches in (1, 8):
        batch = SignalBatch(
            torch.randn(1, len(adapter.channels), patches, shape[-1]),
            ("trial",),
            adapter.channels,
            fs,
            "geometry-boundary-v1",
        )

        def native():
            if name == "diver":
                info = [dict(xyz_id=adapter.positions.clone(), modality="EEG", coord_subtype=None)]
                return model(batch.data, data_info_list=info, use_mask=False)["y"]
            return model.forward_features(
                batch.data.flatten(2), torch.tensor(adapter.channel_indices).unsqueeze(0)
            )

        clean = execute(lens.run_with_cache, batch)
        with torch.no_grad():
            torch.testing.assert_close(clean.output, execute(native), rtol=0, atol=0)
        bad_inputs = [
            replace(batch, sampling_rate=fs + 1),
            replace(batch, channels=adapter.channels[::-1]),
            replace(batch, patch_stride_samples=shape[-1] // 2),
        ]
        if name == "diver":
            bad_inputs.append(replace(batch, data=batch.data[..., :-1]))
        else:
            bad_inputs.append(
                replace(
                    batch, data=torch.randn(1, len(adapter.channels), 1, adapter.patch_size - 1)
                )
            )
            bad_inputs.append(
                replace(
                    batch,
                    data=torch.randn(
                        1, len(adapter.channels), adapter.max_patches + 1, adapter.patch_size
                    ),
                )
            )
            regrouped = replace(
                batch,
                data=batch.data.reshape(1, len(adapter.channels), patches * 2, shape[-1] // 2),
            )
            result = lens.run_with_cache(regrouped)
            torch.testing.assert_close(result.output, clean.output, rtol=0, atol=0)
            for site in clean.cache:
                torch.testing.assert_close(
                    result.cache[site].tensor, clean.cache[site].tensor, rtol=0, atol=0
                )
        calls = []
        first = model if name == "diver" else model.patch_embed
        handle = first.register_forward_pre_hook(lambda *args: calls.append(1))

        def expect_rejection(bad):
            count = len(calls)
            try:
                execute(lens.run_with_cache, bad)
            except ValidationError:
                pass
            else:
                raise AssertionError("Invalid geometry/configuration accepted")
            assert len(calls) == count

        try:
            for bad in bad_inputs:
                expect_rejection(bad)
            if name == "diver":
                positions = adapter.positions.clone()
                try:
                    adapter.positions.add_(1)
                    expect_rejection(batch)
                finally:
                    adapter.positions.copy_(positions)
            else:
                original_pool = model.global_pool
                try:
                    model.global_pool = True
                    expect_rejection(batch)
                finally:
                    model.global_pool = original_pool
            torch.testing.assert_close(
                execute(lens.run_with_cache, batch).output, clean.output, rtol=0, atol=0
            )
            assert len(calls) == 1  # The observer must actually detect valid input computation.
        finally:
            handle.remove()
        assert hooks() == original_hooks
        records.append(
            dict(
                patches=patches,
                native_output_exact=True,
                invalid_inputs_rejected_before_native=len(bad_inputs),
                mutable_configuration_rejected=True,
                exact_restoration_and_recovery=True,
                native_hooks_preserved=True,
                resegmentation_checked=name == "steegformer",
            )
        )
    folder = Path("eeglens/validation/results/geometry-boundaries-v1")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(
        json.dumps(
            dict(
                model=name,
                records=records,
                runner_sha256=sha(__file__),
                builder_sha256=sha(inspect.getfile(build)),
                checkpoint_sha256=sha(checkpoint),
                native_source_sha256=sha(source),
                adapter_sha256=sha(inspect.getfile(type(adapter))),
                runtime_sha256=sha(inspect.getfile(type(lens))),
                metadata=metadata,
                scope="CPU float32 synthetic geometry; sampled 1/8 native patches, not all lengths; DIVER uses paired RNG for upstream eval dropout. ST temporal over-capacity rejected without asserting maximum valid-length runtime coverage.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(name, records, flush=True)


if __name__ == "__main__":
    for name in ("diver", "steegformer"):
        run(name)
