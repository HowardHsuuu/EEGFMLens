"""Admit the public masked EEGPT adapter against the actual full checkpoint.

This is a technical, synthetic-input admission. It contains no participant
effects and does not turn EEGPT into a fourth serial-route finding.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import load_file

from eegfmlens import (
    EEGLens, EEGPTMaskedReconstructionAdapter, Replacement, Selection, SignalBatch,
)

CHECKPOINT_SHA = "d27f2b2d728224fde538d82d84d2804cfebcc162f2ffa6c0c67f559ae8bb2460"
NATIVE_SOURCE_SHA = "4d40c2e5ce5de7a45e9a4ad7c93cab21d22a47d3e43228eb7699186706ae88c6"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--findings-root", type=Path, required=True)
    p.add_argument("--research", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(args.research.resolve()):
        raise ValueError("fresh local-only EEGPT technical admission required")
    checkpoint = args.research / "context-mechanisms/eegpt-full-admission/tensors-v1/state.safetensors"
    if digest(checkpoint) != CHECKPOINT_SHA:
        raise AssertionError("actual EEGPT full checkpoint changed")
    sys.path.insert(0, str(args.findings_root / "experiments/context_mechanisms"))
    from eegmat_data import CHANNELS  # noqa: E402

    torch.set_num_threads(2)
    native_path = args.research / "eeglens_model_validation/repos/EEGPT/pretrain/modeling_pretraining.py"
    if digest(native_path) != NATIVE_SOURCE_SHA:
        raise AssertionError("pinned native EEGPT pretraining source changed")
    spec = importlib.util.spec_from_file_location("native_eegpt_pretraining_components", native_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    config = dict(
        embed_dim=512, embed_num=4, depth=8, num_heads=8,
        mlp_ratio=4.0, qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    )
    model = torch.nn.Module()
    model.encoder = native.EEGTransformer(img_size=(19, 1024), patch_size=64, **config)
    model.target_encoder = native.EEGTransformer(img_size=(19, 1024), patch_size=64, **config)
    model.predictor = native.EEGTransformerPredictor(
        num_patches=(19, 16), predictor_embed_dim=512, use_part_pred=True, **config
    )
    model.reconstructor = native.EEGTransformerReconstructor(
        num_patches=(19, 16), patch_size=64, reconstructor_embed_dim=512, **config
    )
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    model.eval().requires_grad_(False)
    lens = EEGLens(model, EEGPTMaskedReconstructionAdapter(model))
    torch.manual_seed(847)
    baseline_data = torch.randn(2, 19, 16, 64)
    original = SignalBatch(
        baseline_data, ("synthetic-a", "synthetic-b"), CHANNELS, 256,
        "eegpt-package-actual-checkpoint-a25", "model_scaled",
    )
    c3, c4 = CHANNELS.index("C3"), CHANNELS.index("C4")
    mask_x = [
        [time * 19 + sensor for sensor in range(19)
         if sensor != (c3 if 4 <= time < 8 else c4)]
        for time in range(15)
    ]
    mask_y = [time * 19 + c3 for time in range(4, 8)]
    kwargs = {"mask_x": mask_x, "mask_y": mask_y}
    observed = torch.zeros(16 * 19, dtype=torch.bool)
    observed[torch.tensor(mask_x).flatten()] = True
    donor_data = baseline_data.clone()
    donor_data[:, c4, :4] += 0.2
    donor = SignalBatch(
        donor_data, original.trial_ids, CHANNELS, 256,
        original.preprocessing_id, "model_scaled",
    )
    hidden_data = baseline_data.transpose(1, 2).flatten(1, 2).clone()
    hidden_data[:, ~observed] += 1000
    hidden = SignalBatch(
        hidden_data.reshape(2, 16, 19, 64).transpose(1, 2),
        original.trial_ids, CHANNELS, 256, original.preprocessing_id, "model_scaled",
    )
    selected = Selection(patches=(4, 5, 6, 7))
    site = "predictor.combined"
    hooks_before = tuple(
        (name, tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
        for name, module in model.named_modules()
    )
    lens.assert_model_unchanged()
    with torch.inference_mode():
        baseline = lens.run_with_cache(original, **kwargs)
        source_run = lens.run_with_cache(donor, **kwargs)
        channels = model.encoder.prepare_chan_ids(CHANNELS)
        encoded = model.encoder(original.data.flatten(2), channels, mask_x=torch.tensor(mask_x))
        _, combined = model.predictor(encoded, mask_x=torch.tensor(mask_x))
        native_output = model.reconstructor(combined, channels, mask_y=torch.tensor(mask_y))
        torch.testing.assert_close(baseline.output, native_output, rtol=0, atol=0)
        if tuple(baseline.cache[site].tensor.shape) != (2, 16, 4, 512):
            raise AssertionError("actual predictor time/summary/feature geometry differs")
        torch.testing.assert_close(
            lens.run_with_cache(hidden, **kwargs).output, baseline.output,
            rtol=0, atol=0,
        )
        own = lens.run_with_interventions(
            original, interventions=(Replacement(site, baseline.cache[site], selected),), **kwargs
        )
        torch.testing.assert_close(own.output, baseline.output, rtol=0, atol=0)
        edited = lens.run_with_interventions(
            original, interventions=(Replacement(site, source_run.cache[site], selected),), **kwargs
        )

        def native_hook(module, inputs, output):
            changed = output[1].clone()
            changed[:, selected.patches] = source_run.cache[site].tensor[:, selected.patches]
            return output[0], changed

        handle = model.predictor.register_forward_hook(native_hook)
        try:
            oracle = lens.adapter.forward(model, original, **kwargs)
        finally:
            handle.remove()
        torch.testing.assert_close(edited.output, oracle, rtol=0, atol=0)
        changed = baseline.cache[site].tensor.clone()
        changed[:, selected.patches] = source_run.cache[site].tensor[:, selected.patches]
        assert torch.equal(changed[:, :4], baseline.cache[site].tensor[:, :4])
        assert torch.equal(changed[:, 8:], baseline.cache[site].tensor[:, 8:])
        if not (edited.output - baseline.output).norm() > 0:
            raise AssertionError("actual checkpoint selected patch has no effect")
    lens.assert_model_unchanged()
    if hooks_before != tuple(
        (name, tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
        for name, module in model.named_modules()
    ):
        raise AssertionError("native EEGPT hooks remain after package intervention")
    result = {
        "status": "pass", "scope": "actual-checkpoint technical admission; no participant effects",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "native_source_sha256": NATIVE_SOURCE_SHA,
        "adapter_source_sha256": digest(Path(sys.modules[
            EEGPTMaskedReconstructionAdapter.__module__
        ].__file__)),
        "package_adapter": "EEGPTMaskedReconstructionAdapter",
        "predictor_combined_shape": [2, 16, 4, 512],
        "physical_selection": {"time_patches": list(selected.patches), "sensor_selection": False},
        "native_output_exact": True, "hidden_input_independence_exact": True,
        "own_state_exact": True, "native_intervention_exact": True,
        "unselected_state_exact": True, "hook_and_model_integrity_exact": True,
        "script_sha256": digest(Path(__file__)),
        "participant_or_trial_effects_included": False,
    }
    args.output.mkdir(parents=True)
    (args.output / "admission.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "pass", "checkpoint_sha256": CHECKPOINT_SHA}))


if __name__ == "__main__":
    main()
