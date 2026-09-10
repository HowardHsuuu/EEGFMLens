"""Strictly load both original BENDR components and verify their native composition."""

import hashlib
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from semantic_checks import check_features, check_rejected_selectors

from eeglens import Ablation, EEGLens, Replacement, SignalBatch
from eeglens.adapters.bendr import BENDRAdapter


def run():
    root = Path("research/eeglens_model_validation")
    sys.path[:0] = [str(root / "repos/dn3"), str(root / "repos/BENDR")]
    import dn3_ext

    torch.set_num_threads(2)
    torch.manual_seed(4311)
    encoder = dn3_ext.ConvEncoderBENDR(in_features=20, encoder_h=512)
    context = dn3_ext.BENDRContextualizer(in_features=512)
    encoder_checkpoint = root / "checkpoints/bendr-mirror-encoder.pt"
    checkpoint = root / "expansion/bendr-contextualizer.pt"
    encoder.load_state_dict(
        torch.load(encoder_checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    context.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model = torch.nn.Sequential(encoder, context).eval().requires_grad_(False)
    adapter = BENDRAdapter(model, channels=tuple(f"input-{i}" for i in range(20)))
    lens = EEGLens(model, adapter)
    records = []
    for b in (1, 2):
        batch = SignalBatch(
            torch.randn(b, len(adapter.channels), 4, 256),
            tuple(f"t{i}" for i in range(b)),
            adapter.channels,
            256,
            "synthetic-bendr-context-v1",
        )
        with torch.no_grad():
            native = model(batch.data.flatten(2))
        clean = lens.run_with_cache(batch)
        torch.testing.assert_close(clean.output, native, rtol=0, atol=0)
        for site in lens.sites():
            restored = lens.run_with_interventions(
                batch, interventions=[Replacement(site.name, clean.cache[site.name])]
            )
            torch.testing.assert_close(restored.output, native, rtol=0, atol=0)
            edited = lens.run_with_interventions(batch, interventions=[Ablation(site.name)])
            h = model.get_submodule(site.module_path).register_forward_hook(
                lambda m, a, o: torch.zeros_like(o)
            )
            try:
                with torch.no_grad():
                    manual = model(batch.data.flatten(2))
            finally:
                h.remove()
            torch.testing.assert_close(edited.output, manual, rtol=0, atol=0, msg=site.name)
            assert torch.isfinite(edited.output).all()
            # Upstream convolution outputs are B,D,T; input_conditioning ends
            # in Permute([2,0,1]) and TransformerEncoderLayer uses T,B,D.
            axis = 1 if site.module_path in {"0", "1.output_layer"} else -1
            semantic = check_features(
                lens, batch, site, clean, lambda: model(batch.data.flatten(2)), native_axis=axis
            )
            semantic.update(check_rejected_selectors(lens, batch, site))
            records.append(
                dict(
                    batch=b,
                    site=site.name,
                    delta=float((edited.output - native).norm()),
                    **semantic,
                )
            )
        order = list(reversed(range(b)))
        recipient = replace(
            batch, data=batch.data[order] + 0.3, trial_ids=tuple(batch.trial_ids[i] for i in order)
        )
        recovered = lens.run_with_interventions(
            recipient, interventions=[Replacement("context.output", clean.cache["context.output"])]
        )
        torch.testing.assert_close(recovered.output, native[order], rtol=0, atol=0)
        assert all(not m._forward_hooks for m in model.modules())
    result = dict(
        records=records,
        feature_subspace_verified=True,
        strict_full_load=True,
        reordered_donor_recovery=True,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        semantic_helper_sha256=hashlib.sha256(
            Path(inspect.getfile(check_features)).read_bytes()
        ).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(inspect.getfile(BENDRAdapter)).read_bytes()).hexdigest(),
        encoder_checkpoint_sha256=hashlib.sha256(encoder_checkpoint.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        native_source_sha256=hashlib.sha256(
            Path(inspect.getfile(dn3_ext.BENDRContextualizer)).read_bytes()
        ).hexdigest(),
        torch_version=torch.__version__,
    )
    (root / "semantic-v1").mkdir(exist_ok=True)
    (root / "semantic-v1/bendr-context.json").write_text(json.dumps(result, indent=2) + "\n")
    print("BENDR contextualizer", len(records), "conditions passed")


if __name__ == "__main__":
    run()
