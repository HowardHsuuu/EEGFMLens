"""Compare the vendored encoder with an independently imported upstream checkout.

Pass a checkout at the revision documented in docs/models.md. This intentionally
imports that local source; use a trusted checkout. No network access is performed.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
from real_eeg import prepare_edf
from torch import nn

from eeglens import load_cbramod, load_labram
from eeglens.loading import CBRAMOD_REVISION, LABRAM_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cbramod", "labram"], required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--edf", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    expected = CBRAMOD_REVISION if args.model == "cbramod" else LABRAM_REVISION
    actual = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise ValueError(f"Expected upstream revision {expected}, got {actual}")
    batch = prepare_edf(args.edf)
    if args.model == "cbramod":
        sys.path.insert(0, str(args.upstream.resolve()))
        from models.cbramod import CBraMod

        native = CBraMod()
        native.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        native.proj_out = nn.Identity()
        lens = load_cbramod(args.checkpoint)

        def forward():
            return native(batch.data)
    else:
        spec = importlib.util.spec_from_file_location(
            "upstream_labram", args.upstream / "modeling_finetune.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        native = module.labram_base_patch200_200(
            num_classes=0, init_values=0.1, use_mean_pooling=False
        )
        lens = load_labram(args.checkpoint)
        native.load_state_dict(lens.model.state_dict(), strict=True)

        def forward():
            return native.forward_features(
                batch.data,
                input_chans=lens.adapter.channel_indices(batch),
                return_patch_tokens=True,
            )

    native.eval()
    with torch.no_grad():
        expected_output = forward()
    result = lens.run_with_cache(batch, sites=["blocks.0.output"])
    torch.testing.assert_close(result.output, expected_output, atol=1e-6, rtol=1e-5)
    print(
        json.dumps(
            {
                "model": args.model,
                "upstream_revision": actual,
                "upstream_max_diff": float((result.output - expected_output).abs().max()),
                "scope": "Independent source forward; identical encoder weights/output path",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
