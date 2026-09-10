"""Official-checkpoint parity and patching on a local EEGMMIDB EDF.

This is an instrumentation demonstration, not a trained task classifier or
evidence of a physiological mechanism. No implicit downloads.

Run: python examples/real_eeg.py --model cbramod --checkpoint FILE --edf FILE
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from eeglens import Ablation, Replacement, Selection, SignalBatch, load_cbramod, load_labram
from eeglens.loading import sha256_file


def native_forward(model_name, model, batch):
    """Call the encoder directly, independently of the adapter's forward method."""
    if model_name == "cbramod":
        return model(batch.data)
    from eeglens._vendor.labram_channels import CHANNELS

    indices = [0, *(CHANNELS.index(channel) + 1 for channel in batch.channels)]
    return model.forward_features(batch.data, input_chans=indices, return_patch_tokens=True)


def native_local_ablation(model_name, model, batch):
    """Independent raw-coordinate oracle: C3, second patch, all features."""
    channel = batch.channels.index("C3")
    module = model.encoder.layers[0] if model_name == "cbramod" else model.blocks[0]
    observed = []

    def edit(_module, _args, output):
        edited = output.clone()
        selected = torch.zeros_like(output, dtype=torch.bool)
        if model_name == "cbramod":
            selected[:, channel, 1, :] = True
        else:
            # Native channel-major token order; the first token is CLS.
            selected[:, 1 + channel * batch.data.shape[2] + 1, :] = True
        edited[selected] = 0
        observed.append((output.detach().clone(), edited.detach().clone(), selected))
        return edited

    handle = module.register_forward_hook(edit)
    try:
        with torch.no_grad():
            output = native_forward(model_name, model, batch)
    finally:
        handle.remove()
    if len(observed) != 1:
        raise RuntimeError("Expected one invocation of the first native block")
    return output, observed[0]


def prepare_edf(path):
    import mne

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.rename_channels({name: name.rstrip(".").upper() for name in raw.ch_names})
    channels = (
        "FP1",
        "FP2",
        "F7",
        "F3",
        "FZ",
        "F4",
        "F8",
        "T7",
        "C3",
        "CZ",
        "C4",
        "T8",
        "P7",
        "P3",
        "PZ",
        "P4",
        "P8",
        "O1",
        "O2",
    )
    raw.pick(list(channels))
    raw.reorder_channels(list(channels))
    raw.crop(tmin=0, tmax=15.99)
    raw.filter(0.5, 75, verbose="ERROR")
    raw.resample(200, verbose="ERROR")
    values = torch.tensor(raw.get_data(units="uV")[:, :1600], dtype=torch.float32) / 100
    patches = values.reshape(19, 2, 4, 200).permute(1, 0, 2, 3).contiguous()
    recipe = "eegmmidb-demo:v1:19ch:0.5-75Hz:200Hz:uV/100:no-rereference:4s"
    return SignalBatch(patches, ("recording:0-4s", "recording:4-8s"), channels, 200, recipe)


def run(model_name, checkpoint, edf):
    torch.set_num_threads(2)
    batch = prepare_edf(edf)
    loader = load_cbramod if model_name == "cbramod" else load_labram
    lens = loader(checkpoint)
    site = "blocks.0.output"
    last = "blocks.11.output"
    with torch.no_grad():
        native = native_forward(model_name, lens.model, batch)
    clean = lens.run_with_cache(batch, sites=[site, last])
    identity = lens.run_with_interventions(
        batch, interventions=[Replacement(site, clean.cache[site])]
    )
    corrupted_values = batch.data.clone()
    corrupted_values[:, :, -1] = 0
    recipient = replace(batch, data=corrupted_values)
    corrupt = lens.run_with_cache(recipient, sites=[])
    # Full last-block replacement must recover the native clean output.
    restored = lens.run_with_interventions(
        recipient, interventions=[Replacement(last, clean.cache[last])]
    )
    partial = lens.run_with_interventions(
        batch,
        interventions=[Ablation(site, Selection(sensors=("C3",), patches=(1,)))],
        sites=[site],
    )
    expected, (before, edited, selected) = native_local_ablation(model_name, lens.model, batch)
    torch.testing.assert_close(partial.output, expected, atol=0, rtol=0)
    torch.testing.assert_close(clean.cache[site].tensor, before, atol=0, rtol=0)
    torch.testing.assert_close(partial.cache[site].tensor, edited, atol=0, rtol=0)
    torch.testing.assert_close(edited[~selected], before[~selected], atol=0, rtol=0)
    assert torch.count_nonzero(edited[selected]) == 0
    after = lens.run_with_cache(batch, sites=[])
    for result in (clean, identity, restored, after):
        torch.testing.assert_close(result.output, native, atol=1e-6, rtol=1e-5)
    assert not torch.equal(partial.output, native)
    assert not torch.equal(corrupt.output, native)
    return {
        "model": model_name,
        "manifest": lens.manifest,
        "edf_sha256": sha256_file(edf),
        "example_sha256": sha256_file(__file__),
        "preprocessing": batch.preprocessing_id,
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "output_shape": list(native.shape),
        "calls": clean.calls,
        "native_cache_max_diff": float((native - clean.output).abs().max()),
        "identity_max_diff": float((native - identity.output).abs().max()),
        "last_block_restore_max_diff": float((native - restored.output).abs().max()),
        "selected_ablation_max_diff": float((native - partial.output).abs().max()),
        "local_ablation_native_max_diff": float((expected - partial.output).abs().max()),
        "local_ablation_cache_exact": True,
        "unselected_activation_exact": True,
        "local_selection": {"channel": "C3", "patch_index": 1, "features": "all"},
        "cleanup_max_diff": float((native - after.output).abs().max()),
        "scope": "2 EEG windows; pretrained feature outputs, no task accuracy or mechanism claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cbramod", "labram"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--edf", required=True)
    parser.add_argument("--output", type=Path, help="Save JSON; refuses an existing path")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    report = json.dumps(run(args.model, args.checkpoint, args.edf), indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x") as destination:
            destination.write(report)
    print(report, end="")
