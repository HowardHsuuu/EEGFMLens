"""Explicit preprocessing used by the public EEGMMIDB examples."""

import torch

from eegfmlens import SignalBatch

CHANNELS = (
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


def prepare_eegmmidb(path) -> SignalBatch:
    """Prepare two four-second windows without downloading or rereferencing."""

    import mne

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.rename_channels({name: name.rstrip(".").upper() for name in raw.ch_names})
    raw.pick(list(CHANNELS))
    raw.reorder_channels(list(CHANNELS))
    raw.crop(tmin=0, tmax=15.99)
    raw.filter(0.5, 75, verbose="ERROR")
    raw.resample(200, verbose="ERROR")
    values = torch.tensor(raw.get_data(units="uV")[:, :1600], dtype=torch.float32) / 100
    patches = values.reshape(19, 2, 4, 200).permute(1, 0, 2, 3).contiguous()
    recipe = "eegmmidb-demo:v1:19ch:0.5-75Hz:200Hz:uV/100:no-rereference:4s"
    return SignalBatch(
        patches,
        ("recording:0-4s", "recording:4-8s"),
        CHANNELS,
        200,
        recipe,
    )
