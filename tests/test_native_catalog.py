"""Live-source conformance for catalog integrations beyond the loader pair."""

import importlib
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import pytest
import torch

from eegfmlens import Ablation, Replacement, SignalBatch, attribute, connect, model_info

pytestmark = pytest.mark.native


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load upstream module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source_module(source, module_name):
    sys.path.insert(0, str(source))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(source))


def _eegpt(source):
    upstream = _load_file("eegfmlens_native_eegpt", source / "pretrain/modeling_pretraining.py")
    model = upstream.EEGTransformer(
        img_size=(3, 64),
        patch_size=16,
        embed_dim=32,
        embed_num=1,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
    ).eval()
    channels = ("C3", "CZ", "C4")
    batch = SignalBatch(
        torch.randn(2, 3, 4, 16),
        ("trial-a", "trial-b"),
        channels,
        256,
        "native-source-smoke:v1",
    )
    lens = connect(model, "eegpt", model_id="native-source:eegpt")

    def native():
        return model(batch.data.flatten(2), chan_ids=model.prepare_chan_ids(channels))

    return lens, batch, native, "blocks.0.output"


def _csbrain(source):
    upstream = _source_module(source, "models.CSBrain")
    channels = (
        "FP1-REF",
        "FP2-REF",
        "F3-REF",
        "F4-REF",
        "C3-REF",
        "C4-REF",
        "P3-REF",
        "P4-REF",
        "O1-REF",
        "O2-REF",
        "F7-REF",
        "F8-REF",
        "T3-REF",
        "T4-REF",
        "T5-REF",
        "T6-REF",
        "FZ-REF",
        "CZ-REF",
        "PZ-REF",
    )
    regions = [0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1]
    topology = {
        0: ("FP1-REF", "F7-REF", "F3-REF", "FZ-REF", "F4-REF", "F8-REF", "FP2-REF"),
        1: ("P3-REF", "PZ-REF", "P4-REF"),
        2: ("T3-REF", "T5-REF", "T6-REF", "T4-REF"),
        3: ("O1-REF", "O2-REF"),
        4: ("C3-REF", "CZ-REF", "C4-REF"),
    }
    order = tuple(
        sorted(
            range(len(channels)),
            key=lambda index: (regions[index], topology[regions[index]].index(channels[index])),
        )
    )
    model = upstream.CSBrain(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=400,
        seq_len=2,
        n_layer=1,
        nhead=8,
        brain_regions=regions,
        sorted_indices=list(order),
    ).eval()
    batch = SignalBatch(
        torch.randn(1, 19, 2, 200),
        ("trial-a",),
        channels,
        200,
        "native-source-smoke:v1",
    )
    lens = connect(model, "csbrain", channels=channels, model_id="native-source:csbrain")
    return lens, batch, lambda: model(batch.data), "blocks.0.output"


def _steegformer(source):
    upstream = _load_file("eegfmlens_native_steegformer", source / "easy_start/models_vit_eeg.py")
    with (source / "pretrain/senloc_file/sen_chan_idx.pkl").open("rb") as stream:
        mapping = pickle.load(stream)["channels_mapping"]
    channels = ("C1", "C4", "F4")
    model = upstream.vit_small_patch16(num_classes=0, global_pool=False).eval()
    model.register_parameter("pos_embed", None)
    batch = SignalBatch(
        torch.randn(1, 3, 2, 16),
        ("trial-a",),
        channels,
        128,
        "native-source-smoke:v1",
    )
    lens = connect(
        model,
        "steegformer",
        channels=channels,
        channel_mapping=mapping,
        model_id="native-source:steegformer",
    )

    def native():
        indices = torch.tensor([mapping[channel] for channel in channels]).expand(1, -1)
        return model.forward_features(batch.data.flatten(2), indices)

    return lens, batch, native, "blocks.0.output"


def _diver(source):
    upstream = _source_module(source, "models.diver")
    channels = ("C3", "CZ", "C4")
    positions = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    model = upstream.DIVER(d_model=256, e_layer=1, mup=False).eval()
    batch = SignalBatch(
        torch.randn(1, 3, 2, 500),
        ("trial-a",),
        channels,
        500,
        "native-source-smoke:v1",
    )
    lens = connect(
        model,
        "diver",
        channels=channels,
        positions=positions,
        model_id="native-source:diver",
    )

    def native():
        return model(
            batch.data,
            data_info_list=lens.adapter.data_info(batch),
            use_mask=False,
        )["y"]

    return lens, batch, native, "embedding.output"


def _biot(source):
    upstream = _source_module(source, "model.biot")
    channels = ("C3", "CZ", "C4")
    model = upstream.BIOTEncoder(
        emb_size=256,
        heads=8,
        depth=1,
        n_channels=len(channels),
        n_fft=200,
        hop_length=100,
    ).eval()
    batch = SignalBatch(
        torch.randn(1, 3, 2, 200),
        ("trial-a",),
        channels,
        200,
        "native-source-smoke:v1",
    )
    lens = connect(model, "biot", channels=channels, model_id="native-source:biot")
    return lens, batch, lambda: model(batch.data.flatten(2)), "transformer.output"


BUILDERS = {
    "biot": _biot,
    "csbrain": _csbrain,
    "diver": _diver,
    "eegpt": _eegpt,
    "steegformer": _steegformer,
}


def _seeded(call):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260921)
        return call()


def _objective(output, batch):
    if not isinstance(output, torch.Tensor):
        raise TypeError("Native gradient smoke expects a tensor output")
    flattened = output.reshape(output.shape[0], -1)
    weights = torch.linspace(0.5, 1.5, flattened.shape[1], device=output.device)
    return (flattened * weights).sum(dim=1)


def test_pinned_native_catalog_observation_identity_and_ablation():
    family = os.environ.get("EEGLENS_NATIVE_MODEL")
    if family is None:
        pytest.skip("Set EEGLENS_NATIVE_MODEL for one pinned-source matrix job")
    if family not in BUILDERS:
        pytest.fail(f"Unknown live-source smoke family: {family}")
    source_value = os.environ.get("EEGLENS_NATIVE_SOURCE")
    if source_value is None:
        pytest.fail("Set EEGLENS_NATIVE_SOURCE to the selected upstream checkout")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from native_sources import REVISIONS, verified_source

    assert model_info(family).reference == REVISIONS[family]
    source = verified_source(family, source_value)
    lens, batch, native_call, site = BUILDERS[family](source)
    torch.set_num_threads(2)
    lens.model.requires_grad_(False)

    module = lens.model.get_submodule(lens.adapter.require(site).module_path)
    original_hooks = tuple(module._forward_hooks)
    native = _seeded(native_call)
    observed = _seeded(lambda: lens.run_with_cache(batch))
    torch.testing.assert_close(observed.output, native, rtol=0, atol=0)
    assert set(observed.cache) == {declared.name for declared in lens.sites()}

    identity = _seeded(
        lambda: (
            lens.run_with_interventions(
                batch,
                interventions=[Replacement(site, observed.cache[site])],
            ).output
        )
    )
    torch.testing.assert_close(identity, native, rtol=0, atol=0)

    handle = module.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
    try:
        native_zero = _seeded(native_call)
    finally:
        handle.remove()
    ablated = _seeded(
        lambda: (
            lens.run_with_interventions(
                batch,
                interventions=[Ablation(site)],
            ).output
        )
    )
    torch.testing.assert_close(ablated, native_zero, rtol=0, atol=0)
    assert not torch.equal(ablated, native)
    assert tuple(module._forward_hooks) == original_hooks

    attributed = _seeded(
        lambda: attribute(
            lens,
            batch,
            _objective,
            method="input_x_gradient",
            sites=(site,),
        )
    )
    assert attributed.input_attribution.shape == batch.data.shape
    assert attributed.site_attributions[site].shape == observed.cache[site].tensor.shape
    assert torch.isfinite(attributed.input_attribution).all()
    assert torch.isfinite(attributed.site_attributions[site]).all()
    assert all(parameter.grad is None for parameter in lens.model.parameters())
    assert tuple(module._forward_hooks) == original_hooks
