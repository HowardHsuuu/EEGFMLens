"""Native checkpoint component constructors used by the integration runner.

Input assets follow the documented layout. No source downloads or modifications.
Explicit compatibility choices retain historical component scope.
"""

import argparse
import importlib
import importlib.util
import inspect
import json
import pickle
import sys
import types
from functools import partial
from pathlib import Path

import torch
from run_checkpoint_checks import digest as sha
from safetensors.torch import load_file

from eegfmlens import (
    BENDRAdapter,
    BENDREncoderAdapter,
    BIOTAdapter,
    BrainOmniAdapter,
    CSBrainAdapter,
    DIVERAdapter,
    EEGPTAdapter,
    NeuroRVQAdapter,
    SignalJEPAAdapter,
    STEEGFormerAdapter,
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def encoders(root, name):
    if name == "eegpt":
        from functools import partial

        from safetensors.torch import load_file

        path = root / "repos/EEGPT/downstream/Modules/models/EEGPT_mcae_finetune.py"
        module = load_module(path, "native_eegpt")
        model = module.EEGTransformer(
            img_size=(4, 1024),
            patch_size=64,
            embed_dim=512,
            embed_num=4,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        )
        checkpoint = root / "checkpoints/eegpt-braindecode.safetensors"
        state = load_file(str(checkpoint))
        assert set(k for k in state if not k.startswith("target_encoder.")) == {"chans_id"}
        state = {
            k.removeprefix("target_encoder."): v
            for k, v in state.items()
            if k.startswith("target_encoder.")
        }
        model.load_state_dict(state, strict=True)
        adapter = EEGPTAdapter(model)
        shape = (2, 4, 16, 64)
        channels = ("C3", "C4", "F3", "F4")
        fs = 256
    elif name in {"biot", "biot-prest"}:
        path = root / "biot.py"
        module = load_module(path, "native_biot")
        n_channels = 16 if name == "biot-prest" else 18
        model = module.BIOTEncoder(n_channels=n_channels)
        checkpoint = (
            root
            / "checkpoints"
            / ("biot-prest.ckpt" if name == "biot-prest" else "biot-six-mirror.ckpt")
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        shape = (2, n_channels, 10, 200)
        fs = 200
        channels = (
            "FP1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "FP2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
            "C3-A2",
            "C4-A1",
        )
        channels = channels[:n_channels]
    else:
        sys.path[:0] = [str(root / "repos/dn3"), str(root / "repos/BENDR")]
        import dn3_ext

        path = root / "repos/BENDR/dn3_ext.py"
        checkpoint = root / "checkpoints/bendr-mirror-encoder.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = dn3_ext.ConvEncoderBENDR(in_features=20, encoder_h=512)
        model.load_state_dict(state, strict=True)
        shape = (2, 20, 4, 256)
        fs = 256
        channels = tuple(f"input-{i}" for i in range(20))

    if name in {"biot", "biot-prest"}:
        adapter = BIOTAdapter(model, channels=channels)
    elif name == "bendr":
        adapter = BENDREncoderAdapter(model, channels=channels)
    model.eval().requires_grad_(False)
    return model, adapter, shape[1:], channels, fs


def bendr_context(root):
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
    return model, adapter, (20, 4, 256), adapter.channels, 256


def brainomni(root):
    source = root / "repos/BrainOmni"
    sys.path.insert(0, str(source))
    from brainomni.model import BrainOmni

    config = root / "expansion/OpenTSLab-BrainOmni/tiny/model_cfg.json"
    checkpoint = config.with_name("BrainOmni.pt")
    torch.set_num_threads(2)
    torch.manual_seed(4311)
    model = BrainOmni(**json.loads(config.read_text()))
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().requires_grad_(False)
    positions = torch.randn(4, 6)
    kinds = torch.zeros(4, dtype=torch.long)
    adapter = BrainOmniAdapter(
        model, channels=("C3", "C4", "F3", "F4"), positions=positions, sensor_types=kinds
    )
    return model, adapter, (4, 3, 512), adapter.channels, 256


def csbrain(root):
    sys.path.insert(0, str(root / "repos/CSBrain"))
    from models.CSBrain import CSBrain

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
    order = [0, 10, 2, 16, 3, 11, 1, 6, 18, 7, 12, 14, 15, 13, 8, 9, 4, 17, 5]
    torch.set_num_threads(2)
    torch.manual_seed(4311)
    model = CSBrain(brain_regions=regions, sorted_indices=order)
    checkpoint = root / "expansion/CSBrain.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert all(k.startswith("module.") for k in state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    adapter = CSBrainAdapter(model, channels=channels)
    return model, adapter, (19, 4, 200), channels, 200


def neurorvq(root):
    sys.path.insert(0, str(root / "repos/NeuroRVQ"))
    from inference.modules.NeuroRVQ_EEG_tokenizer_inference_modules import (
        ch_names_global,
    )
    from NeuroRVQ_EEG.NeuroRVQ import NeuroRVQFM

    torch.set_num_threads(2)
    torch.manual_seed(4311)
    checkpoint = root / "expansion/NeuroRVQ_EEG_FM.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = NeuroRVQFM(
        num_classes=0,
        qkv_bias=True,
        qk_norm=partial(torch.nn.LayerNorm, eps=1e-6),
        init_values=1e-5,
        n_global_electrodes=len(ch_names_global),
    )
    # Upstream inference declares fresh downstream norms absent from pretraining.
    # Return raw branch features instead: no untrained affine parameters are used.
    for i in range(1, 5):
        setattr(model, f"fc_norm_{i}", torch.nn.Identity())
    extra = set(state) - set(model.state_dict())
    assert extra == {"mask_token", "norm_pre.weight", "norm_pre.bias"} | {
        f"head_pre_{i}.{p}" for i in range(1, 33) for p in ["weight", "bias"]
    }
    selected = {k: v for k, v in state.items() if k not in extra}
    model.load_state_dict(selected, strict=True)
    model.eval().requires_grad_(False)
    adapter = NeuroRVQAdapter(model, channel_vocabulary=[c.decode() for c in ch_names_global])
    return model, adapter, (2, 2, 200), ("C3", "C4"), 200


def signaljepa(root, source):
    for name, path in (
        ("braindecode", source),
        ("braindecode.models", source / "models"),
        ("braindecode.modules", source / "modules"),
    ):
        if name in sys.modules:
            raise RuntimeError("Direct-source validation requires a fresh process")
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    SignalJEPA = importlib.import_module("braindecode.models.signal_jepa").SignalJEPA
    checkpoint_dir = root / "expansion/braindecode-signal-jepa"
    config = json.loads((checkpoint_dir / "config.json").read_text())
    config.pop("braindecode_version")
    assert config.pop("activation") == "torch.nn.modules.activation.GELU"
    config["activation"] = torch.nn.GELU
    torch.set_num_threads(2)
    torch.manual_seed(4311)
    model = SignalJEPA(**config)
    checkpoint = checkpoint_dir / "model.safetensors"
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    model.eval().requires_grad_(False)
    adapter = SignalJEPAAdapter(model)
    return model, adapter, (len(adapter.channels), 3, 128), adapter.channels, 128


def spatial(root, name):
    if name == "steegformer":
        source = root / "repos/STEEGFormer/easy_start/models_vit_eeg.py"
        spec = importlib.util.spec_from_file_location("native_steegformer", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        checkpoint = root / "expansion/steegformer-small.pth"
        with torch.serialization.safe_globals([argparse.Namespace]):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["model"]
        model = module.vit_small_patch16(num_classes=0, global_pool=False)
        # Parent timm adds image position embeddings unused by native EEG forward.
        # Remove that unused, randomly initialized parameter from the load contract.
        model.register_parameter("pos_embed", None)
        extra = sorted(set(state) - set(model.state_dict()))
        assert all(k.startswith(("decoder_", "dec_")) or k == "mask_token" for k in extra), extra
        model.load_state_dict({k: v for k, v in state.items() if k not in extra}, strict=True)
        mapping_path = root / "repos/STEEGFormer/pretrain/senloc_file/sen_chan_idx.pkl"
        mapping = pickle.loads(mapping_path.read_bytes())["channels_mapping"]
        channels = ("C3", "C4", "F3")
        adapter = STEEGFormerAdapter(model, channels=channels, channel_mapping=mapping)
        shape, fs = (3, 48, 16), 128
        metadata = dict(
            strict_backbone_load=True,
            excluded_pretraining_keys=extra,
            unused_parent_parameter="pos_embed removed; native EEG forward does not use it",
            channel_mapping_sha256=sha(mapping_path),
        )
    else:
        source_root = root / "repos/DIVER-1"
        sys.path.insert(0, str(source_root))
        from models.diver import DIVER
        from mup import set_base_shapes

        checkpoint = root / "expansion/diver1-eeg.pt"
        # The author's LFS pointer provides an independent expected digest.
        expected = "dbfa48289989475a52719b1bcb868e62a82877120ac8272ca7dab772e407b891"
        assert sha(checkpoint) == expected
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["module"]  # Official release uses the documented DeepSpeed format.
        width = state["token_manager.special_tokens.N_token.param"].shape[-1]
        depth = len(
            {int(k.split(".")[3]) for k in state if k.startswith("encoder.encoder.layers.")}
        )
        model = DIVER(d_model=width, e_layer=depth, mup=True, patch_size=500)
        shapes = (
            root
            / "expansion/diver-mup-shapes"
            / f"EEGLens_DIVER_EEG_dmodel_{width}_layer{depth}_256_512.bsh"
        )
        set_base_shapes(model, str(shapes))
        model.load_state_dict(state, strict=True)
        channels = ("C3", "C4", "F3")
        positions = torch.tensor([[-0.06, 0.0, 0.06], [0.06, 0.0, 0.06], [-0.04, 0.05, 0.07]])
        adapter = DIVERAdapter(model, channels=channels, positions=positions)
        shape, fs = (3, 5, 500), 500
        source = Path(inspect.getfile(DIVER))
        metadata = dict(
            strict_full_load=True,
            width=width,
            depth=depth,
            mup=True,
            official_lfs_digest_matched=True,
            geometry="synthetic xyz, EEG only",
        )
    model.eval().requires_grad_(False)
    model._validation_component = metadata
    return model, adapter, shape, channels, fs


def build(root, name, signaljepa_source=None):
    if name in {"eegpt", "biot", "biot-prest", "bendr"}:
        return encoders(root, name)
    if name in {"diver", "steegformer"}:
        return spatial(root, name)
    if name == "signaljepa":
        return signaljepa(root, signaljepa_source)
    return {
        "bendr-context": bendr_context,
        "brainomni": brainomni,
        "csbrain": csbrain,
        "neurorvq": neurorvq,
    }[name](root)


def native_call(model, adapter, name, batch):
    """Direct native call; never uses adapter.forward or adapter layout conversion."""
    b = batch.data.shape[0]
    if name == "eegpt":
        return model(batch.data.flatten(2), chan_ids=model.prepare_chan_ids(batch.channels))
    if name == "brainomni":
        return model.encode(
            batch.data.flatten(2),
            adapter.positions.unsqueeze(0).expand(b, -1, -1),
            adapter.sensor_types.unsqueeze(0).expand(b, -1),
        )
    if name == "csbrain":
        return model(batch.data)
    if name == "neurorvq":
        from inference.modules.NeuroRVQ_EEG_tokenizer_inference_modules import (
            ch_names_global,
            create_embedding_ix,
        )

        t, s = create_embedding_ix(
            batch.data.shape[2], 256, [c.lower().encode() for c in batch.channels], ch_names_global
        )
        return torch.stack(
            model(batch.data, t.expand(b, -1), s.expand(b, -1), return_patch_tokens=True)[:4], dim=2
        )
    if name == "diver":
        info = [
            {"xyz_id": adapter.positions.clone(), "modality": "EEG", "coord_subtype": None}
            for _ in range(b)
        ]
        return model(batch.data, data_info_list=info, use_mask=False)["y"]
    if name == "steegformer":
        return model.forward_features(
            batch.data.flatten(2), torch.tensor(adapter.channel_indices).expand(b, -1)
        )
    return model(batch.data.flatten(2))
