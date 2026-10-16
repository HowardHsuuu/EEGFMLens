"""Native masked-decoder parity and physical-token oracle for ST-EEGFormer."""

import importlib.util
import os
import pickle
import sys
from pathlib import Path

import pytest
import torch

from eegfmlens import Ablation, EEGLens, Selection, SignalBatch, STEEGFormerMAEAdapter
from eegfmlens.errors import ValidationError

pytestmark = pytest.mark.native


def _upstream():
    source = os.environ.get("EEGFMLENS_STEEGFORMER_SOURCE")
    if not source:
        pytest.skip("Set EEGFMLENS_STEEGFORMER_SOURCE for the native MAE test")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from native_sources import verified_source

    root = verified_source("steegformer", source)
    path = root / "pretrain/models_mae_eeg.py"
    spec = importlib.util.spec_from_file_location("eegfmlens_native_steegformer_mae", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load native ST-EEGFormer MAE")
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    with (root / "pretrain/senloc_file/sen_chan_idx.pkl").open("rb") as stream:
        mapping = pickle.load(stream)["channels_mapping"]
    return upstream, {channel.upper(): index for channel, index in mapping.items()}


def test_native_mae_forward_and_selected_decoder_token():
    upstream, mapping = _upstream()
    torch.manual_seed(31)
    torch.set_num_threads(2)
    model = upstream.mae_vit_small_patch16(norm_pix_loss=False).eval()
    channels = ("C3", "CZ", "C4")
    batch = SignalBatch(torch.randn(1, 3, 4, 16), ("trial-a",), channels, 128, "native-mae-fixture")
    mask = torch.zeros(batch.data.shape[:-1], dtype=torch.bool)
    mask[:, 0, 1] = True
    adapter = STEEGFormerMAEAdapter(model, channels=channels, channel_mapping=mapping)
    lens = EEGLens(model, adapter)

    with torch.no_grad():
        observed = lens.run_with_cache(
            batch,
            sites=(
                "decoder.0.attention_input",
                "decoder.1.attention_input",
                "decoder.0.output",
            ),
            mask=mask.tolist(),
        )
        flat = mask.transpose(1, 2).flatten(1)
        visible = torch.where(~flat[0])[0]
        hidden = torch.where(flat[0])[0]
        keep = visible.unsqueeze(0)
        restore = torch.argsort(torch.cat((visible, hidden))).unsqueeze(0)
        channel_ids = torch.tensor([mapping[channel] for channel in channels]).unsqueeze(0)
        native_prediction = model.forward_demo(
            batch.data.flatten(2),
            channel_ids,
            ids_keep=keep,
            mask=flat.float(),
            ids_restore=restore,
        )[1]
        expected = native_prediction.reshape(1, 4, 3, 16).transpose(1, 2)
        torch.testing.assert_close(observed.output, expected, atol=0, rtol=0)

        hidden = batch.data.clone()
        hidden[:, 0, 1] += 1000
        hidden_batch = SignalBatch(
            hidden, batch.trial_ids, batch.channels, 128, batch.preprocessing_id
        )
        hidden_run = lens.run_with_cache(hidden_batch, sites=(), mask=mask.tolist())
        torch.testing.assert_close(hidden_run.output, observed.output, atol=0, rtol=0)

        native_site = {}

        def capture(_module, _args, output):
            native_site["output"] = output.detach().clone()

        handle = model.decoder_blocks[0].register_forward_hook(capture)
        norm_handle = model.decoder_blocks[0].norm1.register_forward_hook(
            lambda _module, _args, output: native_site.__setitem__(
                "attention_input", output.detach().clone()
            )
        )
        late_norm_handle = model.decoder_blocks[1].norm1.register_forward_hook(
            lambda _module, _args, output: native_site.__setitem__(
                "late_attention_input", output.detach().clone()
            )
        )
        try:
            model.forward_demo(
                batch.data.flatten(2),
                channel_ids,
                ids_keep=keep,
                mask=flat.float(),
                ids_restore=restore,
            )
        finally:
            late_norm_handle.remove()
            norm_handle.remove()
            handle.remove()
        physical_index = 1 + 0 * 4 + 1
        native_index = 1 + 1 * 3 + 0
        torch.testing.assert_close(
            observed.cache["decoder.0.output"].tensor[:, physical_index],
            native_site["output"][:, native_index],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            observed.cache["decoder.0.attention_input"].tensor[:, physical_index],
            native_site["attention_input"][:, native_index],
            atol=0,
            rtol=0,
        )
        attention_input = observed.cache["decoder.0.attention_input"]
        restored_input = adapter.restore(
            attention_input.tensor,
            adapter.require("decoder.0.attention_input"),
            attention_input.native_shape,
        )
        torch.testing.assert_close(restored_input, native_site["attention_input"], atol=0, rtol=0)
        late_input = observed.cache["decoder.1.attention_input"]
        restored_late_input = adapter.restore(
            late_input.tensor,
            adapter.require("decoder.1.attention_input"),
            late_input.native_shape,
        )
        torch.testing.assert_close(
            restored_late_input, native_site["late_attention_input"], atol=0, rtol=0
        )

        selected = Selection(sensors=("C3",), patches=(1,))
        input_ablated = lens.run_with_interventions(
            batch,
            interventions=[Ablation("decoder.0.attention_input", selected)],
            mask=mask.tolist(),
        ).output

        def native_input_zero(_module, _args, output):
            altered = output.clone()
            altered[:, native_index] = 0
            return altered

        handle = model.decoder_blocks[0].norm1.register_forward_hook(native_input_zero)
        try:
            input_oracle = (
                model.forward_demo(
                    batch.data.flatten(2),
                    channel_ids,
                    ids_keep=keep,
                    mask=flat.float(),
                    ids_restore=restore,
                )[1]
                .reshape(1, 4, 3, 16)
                .transpose(1, 2)
            )
        finally:
            handle.remove()
        torch.testing.assert_close(input_ablated, input_oracle, atol=0, rtol=0)

        ablated = lens.run_with_interventions(
            batch,
            interventions=[Ablation("decoder.0.output", selected)],
            mask=mask.tolist(),
        ).output

        def native_zero(_module, _args, output):
            altered = output.clone()
            altered[:, native_index] = 0
            return altered

        handle = model.decoder_blocks[0].register_forward_hook(native_zero)
        try:
            oracle = (
                model.forward_demo(
                    batch.data.flatten(2),
                    channel_ids,
                    ids_keep=keep,
                    mask=flat.float(),
                    ids_restore=restore,
                )[1]
                .reshape(1, 4, 3, 16)
                .transpose(1, 2)
            )
        finally:
            handle.remove()
        torch.testing.assert_close(ablated, oracle, atol=0, rtol=0)
        assert not any(module._forward_hooks for module in model.modules())

    with pytest.raises(ValidationError, match="boolean B,C,P"):
        lens.run_with_cache(batch, mask=mask.int().tolist())
