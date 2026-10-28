"""Small native tokenizer tests; no checkpoint or cohort admission."""

import copy
import hashlib
import importlib
import sys
from pathlib import Path

import pytest
import torch

from eegfmlens import EEGLens, LaBraMTokenizerAdapter, SignalBatch, load_labram_tokenizer
from eegfmlens.errors import ValidationError

pytestmark = pytest.mark.native


@pytest.fixture
def native_tokenizer(native_labram):
    root = Path(native_labram.__file__).parent
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("modeling_vqnsp")
    finally:
        sys.path.remove(str(root))
    assert Path(module.__file__).parent == root
    torch.set_num_threads(1)
    torch.manual_seed(35)
    encoder = module.get_model_default_params()
    encoder.update(depth=1, EEG_size=800, num_classes=0)
    decoder = module.get_model_default_params()
    decoder.update(depth=1, EEG_size=4, patch_size=1, in_chans=8, num_classes=0)
    model = (
        module.VQNSP(encoder, decoder, n_embed=7, embed_dim=8, quantize_kmeans_init=False)
        .eval()
        .requires_grad_(False)
    )
    batch = SignalBatch(
        torch.randn(2, 3, 4, 200), ("one", "two"), ("C3", "CZ", "C4"), 200, "synthetic"
    )
    return model, batch


def snapshot(model):
    return {name: value.clone() for name, value in model.state_dict().items()}


def assert_same(model, before):
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_native_codes_usage_update_and_persistent_state(native_tokenizer):
    model, batch = native_tokenizer
    adapter = LaBraMTokenizerAdapter(model)
    initial = snapshot(model)
    with torch.inference_mode():
        expected = model.get_codebook_indices(
            batch.data, input_chans=adapter.channel_indices(batch)
        )
    assert not torch.equal(model.quantize.cluster_size, initial["quantize.cluster_size"])
    # Restore before creating the wrapper; no post-wrap copy/version suppression.
    model.load_state_dict(initial, strict=True)
    lens = EEGLens(model, adapter)
    counter = model.quantize.cluster_size
    version = counter._version
    for _ in range(2):
        result = lens.run_with_cache(batch)
        torch.testing.assert_close(result.output, expected, atol=0, rtol=0)
        assert result.output.shape == (2, 12)
        assert result.output.dtype == torch.long
        projection = result.cache["code_projection.output"].tensor.double()
        normalized = projection / torch.linalg.vector_norm(projection, dim=-1, keepdim=True)
        weights = model.quantize.embedding.weight.double()
        distances = (
            (normalized**2).sum(-1, keepdim=True)
            + (weights**2).sum(-1)
            - 2 * normalized @ weights.T
        )
        torch.testing.assert_close(result.output, distances.argmin(-1), atol=0, rtol=0)
        assert model.quantize.cluster_size is counter
        assert counter._version == version
        assert_same(model, initial)
        lens.assert_model_unchanged()


def test_counter_restored_after_native_exception(native_tokenizer):
    model, batch = native_tokenizer
    lens = EEGLens(model, LaBraMTokenizerAdapter(model))
    before = snapshot(model)
    counter = model.quantize.cluster_size

    def fail_after_counter_update(module, args, output):
        raise RuntimeError("injected after native update")

    handle = model.quantize.register_forward_hook(fail_after_counter_update)
    try:
        with pytest.raises(RuntimeError, match="injected after native update"):
            lens.run_with_cache(batch)
    finally:
        handle.remove()
    assert model.quantize.cluster_size is counter
    assert_same(model, before)
    assert all(not m._forward_hooks for m in model.modules())
    lens.assert_model_unchanged()
    lens.run_with_cache(batch)


@pytest.mark.parametrize("problem", ["uninitialized", "training", "counter", "options"])
def test_unsafe_tokenizer_rejected(native_tokenizer, problem):
    model, batch = native_tokenizer
    if problem == "uninitialized":
        model.quantize.embedding.initted.zero_()
    elif problem == "training":
        model.encoder.train()
    elif problem == "counter":
        del model.quantize.cluster_size
    with pytest.raises(ValidationError):
        adapter = LaBraMTokenizerAdapter(model)
        EEGLens(model, adapter).run_with_cache(batch, unsupported=True)


def test_unrelated_buffer_mutation_still_rejected(native_tokenizer):
    model, batch = native_tokenizer
    lens = EEGLens(model, LaBraMTokenizerAdapter(model))
    model.quantize.embedding.initted.add_(0)
    with pytest.raises(ValidationError, match="state changed"):
        lens.run_with_cache(batch)


def test_tokenizer_strict_load_roundtrip_and_corruption(native_tokenizer, tmp_path):
    model, batch = native_tokenizer
    path = tmp_path / "tokenizer.pt"
    torch.save({"model": model.state_dict()}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lens = load_labram_tokenizer(
        path, model_factory=lambda: copy.deepcopy(model), expected_sha256=digest
    )
    assert lens.manifest["strict_loaded_tensor_count"] == len(model.state_dict())
    assert lens.manifest["omitted_checkpoint_keys"] == []
    result = lens.run_with_cache(batch, sites=())
    with torch.inference_mode():
        expected = model.get_codebook_indices(
            batch.data, input_chans=lens.adapter.channel_indices(batch)
        )
    torch.testing.assert_close(result.output, expected, atol=0, rtol=0)
    state = model.state_dict()
    del state["quantize.embedding.weight"]
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_labram_tokenizer(
            path,
            model_factory=lambda: copy.deepcopy(model),
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
