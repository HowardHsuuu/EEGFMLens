"""Small published-native architecture tests; not checkpoint/cohort admission."""

import copy
import hashlib
import importlib
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch

from eegfmlens import (
    Ablation,
    EEGLens,
    LaBraMPretrainingAdapter,
    Replacement,
    Selection,
    SignalBatch,
    load_labram_pretraining,
)
from eegfmlens.errors import ValidationError

pytestmark = pytest.mark.native


@pytest.fixture
def native_masked(native_labram):
    # native_labram already verifies the pinned revision and clean source.
    root = Path(native_labram.__file__).parent
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("modeling_pretrain")
    finally:
        sys.path.remove(str(root))
    assert Path(module.__file__).parent == root
    torch.set_num_threads(1)
    torch.manual_seed(31)
    model = (
        module.NeuralTransformerForMEM(
            depth=2,
            vocab_size=7,
            init_values=0.1,
            qkv_bias=False,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            qk_norm=partial(torch.nn.LayerNorm, eps=1e-6),
        )
        .eval()
        .requires_grad_(False)
    )
    batch = SignalBatch(
        torch.randn(2, 3, 4, 200), ("one", "two"), ("C3", "CZ", "C4"), 200, "synthetic"
    )
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    mask[:, 0, 1] = True
    mask[:, 2, 3] = True
    return model, batch, mask


def test_both_native_training_arms_and_outer_head(native_masked):
    model, batch, mask = native_masked
    lens = EEGLens(model, LaBraMPretrainingAdapter(model))
    channels = lens.adapter.channel_indices(batch)
    with torch.inference_mode():
        original = model(batch.data, input_chans=channels, bool_masked_pos=mask.flatten(1))
    for actual_mask, expected in zip((mask, ~mask), original, strict=True):
        result = lens.run_with_cache(batch, mask=actual_mask.tolist())
        assert result.output.shape == (2, 12, 7)
        torch.testing.assert_close(result.output[actual_mask.flatten(1)], expected)
        norm = result.cache["normalization.output"].tensor[:, 1:]
        torch.testing.assert_close(result.output, model.lm_head(norm), atol=0, rtol=0)
        assert not torch.allclose(result.output, model.student.lm_head(norm))
        assert result.cache["prediction.output"].tensor.shape == result.output.shape
        assert result.cache["blocks.0.output"].tensor.shape == (2, 13, 200)


def test_physical_support_and_fault_cleanup(native_masked):
    model, batch, mask = native_masked
    lens = EEGLens(model, LaBraMPretrainingAdapter(model))
    baseline = lens.run_with_cache(batch, mask=mask.tolist())
    for site in (
        "embedding.output",
        "blocks.0.output",
        "normalization.output",
        "prediction.output",
    ):
        own = lens.run_with_interventions(
            batch, mask=mask.tolist(), interventions=[Replacement(site, baseline.cache[site])]
        )
        torch.testing.assert_close(own.output, baseline.output, atol=0, rtol=0)
        run = lens.run_with_interventions(
            batch,
            mask=mask.tolist(),
            sites=(site,),
            interventions=[Ablation(site, Selection(sensors=("C3",), patches=(1,)))],
        )
        index = 1 + int(site not in ("embedding.output", "prediction.output"))
        expected = baseline.cache[site].tensor.clone()
        expected[:, index] = 0
        torch.testing.assert_close(run.cache[site].tensor, expected, atol=0, rtol=0)
    with pytest.raises(ValidationError, match="execution configuration"):
        lens.run_with_interventions(
            batch,
            mask=(~mask).tolist(),
            interventions=[Replacement("blocks.0.output", baseline.cache["blocks.0.output"])],
        )
    assert all(not m._forward_hooks for m in model.modules())
    replay = lens.run_with_cache(batch, mask=mask.tolist())
    torch.testing.assert_close(replay.output, baseline.output, atol=0, rtol=0)


def test_native_token_mask_is_not_raw_hiding(native_masked):
    model, batch, mask = native_masked
    lens = EEGLens(model, LaBraMPretrainingAdapter(model))
    altered = batch.data.clone()
    altered[:, 0, 1] += 100
    before = lens.run_with_cache(batch, mask=mask.tolist(), sites=())
    after = lens.run_with_cache(replace(batch, data=altered), mask=mask.tolist(), sites=())
    assert not torch.equal(before.output[:, 1], after.output[:, 1])
    raw_occluded = batch.data.clone()
    raw_occluded[:, 0, 1] = 0
    altered[:, 0, 1] = 0
    first = lens.run_with_cache(replace(batch, data=raw_occluded), mask=mask.tolist(), sites=())
    second = lens.run_with_cache(replace(batch, data=altered), mask=mask.tolist(), sites=())
    torch.testing.assert_close(first.output, second.output, atol=0, rtol=0)
    assert lens.adapter.metadata()["raw_target_hiding_guaranteed"] is False


@pytest.mark.parametrize(
    "kwargs", [{}, {"mask": [True]}, {"mask": [[[1] * 4] * 3] * 2}, {"unexpected": True}]
)
def test_invalid_mask_rejected(native_masked, kwargs):
    model, batch, _ = native_masked
    lens = EEGLens(model, LaBraMPretrainingAdapter(model))
    with pytest.raises(ValidationError):
        lens.run_with_cache(batch, **kwargs)
    assert all(not m._forward_hooks for m in model.modules())


def test_complete_pretraining_load_and_native_parity(native_masked, tmp_path):
    model, batch, mask = native_masked
    path = tmp_path / "complete.pt"
    torch.save({"model": model.state_dict()}, path)
    lens = load_labram_pretraining(
        path,
        model_factory=lambda: copy.deepcopy(model),
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert lens.manifest["strict_loaded_tensor_count"] == len(model.state_dict())
    assert lens.manifest["training_run_attested"] is False
    for name, value in model.state_dict().items():
        torch.testing.assert_close(lens.model.state_dict()[name], value, atol=0, rtol=0)
    with torch.inference_mode():
        expected = model(
            batch.data,
            input_chans=lens.adapter.channel_indices(batch),
            bool_masked_pos=mask.flatten(1),
        )[0]
    result = lens.run_with_cache(batch, mask=mask.tolist(), sites=())
    # Native projects selected rows; the adapter projects every row. The float32
    # GEMM shape differs, so native parity is numerical, while weights above are exact.
    torch.testing.assert_close(result.output[mask.flatten(1)], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("problem", ["outer_head", "student_head", "extra", "shape"])
def test_full_pretraining_load_rejects_partial_or_changed_state(native_masked, tmp_path, problem):
    model, _, _ = native_masked
    state = model.state_dict()
    if problem == "outer_head":
        del state["lm_head.weight"]
    elif problem == "student_head":
        del state["student.lm_head.weight"]
    elif problem == "extra":
        state["unknown.weight"] = torch.ones(1)
    else:
        state["lm_head.weight"] = state["lm_head.weight"][:1]
    path = tmp_path / "changed.pt"
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError):
        load_labram_pretraining(
            path,
            model_factory=lambda: copy.deepcopy(model),
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_unused_scale_requires_explicit_known_hash_exception(native_masked, tmp_path, monkeypatch):
    from eegfmlens import pretraining_loading

    model, _, _ = native_masked
    state = model.state_dict()
    state["logit_scale"] = torch.tensor(2.0)
    path = tmp_path / "extra_scale.pt"
    torch.save({"model": state}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    kwargs = dict(model_factory=lambda: copy.deepcopy(model), expected_sha256=digest)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_labram_pretraining(path, **kwargs)
    with pytest.raises(ValidationError, match="exact official checkpoint"):
        load_labram_pretraining(path, **kwargs, acknowledge_unused_logit_scale=True)
    # Exercise the exception branch with synthetic bytes; not official admission.
    monkeypatch.setattr(pretraining_loading, "LABRAM_BASE_SHA256", digest)
    lens = load_labram_pretraining(path, **kwargs, acknowledge_unused_logit_scale=True)
    receipt = lens.manifest["omitted_checkpoint_keys"]
    assert len(receipt) == 1 and receipt[0]["key"] == "logit_scale"
    assert receipt[0]["applied_to_logits"] is False
    assert "student.lm_head.weight" in lens.model.state_dict()


@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_required_hash_rejected_before_factory(tmp_path, digest):
    path = tmp_path / "checkpoint.pt"
    torch.save({}, path)

    def forbidden_factory():
        pytest.fail("factory must not be called before verified checkpoint")

    with pytest.raises(ValidationError, match="SHA256"):
        load_labram_pretraining(path, model_factory=forbidden_factory, expected_sha256=digest)
