import json
from collections import namedtuple

import pytest
import torch
from torch import nn

from eegfmlens import (
    ActivationSite,
    Adapter,
    EEGLens,
    SignalBatch,
    SubspaceAblation,
    load_run,
    save_run,
)
from eegfmlens.errors import ValidationError


def setup():
    model = nn.Sequential(nn.Identity()).eval()
    lens = EEGLens(model, Adapter([ActivationSite("output", "0")]))
    batch = SignalBatch(torch.tensor([[[[3.0, 4.0, 5.0]]]]), ("a",), ("C3",), 200, "fixture")
    return lens, batch


def test_subspace_removes_selected_direction_and_preserves_complement():
    lens, batch = setup()
    patch = SubspaceAblation(
        "output", torch.tensor([[1.0], [0.0], [0.0]]), torch.tensor([1.0, 0.0, 0.0])
    )
    result = lens.run_with_interventions(batch, interventions=[patch])
    torch.testing.assert_close(result.output, torch.tensor([[[[1.0, 4.0, 5.0]]]]))
    with pytest.raises(ValidationError, match="orthonormal"):
        lens.run_with_interventions(
            batch, interventions=[SubspaceAblation("output", patch.basis * 2, patch.center)]
        )


def test_export_roundtrip_and_integrity(tmp_path):
    lens, batch = setup()
    run = lens.run_with_cache(batch)
    target = save_run(run, tmp_path / "run")
    loaded = load_run(target)
    torch.testing.assert_close(loaded.output, run.output)
    torch.testing.assert_close(loaded.cache["output"].tensor, run.cache["output"].tensor)
    assert loaded.cache["output"].trial_ids == ("a",)
    with pytest.raises(FileExistsError):
        save_run(run, target)
    with (target / "tensors.pt").open("ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValidationError, match="checksum"):
        load_run(target)


def test_failed_export_leaves_no_partial_bundle(tmp_path):
    lens, batch = setup()
    run = lens.run_with_cache(batch)
    run.metadata["unsupported"] = object()
    with pytest.raises(TypeError):
        save_run(run, tmp_path / "bad")
    assert list(tmp_path.iterdir()) == []


def test_unknown_schema_rejected(tmp_path):
    lens, batch = setup()
    path = save_run(lens.run_with_cache(batch), tmp_path / "run")
    doc = json.loads((path / "run.json").read_text())
    doc["schema_version"] = 999
    (path / "run.json").write_text(json.dumps(doc))
    with pytest.raises(ValidationError, match="schema"):
        load_run(path)


def test_namedtuple_and_list_subclass_export_as_plain_containers(tmp_path):
    NativeOutput = namedtuple("NativeOutput", "features details")

    class NativeList(list):
        pass

    lens, batch = setup()
    run = lens.run_with_cache(batch)
    original = run.output.clone()
    run.output = NativeOutput(run.output, NativeList([{"score": 2.0}, None]))
    path = save_run(run, tmp_path / "structured")
    # The bundle must remain loadable without importing either native class.
    loaded = load_run(path)
    assert type(loaded.output) is tuple
    assert type(loaded.output[1]) is list
    torch.testing.assert_close(loaded.output[0], original, atol=0, rtol=0)
    assert loaded.output[1] == [{"score": 2.0}, None]
    assert isinstance(run.output, NativeOutput)
