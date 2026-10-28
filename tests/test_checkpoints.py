"""Optional official-file integration tests; never download during collection."""

import os
from pathlib import Path

import pytest
import torch

from eegfmlens import Ablation, Replacement, Selection, SignalBatch, load_cbramod, load_labram
from eegfmlens.errors import ValidationError


@pytest.mark.integration
@pytest.mark.parametrize("name,loader", [("cbramod", load_cbramod), ("labram", load_labram)])
def test_official_checkpoint_strict_load_and_identity(name, loader, request):
    path = os.environ.get(f"EEGFMLENS_{name.upper()}_CHECKPOINT")
    if not path:
        pytest.skip(f"Set EEGFMLENS_{name.upper()}_CHECKPOINT for checkpoint integration")
    assert Path(path).is_file()
    torch.set_num_threads(2)
    factory = request.getfixturevalue("native_" + name)
    factory = factory.CBraMod if name == "cbramod" else factory.labram_base_patch200_200
    lens = loader(path, model_factory=factory)
    batch = SignalBatch(
        torch.randn(1, 3, 2, 200), ("trial",), ("C3", "CZ", "C4"), 200, "synthetic-checkpoint-test"
    )
    result = lens.run_with_cache(batch)
    for site in result.cache:
        identity = lens.run_with_interventions(
            batch, interventions=[Replacement(site, result.cache[site])]
        )
        torch.testing.assert_close(result.output, identity.output, atol=1e-6, rtol=1e-5)
        effect = lens.run_with_interventions(
            batch, interventions=[Ablation(site, Selection(sensors=("C3",), patches=(1,)))]
        )
        assert not torch.equal(result.output, effect.output), site
    assert lens.manifest["missing_keys"] == lens.manifest["unexpected_keys"] == []
    with pytest.raises(ValidationError, match="SHA256"):
        loader(path, model_factory=factory, expected_sha256="wrong-hash")
