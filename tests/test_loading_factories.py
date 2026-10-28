"""External construction must not weaken integrity or strict checkpoint loading."""

import pytest
import torch
from torch import nn

from eegfmlens import load_cbramod, load_labram
from eegfmlens.errors import ValidationError


@pytest.mark.parametrize("loader", [load_cbramod, load_labram])
def test_bad_hash_rejected_before_calling_external_code(loader, tmp_path):
    checkpoint = tmp_path / "weights.pth"
    torch.save({}, checkpoint)

    def forbidden(**kwargs):
        raise AssertionError("Must not construct before verifying checkpoint")

    with pytest.raises(ValidationError, match="SHA256"):
        loader(checkpoint, model_factory=forbidden, expected_sha256="wrong")


@pytest.mark.parametrize("loader", [load_cbramod, load_labram])
def test_external_factory_does_not_allow_missing_weights(loader, tmp_path):
    checkpoint = tmp_path / "weights.pth"
    torch.save({}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        loader(checkpoint, model_factory=lambda **kwargs: nn.Linear(2, 2))


@pytest.mark.parametrize("loader", [load_cbramod, load_labram])
def test_non_callable_factory_rejected(loader, tmp_path):
    with pytest.raises(ValidationError, match="callable"):
        loader(tmp_path / "unused", model_factory=None)
