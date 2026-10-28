"""Known coordinate units/order and malformed-source checks; no model run."""

import hashlib

import pytest
import torch

from eegfmlens import ElectrodeGeometry, load_elc_geometry
from eegfmlens.errors import ValidationError

SOURCE = "# ASA electrode file\nReferenceLabel avg\nUnitPosition mm\nNumberPositions= 3\nPositions\n-20 4 80\n20 4 80\n0 10 90\nLabels\nC3\nC4\nCz\n"


def load(tmp_path, text=SOURCE, **kwargs):
    path = tmp_path / "coordinates.elc"
    path.write_text(text)
    arguments = dict(
        channels=("C4", "C3"),
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        unit="mm",
        coordinate_frame="declared_native_frame",
    )
    arguments.update(kwargs)
    return load_elc_geometry(path, **arguments)


def test_native_units_order_and_fresh_tensors(tmp_path):
    geometry = load(tmp_path, channels=("cz", "C3"))
    assert geometry.channels == ("cz", "C3")
    torch.testing.assert_close(
        geometry.as_tensor(), torch.tensor([[0.0, 10.0, 90.0], [-20.0, 4.0, 80.0]]), atol=0, rtol=0
    )
    changed = geometry.as_tensor()
    changed.zero_()
    assert geometry.as_tensor()[0, 2] == 90
    assert geometry.as_tensor(dtype=torch.float64).dtype == torch.float64
    assert geometry.metadata()["unit"] == "mm"
    assert geometry.metadata()["coordinate_transform"] == "none"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unit": "m"},
        {"expected_sha256": "0" * 64},
        {"channels": ("C3", "c3")},
        {"channels": ("missing",)},
        {"channels": ()},
        {"coordinate_frame": ""},
    ],
)
def test_explicit_contract_rejections(tmp_path, kwargs):
    with pytest.raises(ValidationError):
        load(tmp_path, **kwargs)


@pytest.mark.parametrize(
    "text",
    [
        SOURCE.replace("NumberPositions= 3", "NumberPositions= 2"),
        SOURCE.replace("Cz\n", "c3\n"),
        SOURCE.replace("0 10 90", "0 nan 90"),
        SOURCE.replace("0 10 90", "0 10"),
        SOURCE.replace("UnitPosition mm", "UnitPosition mm\nUnitPosition mm"),
        SOURCE.replace("Positions\n", "Positions\nPositions\n"),
    ],
)
def test_invalid_full_inventory_rejected(tmp_path, text):
    with pytest.raises(ValidationError):
        load(tmp_path, text)


def test_geometry_constructor_copies_values_and_refuses_low_precision():
    values = [[1.0, 2.0, 3.0]]
    geometry = ElectrodeGeometry(["C3"], values, "mm", "native", "a" * 64)
    values[0][0] = 100
    assert geometry.coordinates[0][0] == 1
    with pytest.raises(ValidationError):
        geometry.as_tensor(dtype=torch.float16)
    with pytest.raises(ValidationError):
        ElectrodeGeometry(("C3",), ((1e40, 0.0, 0.0),), "mm", "native", "a" * 64).as_tensor()
