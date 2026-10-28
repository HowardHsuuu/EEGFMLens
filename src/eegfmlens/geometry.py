"""Explicit, checksum-bound electrode coordinates for native model inputs."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .errors import ValidationError


@dataclass(frozen=True)
class ElectrodeGeometry:
    """Immutable coordinate values and provenance; no inferred coordinate frame."""

    channels: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    unit: str
    coordinate_frame: str
    source_sha256: str

    def __post_init__(self):
        channels = tuple(self.channels)
        coordinates = tuple(tuple(float(value) for value in row) for row in self.coordinates)
        if (
            not channels
            or any(
                not isinstance(name, str) or not name or name.strip() != name for name in channels
            )
            or len({name.upper() for name in channels}) != len(channels)
            or len(coordinates) != len(channels)
            or any(
                len(row) != 3 or not all(math.isfinite(value) for value in row)
                for row in coordinates
            )
            or self.unit not in {"m", "cm", "mm"}
            or not isinstance(self.coordinate_frame, str)
            or not self.coordinate_frame.strip()
            or not isinstance(self.source_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256)
        ):
            raise ValidationError("Invalid explicit electrode geometry or provenance")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "coordinates", coordinates)

    def as_tensor(self, *, dtype=torch.float32, device=None) -> torch.Tensor:
        """Return a fresh tensor in the declared native unit, without transforms."""
        if dtype not in {torch.float32, torch.float64}:
            raise ValidationError("Electrode geometry requires float32 or float64")
        result = torch.tensor(self.coordinates, dtype=dtype, device=device)
        if not torch.isfinite(result).all():
            raise ValidationError("Electrode coordinates overflow the requested dtype")
        return result

    def metadata(self) -> dict:
        return {
            "channels": list(self.channels),
            "unit": self.unit,
            "coordinate_frame": self.coordinate_frame,
            "source_sha256": self.source_sha256,
            "format": "ASA ELC",
            "coordinate_transform": "none",
        }


def load_elc_geometry(
    path,
    *,
    channels,
    expected_sha256: str,
    unit: str,
    coordinate_frame: str,
) -> ElectrodeGeometry:
    """Select ordered electrodes from an explicit ASA ELC source.

    Labels are matched case-insensitively, without aliases. The caller must
    declare the expected source hash, native unit and coordinate frame. This
    parser neither transforms positions nor validates an anatomical frame or
    a checkpoint's choice of montage. MNE is not required to read the file.
    """
    blob = Path(path).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected_sha256:
        raise ValidationError("Electrode coordinate source SHA-256 mismatch")
    lines = [
        line.strip()
        for line in blob.decode("utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if lines.count("Positions") != 1 or lines.count("Labels") != 1:
        raise ValidationError("ELC requires one Positions and one Labels section")
    start, stop = lines.index("Positions"), lines.index("Labels")
    if stop <= start:
        raise ValidationError("ELC coordinate sections are out of order")
    units = [line.split()[1:] for line in lines[:start] if line.split()[0] == "UnitPosition"]
    counts = [
        line.split("=", 1)[1].strip()
        for line in lines[:start]
        if line.startswith("NumberPositions=")
    ]
    if units != [[unit]] or len(counts) != 1:
        raise ValidationError("ELC unit or position count is missing, duplicated or changed")
    try:
        count = int(counts[0])
        coordinates = tuple(
            tuple(float(value) for value in line.split()) for line in lines[start + 1 : stop]
        )
    except ValueError as error:
        raise ValidationError("Invalid numeric ELC coordinates or count") from error
    labels = lines[stop + 1 :]
    normalized = [label.upper() for label in labels]
    if (
        count <= 0
        or count != len(coordinates)
        or count != len(labels)
        or len(set(normalized)) != count
        or any(len(row) != 3 or not all(math.isfinite(v) for v in row) for row in coordinates)
    ):
        raise ValidationError("Invalid ELC coordinate/label inventory")
    requested = tuple(channels)
    if any(not isinstance(name, str) or name.upper() not in normalized for name in requested):
        raise ValidationError("Requested electrode is missing from the ELC source")
    selected = tuple(
        (row[0], row[1], row[2])
        for row in (coordinates[normalized.index(name.upper())] for name in requested)
    )
    return ElectrodeGeometry(requested, selected, unit, coordinate_frame, digest)
