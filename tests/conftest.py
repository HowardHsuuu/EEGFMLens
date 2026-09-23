"""Native sources are opt-in; the core suite needs no upstream checkout."""

import os
import sys
from pathlib import Path

import pytest


def _native(name):
    source = os.environ.get(f"EEGFMLENS_{name.upper()}_SOURCE")
    if not source:
        pytest.fail(f"Set EEGFMLENS_{name.upper()}_SOURCE or deselect native/integration tests")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from native_sources import native_module

    return native_module(name, source)


@pytest.fixture(scope="session")
def native_cbramod():
    return _native("cbramod")


@pytest.fixture(scope="session")
def native_labram():
    return _native("labram")
