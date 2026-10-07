"""Shared pytest fixtures and options for the process_petsys suite (spec 006).

Markers are registered in ``pyproject.toml``; the pytester plugin is loaded there
with ``-p pytester`` for the infrastructure tests. Shared non-fixture code lives
in ``tests/helpers.py``.
"""

import os
import sys
from pathlib import Path

import pytest

DATA_ENV = "PETSYS_DATA_DIR"


def pytest_addoption(parser):
    parser.addoption("--fr", action="append", default=[], metavar="ID",
                     help="run only tests citing requirement ID with @pytest.mark.fr (repeatable)")


def pytest_collection_modifyitems(config, items):
    wanted = set(config.getoption("fr"))
    if wanted:
        selected, deselected = [], []
        for item in items:
            cited = {i for mark in item.iter_markers("fr") for i in mark.args}
            (selected if cited & wanted else deselected).append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = selected

    if sys.platform != "linux":
        skip = pytest.mark.skip(reason=f"linux: runs only on Linux (Cornell PC), not {sys.platform}")
        for item in items:
            if item.get_closest_marker("linux"):
                item.add_marker(skip)


@pytest.fixture
def tk_root():
    """A withdrawn Tk root, destroyed after the test; skips without a display."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"gui: no display available ({exc})")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def real_data_dir():
    """Root of acquisitions/calibrations from PETSYS_DATA_DIR; skips when unavailable."""
    value = os.environ.get(DATA_ENV)
    if not value:
        pytest.skip(f"real_data: {DATA_ENV} is not set")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"real_data: {DATA_ENV}={value} is not a directory")
    return path


@pytest.fixture
def real_data_file(real_data_dir):
    """Return a finder for files under PETSYS_DATA_DIR that skips, naming the file, when one is missing."""
    def find(relative):
        path = real_data_dir / relative
        if not path.is_file():
            pytest.skip(f"real_data: missing {relative} under {DATA_ENV}={real_data_dir}")
        return path
    return find
