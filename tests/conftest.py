"""Shared pytest fixtures and options for the process_petsys suite (specs 006, 007).

Markers are registered in ``pyproject.toml``; the pytester plugin is loaded there
with ``-p pytester`` for the infrastructure tests. Shared non-fixture code lives
in ``tests/helpers.py``.
"""

import os
import sys
from pathlib import Path

import pytest

DATA_ENV = "PETSYS_DATA_DIR"
CAL_ENV = "PETSYS_CAL_DIR"

# OpenBLAS (numpy, scipy) commits a buffer per thread: ~1.5 GB per process with 24 threads. 24 parallel
# workers then reach the Windows commit limit, and a child process spawned meanwhile fails to start
# (OSError: [Errno 22]). One thread per test process, inherited by workers and children (spec 007 T20);
# set before numpy loads, and an explicit environment value wins.
BLAS_THREADS = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
for _name, _value in BLAS_THREADS.items():
    os.environ.setdefault(_name, _value)


def pytest_addoption(parser):
    parser.addoption("--fr", action="append", default=[], metavar="ID",
                     help="run only tests citing requirement ID with @pytest.mark.fr (repeatable)")
    parser.addoption("--slow-limit", type=float, default=None, metavar="SECONDS",
                     help="fail a passing test not marked slow whose call takes SECONDS or more")


def pytest_configure(config):
    # Parallel workers compete for the CPU and stretch test durations, so slow is measured serially.
    if config.getoption("slow_limit") is not None and getattr(config.option, "numprocesses", None):
        raise pytest.UsageError("--slow-limit measures serial durations; add -n 0")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    limit = item.config.getoption("slow_limit")
    if (limit is not None and report.when == "call" and report.passed
            and report.duration >= limit and not item.get_closest_marker("slow")):
        report.outcome = "failed"
        report.longrepr = (f"slow-limit: took {report.duration:.2f} s >= {limit:g} s "
                           f"without @pytest.mark.slow")


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


def _env_dir(env):
    """Directory named by environment variable ``env``; skips, naming it, when unavailable."""
    value = os.environ.get(env)
    if not value:
        pytest.skip(f"real_data: {env} is not set")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"real_data: {env}={value} is not a directory")
    return path


def _file_finder(root, env):
    """Return a finder for files under ``root`` that skips, naming the file, when one is missing."""
    def find(relative):
        path = root / relative
        if not path.is_file():
            pytest.skip(f"real_data: missing {relative} under {env}={root}")
        return path
    return find


@pytest.fixture
def real_data_dir():
    """Root of acquisitions from PETSYS_DATA_DIR; skips when unavailable."""
    return _env_dir(DATA_ENV)


@pytest.fixture
def real_data_file(real_data_dir):
    """Finder for files under PETSYS_DATA_DIR; skips, naming the file, when one is missing."""
    return _file_finder(real_data_dir, DATA_ENV)


@pytest.fixture
def real_cal_dir():
    """Root of calibration files from PETSYS_CAL_DIR; skips when unavailable."""
    return _env_dir(CAL_ENV)


@pytest.fixture
def real_cal_file(real_cal_dir):
    """Finder for files under PETSYS_CAL_DIR; skips, naming the file, when one is missing."""
    return _file_finder(real_cal_dir, CAL_ENV)
