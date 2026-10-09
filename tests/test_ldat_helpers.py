"""tests/ldat_helpers.py smoke tests (spec 007 FR-6, T18): tracked configs, one fixture per system."""

import os
import subprocess
import sys

import pytest

from helpers import REPO
from ldat_helpers import CONFIGS, RECOVERY_CASES, build, fixture_files, side, write_ldat, write_pairs
from src.ldat_inspector.engine import Settings, load_setup
from src.read_compact import read_binary_file

pytestmark = pytest.mark.fr("007-FR-6")
SYSTEMS = ("IMAS", "CORNELL")


def tracked(path):
    relative = path.resolve().relative_to(REPO).as_posix()
    result = subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=REPO, capture_output=True)
    return result.returncode == 0


def test_imports_without_tk():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO), str(REPO / "tests")]))
    code = "import sys, ldat_helpers; sys.exit('tkinter' in sys.modules or 'customtkinter' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("system", SYSTEMS)
def test_configs_and_their_maps_are_tracked(system):
    setup = load_setup(Settings(str(CONFIGS[system]), "", system, calibrated=False))
    assert tracked(CONFIGS[system]) and tracked(REPO / setup.config["map_file"])


@pytest.mark.parametrize("system", SYSTEMS)
def test_fixture_files_write_a_readable_calibrated_fixture(tmp_path, system):
    config, calibration, channels = fixture_files(tmp_path, system)
    setup = load_setup(Settings(str(config), str(calibration), system))
    assert [setup.channel_modules[part["t"]][0] for part in channels] == [0, 1]
    path = tmp_path / "pairs.ldat"
    write_pairs(path, channels, 3)
    records = list(read_binary_file(str(path)))
    assert len(records) == 3
    assert [[hit[2] for hit in det] for det in records[0]] == [[part["t"], part["e"]] for part in channels]


def test_recovery_sides_and_tail(tmp_path):
    setup = load_setup(Settings(str(CONFIGS["CORNELL"]), "", "CORNELL", calibrated=False))
    full = [30.0, 20.0, 10.0, 5.0]
    pairs = [(side(setup, 0, 0, case[2], full, 10 ** 12), side(setup, 1, 0, {3: 12.0, 2: 5.0}, full, 10 ** 12 + 99))
             for case in RECOVERY_CASES]
    path = write_ldat(tmp_path / "recovery.ldat", pairs, tail=b"\x04")
    assert path.stat().st_size == sum(2 + 16 * (len(a) + len(b)) for a, b in pairs) + 1
    assert all(setup.channel_modules[hit[2]] == (0, 0) for hit in pairs[0][0])


@pytest.mark.parametrize("system", SYSTEMS)
def test_build_assigns_every_case(system):
    dataset, assignment, truth, _ = build(system)
    assert set(assignment.values()) == set(truth) <= set(dataset.expected_time)
    assert ("half-populated" in assignment) == (system == "CORNELL")
