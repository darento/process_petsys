"""Suite infrastructure (specs 006, 007): markers, --fr selection, real-data skips, platform
skips, default run without slow and in parallel workers, --slow-limit.

Each check runs a throwaway project in a pytester subprocess with copies of the real
``pyproject.toml`` and ``tests/conftest.py``, so it exercises the actual configuration.
"""

import ast
import sys
from types import SimpleNamespace

import pytest

from helpers import REPO

CONFTEST = REPO / "tests" / "conftest.py"


@pytest.fixture
def project(pytester):
    """Pytester project with the suite's config; ``project(source)`` writes tests/test_inner.py.

    Its ``runpytest_subprocess`` runs serially (``-n 0``), because pytest-xdist leaves the
    deselected count out of the summary; ``parallel`` runs with the configured workers.
    """
    pytester.makefile(".toml", pyproject=(REPO / "pyproject.toml").read_text(encoding="utf-8"))
    tests = pytester.mkdir("tests")
    (tests / "conftest.py").write_text(CONFTEST.read_text(encoding="utf-8"), encoding="utf-8")

    def write(source):
        (tests / "test_inner.py").write_text(source, encoding="utf-8")
        return SimpleNamespace(runpytest_subprocess=lambda *args: pytester.runpytest_subprocess(*args, "-n", "0"),
                               parallel=pytester.runpytest_subprocess)
    return write


FR_TESTS = """
import pytest

@pytest.mark.fr("006-FR-5")
def test_a(): pass

@pytest.mark.fr("006-FR-5", "bug-x")
def test_b(): pass

@pytest.mark.fr("bug-x")
def test_c(): pass

def test_d(): pass
"""


@pytest.mark.fr("006-FR-5")
@pytest.mark.parametrize("fr_id, expected", [("006-FR-5", {"test_a", "test_b"}),
                                             ("bug-x", {"test_b", "test_c"})])
def test_fr_selects_exactly_citing_tests(project, fr_id, expected):
    result = project(FR_TESTS).runpytest_subprocess("--fr", fr_id, "-v")
    result.assert_outcomes(passed=2, deselected=2)
    ran = {name for name in ("test_a", "test_b", "test_c", "test_d")
           if any(f"::{name} PASSED" in line for line in result.outlines)}
    assert ran == expected


@pytest.mark.fr("006-FR-5")
def test_fr_without_option_runs_everything(project):
    project(FR_TESTS).runpytest_subprocess().assert_outcomes(passed=4)


@pytest.mark.fr("006-FR-1", "006-FR-3")
def test_unregistered_marker_is_an_error(project):
    result = project("import pytest\n\n@pytest.mark.unknownmark\ndef test_x(): pass\n"
                     ).runpytest_subprocess()
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*'unknownmark' not found in `markers` configuration option*"])


REAL_DATA_TESTS = """
import pytest

@pytest.mark.real_data
def test_needs_file(real_data_file):
    assert real_data_file("acq/run.ldat").stat().st_size == 3

def test_plain(): pass
"""


@pytest.mark.fr("006-FR-3")
def test_default_run_deselects_real_data(project, monkeypatch):
    monkeypatch.delenv("PETSYS_DATA_DIR", raising=False)
    project(REAL_DATA_TESTS).runpytest_subprocess().assert_outcomes(passed=1, deselected=1)


@pytest.mark.fr("006-FR-4")
def test_real_data_skips_naming_unset_variable(project, monkeypatch):
    monkeypatch.delenv("PETSYS_DATA_DIR", raising=False)
    result = project(REAL_DATA_TESTS).runpytest_subprocess("-m", "real_data", "-rs")
    result.assert_outcomes(skipped=1, deselected=1)
    result.stdout.fnmatch_lines(["*SKIPPED*real_data: PETSYS_DATA_DIR is not set*"])


@pytest.mark.fr("006-FR-4")
def test_real_data_skips_naming_missing_file(project, monkeypatch, tmp_path):
    monkeypatch.setenv("PETSYS_DATA_DIR", str(tmp_path))
    result = project(REAL_DATA_TESTS).runpytest_subprocess("-m", "real_data", "-rs")
    result.assert_outcomes(skipped=1, deselected=1)
    result.stdout.fnmatch_lines(["*SKIPPED*real_data: missing acq/run.ldat under PETSYS_DATA_DIR=*"])


@pytest.mark.fr("006-FR-4")
def test_real_data_runs_when_file_present(project, monkeypatch, tmp_path):
    (tmp_path / "acq").mkdir()
    (tmp_path / "acq" / "run.ldat").write_bytes(b"abc")
    monkeypatch.setenv("PETSYS_DATA_DIR", str(tmp_path))
    project(REAL_DATA_TESTS).runpytest_subprocess("-m", "real_data").assert_outcomes(
        passed=1, deselected=1)


REAL_CAL_TESTS = """
import pytest

@pytest.mark.real_data
def test_needs_cal(real_cal_file):
    assert real_cal_file("jan/slab.encal").stat().st_size == 3
"""


@pytest.mark.fr("007-FR-4")
def test_real_cal_skips_naming_unset_variable(project, monkeypatch):
    monkeypatch.delenv("PETSYS_CAL_DIR", raising=False)
    result = project(REAL_CAL_TESTS).runpytest_subprocess("-m", "real_data", "-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*SKIPPED*real_data: PETSYS_CAL_DIR is not set*"])


@pytest.mark.fr("007-FR-4")
def test_real_cal_skips_naming_missing_file(project, monkeypatch, tmp_path):
    monkeypatch.setenv("PETSYS_CAL_DIR", str(tmp_path))
    result = project(REAL_CAL_TESTS).runpytest_subprocess("-m", "real_data", "-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*SKIPPED*real_data: missing jan/slab.encal under PETSYS_CAL_DIR=*"])


@pytest.mark.fr("007-FR-4")
def test_real_cal_runs_when_file_present(project, monkeypatch, tmp_path):
    (tmp_path / "jan").mkdir()
    (tmp_path / "jan" / "slab.encal").write_bytes(b"abc")
    monkeypatch.setenv("PETSYS_CAL_DIR", str(tmp_path))
    project(REAL_CAL_TESTS).runpytest_subprocess("-m", "real_data").assert_outcomes(passed=1)


SLOW_TESTS = """
import pytest

@pytest.mark.slow
def test_slow(): pass

@pytest.mark.real_data
def test_real(): pass

def test_plain(): pass
"""


@pytest.mark.fr("007-FR-5")
def test_default_run_deselects_slow_and_real_data(project):
    project(SLOW_TESTS).runpytest_subprocess().assert_outcomes(passed=1, deselected=2)


@pytest.mark.fr("007-FR-5")
def test_full_run_includes_slow(project):
    project(SLOW_TESTS).runpytest_subprocess("-m", "not real_data").assert_outcomes(
        passed=2, deselected=1)


SLEEP_TESTS = """
import time
import pytest

@pytest.mark.slow
def test_slow_sleeper(): time.sleep(1.5)

def test_sleeper(): time.sleep(1.5)

def test_quick(): pass
"""


@pytest.mark.fr("007-FR-5")
def test_slow_limit_fails_unmarked_slow_test(project):
    result = project(SLEEP_TESTS).runpytest_subprocess("-m", "not real_data", "--slow-limit", "1", "-v")
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(["*::test_slow_sleeper PASSED*", "*::test_sleeper FAILED*",
                                 "*slow-limit: took 1.* s >= 1 s without @pytest.mark.slow*"])


@pytest.mark.fr("007-FR-5")
def test_slow_limit_is_off_by_default(project):
    project(SLEEP_TESTS).runpytest_subprocess().assert_outcomes(passed=2, deselected=1)


@pytest.mark.fr("007-FR-5")
def test_slow_limit_refuses_parallel_workers(project):
    result = project(SLEEP_TESTS).parallel("-m", "not real_data", "--slow-limit", "1")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--slow-limit measures serial durations; add -n 0*"])


WORKER_TESTS = """
import os
import unittest
from pathlib import Path

def record(test):
    log = Path(os.environ["WORKER_LOG"])
    (log / test.id().split(".", 1)[1]).write_text(os.environ.get("PYTEST_XDIST_WORKER", "serial"))

class First(unittest.TestCase):
    def test_1(self): record(self)
    def test_2(self): record(self)
    def test_3(self): record(self)

class Second(unittest.TestCase):
    def test_1(self): record(self)
    def test_2(self): record(self)
    def test_3(self): record(self)
"""


@pytest.mark.fr("007-FR-5")
def test_default_run_keeps_each_class_on_one_worker(project, monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_LOG", str(tmp_path))
    project(WORKER_TESTS).parallel().assert_outcomes(passed=6)
    workers = {}
    for log in tmp_path.iterdir():
        workers.setdefault(log.name.split(".")[0], set()).add(log.read_text())
    assert sorted(workers) == ["First", "Second"]
    for cls, used in workers.items():
        assert len(used) == 1 and next(iter(used)).startswith("gw"), (cls, used)


@pytest.mark.fr("006-FR-3", "006-FR-11")
def test_linux_marker_skips_elsewhere_with_reason(project):
    result = project("import pytest\n\n@pytest.mark.linux\ndef test_x(): pass\n"
                     ).runpytest_subprocess("-rs")
    if sys.platform == "linux":
        result.assert_outcomes(passed=1)
    else:
        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines([f"*SKIPPED*linux: runs only on Linux (Cornell PC), not {sys.platform}*"])


@pytest.mark.gui
@pytest.mark.fr("006-FR-3")
def test_tk_root_is_withdrawn(tk_root):
    assert tk_root.state() == "withdrawn"


@pytest.mark.fr("006-FR-1", "006-FR-6")
@pytest.mark.parametrize("path", sorted((REPO / "tests").glob("*.py")), ids=lambda p: p.name)
def test_no_sys_path_edits_or_script_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "path" \
                and isinstance(node.value, ast.Name) and node.value.id == "sys":
            pytest.fail(f"{path.name}:{node.lineno} touches sys.path")
        modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                   else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        for module in modules:
            top = module.split(".")[0]
            assert top not in {"scripts", "scripts_cornell", "scripts_imas"}, \
                f"{path.name}:{node.lineno} imports {module}"
            assert not top.startswith("test_"), f"{path.name}:{node.lineno} imports test file {module}"
