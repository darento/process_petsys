"""Tk opens without an X input method on Linux (bug-tk-ibus-destroy-hang).

On the Cornell desktop (ibus as XIM, Tk 8.6.13) ``tests/test_ldat_processing.py`` hung in Tk
``destroy`` of a text widget on its first run; with ``XMODIFIERS=@im=none`` it passed 3 of 3. The
hang needs ibus, so these tests check that the tests and both GUIs set the variable before a root.
"""

import json
import os
import subprocess
import sys

import pytest

from helpers import REPO
from src.tk_input import NO_INPUT_METHOD, disable_x_input_method

pytestmark = pytest.mark.fr("bug-tk-ibus-destroy-hang")
LINUX = sys.platform.startswith("linux")


@pytest.mark.parametrize("platform, expected", [("linux", NO_INPUT_METHOD), ("win32", "@im=ibus")])
def test_input_method_disabled_only_on_linux(monkeypatch, platform, expected):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("XMODIFIERS", "@im=ibus")
    disable_x_input_method()
    assert os.environ["XMODIFIERS"] == expected


@pytest.mark.skipif(not LINUX, reason="X input methods exist only on Linux")
def test_test_processes_inherit_no_input_method():
    # conftest sets it before any test, so workers and the children they start inherit it
    child = subprocess.run([sys.executable, "-c", "import os; print(os.environ.get('XMODIFIERS'))"],
                           capture_output=True, text=True, check=True)
    assert os.environ["XMODIFIERS"] == NO_INPUT_METHOD and child.stdout.strip() == NO_INPUT_METHOD


def test_both_guis_disable_the_input_method_on_import():
    # a launcher under ibus: importing either GUI module (before its root exists) overrides the variable
    code = ("import json, os; os.environ['XMODIFIERS'] = '@im=ibus'; seen = {}\n"
            "import exe_programs.ldat_inspector_gui; seen['ldat'] = os.environ['XMODIFIERS']\n"
            "os.environ['XMODIFIERS'] = '@im=ibus'\n"
            "import exe_programs.petsys_manager_gui; seen['manager'] = os.environ['XMODIFIERS']\n"
            "print(json.dumps(seen))")
    run = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    expected = NO_INPUT_METHOD if LINUX else "@im=ibus"
    assert json.loads(run.stdout.strip().splitlines()[-1]) == {"ldat": expected, "manager": expected}
