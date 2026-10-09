"""Pairs-per-file choices and the 30 M prefix cap (spec 002 FR-22; spec 007 T21).

Moved from ``scripts/ldat_pair_choices_check.py``: each ``check`` is one test. A worker returns each
file's result in one pickled message; on Windows a message over 4 GiB fails with "[WinError 87] El
parámetro no es correcto" (reproduced 2026-09-28 on the 25 GB Cornell Source_200microCi_60s file read
whole: 99.3 M pairs, 6.61 GB result). The prefix is therefore chosen from a fixed list whose largest
entry stays below that limit even at 100 % acceptance, and whole files estimated above the cap are
refused before processing. The GUI checks are consecutive steps on one hidden window, so ``gui`` runs
them once in order and records what each step observed. ``--real`` (25 GB file, 30 M prefix) is
retired (spec 007 Clarify).
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ldat_helpers import destroy, fixture_files, write_pairs
from src.ldat_inspector.engine import _TABLE_DTYPES, MAX_PAIRS_PER_FILE, Settings
from src.ldat_inspector.memory import MemoryEstimate, estimate_memory

pytestmark = pytest.mark.fr("002-FR-22")
PIPE_LIMIT = 2**32  # Windows WriteFile length is a 32-bit DWORD


def test_choices_parse():
    from exe_programs.ldat_inspector_gui import PAIR_CHOICES, _pair_count

    values = [_pair_count(choice) for choice in PAIR_CHOICES]
    assert values == [10_000, 100_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000]


def test_largest_choice_is_the_cap():
    from exe_programs.ldat_inspector_gui import PAIR_CHOICES, _pair_count

    assert _pair_count(PAIR_CHOICES[-1]) == MAX_PAIRS_PER_FILE, f"{MAX_PAIRS_PER_FILE:,}"


def test_plain_integer_still_parses():
    from exe_programs.ldat_inspector_gui import _pair_count

    assert _pair_count("200") == 200


@pytest.fixture
def rejected():
    """``rejected(pairs)``: whether ``Settings.validate`` refuses that pairs-per-file value."""
    with tempfile.TemporaryDirectory() as name:
        config = Path(name) / "config.yaml"
        config.write_text("map_file: x\n")

        def rejected(pairs):
            try:
                Settings(str(config), "", "CORNELL", max_pairs=pairs, calibrated=False).validate()
                return False
            except ValueError:
                return True

        yield rejected


def test_cap_and_whole_files_accepted(rejected):
    assert not rejected(MAX_PAIRS_PER_FILE) and not rejected(None)


def test_above_cap_and_below_1_rejected(rejected):
    assert rejected(MAX_PAIRS_PER_FILE + 1) and rejected(0)


def test_worst_case_result_below_4_gib():
    # Worst case: every pair accepted, calibration keys widened to int64 by _narrowest.
    side = sum(np.dtype(d).itemsize for d in _TABLE_DTYPES.values()) + 4
    worst = 2 * MAX_PAIRS_PER_FILE * side
    assert worst < 0.9 * PIPE_LIMIT, f"{side} B/side x {2 * MAX_PAIRS_PER_FILE:,} sides = {worst / 1e9:.2f} GB"


@pytest.fixture(scope="module")
def refusal():
    """Whole-file refusal: a synthetic 140-pair Cornell file against a cap lowered to 100."""
    with tempfile.TemporaryDirectory() as name:
        config, _, channels = fixture_files(Path(name), "CORNELL")
        path = Path(name) / "fixture.ldat"
        write_pairs(path, channels, 140)
        settings = Settings(str(config), "", "CORNELL", max_pairs=None, calibrated=False)
        with patch("src.ldat_inspector.memory.MAX_PAIRS_PER_FILE", 100):
            whole = estimate_memory([str(path)], settings, None)
            prefix = estimate_memory([str(path)], settings, 100)
        yield SimpleNamespace(path=path, whole=whole, prefix=prefix,
                              unpatched=estimate_memory([str(path)], settings, None))


def test_whole_file_over_the_cap_listed(refusal):
    assert [(p, round(n)) for p, n in refusal.whole.over_cap] == [(str(refusal.path), 140)], refusal.whole.over_cap


def test_prefix_never_listed(refusal):
    assert refusal.prefix.over_cap == ()


def test_whole_file_under_the_cap_not_listed(refusal):
    assert refusal.unpatched.over_cap == ()


# --- hidden GUI, consecutive steps on one window ------------------------------

@pytest.fixture(scope="module")
def gui():
    import tkinter as tk

    from exe_programs.ldat_inspector_gui import PAIR_CHOICES, LDATWorkbench

    try:
        window = LDATWorkbench()
    except tk.TclError as exc:
        pytest.skip(f"gui: no display available ({exc})")
    window.withdraw()
    seen = SimpleNamespace(choices=PAIR_CHOICES)
    try:
        seen.offered = tuple(window.max_pairs_entry.cget("values"))
        seen.default = window._pairs_limit()
        window.max_pairs.set("30M")
        seen.limit_30m = window._pairs_limit()
        window.whole_files.set(True)
        window._whole_files_changed()
        seen.whole_disabled = window.max_pairs_entry.cget("state") == "disabled" and window._pairs_limit() is None
        window.whole_files.set(False)
        window._whole_files_changed()
        seen.prefix_state = window.max_pairs_entry.cget("state")
        over = MemoryEstimate(1, 99_000_000, 120_000_000, 10**10, 10**8, 5 * 10**10, True,
                              over_cap=(("big.ldat", 99_273_393),))
        window._show_estimate(over)
        seen.shown = window.estimate_text.cget("text")
        window._busy = True
        with patch("exe_programs.ldat_inspector_gui.messagebox.showerror") as error, \
                patch.object(window, "_launch_pool") as launch:
            window._confirm_and_launch(over, None, ("big.ldat",))
        seen.refused = (error.called, launch.called, window._busy)
        yield seen
    finally:
        destroy(window)


@pytest.mark.gui
def test_gui_offers_the_choices(gui):
    assert gui.offered == gui.choices


@pytest.mark.gui
def test_gui_default_10k(gui):
    assert gui.default == 10_000


@pytest.mark.gui
def test_gui_30m(gui):
    assert gui.limit_30m == 30_000_000


@pytest.mark.gui
def test_gui_whole_files_disables_the_choice(gui):
    assert gui.whole_disabled and gui.prefix_state == "readonly"


@pytest.mark.gui
def test_gui_estimate_shows_the_refusal(gui):
    assert "Whole files unavailable above 30 M" in gui.shown and "big.ldat ~99 M" in gui.shown, \
        gui.shown.splitlines()[-1]


@pytest.mark.gui
def test_gui_processing_refused_no_workers(gui):
    error_called, launch_called, busy = gui.refused
    assert error_called and not launch_called and not busy
