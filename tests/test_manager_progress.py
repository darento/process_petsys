"""Spec 005 run-progress logic without Tk (T1): time-remaining estimates."""

import os
import subprocess
import sys

import pytest

from helpers import REPO
from src.petsys_manager.progress import Progress, RateEstimate, split_remaining


@pytest.mark.fr("005-FR-1")
def test_rate_estimate_from_this_stage_rate():
    # 25 % done after 10 s: 30 s left at the same rate.
    estimate = RateEstimate()
    estimate.start(100.0)
    progress = estimate.update(110.0, 250, 1000)
    assert progress.fraction == 0.25
    assert progress.remaining_s == pytest.approx(30.0)


@pytest.mark.fr("005-FR-1")
@pytest.mark.parametrize("now, done, fraction", [(104.9, 500, 0.5),    # half done, but under 5 s
                                                 (200.0, 9, 0.009)])  # 100 s, but under 1 %
def test_rate_estimate_needs_5_s_and_1_percent(now, done, fraction):
    estimate = RateEstimate()
    estimate.start(100.0)
    progress = estimate.update(now, done, 1000)
    assert progress.fraction == fraction
    assert progress.remaining_s is None


@pytest.mark.fr("005-FR-1")
def test_rate_estimate_at_the_gate():
    # Exactly 5 s and exactly 1 %: 99 times the elapsed time is left.
    estimate = RateEstimate()
    estimate.start(100.0)
    assert estimate.update(105.0, 10, 1000).remaining_s == pytest.approx(495.0)


@pytest.mark.fr("005-FR-1")
@pytest.mark.parametrize("total", [None, 0])
def test_rate_estimate_unknown_total(total):
    estimate = RateEstimate()
    estimate.start(100.0)
    assert estimate.update(160.0, 500, total) == Progress(None, None)


@pytest.mark.fr("005-FR-1")
def test_rate_estimate_new_phase_restarts_the_clock():
    # Read ends at 100 s; pass 2 starts there: 2 s later is under the 5 s gate,
    # and 10 s into pass 2 at 50 % leaves 10 s (not 110 s from the stage start).
    estimate = RateEstimate()
    estimate.start(0.0, phase="read")
    assert estimate.update(100.0, 1000, 1000, phase="read").remaining_s == 0.0
    assert estimate.update(102.0, 100, 1000, phase="pass 2").remaining_s is None
    assert estimate.update(110.0, 500, 1000, phase="pass 2").remaining_s == pytest.approx(10.0)


@pytest.mark.fr("005-FR-1")
def test_rate_estimate_without_start_begins_at_the_first_event():
    estimate = RateEstimate()
    assert estimate.update(50.0, 100, 1000) == Progress(0.1, None)
    assert estimate.update(60.0, 200, 1000).remaining_s == pytest.approx(40.0)


@pytest.mark.fr("005-FR-1")
def test_split_remaining_from_closed_split_durations():
    # 2 of 5 splits closed after 10 s and 20 s: 3 left at 15 s each.
    assert split_remaining(5, [10.0, 20.0]) == pytest.approx(45.0)


@pytest.mark.fr("005-FR-1")
@pytest.mark.parametrize("splits, durations", [(1, []), (1, [30.0]),   # one split: unknown total
                                               (5, [])])               # none closed yet
def test_split_remaining_unavailable(splits, durations):
    assert split_remaining(splits, durations) is None


@pytest.mark.fr("005-FR-1")
def test_split_remaining_all_closed():
    assert split_remaining(3, [10.0, 12.0, 11.0]) == 0.0


@pytest.mark.fr("005-FR-1")
def test_split_remaining_never_negative():
    # The converter wrote one split more than requested.
    assert split_remaining(2, [10.0, 10.0, 1.0]) == 0.0


@pytest.mark.fr("005-FR-1")
def test_rate_estimate_overshoot_is_done_not_negative():
    estimate = RateEstimate()
    estimate.start(0.0)
    assert estimate.update(20.0, 1100, 1000) == Progress(1.0, 0.0)


@pytest.mark.fr("005-FR-1")
def test_progress_imports_without_tk():
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["PYTHONPATH"] = str(REPO)
    code = ("import sys, src.petsys_manager.progress; "
            "sys.exit('tkinter' in sys.modules or 'customtkinter' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
