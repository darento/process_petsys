"""Spec 005 run-progress logic without Tk: time-remaining estimates (T1), the stage overview (T6), the
warning banners (T7) and the acquisition growth banner (T8)."""

import os
import subprocess
import sys

import pytest

from helpers import REPO
from src.petsys_manager.progress import Banners, GrowthBanner, GrowthView, Progress, RateEstimate, StageOverview, split_remaining


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


PIPELINE = ["acquisition", "conversion", "calibration", "listmode"]


def states(overview, now):
    return [(row.stage, row.state, row.elapsed_s) for row in overview.rows(now)]


@pytest.mark.fr("005-FR-2")
def test_overview_success_sequence():
    # Acquisition reports no stage events of its own: it runs from its first event until conversion starts.
    overview = StageOverview(PIPELINE)
    assert states(overview, 0.0) == [(stage, "pending", None) for stage in PIPELINE]
    overview.event(10.0, "acquisition_attempt_started", "acquisition", {"attempt": 1})
    overview.event(70.0, "stage_started", "conversion", {})
    overview.event(100.0, "stage_finished", "conversion", {"status": "succeeded", "elapsed_s": 30.2})
    overview.event(100.0, "stage_started", "calibration", {})
    overview.event(160.0, "stage_finished", "calibration", {"status": "succeeded", "elapsed_s": 59.9})
    overview.event(160.0, "stage_started", "listmode", {})
    overview.event(200.0, "stage_finished", "listmode", {"status": "succeeded", "elapsed_s": 40.1})
    overview.event(200.0, "workflow_finished", "workflow", {"status": "succeeded"})
    assert states(overview, 900.0) == [("acquisition", "succeeded", 60.0), ("conversion", "succeeded", 30.2),
                                       ("calibration", "succeeded", 59.9), ("listmode", "succeeded", 40.1)]


@pytest.mark.fr("005-FR-2")
@pytest.mark.parametrize("status", ["failed", "launch_error"])
def test_overview_failure_leaves_later_stages_not_run(status):
    overview = StageOverview(PIPELINE)
    overview.event(10.0, "acquisition_attempt_started", "acquisition", {"attempt": 1})
    overview.event(70.0, "stage_started", "conversion", {})
    overview.event(75.0, "stage_finished", "conversion", {"status": status, "elapsed_s": 5.0})
    overview.event(75.0, "workflow_finished", "workflow", {"status": "failed"})
    assert states(overview, 900.0) == [("acquisition", "succeeded", 60.0), ("conversion", "failed", 5.0),
                                       ("calibration", "pending", None), ("listmode", "pending", None)]


@pytest.mark.fr("005-FR-2")
def test_overview_stop_during_the_first_stage_stops_the_rest():
    overview = StageOverview(PIPELINE)
    overview.event(10.0, "acquisition_attempt_started", "acquisition", {"attempt": 1})
    overview.event(40.0, "workflow_finished", "workflow", {"status": "cancelled"})
    assert states(overview, 900.0) == [("acquisition", "stopped", 30.0), ("conversion", "stopped", None),
                                       ("calibration", "stopped", None), ("listmode", "stopped", None)]


@pytest.mark.fr("005-FR-2")
def test_overview_failed_acquisition_ends_at_the_workflow_result():
    # Acquisition reports no stage_finished: the workflow's failure is its own.
    overview = StageOverview(["acquisition", "conversion", "qc"])
    overview.event(10.0, "acquisition_attempt_started", "acquisition", {"attempt": 1})
    overview.event(25.0, "workflow_finished", "workflow", {"status": "failed"})
    assert states(overview, 900.0) == [("acquisition", "failed", 15.0), ("conversion", "pending", None),
                                       ("qc", "pending", None)]


@pytest.mark.fr("005-FR-2")
def test_overview_running_stage_elapsed_ticks_until_recorded():
    overview = StageOverview(["acquisition", "conversion", "qc"])
    overview.event(10.0, "stage_started", "conversion", {})
    assert states(overview, 12.5)[1] == ("conversion", "running", 2.5)
    assert states(overview, 70.0)[1] == ("conversion", "running", 60.0)
    overview.event(71.0, "stage_finished", "conversion", {"status": "succeeded", "elapsed_s": 60.4})
    assert states(overview, 500.0)[1] == ("conversion", "succeeded", 60.4)


def shown(banners):
    return [(banner.kind, banner.text, banner.dismissible) for banner in banners.items()]


@pytest.mark.fr("005-FR-7")
def test_banners_show_in_fixed_order_and_a_kind_keeps_its_latest_text():
    banners = Banners()
    assert shown(banners) == []
    banners.show("stopped", "Conversion cancelled")
    banners.show("failure", "DAQD FAILED: daemon exited")
    banners.show("bias", "SiPM bias state unknown")
    banners.show("failure", "Initialization failed: no answer")
    assert shown(banners) == [("bias", "SiPM bias state unknown", False),
                              ("failure", "Initialization failed: no answer", True),
                              ("stopped", "Conversion cancelled", True)]


@pytest.mark.fr("005-FR-7")
def test_banners_dismiss_and_new_run_keep_the_bias_banner():
    banners = Banners()
    for kind in ("bias", "failure", "stopped"):
        banners.show(kind, kind)
    banners.dismiss("failure")
    banners.dismiss("bias")         # can't be dismissed
    assert [banner.kind for banner in banners.items()] == ["bias", "stopped"]
    banners.show("failure", "again")
    banners.new_run()
    assert shown(banners) == [("bias", "bias", False)]
    banners.new_run()
    assert [banner.kind for banner in banners.items()] == ["bias"]


@pytest.mark.fr("005-FR-7")
def test_banners_only_acknowledge_bias_clears_the_bias_banner():
    banners = Banners()
    banners.show("bias", "SiPM bias state unknown")
    banners.show("stopped", "stopped")
    banners.acknowledge_bias()
    assert shown(banners) == [("stopped", "stopped", True)]


@pytest.mark.fr("005-FR-7")
def test_banners_reject_an_unknown_kind():
    with pytest.raises(ValueError):
        Banners().show("warning", "processing warnings stay in the log")


def growing_acquisition(duration_s=60.0):
    """Started at 10, attempt at 12, RAW growing, growth check passed at 32."""
    banner = GrowthBanner()
    banner.reset(duration_s)
    banner.event(10.0, "workflow_started", "workflow", {"run_root": "/data/run_2026-10-10_1200", "stages": ["acquisition"]})
    banner.event(12.0, "acquisition_attempt_started", "acquisition", {"attempt": 1, "max_attempts": 3})
    banner.event(14.0, "acquisition_growth_started", "acquisition", {"size": 4096})
    banner.event(17.0, "acquisition_rawf_progress", "acquisition", {"size": 10_000_000, "bytes_per_s": 3e6, "growing": True})
    assert banner.view(20.0) is None            # hidden until the growth check passes
    banner.event(32.0, "acquisition_growth_passed", "acquisition", {"growth_bytes": 50_000_000})
    return banner


@pytest.mark.fr("005-FR-8")
def test_growth_banner_green_after_the_growth_check_with_run_name_and_times():
    banner = growing_acquisition()
    banner.event(37.0, "acquisition_rawf_progress", "acquisition", {"size": 80_000_000, "bytes_per_s": 2.5e6, "growing": True})
    assert banner.view(38.0) == GrowthView("growing", "run_2026-10-10_1200", 26.0, 34.0, 80_000_000, 2.5e6, None)


@pytest.mark.fr("005-FR-8")
def test_growth_banner_red_while_stalled_then_green_again():
    banner = growing_acquisition()
    banner.event(37.0, "acquisition_rawf_progress", "acquisition", {"size": 80_000_000, "bytes_per_s": 2.5e6, "growing": True})
    banner.event(42.0, "acquisition_rawf_progress", "acquisition", {"size": 80_000_000, "bytes_per_s": 0.0, "growing": False})
    assert banner.view(43.0) == GrowthView("stalled", "run_2026-10-10_1200", 31.0, 29.0, 80_000_000, 0.0, 6.0)
    banner.event(47.0, "acquisition_rawf_progress", "acquisition", {"size": 80_000_000, "bytes_per_s": 0.0, "growing": False})
    assert banner.view(50.0).since_growth_s == 13.0      # still counted from the last growth at 37
    banner.event(52.0, "acquisition_rawf_progress", "acquisition", {"size": 90_000_000, "bytes_per_s": 2e6, "growing": True})
    assert banner.view(52.0) == GrowthView("growing", "run_2026-10-10_1200", 40.0, 20.0, 90_000_000, 2e6, None)


@pytest.mark.fr("005-FR-8")
def test_growth_banner_stalled_since_the_growth_check_when_no_growth_followed():
    banner = growing_acquisition()
    banner.event(36.0, "acquisition_rawf_progress", "acquisition", {"size": 10_000_000, "bytes_per_s": 0.0, "growing": False})
    assert banner.view(40.0).state == "stalled"
    assert banner.view(40.0).since_growth_s == 8.0


@pytest.mark.fr("005-FR-8")
def test_growth_banner_retry_hides_it_with_a_new_clock():
    banner = growing_acquisition()
    banner.event(50.0, "acquisition_attempt_started", "acquisition", {"attempt": 2, "max_attempts": 3})
    assert banner.view(51.0) is None
    banner.event(60.0, "acquisition_growth_passed", "acquisition", {"growth_bytes": 1})
    view = banner.view(61.0)
    assert (view.state, view.elapsed_s, view.remaining_s, view.size) == ("growing", 11.0, 49.0, None)


@pytest.mark.fr("005-FR-8")
@pytest.mark.parametrize("kind, stage, payload", [
    ("acquisition_aborting", "acquisition", {"reason": "prerequisite_lost"}),
    ("acquisition_attempt_finished", "acquisition", {"status": "succeeded"}),
    ("acquisition_finished", "acquisition", {"status": "succeeded"}),
    ("stage_finished", "acquisition", {"status": "succeeded"}),
    ("workflow_finished", "workflow", {"status": "cancelled"})])
def test_growth_banner_removed_when_the_acquisition_ends(kind, stage, payload):
    banner = growing_acquisition()
    banner.event(40.0, kind, stage, payload)
    assert banner.view(41.0) is None
    banner.event(42.0, "acquisition_rawf_progress", "acquisition", {"size": 1, "bytes_per_s": 1.0, "growing": False})
    assert banner.view(43.0) is None


@pytest.mark.fr("005-FR-8")
def test_growth_banner_later_stages_leave_it_alone_until_the_acquisition_ends():
    banner = growing_acquisition()
    banner.event(40.0, "stage_finished", "conversion", {"status": "succeeded"})
    assert banner.view(41.0).state == "growing"


@pytest.mark.fr("005-FR-8")
def test_growth_banner_time_remaining_never_negative_or_made_up():
    banner = growing_acquisition(duration_s=10.0)
    assert banner.view(100.0).remaining_s == 0.0
    banner = growing_acquisition(duration_s=None)
    assert banner.view(40.0).remaining_s is None


@pytest.mark.fr("005-FR-8")
def test_growth_banner_reset_forgets_the_previous_run():
    banner = growing_acquisition()
    banner.reset(30.0)
    assert banner.view(40.0) is None
    banner.event(50.0, "acquisition_attempt_started", "acquisition", {"attempt": 1})
    banner.event(55.0, "acquisition_growth_passed", "acquisition", {})
    assert banner.view(56.0).run_name == ""
