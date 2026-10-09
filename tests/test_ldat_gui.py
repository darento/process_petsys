"""Linked views of a hidden LDATWorkbench on LDAT fixtures (spec 001 T4; spec 007 T20).

Moved from ``scripts/ldat_gui_check.py``: each ``checks += 1`` block is one test. The script drives
one window through 14 consecutive steps per system. ``run`` replays them once per system on the
shared ``window`` and records each step's observations, or the exception it raised; each test
asserts one step. Cornell uses the tracked January config (spec 007 Clarify, T18).
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pypdf import PdfReader

from ldat_helpers import destroy, fixture_files, pump, write_pairs
from src.ldat_inspector.engine import Settings, load_setup, merge_results, process_file

pytestmark = [pytest.mark.gui, pytest.mark.fr("001-FR-15")]
fr = pytest.mark.fr
# ``run`` replays the 14 steps in its setup: 10 s for IMAS, 7 s for Cornell.
RUN = pytest.mark.slow


@pytest.fixture(scope="module")
def window():
    import tkinter as tk

    from exe_programs.ldat_inspector_gui import LDATWorkbench

    try:
        app = LDATWorkbench()
    except tk.TclError as exc:
        pytest.skip(f"gui: no display available ({exc})")
    app.withdraw()
    app.tab_names = list(app.tabs._name_list)
    # spec 002: T10 retires the SuperModule Status tab, T11 adds Coincidences; only the visible tab
    # redraws, so show the SuperModule tab these checks read
    app.tabs.set("SuperModule")
    yield app
    destroy(app)


def record(seen, name, action):
    """Run one step; keep its observations, or the exception it raised for its test to re-raise."""
    try:
        setattr(seen, name, action())
    except Exception as exc:
        setattr(seen, name, exc)


def step(run, name):
    value = getattr(run, name)
    if isinstance(value, Exception):
        raise value
    return value


@pytest.fixture(scope="module", params=["IMAS", "CORNELL"])
def run(request, window):
    system, w = request.param, window
    seen = SimpleNamespace()
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        config, calibration, channels = fixture_files(root, system)
        path = root / "fixture.ldat"
        write_pairs(path, channels, 140)
        settings = Settings(str(config), str(calibration), system, 200)
        dataset = merge_results(settings, [process_file(str(path), settings)], load_setup(settings))
        w.dataset = dataset
        w._update_after_processing()
        w.update_idletasks()

        def wait(busy):
            # An update loop rather than the script's wait_variable: a BooleanVar per wait, left in a
            # closure cycle, can be collected on a worker thread ("main thread is not in main loop").
            pump(w, lambda: not busy(), 40)

        def explorer():
            return w.explorer_info.cget("text")

        record(seen, "controls", lambda: (len(w._sliders), w.energy_group_title.cget("text")))
        record(seen, "selected_sm", w._selected_sm)
        record(seen, "status", lambda: (w.status.cget("text"), len(w.channel_tree.get_children())))

        def energy_cut():
            w.energy_low.set("550")
            w._refresh_all()
            return explorer()
        record(seen, "energy_cut", energy_cut)

        def reset():
            w._reset_filters()
            return explorer()
        record(seen, "reset", reset)

        def flood():
            w.color_min.set("0.1")
            w.color_max.set("3.5")
            w._refresh_all()
            image = w.flood_ax.collections[0]
            return image.get_clim(), bool(image.get_array().mask.any()), image.cmap.get_bad()[:3].tolist()
        record(seen, "flood", flood)

        def slider():
            w.energy_low.set("550")
            w._refresh_all()
            value = w._sliders[0][0].get()
            w._reset_filters()
            return value
        record(seen, "slider", slider)

        def flood_overview():
            w.overview_mode.set("Flood maps")
            w._draw_overview()
            return True
        record(seen, "flood_overview", flood_overview)

        def paired_dt():
            w.overview_mode.set("Ingest sides")  # spec 002 T9: minimodule tiles replace "Counts"
            w._draw_overview()
            w.time_view.set("Paired delta-t")
            w._draw_timestamps()
            return len(w.time_fig.axes)
        record(seen, "paired_dt", paired_dt)

        def processing():
            w.time_view.set("Event rate")
            w.experimental.set(True)
            w._refresh_selected()
            w.experimental.set(False)
            w.config_path = str(config)
            w.calibration_path = str(calibration)
            w.system.set(system)
            w.files = [str(path)]
            w.max_pairs.set("200")
            w.min_channels.set("1")
            w.min_channel_energy.set("0")
            w._start_processing()
            wait(lambda: w._busy)
            return w._busy, w.dataset.files[0].pairs_accepted
        record(seen, "processing", processing)

        def report():
            pdf_path = root / "gui_report.pdf"
            with patch("exe_programs.ldat_inspector_gui.filedialog.asksaveasfilename", return_value=str(pdf_path)):
                w._report(0)
            started = w._report_busy
            wait(lambda: w._report_busy)
            return started, w._report_busy, pdf_path.exists() and len(PdfReader(str(pdf_path)).pages)
        record(seen, "report", report)

        def raw_mode():
            w._invalidate_data()
            cleared = w.dataset is None and not w.channel_tree.get_children()
            w.calibrated.set(False)
            w._toggle_calibration()
            w.calibration_path = ""
            return cleared, w.energy_low.get(), w.energy_high.get(), w.energy_group_title.cget("text")
        record(seen, "raw_mode", raw_mode)

        def raw_processing():
            w._start_processing()
            wait(lambda: w._busy)
            return (w._busy, w.dataset.settings.calibrated, w.dataset.files[0].pairs_accepted, explorer(),
                    w.energy_ax.get_xlabel())
        record(seen, "raw_processing", raw_processing)

        def calibrated_again():
            w.calibration_path = str(calibration)
            w.calibrated.set(True)
            w._toggle_calibration()
            wait(lambda: w._busy)
            observed = (w.dataset.settings.calibrated, w.dataset.files[0].pairs_accepted)
            w.dataset = dataset
            w._update_after_processing()
            return observed + (explorer(),)
        record(seen, "calibrated_again", calibrated_again)
        yield seen


@fr("001-FR-1")
def test_five_tabs(window):
    assert len(window.tab_names) == 5  # T11 adds Coincidences


@RUN
@fr("001-FR-6", "001-FR-19")
def test_explorer_has_six_sliders_and_an_energy_group(run):
    sliders, title = step(run, "controls")
    assert sliders == 6 and title.startswith("Energy")


@RUN
@fr("001-FR-6")
def test_first_supermodule_selected(run):
    assert step(run, "selected_sm") == 0


@RUN
@fr("001-FR-7", "002-FR-5")
def test_status_counts_and_channel_rows(run):
    status, rows = step(run, "status")
    assert "280" in status and rows >= 2


@RUN
@fr("001-FR-4")
def test_paired_energy_cut_updates_the_explorer(run):
    assert "0 after paired energy" in step(run, "energy_cut")


@RUN
@fr("001-FR-4")
def test_reset_restores_the_population(run):
    assert "140 after paired energy" in step(run, "reset")


@RUN
@fr("001-FR-6", "001-FR-21")
def test_flood_colour_limits_and_white_empty_bins(run):
    clim, masked, bad = step(run, "flood")
    assert clim == (0.1, 3.5) and masked
    assert bad == [1.0, 1.0, 1.0]


@RUN
@fr("001-FR-19")
def test_energy_slider_follows_the_numeric_field(run):
    assert abs(step(run, "slider") - 550) < 10


@RUN
@fr("001-FR-7")
def test_overview_flood_maps_draw(run):
    assert step(run, "flood_overview")


@RUN
@fr("001-FR-11")
def test_paired_delta_t_view_draws(run):
    assert step(run, "paired_dt") >= 1


@RUN
@fr("001-FR-13")
def test_worker_processing_updates_the_dataset(run):
    busy, accepted = step(run, "processing")
    assert not busy and accepted == 140


@RUN
@fr("001-FR-12", "001-FR-13")
def test_report_in_the_background_writes_five_pages(run):
    started, busy, pages = step(run, "report")
    assert started and not busy and pages == 5


@RUN
@fr("001-FR-22", "001-FR-24")
def test_invalidate_and_raw_mode_reset_energy_controls(run):
    cleared, low, high, title = step(run, "raw_mode")
    assert cleared
    assert low == "0" and high == "300"
    assert "raw" in title.lower()


@RUN
@fr("001-FR-22")
def test_raw_processing_labels_raw_units(run):
    busy, calibrated, accepted, explorer, xlabel = step(run, "raw_processing")
    assert not busy and calibrated is False
    assert accepted == 140
    assert "raw a.u." in explorer
    assert "Raw energy" in xlabel


@RUN
@fr("001-FR-22", "001-FR-23")
def test_calibration_on_again_and_previous_dataset_restored(run):
    calibrated, accepted, explorer = step(run, "calibrated_again")
    assert calibrated and accepted == 140
    assert "140 after paired energy" in explorer
