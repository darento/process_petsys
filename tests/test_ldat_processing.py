"""Whole files, all cores, Cancel and the memory estimate (spec 002 T4, FR-2-FR-4; spec 007 T20).

Moved from ``scripts/ldat_processing_check.py``: each ``check`` is one test. Engine checks run
in-process on four copies of the IMAS mixed fixture. The GUI checks are consecutive steps on one
hidden LDATWorkbench with real worker processes, so ``gui_run`` runs them once in order and records
what each step observed. Cancel and close use ``ldat_helpers.slow_reader``, which honours the same
cancel flag as the fast reader.
"""

from dataclasses import replace
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from helpers import REPO
from ldat_helpers import CONFIGS, console, fixture_pairs, pump, slow_reader, workbench, write_ldat
import src.ldat_inspector.fastread as ldat_fastread
from src.ldat_inspector.engine import Settings, load_setup, merge_results, process_file_reference
from src.ldat_inspector.memory import PEAK_BYTES_PER_SIDE, WORKER_BYTES_PER_SIDE, estimate_memory

pytestmark = pytest.mark.fr("002-FR-2")
fr = pytest.mark.fr
# ``gui_run`` runs the GUI steps with real workers in its setup: about 7 s.
GUI_RUN = pytest.mark.slow
CONFIG = CONFIGS["IMAS"]


@pytest.fixture(scope="module")
def fixtures():
    """Four copies of the IMAS mixed fixture under a short tempfile root, as in the script."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        settings = Settings(str(CONFIG), "", "IMAS", max_pairs=None, min_channels=4,
                            min_channel_energy=0.2, calibrated=False)
        cases = fixture_pairs("IMAS", load_setup(settings))
        pairs = [builder(i) for _, count, _, builder in cases for i in range(count)]
        accepted = sum(count for _, count, outcome, _ in cases if outcome == "accepted")
        paths = [write_ldat(root / f"fixture_{n}.ldat", pairs) for n in range(4)]
        yield SimpleNamespace(root=root, settings=settings, paths=paths, n_pairs=len(pairs), accepted=accepted)


@pytest.fixture
def worker_state():
    """Reset the fast reader's per-process cancel/progress globals after the test."""
    yield
    ldat_fastread.init_worker(None, None)


# --- engine ------------------------------------------------------------------

def test_whole_file_mode_reads_every_pair_not_a_prefix(fixtures):
    whole = ldat_fastread.process_file_fast(str(fixtures.paths[0]), fixtures.settings, 0)
    reference = process_file_reference(str(fixtures.paths[0]), fixtures.settings, 0)
    assert (whole.pairs_read == reference.pairs_read == fixtures.n_pairs and not whole.prefix_limited
            and whole.pairs_accepted == reference.pairs_accepted == fixtures.accepted), \
        f"{whole.pairs_read} read, {whole.pairs_accepted} accepted"


@fr("002-FR-3")
def test_fast_reader_reports_per_file_progress(fixtures, worker_state):
    progress = multiprocessing.get_context("spawn").Array("d", 1)
    ldat_fastread.init_worker(None, progress)
    ldat_fastread.process_file_fast(str(fixtures.paths[0]), fixtures.settings, 0)
    assert abs(progress[0] - 1.0) < 1e-9, f"{progress[0]:.3f}"


@fr("002-FR-4")
def test_cancel_stops_the_fast_reader_between_chunks_with_no_partial_data(fixtures, worker_state):
    event = multiprocessing.get_context("spawn").Event()
    event.set()
    ldat_fastread.init_worker(event, None)
    with patch.object(ldat_fastread, "CHUNK_PAIRS", 5):
        stopped = ldat_fastread.process_file_fast(str(fixtures.paths[0]), fixtures.settings, 0)
    assert stopped.error == "Cancelled" and stopped.table is None and stopped.pairs_accepted == 0


def test_merging_that_releases_per_file_tables_gives_the_same_table(fixtures):
    def read():
        return [ldat_fastread.process_file_fast(str(p), fixtures.settings, i) for i, p in enumerate(fixtures.paths)]

    kept = merge_results(fixtures.settings, read())
    consumed = merge_results(fixtures.settings, read(), consume=True)
    assert (all(np.array_equal(kept.table.columns[k], consumed.table.columns[k]) for k in kept.table.columns)
            and len(kept.table) == 2 * fixtures.accepted * len(fixtures.paths))


@pytest.fixture(scope="module")
def estimate(fixtures):
    return estimate_memory([str(p) for p in fixtures.paths], fixtures.settings, None)


def test_estimate_pairs_and_accepted_sides_from_sampled_files(fixtures, estimate):
    prefix = estimate_memory([str(p) for p in fixtures.paths], fixtures.settings, 7)
    n = len(fixtures.paths)
    assert (estimate.pairs == fixtures.n_pairs * n and estimate.sides == 2 * fixtures.accepted * n
            and prefix.pairs == 7 * n and estimate.acceptance_sampled), \
        f"{estimate.pairs} pairs, {estimate.sides} sides; prefix {prefix.pairs} pairs"


def test_estimate_baseline_plus_bytes_per_side(estimate):
    assert estimate.bytes == estimate.baseline + estimate.sides * PEAK_BYTES_PER_SIDE, estimate.text()


@fr("002-FR-3")
def test_estimate_adds_the_concurrent_workers_memory(fixtures, estimate):
    two = estimate_memory([str(p) for p in fixtures.paths], fixtures.settings, None, workers=2)
    assert (estimate.workers == min(len(fixtures.paths), max(1, (os.cpu_count() or 1) - 2))
            and estimate.worker_bytes == int(estimate.sides * WORKER_BYTES_PER_SIDE)
            and two.worker_bytes == int(2 * 2 * fixtures.accepted * WORKER_BYTES_PER_SIDE)
            and estimate.total == estimate.bytes + estimate.worker_bytes
            and "workers while reading" in estimate.text()), \
        f"{estimate.workers} workers: {estimate.worker_bytes:,} B; 2 workers: {two.worker_bytes:,} B"


def test_process_memory_and_free_ram_are_measurable_on_windows(estimate):
    from src.ldat_inspector.memory import available_bytes, working_set

    current, peak = working_set(), working_set(peak=True)
    # A None here once left the estimate without its baseline (64-bit handle truncated).
    assert (sys.platform != "win32" or (current and peak and peak >= current and estimate.baseline > 0
                                        and available_bytes())), \
        f"working set {current}, peak {peak}, baseline {estimate.baseline}"


def test_estimate_without_a_config_is_a_labelled_upper_bound(fixtures):
    upper = estimate_memory([str(p) for p in fixtures.paths], None, None)
    assert upper.sides == 2 * fixtures.n_pairs * len(fixtures.paths) and "upper bound" in upper.text()


# --- hidden GUI, consecutive steps -------------------------------------------

@pytest.fixture(scope="module")
def gui_run(fixtures):
    import tkinter as tk
    from tkinter import messagebox

    try:
        app = workbench(fixtures.settings, fixtures.paths)
    except tk.TclError as exc:
        pytest.skip(f"gui: no display available ({exc})")
    seen = SimpleNamespace(errors=[])
    try:
        with patch.object(messagebox, "showerror", side_effect=lambda *a, **k: seen.errors.append(a)):
            app._schedule_estimate()
            seen.estimate_shown = pump(app, lambda: "M pairs" in app.estimate_text.cget("text"), 20)
            seen.estimate_text = app.estimate_text.cget("text")

            asked = []
            with patch("src.ldat_inspector.memory.available_bytes", return_value=1), \
                    patch.object(messagebox, "askyesno", side_effect=lambda *a, **k: asked.append(a) or False):
                app._start_processing()
                pump(app, lambda: not app._busy, 20)
            seen.asked, seen.declined_dataset, seen.declined_log = len(asked), app.dataset, console(app)

            app.whole_files.set(True)
            app._whole_files_changed()
            app._start_processing()
            seen.done = pump(app, lambda: not app._busy and app.dataset is not None, 120)
            seen.dataset, seen.workers, seen.whole_log = app.dataset, app._workers, console(app)

            app._reader = slow_reader
            app._start_processing()
            seen.launched = pump(app, lambda: app._pool is not None and "worker processes" in console(app).split(
                "Processing complete")[-1], 30)
            pump(app, lambda: False, 1.0)
            start = time.monotonic()
            app._cancel_processing()
            seen.returned = pump(app, lambda: not app._busy, 10)
            seen.cancel_s = time.monotonic() - start
            seen.dataset_after_cancel, seen.cancel_log = app.dataset, console(app)
        yield seen
    finally:
        app._close()


@pytest.mark.gui
@GUI_RUN
def test_memory_estimate_appears_after_files_are_selected(gui_run):
    assert gui_run.estimate_shown, gui_run.estimate_text


@pytest.mark.gui
@GUI_RUN
def test_estimate_above_free_ram_asks_first_and_declining_does_not_process(gui_run):
    assert gui_run.asked == 1 and gui_run.declined_dataset is None and "not started" in gui_run.declined_log


@pytest.mark.gui
@GUI_RUN
@fr("002-FR-3")
def test_whole_file_multi_file_run_uses_more_than_2_worker_processes(gui_run, fixtures):
    dataset = gui_run.dataset
    assert (gui_run.done and gui_run.workers > 2 and dataset is not None
            and sum(f.pairs_accepted for f in dataset.files) == fixtures.accepted * len(fixtures.paths)), \
        f"{gui_run.workers} workers for {len(fixtures.paths)} files"


@pytest.mark.gui
@GUI_RUN
def test_log_labels_whole_file_scope_per_run_and_per_file(gui_run, fixtures):
    assert "(whole files)" in gui_run.whole_log and gui_run.whole_log.count("(whole file)") >= len(fixtures.paths)


@pytest.mark.gui
@GUI_RUN
def test_report_provenance_labels_whole_files(gui_run, fixtures):
    from pypdf import PdfReader

    from src.ldat_inspector.engine import Selection
    from src.ldat_inspector.report import write_report

    pdf = fixtures.root / "whole.pdf"
    write_report(pdf, gui_run.dataset, Selection(0, 300), sm=0)
    text = "\n".join(page.extract_text() for page in PdfReader(str(pdf)).pages)
    assert "whole files (no pair limit)" in text and "(whole file)" in text


@pytest.mark.gui
@GUI_RUN
@fr("002-FR-4")
def test_cancel_returns_control_within_2_s_and_keeps_the_previous_dataset(gui_run):
    assert (gui_run.launched and gui_run.returned and gui_run.cancel_s <= 2.0
            and gui_run.dataset_after_cancel is gui_run.dataset
            and "previous dataset kept" in gui_run.cancel_log), f"{gui_run.cancel_s:.2f} s"


@pytest.mark.gui
@GUI_RUN
def test_no_error_dialogs_during_the_gui_checks(gui_run):
    assert not gui_run.errors, f"{gui_run.errors[:1]}"


# --- closing mid-run ---------------------------------------------------------

@pytest.mark.gui
@fr("002-FR-4")
def test_closing_the_window_mid_run_exits_within_5_s(fixtures):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO), str(REPO / "tests")]))
    code = "import sys; from ldat_helpers import close_child; close_child(sys.argv[1], sys.argv[2])"
    child = subprocess.Popen([sys.executable, "-c", code, str(CONFIG), str(fixtures.paths[0])],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(REPO), env=env)
    closing_at = None
    output = []
    for line in child.stdout:
        output.append(line)
        if line.startswith("CLOSING"):
            closing_at = time.monotonic()
            break
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    exit_s = time.monotonic() - closing_at if closing_at else float("inf")
    assert closing_at is not None and child.returncode == 0, \
        f"{exit_s:.2f} s, exit code {child.returncode}: {''.join(output)[-2000:]}"
    # The 5 s bound is measured serially (-n 0): parallel workers' CPU load stretches it to 6-7 s (owner, T20).
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        assert exit_s <= 5.0, f"{exit_s:.2f} s"
