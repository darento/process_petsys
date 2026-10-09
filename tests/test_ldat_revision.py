"""One-ingest calibration and explorer layout regressions (spec 001 T13; spec 007 T19).

Moved from ``scripts/ldat_revision_check.py``: each ``checks += 1`` block is one test, per
system. The engine blocks run on a shared fixture (``fixture``); the four GUI blocks are
consecutive steps on one hidden LDATWorkbench, so ``gui`` runs them once in order and
records what each step observed, and each test asserts its step's observations.
"""

from pathlib import Path
import tempfile
import time
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pypdf import PdfReader

from ldat_helpers import destroy, fixture_files, write_pairs
from src.ldat_inspector.engine import Selection, Settings, apply_calibration, load_setup, merge_results, process_file
from src.ldat_inspector.report import report_rows, write_report

pytestmark = pytest.mark.fr("001-FR-15")
fr = pytest.mark.fr


def pdf_text(path):
    return "\n".join(page.extract_text() for page in PdfReader(str(path)).pages)


@pytest.fixture(scope="module", params=["IMAS", "CORNELL"])
def fixture(request):
    """The script's per-system fixture, under a short tempfile root (report lines wrap at 118 characters)."""
    system = request.param
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config, calibration, channels = fixture_files(root, system)
        acquisition = root / "sample.ldat"
        write_pairs(acquisition, channels, 240)
        settings = Settings(str(config), "", system, max_pairs=300, calibrated=False)
        files = [process_file(str(acquisition), settings)]
        alternate = root / ("2026_09_24_long_cornell_or_imas_energy_calibration_"
                            "that_must_not_expand_the_input_files_card.encal")
        alternate.write_text(calibration.read_text(encoding="utf-8").replace("\t100", "\t200"), encoding="utf-8")
        yield SimpleNamespace(system=system, root=root, config=config, calibration=calibration, files=files,
                              dataset=merge_results(settings, files, load_setup(settings)), alternate=alternate)


# --- engine ------------------------------------------------------------------

@fr("001-FR-22", "001-FR-23")
def test_raw_ingest_without_calibration(fixture):
    assert fixture.files[0].pairs_accepted == 240
    assert sum(Selection(0, 300).mask(d).sum() for d in fixture.dataset.modules.values()) == 480


@fr("001-FR-12", "001-FR-23")
def test_calibration_switches_reuse_the_ingest(fixture):
    dataset = fixture.dataset
    original = dataset.modules[0]
    calibrated = apply_calibration(dataset, str(fixture.calibration), True)
    assert calibrated.files is dataset.files and calibrated.modules[0].raw_energy is original.raw_energy
    assert np.allclose(calibrated.modules[0].energy, 511)
    assert sum(Selection(400, 650).mask(d).sum() for d in calibrated.modules.values()) == 480
    changed = apply_calibration(calibrated, str(fixture.alternate), True)
    assert np.allclose(changed.modules[0].energy, 255.5)
    assert np.allclose(apply_calibration(changed, "", False).modules[0].energy, 100)
    assert sum(r["selected"] for r in report_rows(changed, Selection(200, 300))) == 480
    pdf_path = fixture.root / "changed.pdf"
    write_report(pdf_path, changed, Selection(200, 300), sm=0)
    text = pdf_text(pdf_path)
    assert "200..300 keV" in text and fixture.alternate.name[:18] in text


@fr("001-FR-12", "001-FR-23")
def test_missing_factors_are_unavailable_not_zero(fixture):
    dataset = fixture.dataset
    missing = fixture.root / "missing.encal"
    missing.write_text(fixture.calibration.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    unavailable = apply_calibration(dataset, str(missing), True)
    assert unavailable.files is dataset.files and np.isnan(unavailable.modules[0].energy).all()
    assert not Selection(0, 1500).mask(unavailable.modules[0]).any()
    pdf_path = fixture.root / "missing.pdf"
    write_report(pdf_path, unavailable, Selection(0, 1500), sm=0)
    assert "Unavailable calibrated energy: 480 detector sides" in pdf_text(pdf_path)


# --- hidden GUI, four consecutive steps --------------------------------------

@pytest.fixture(scope="module")
def gui(fixture):
    from exe_programs.ldat_inspector_gui import LDATWorkbench

    try:
        app = LDATWorkbench()
    except tk.TclError as exc:
        pytest.skip(f"gui: no display available ({exc})")
    app.withdraw()
    seen = SimpleNamespace()
    try:
        app.config_path = str(fixture.config)
        app.calibration_path = str(fixture.calibration)
        app.tabs.set("SuperModule")  # spec 002 T10: only the visible tab redraws
        app.dataset = fixture.dataset
        app._update_after_processing()
        app.update_idletasks()
        width = app.inputs_card.winfo_reqwidth()
        # Step 1: layout and fixed ranges.
        seen.slider_groups = [app._sliders[i][0].master.master for i in (0, 2, 4)]
        seen.control_groups = [app.energy_control_group, app.doi_control_group, app.spatial_control_group]
        app.energy_low.set("50")
        app.energy_high.set("200")
        app.doi_low.set("0.05")
        app.doi_high.set("0.95")
        app._refresh_all()
        seen.raw_xlim = app.energy_ax.get_xlim()
        seen.doi_xlim = app.doi_ax.get_xlim()
        app.doi_low.set("0.1")
        app.doi_high.set("0.8")
        app._refresh_all()
        seen.doi_xlim_after_cut = app.doi_ax.get_xlim()

        original_files = app.dataset.files
        deadline = time.monotonic() + 20

        def wait_for_calibration():
            ready = tk.BooleanVar(master=app, value=False)

            def check():
                if not app._busy or time.monotonic() > deadline:
                    ready.set(True)
                else:
                    app.after(30, check)

            app.after(30, check)
            app.wait_variable(ready)
            return not app._busy

        # Step 2: calibration on, without rereading the LDAT.
        with patch("exe_programs.ldat_inspector_gui.process_file", side_effect=AssertionError("LDAT reread")):
            app.calibrated.set(True)
            app._toggle_calibration()
            seen.on_idle = wait_for_calibration()
        seen.on_same_files = app.dataset is not None and app.dataset.files is original_files
        seen.on_energy = np.array(app.dataset.modules[0].energy) if app.dataset is not None else None
        seen.on_xlim = app.energy_ax.get_xlim()

        # Step 3: another calibration file.
        with patch("exe_programs.ldat_inspector_gui.filedialog.askopenfilename", return_value=str(fixture.alternate)):
            app._select_calibration()
        seen.alternate_idle = wait_for_calibration()
        app.update_idletasks()
        seen.widths = (app.inputs_card.winfo_reqwidth(), width)
        seen.calib_text = app.calib_text.cget("text")
        seen.alternate_energy = np.array(app.dataset.modules[0].energy)
        seen.first_row = [str(v) for v in app.channel_tree.item("0", "values")]
        seen.channel_summary = app.channel_summary.cget("text")
        with patch("exe_programs.ldat_inspector_gui.messagebox.showinfo") as info:
            app._show_calibration_path()
            seen.shown_path = info.call_args.args[1]

        # Step 4: back to raw.
        app.calibrated.set(False)
        app._toggle_calibration()
        seen.off_idle = wait_for_calibration()
        seen.off_same_files = app.dataset.files is original_files
        seen.off_xlim = app.energy_ax.get_xlim()
        yield seen
    finally:
        app.update_idletasks()
        destroy(app)


@pytest.mark.gui
@fr("001-FR-19", "001-FR-24")
def test_explorer_groups_and_fixed_raw_ranges(gui):
    assert gui.slider_groups == gui.control_groups
    assert gui.raw_xlim == (0, 300)
    assert gui.doi_xlim_after_cut == gui.doi_xlim


@pytest.mark.gui
@fr("001-FR-23", "001-FR-24")
def test_calibration_toggle_does_not_reread_the_ldat(gui):
    assert gui.on_idle
    assert gui.on_same_files
    assert np.allclose(gui.on_energy, 511)
    assert gui.on_xlim == (0, 1500)


@pytest.mark.gui
@fr("001-FR-23", "001-FR-26", "002-FR-5")
def test_selected_calibration_keeps_card_width_and_channel_status(gui, fixture):
    assert gui.alternate_idle
    assert gui.widths[0] == gui.widths[1] == 370
    assert len(gui.calib_text) < len(fixture.alternate.name)
    assert np.allclose(gui.alternate_energy, 255.5)
    # Spec 002 T7: Channel Status shows ingest-population channel findings; the
    # selected-side count moved out of this tab (it does not depend on display cuts).
    first_row = gui.first_row
    assert first_row[0] == "SM 0" and first_row[1] == "240" and first_row[2].startswith("1/")
    assert first_row[-1] in ("OK", "NOT OBSERVED", "LOW", "HIGH", "INSUFFICIENT EVENTS", "NO DATA")
    assert "ADC" not in gui.channel_summary
    assert gui.shown_path == str(fixture.alternate)


@pytest.mark.gui
@fr("001-FR-23", "001-FR-24")
def test_calibration_off_returns_to_raw_on_the_same_ingest(gui):
    assert gui.off_idle and gui.off_same_files and gui.off_xlim == (0, 300)
