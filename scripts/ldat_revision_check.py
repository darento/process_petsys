"""Focused regression checks for one-ingest calibration and explorer layout."""

from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pypdf import PdfReader

from exe_programs.ldat_inspector_gui import LDATWorkbench
from scripts.ldat_inspector_check import fixture_files, write_pairs
from src.ldat_inspector import Selection, Settings, apply_calibration, load_setup, merge_results, process_file
from src.ldat_report import report_rows, write_report


def main():
    checks = 0
    app = None
    for system in ("IMAS", "CORNELL"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, calibration, channels = fixture_files(root, system)
            acquisition = root / "sample.ldat"
            write_pairs(acquisition, channels, 240)
            settings = Settings(str(config), "", system, max_pairs=300, calibrated=False)
            files = [process_file(str(acquisition), settings)]
            dataset = merge_results(settings, files, load_setup(settings))
            assert files[0].pairs_accepted == 240
            assert sum(Selection(0, 300).mask(d).sum() for d in dataset.modules.values()) == 480
            checks += 1
            original = dataset.modules[0]
            calibrated = apply_calibration(dataset, str(calibration), True)
            assert calibrated.files is dataset.files and calibrated.modules[0].raw_energy is original.raw_energy
            assert np.allclose(calibrated.modules[0].energy, 511)
            assert sum(Selection(400, 650).mask(d).sum() for d in calibrated.modules.values()) == 480
            alternate = root / ("2026_09_24_long_cornell_or_imas_energy_calibration_"
                                "that_must_not_expand_the_input_files_card.encal")
            alternate.write_text(calibration.read_text(encoding="utf-8").replace("\t100", "\t200"),
                                 encoding="utf-8")
            changed = apply_calibration(calibrated, str(alternate), True)
            assert np.allclose(changed.modules[0].energy, 255.5)
            assert np.allclose(apply_calibration(changed, "", False).modules[0].energy, 100)
            assert sum(r["selected"] for r in report_rows(changed, Selection(200, 300))) == 480
            pdf_path = root / "changed.pdf"
            write_report(pdf_path, changed, Selection(200, 300), sm=0)
            pdf_text = "\n".join(page.extract_text() for page in PdfReader(str(pdf_path)).pages)
            assert "200..300 keV" in pdf_text and alternate.name[:18] in pdf_text
            checks += 1
            missing = root / "missing.encal"
            missing.write_text(calibration.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
            unavailable = apply_calibration(dataset, str(missing), True)
            assert unavailable.files is dataset.files and np.isnan(unavailable.modules[0].energy).all()
            assert not Selection(0, 1500).mask(unavailable.modules[0]).any()
            pdf_path = root / "missing.pdf"
            write_report(pdf_path, unavailable, Selection(0, 1500), sm=0)
            pdf_text = "\n".join(page.extract_text() for page in PdfReader(str(pdf_path)).pages)
            assert "Unavailable calibrated energy: 480 detector sides" in pdf_text
            checks += 1

            if app is None:
                app = LDATWorkbench()
                app.withdraw()
            try:
                app.config_path = str(config)
                app.calibration_path = str(calibration)
                app.dataset = dataset
                app._update_after_processing()
                app.update_idletasks()
                width = app.inputs_card.winfo_reqwidth()
                assert app._sliders[0][0].master.master is app.energy_control_group
                assert app._sliders[2][0].master.master is app.doi_control_group
                assert app._sliders[4][0].master.master is app.spatial_control_group
                app.energy_low.set("50")
                app.energy_high.set("200")
                app.doi_low.set("0.05")
                app.doi_high.set("0.95")
                app._refresh_all()
                assert app.energy_ax.get_xlim() == (0, 300)
                doi_xlim = app.doi_ax.get_xlim()
                app.doi_low.set("0.1")
                app.doi_high.set("0.8")
                app._refresh_all()
                assert app.doi_ax.get_xlim() == doi_xlim
                checks += 1

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
                    assert not app._busy

                with patch("exe_programs.ldat_inspector_gui.process_file", side_effect=AssertionError("LDAT reread")):
                    app.calibrated.set(True)
                    app._toggle_calibration()
                    wait_for_calibration()
                assert app.dataset is not None and app.dataset.files is original_files
                assert np.allclose(app.dataset.modules[0].energy, 511)
                assert app.energy_ax.get_xlim() == (0, 1500)
                checks += 1
                with patch("exe_programs.ldat_inspector_gui.filedialog.askopenfilename", return_value=str(alternate)):
                    app._select_calibration()
                wait_for_calibration()
                app.update_idletasks()
                assert app.inputs_card.winfo_reqwidth() == width == 370
                assert len(app.calib_text.cget("text")) < len(alternate.name)
                assert np.allclose(app.dataset.modules[0].energy, 255.5)
                first_row = app.channel_tree.item("0", "values")
                assert first_row[0] == "SM 0" and first_row[1] == "240" and first_row[2] == "0"
                assert first_row[3].startswith("1/") and "ADC" not in app.channel_ax.get_title()
                with patch("exe_programs.ldat_inspector_gui.messagebox.showinfo") as info:
                    app._show_calibration_path()
                    assert info.call_args.args[1] == str(alternate)
                checks += 1
                app.calibrated.set(False)
                app._toggle_calibration()
                wait_for_calibration()
                assert app.dataset.files is original_files and app.energy_ax.get_xlim() == (0, 300)
                checks += 1
            finally:
                app.update_idletasks()
    if app is not None:
        app.destroy()
    print(f"PASS: {checks} revision checks")


if __name__ == "__main__":
    main()
