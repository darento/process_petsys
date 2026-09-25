"""Build a withdrawn real LDATWorkbench and exercise linked views on LDAT fixtures."""

from pathlib import Path
import tempfile
import sys
import time
import tkinter as tk
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exe_programs.ldat_inspector_gui import LDATWorkbench
from scripts.ldat_inspector_check import fixture_files, write_pairs
from src.ldat_inspector import Settings, load_setup, merge_results, process_file
from pypdf import PdfReader


def main():
    window = LDATWorkbench()
    window.withdraw()
    checks = 0
    try:
        assert len(window.tabs._name_list) == 5
        checks += 1
        for system in ("IMAS", "CORNELL"):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                config, calibration, channels = fixture_files(root, system)
                path = root / "fixture.ldat"
                write_pairs(path, channels, 140)
                settings = Settings(str(config), str(calibration), system, 200)
                dataset = merge_results(settings, [process_file(str(path), settings)], load_setup(settings))
                window.dataset = dataset
                window._update_after_processing()
                window.update_idletasks()
                assert len(window._sliders) == 6 and window.energy_group_title.cget("text").startswith("Energy")
                checks += 1
                assert window._selected_sm() == 0
                checks += 1
                assert "280" in window.status.cget("text") and len(window.channel_tree.get_children()) >= 2
                checks += 1
                window.energy_low.set("550")
                window._refresh_all()
                assert "0 after paired energy" in window.explorer_info.cget("text")
                checks += 1
                window._reset_filters()
                assert "140 after paired energy" in window.explorer_info.cget("text")
                checks += 1
                window.color_min.set("0.1")
                window.color_max.set("3.5")
                window._refresh_all()
                image = window.flood_ax.collections[0]
                assert image.get_clim() == (0.1, 3.5) and image.get_array().mask.any()
                assert image.cmap.get_bad()[:3].tolist() == [1.0, 1.0, 1.0]
                checks += 1
                window.energy_low.set("550")
                window._refresh_all()
                assert abs(window._sliders[0][0].get() - 550) < 10
                window._reset_filters()
                checks += 1
                window.overview_mode.set("Flood maps")
                window._draw_overview()
                checks += 1
                window.overview_mode.set("Counts")
                window._draw_overview()
                window.time_view.set("Paired delta-t")
                window._draw_timestamps()
                assert len(window.time_fig.axes) >= 1
                checks += 1
                window.time_view.set("Event rate")
                window.experimental.set(True)
                window._refresh_selected()
                window.experimental.set(False)
                window.config_path = str(config)
                window.calibration_path = str(calibration)
                window.system.set(system)
                window.files = [str(path)]
                window.max_pairs.set("200")
                window.min_channels.set("1")
                window.min_channel_energy.set("0")
                done = tk.BooleanVar(master=window, value=False)
                deadline = time.monotonic() + 40

                def wait_for_worker():
                    if not window._busy or time.monotonic() > deadline:
                        done.set(True)
                    else:
                        window.after(70, wait_for_worker)

                window._start_processing()
                window.after(70, wait_for_worker)
                window.wait_variable(done)
                assert not window._busy and window.dataset.files[0].pairs_accepted == 140
                checks += 1
                pdf_path = root / "gui_report.pdf"
                with patch("exe_programs.ldat_inspector_gui.filedialog.asksaveasfilename",
                           return_value=str(pdf_path)):
                    window._report(0)
                assert window._report_busy
                done.set(False)

                def wait_for_report():
                    if not window._report_busy or time.monotonic() > deadline:
                        done.set(True)
                    else:
                        window.after(70, wait_for_report)

                deadline = time.monotonic() + 40
                window.after(70, wait_for_report)
                window.wait_variable(done)
                assert not window._report_busy and pdf_path.exists() and len(PdfReader(str(pdf_path)).pages) == 3
                checks += 1
                window._invalidate_data()
                assert window.dataset is None and not window.channel_tree.get_children()
                window.calibrated.set(False)
                window._toggle_calibration()
                window.calibration_path = ""
                assert window.energy_low.get() == "0" and window.energy_high.get() == "300"
                assert "raw" in window.energy_group_title.cget("text").lower()
                checks += 1
                done.set(False)
                deadline = time.monotonic() + 40
                window._start_processing()
                window.after(70, wait_for_worker)
                window.wait_variable(done)
                assert not window._busy and window.dataset.settings.calibrated is False
                assert window.dataset.files[0].pairs_accepted == 140
                assert "raw a.u." in window.explorer_info.cget("text")
                assert "Raw energy" in window.energy_ax.get_xlabel()
                checks += 1
                window.calibration_path = str(calibration)
                window.calibrated.set(True)
                window._toggle_calibration()
                done.set(False)
                deadline = time.monotonic() + 40
                window.after(70, wait_for_worker)
                window.wait_variable(done)
                assert window.dataset.settings.calibrated and window.dataset.files[0].pairs_accepted == 140
                window.dataset = dataset
                window._update_after_processing()
                assert "140 after paired energy" in window.explorer_info.cget("text")
                checks += 1
        print(f"PASS: {checks}/{checks} hidden-GUI checks passed")
    finally:
        window.destroy()


if __name__ == "__main__":
    main()
