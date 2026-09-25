"""Reproduce the Cornell inspector's popup, fit-overlay and tolerance-colour issues.

Run without arguments for a quick hidden-GUI regression loop. Add --real to
check the supplied Cornell acquisition's first 80,000 coincidence pairs.
"""

import argparse
from pathlib import Path
import sys
import tempfile
import tkinter as tk
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import customtkinter as ctk
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from tkinter import ttk

from exe_programs.ldat_inspector_gui import LDATWorkbench
from scripts.ldat_inspector_check import fixture_files, write_pairs
from src.ldat_inspector import Selection, Settings, fit_on_display_bins, fit_peak, fit_peak_background, load_setup, merge_results, process_file
from src.ldat_report import write_report


def _descendant(widget, kind):
    for child in widget.winfo_children():
        if isinstance(child, kind):
            return child
        result = _descendant(child, kind)
        if result is not None:
            return result
    return None


def _check(label, action, failures):
    try:
        action()
        print(f"[ok  ] {label}")
    except (AssertionError, ValueError, RuntimeError) as exc:
        print(f"[FAIL] {label}: {exc}")
        failures.append(label)


def _gaussian_overlay(app, *, experimental=False):
    app.experimental.set(experimental)
    app._refresh_selected()
    prefix = "Photopeak model:"
    fit_lines = [line for line in app.energy_ax.lines if line.get_label().startswith(prefix)]
    assert fit_lines, f"No visible {prefix} overlay"
    line = fit_lines[0]
    data = app.dataset.modules[app._selected_sm()]
    fit = fit_peak(data.energy[app._selection().mask(data, energy=False)])
    assert fit["status"] == "FIT"
    all_bars = [patch for patch in app.energy_ax.patches if isinstance(patch, Rectangle)]
    edges = [patch.get_x() for patch in all_bars] + [all_bars[-1].get_x() + all_bars[-1].get_width()]
    predicted = fit_on_display_bins(fit, edges)
    assert np.allclose(line.get_ydata(), predicted["total"]), "Default curve omits fitted background"
    assert f"{fit['resolution']:.1f}%" in line.get_label(), "Photopeak resolution missing"
    mu_bin = max(line.get_xdata(), key=lambda x: -abs(x - 511))
    curve_peak = max(line.get_ydata())
    bars = [patch for patch in app.energy_ax.patches if isinstance(patch, Rectangle)
            and abs(patch.get_x() + patch.get_width()/2 - mu_bin) < 10]
    assert bars, "No histogram bars at fitted photopeak"
    hist_peak = max(bar.get_height() for bar in bars)
    ratio = curve_peak / hist_peak
    assert 0.8 <= ratio <= 1.2, f"total model / displayed histogram = {ratio:.3f} (expected near 1)"
    if experimental:
        labels = [item.get_label() for item in app.energy_ax.lines]
        assert any(label.startswith("Gaussian component") for label in labels), labels
        assert any(label.startswith("Estimated background") for label in labels), labels
        assert (any("Background-aware fit:" in label and "resolution" in label.lower()
                    for label in labels) or
                any("same background-aware fit" in label and "resolution" in label.lower()
                    for label in labels)), labels
        assert any("Background fit:" in item.get_text() and "resolution:" in item.get_text()
                   for item in app.energy_ax.texts), "Background-fit resolution readout missing"


def _comparison_overlay(app):
    data = app.dataset.modules[app._selected_sm()]
    auto = fit_peak(data.energy[app._selection().mask(data, energy=False)])
    app.experimental.set(True)
    app._refresh_selected()
    labels = [line.get_label() for line in app.energy_ax.lines]
    assert any(label.startswith("Photopeak model:") for label in labels), labels
    assert any(label.startswith("Gaussian component") for label in labels), labels
    assert any(label.startswith("Estimated background") for label in labels), labels
    if auto["status"] == "FIT" and auto["background_model"] == "linear":
        assert not any(label.startswith("Background-aware fit:") for label in labels), labels
        assert any("same background-aware fit" in label for label in labels), labels
    app.fit_low.set("400")
    app.fit_high.set("660")
    app._refresh_selected()
    labels = [line.get_label() for line in app.energy_ax.lines]
    assert any(label.startswith("Background-aware fit:") and "resolution" in label.lower()
               for label in labels), labels
    readout = [text.get_text() for text in app.energy_ax.texts]
    assert any("Background fit:" in text and "resolution:" in text for text in readout), readout
    app.fit_low.set("350")
    app.fit_high.set("700")
    app._refresh_selected()


def _popup(app, action, expected, *, require_parent=True):
    existing = set(app.winfo_children())
    action()
    app.update_idletasks()
    dialogs = [child for child in app.winfo_children() if isinstance(child, ctk.CTkToplevel)
               and child not in existing]
    assert len(dialogs) == 1, f"Expected {expected} dialog; got {len(dialogs)}"
    dialog = dialogs[0]
    try:
        if require_parent:
            assert str(dialog.transient()) == str(app), (
                f"{expected} detached from its parent; can drop behind main window")
        return dialog
    except Exception:
        dialog.destroy()
        raise


def _uniformity_colours(app):
    # Exercise the actual popup table with three verdicts; do not rely on a
    # lucky distribution of real photopeaks to cover all three states.
    rows = [
        {"sm": 0, "events": 1000, "fit": {"status": "FIT", "mu": 511., "resolution": 10.},
         "deviation_pct": 0., "result": "IN TOLERANCE"},
        {"sm": 1, "events": 1000, "fit": {"status": "FIT", "mu": 585., "resolution": 12.},
         "deviation_pct": 14.5, "result": "OUT OF TOLERANCE"},
        {"sm": 2, "events": 0, "fit": {"status": "insufficient events", "mu": None,
         "resolution": None}, "deviation_pct": None, "result": "UNAVAILABLE"},
    ]
    with patch("exe_programs.ldat_inspector_gui.uniformity", return_value=rows):
        dialog = _popup(app, app._uniformity, "Photopeak Uniformity", require_parent=False)
    try:
        tree = _descendant(dialog, ttk.Treeview)
        assert tree is not None and len(tree.get_children()) == 3
        verdicts = [tree.item(item, "tags") for item in tree.get_children()]
        assert len(set(verdicts)) == 3 and all(tags for tags in verdicts), (
            f"no distinct status colours for in/out/unavailable: {verdicts}")
        fills = [tree.tag_configure(tags[0]).get("background") for tags in verdicts]
        assert len(set(fills)) == 3 and all(fills), f"indistinguishable row fills: {fills}"
    finally:
        dialog.destroy()


def _visible_popups(app):
    """Check actual stacking/focus after Tk maps both dialogs on this desktop."""
    app.deiconify()
    app.update()
    try:
        app.experimental.set(True)
        for action, name in ((app._fit_settings, "fit settings"),
                             (app._uniformity, "photopeak uniformity"),
                             (app._residuals, "fit residuals"),
                             (lambda: app._show_profile(0, 102, 0, 102), "ROI profile")):
            if name == "photopeak uniformity":
                with patch("exe_programs.ldat_inspector_gui.uniformity", return_value=[]):
                    dialog = _popup(app, action, name)
            else:
                dialog = _popup(app, action, name)
            try:
                app.update()
                ready = tk.BooleanVar(master=app, value=False)
                app.after(100, lambda: ready.set(True))
                app.wait_variable(ready)
                assert dialog.winfo_viewable(), f"{name} not mapped"
                assert int(app.tk.call("wm", "stackorder", dialog._w, "isabove", app._w)), (
                    f"{name} behind inspector")
                focus = app.focus_displayof()
                assert str(focus) == str(dialog), f"{name} lacks keyboard focus: {focus!s}"
            finally:
                dialog.destroy()
                app.update()
    finally:
        app.withdraw()
        app.update()


def _real_probe():
    folder = Path(r"C:\Users\dsanchez\Desktop\data\Cornell\full_system")
    files = sorted(folder.glob("*coincCompact11s_0000000[3-8].ldat"))
    assert len(files) == 6, f"Expected the six Cornell acquisitions, found {len(files)}"
    root = Path(__file__).resolve().parent.parent
    settings = Settings(str(root / "configs/cornell_full_system.yaml"),
                        str(root / "encal_files/20260119_2NaSourcesAxialSeparated_vBiasCompDiscCalibAdjusted2hits_300s_coincFixed11s_fixed.encal"),
                        "CORNELL", 80_000, 4, 0.2)
    result = process_file(str(files[0]), settings)
    assert result.success and result.pairs_accepted > 10_000, (result.error, result.pairs_accepted)
    print(f"Cornell prefix: {result.pairs_accepted:,}/{result.pairs_read:,} coincidence pairs")
    return merge_results(settings, [result])


def _fit_stability(dataset=None):
    rng = np.random.default_rng(20260924)
    energies = np.concatenate((rng.normal(511, 38, 8000),
                               rng.uniform(300, 800, 4000)))
    fit = fit_peak(energies)
    assert fit["status"] == "FIT", fit["status"]
    assert abs(fit["mu"] - 511) < 8, f"broad photopeak centroid {fit['mu']:.1f} keV"
    assert abs(fit["sigma"] - 38) < 3.8, f"broad photopeak width {fit['sigma']:.1f} vs 38 keV"
    if dataset is not None:
        actual = dataset.modules[10].energy
        gaussian, continuum = fit_peak(actual), fit_peak_background(actual)
        assert continuum["status"] == "FIT"
        assert (gaussian["status"] != "FIT" or abs(gaussian["mu"] - continuum["mu"]) < 25), (
            f"Cornell SM10 photopeak {gaussian['mu']:.1f} vs supported continuum peak {continuum['mu']:.1f}")


def _pdf_overlay(dataset, sm, destination):
    """Capture the plotted PDF line and compare it with observed-bin total counts."""
    observed_lines = []
    original_plot = Axes.plot

    def capture(axis, *args, **kwargs):
        if len(args) >= 2:
            observed_lines.append((np.asarray(args[0]), np.asarray(args[1])))
        return original_plot(axis, *args, **kwargs)

    with patch.object(Axes, "plot", capture):
        write_report(destination, dataset, Selection(0, 1500), sm=sm)
    energies = dataset.modules[sm].energy
    fit = fit_peak(energies)
    assert fit["status"] == "FIT"
    display_edges = np.linspace(0, 1500, 141)
    expected = fit_on_display_bins(fit, display_edges)
    assert any(np.array_equal(x, expected["x"]) and np.allclose(y, expected["total"])
               for x, y in observed_lines), "PDF line is not the total-model counts/bin"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--visible", action="store_true", help="Map dialogs and check desktop stacking/focus")
    options = parser.parse_args()
    failures = []
    with tempfile.TemporaryDirectory() as temporary:
        if options.real:
            dataset = _real_probe()
            fitted = next((sm for sm in sorted(dataset.modules)
                           if fit_peak(dataset.modules[sm].energy)["status"] == "FIT"), None)
            assert fitted is not None, "Real Cornell prefix has no suitable fitted SuperModule"
            print(f"Cornell inspected SuperModule: {fitted}")
        else:
            root = Path(temporary)
            config, calibration, channels = fixture_files(root, "CORNELL")
            file = root / "fixture.ldat"
            write_pairs(file, channels, 140)
            settings = Settings(str(config), str(calibration), "CORNELL", 200)
            dataset = merge_results(settings, [process_file(str(file), settings)], load_setup(settings))
            fitted = 0
            data = dataset.modules[fitted]
            sample = 30_000
            for key in vars(data):
                setattr(data, key, np.resize(getattr(data, key), sample))
            data.energy = np.random.default_rng(42).normal(511, 20, sample)
            data.partner_energy = np.full(sample, 511.)
        app = LDATWorkbench()
        app.withdraw()
        try:
            app.dataset = dataset
            app._update_after_processing()
            app.module_var.set(f"SM {fitted}")
            _check("Gaussian peak follows displayed energy bars", lambda: _gaussian_overlay(app), failures)
            _check("background-inclusive fit follows displayed energy bars", lambda: _gaussian_overlay(app, experimental=True), failures)
            _check("default comparison avoids duplicates; adjusted fit remains distinct",
                   lambda: _comparison_overlay(app), failures)
            _check("advanced settings stay above parent", lambda: _popup(app, app._fit_settings, "fit settings").destroy(), failures)
            _check("photopeak uniformity stays above parent",
                   lambda: _popup(app, app._uniformity, "Photopeak Uniformity").destroy(), failures)
            _check("uniformity tolerance has distinct colours", lambda: _uniformity_colours(app), failures)
            _check("broad and Cornell photopeaks have credible centroid/width",
                   lambda: _fit_stability(dataset if options.real else None), failures)
            _check("PDF line plots the same total photopeak model",
                   lambda: _pdf_overlay(dataset, fitted, Path(temporary) / "fit.pdf"), failures)
            if options.visible:
                _check("mapped fit/uniformity/residual/profile dialogs above parent and focused",
                       lambda: _visible_popups(app), failures)
        finally:
            app.destroy()
    total = 9 if options.visible else 8
    print(f"{'FAIL' if failures else 'PASS'}: {total - len(failures)}/{total} checks passed")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
