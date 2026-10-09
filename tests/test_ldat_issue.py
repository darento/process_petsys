"""Cornell inspector popup, fit-overlay and tolerance-colour regressions (spec 001 T7, T8, T11, T17;
spec 007 T21).

Moved from ``scripts/ldat_issue_check.py`` (fixture mode): each ``_check`` is one test. The fixture is
the Cornell two-module file with SM 0 replaced by a 30,000-side normal(511, 20) keV spectrum. The
window checks are consecutive steps on one hidden window, so ``steps`` runs them once in order and
records each step's exception, which its test re-raises. ``--real`` (photopeak probe) is retired
(spec 007 Clarify); ``--visible`` (mapped dialogs, desktop stacking and focus) is not migrated (T21).
"""

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from ldat_helpers import destroy, fixture_files, write_pairs
from src.ldat_inspector.engine import (Selection, Settings, SideTable, fit_on_display_bins, fit_peak, load_setup,
                                       merge_results, process_file)
from src.ldat_inspector.report import write_report

pytestmark = pytest.mark.fr("001-FR-15")
fr = pytest.mark.fr


def descendant(widget, kind):
    for child in widget.winfo_children():
        if isinstance(child, kind):
            return child
        result = descendant(child, kind)
        if result is not None:
            return result
    return None


def single_module_dataset(dataset, energies):
    """SM 0 alone, as paired sides with the given calibrated energies.

    Spec 002 stores sides once in a SideTable, so the synthetic 30,000-side spectrum is a new table
    rather than overwritten module attributes. Other columns repeat the fixture's SM 0 values.
    """
    n = len(energies) // 2 * 2
    template = dataset.modules[0]
    repeat = lambda values: np.resize(values, n)
    table = SideTable.from_pairs(
        0, raw_energy=energies[:n], calibration_key=repeat(template.calibration_key),
        x=repeat(template.x), y=repeat(template.y), doi=repeat(template.doi),
        timestamp=repeat(template.timestamp), sm=np.zeros(n, int), mm=repeat(template.mm),
        random_slab=np.zeros(n, bool))
    return replace(dataset, table=table, modules=table.by_sm())


@pytest.fixture(scope="module")
def fixture():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config, calibration, channels = fixture_files(root, "CORNELL")
        file = root / "fixture.ldat"
        write_pairs(file, channels, 140)
        settings = Settings(str(config), str(calibration), "CORNELL", 200)
        dataset = merge_results(settings, [process_file(str(file), settings)], load_setup(settings))
        yield SimpleNamespace(root=root, sm=0,
                              dataset=single_module_dataset(dataset, np.random.default_rng(42).normal(511, 20, 30_000)))


# --- window steps --------------------------------------------------------------

def gaussian_overlay(app, *, experimental=False):
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


def comparison_overlay(app):
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


def popup(app, action, expected, *, require_parent=True):
    import customtkinter as ctk

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


def uniformity_colours(app):
    from tkinter import ttk

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
        dialog = popup(app, app._uniformity, "Photopeak Uniformity", require_parent=False)
    try:
        tree = descendant(dialog, ttk.Treeview)
        assert tree is not None and len(tree.get_children()) == 3
        verdicts = [tree.item(item, "tags") for item in tree.get_children()]
        assert len(set(verdicts)) == 3 and all(tags for tags in verdicts), (
            f"no distinct status colours for in/out/unavailable: {verdicts}")
        fills = [tree.tag_configure(tags[0]).get("background") for tags in verdicts]
        assert len(set(fills)) == 3 and all(fills), f"indistinguishable row fills: {fills}"
    finally:
        dialog.destroy()


STEPS = {
    "gaussian": lambda app: gaussian_overlay(app),
    "background": lambda app: gaussian_overlay(app, experimental=True),
    "comparison": comparison_overlay,
    "fit settings": lambda app: popup(app, app._fit_settings, "fit settings").destroy(),
    "uniformity": lambda app: popup(app, app._uniformity, "Photopeak Uniformity").destroy(),
    "colours": uniformity_colours,
}


@pytest.fixture(scope="module")
def steps(fixture):
    """Step name -> None, or the exception it raised; the steps run once, in order, on one window."""
    import tkinter as tk

    from exe_programs.ldat_inspector_gui import LDATWorkbench

    try:
        app = LDATWorkbench()
    except tk.TclError as exc:
        pytest.skip(f"gui: no display available ({exc})")
    app.withdraw()
    seen = {}
    try:
        app.dataset = fixture.dataset
        app._update_after_processing()
        app.module_var.set(f"SM {fixture.sm}")
        for name, action in STEPS.items():
            try:
                action(app)
                seen[name] = None
            except Exception as exc:
                seen[name] = exc
        yield seen
    finally:
        destroy(app)


def step(steps, name):
    if steps[name] is not None:
        raise steps[name]


@pytest.mark.gui
@fr("001-FR-16", "001-FR-20", "001-FR-27")
def test_gaussian_peak_follows_displayed_energy_bars(steps):
    step(steps, "gaussian")


@pytest.mark.gui
@fr("001-FR-10", "001-FR-20", "001-FR-27")
def test_background_inclusive_fit_follows_displayed_energy_bars(steps):
    step(steps, "background")


@pytest.mark.gui
@fr("001-FR-10", "001-FR-27")
def test_default_comparison_avoids_duplicates_adjusted_fit_remains_distinct(steps):
    step(steps, "comparison")


@pytest.mark.gui
@fr("001-FR-17")
def test_advanced_settings_stay_above_parent(steps):
    step(steps, "fit settings")


@pytest.mark.gui
@fr("001-FR-17")
def test_photopeak_uniformity_stays_above_parent(steps):
    step(steps, "uniformity")


@pytest.mark.gui
@fr("001-FR-18")
def test_uniformity_tolerance_has_distinct_colours(steps):
    step(steps, "colours")


# --- engine and report ----------------------------------------------------------

@fr("001-FR-8", "001-FR-16")
def test_broad_photopeak_has_credible_centroid_and_width():
    rng = np.random.default_rng(20260924)
    energies = np.concatenate((rng.normal(511, 38, 8000), rng.uniform(300, 800, 4000)))
    fit = fit_peak(energies)
    assert fit["status"] == "FIT", fit["status"]
    assert abs(fit["mu"] - 511) < 8, f"broad photopeak centroid {fit['mu']:.1f} keV"
    assert abs(fit["sigma"] - 38) < 3.8, f"broad photopeak width {fit['sigma']:.1f} vs 38 keV"


@fr("001-FR-12", "001-FR-16")
def test_pdf_line_plots_the_same_total_photopeak_model(fixture):
    """Capture the plotted PDF line and compare it with observed-bin total counts."""
    observed_lines = []
    original_plot = Axes.plot

    def capture(axis, *args, **kwargs):
        if len(args) >= 2:
            observed_lines.append((np.asarray(args[0]), np.asarray(args[1])))
        return original_plot(axis, *args, **kwargs)

    with patch.object(Axes, "plot", capture):
        write_report(fixture.root / "fit.pdf", fixture.dataset, Selection(0, 1500), sm=fixture.sm)
    fit = fit_peak(fixture.dataset.modules[fixture.sm].energy)
    assert fit["status"] == "FIT"
    expected = fit_on_display_bins(fit, np.linspace(0, 1500, 141))
    assert any(np.array_equal(x, expected["x"]) and np.allclose(y, expected["total"])
               for x, y in observed_lines), "PDF line is not the total-model counts/bin"
