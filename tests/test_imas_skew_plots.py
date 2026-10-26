"""IMAS skew diagnostics written headless from a synthetic LUT (spec 011 T14, FR-11)."""
import numpy as np
import pytest

from imas_skew_helpers import CONFIG
from src.imas.skew import outputs, plots
from src.imas.skew.setup import ImasSetup
from src.timing import plots as core_plots, solver as ts

pytestmark = pytest.mark.fr("011-FR-11")


@pytest.fixture(scope="module")
def setup():
    return ImasSetup.from_files(CONFIG)


def test_layout_places_every_minimodule_once(setup):
    place = plots.layout(setup)
    assert (place["rings"], place["n_azimuth"], place["rows"], place["columns"]) == (5, 24, 4, 4)
    lookup = setup.lookup()
    sm, mm = lookup.sm[setup.time_channels], lookup.mm[setup.time_channels]
    cells = {}
    for key, cell in zip(zip(sm.tolist(), mm.tolist()),
                         zip(place["ring"], place["azimuth"], place["row"], place["column"])):
        cells.setdefault(cell, set()).add(key)
    assert len(cells) == 120 * 16 and all(len(keys) == 1 for keys in cells.values())


def test_every_figure_written_from_synthetic_lut(setup, tmp_path):
    rng = np.random.default_rng(2)
    times = setup.time_channels
    rows = np.zeros(len(times), ts.CHANNEL_DTYPE)
    rows["time_channel"] = times
    applied = rng.random(len(times)) > 0.05
    rows["delay_ps"] = np.where(applied, rng.normal(0, 800, len(times)), np.nan)
    rows["sigma_ps"] = rows["raw_sigma_ps"] = np.where(applied, 25.0, np.nan)
    rows["residual_rms"] = 1.0
    rows["status"] = np.where(applied, ts.LOW_STATISTICS, ts.UNOBSERVED)
    rows["apply"], rows["component"] = applied, np.where(applied, 0, -1)
    path = tmp_path / "lut.csv"
    outputs.write_imas_lut(path, ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {}), {}, setup)
    legacy = {int(c): (float(d) + rng.normal(0, 90), np.nan) for c, d in zip(times[applied], rows["delay_ps"][applied])}
    shifts = rng.normal(0, 20, applied.sum())
    train = {"shift_ps": shifts, "shift_sigma_ps": np.full(applied.sum(), 20.0)}
    heldout = {"shift_ps": shifts * 1.5, "shift_sigma_ps": np.full(applied.sum(), 20.0)}
    result = plots.write_imas_plots(path, setup, legacy=legacy, residuals=(train, heldout))
    expected = list(plots.paths(path).values()) + core_plots.plot_paths(path)
    assert all(p.exists() and p.stat().st_size > 1000 for p in expected if p.suffix == ".png")
    assert sorted(result["files"]) == sorted(str(p) for p in expected)
    assert result["legacy"]["channels"] == applied.sum() and 70 < result["legacy"]["robust_sd_ps"] < 110
    with pytest.raises(core_plots.PlotError, match="exist"):
        plots.write_imas_plots(path, setup)
