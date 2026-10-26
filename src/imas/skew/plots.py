"""IMAS skew diagnostics (spec 011 T14, FR-11). Diagnostics only, not validation evidence.

``python -m src.imas.skew plot LUT.csv [--legacy skew_IMAS.txt] [--observations OBS --train-files 0 1
--heldout-files 2]`` writes next to the LUT, never overwriting:

* the per-ASIC plots of ``src.timing.plots``, for IMAS global IDs;
* ``<lut>_overview.png``: median applied delay per SM and per minimodule laid
  out by ring x azimuth, like the Inspector's System Overview. Minimodules
  are placed by their local transaxial (TIME) and axial (ENERGY) coordinates;
* ``<lut>_legacy.png`` (with ``--legacy``): new against legacy delays (``-skew``,
  median-aligned), the difference histogram and the per-SM median difference
  by ring x azimuth;
* ``<lut>_residuals.png`` (with ``--observations``): per channel, the
  one-step delay change the train and the held-out events ask for, other
  delays fixed (``src.timing.em.evaluate``), normalised by its uncertainty.
  Held-out values much wider than train flag delays that do not generalise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from matplotlib.figure import Figure

from src.imas.skew import outputs, solve
from src.imas.skew.setup import ImasSetup
from src.timing import em, plots as core_plots
from src.timing.cache import ObservationSet
from src.timing.lut import LutError

SUFFIXES = ("_overview.png", "_legacy.png", "_residuals.png")
MODULES_PER_SIDE = 4


def layout(setup):
    """(ring, azimuth, axial row, transaxial column) per IMAS TIME channel, from the map's local frame."""
    lookup = setup.lookup()
    channels = setup.time_channels
    sm, mm = lookup.sm[channels].astype(np.int64), lookup.mm[channels].astype(np.int64)
    energy = np.flatnonzero(lookup.kind == 2)
    x = {}
    for key, value in zip(zip(sm.tolist(), mm.tolist()), lookup.coord_x[channels].tolist()):
        x.setdefault(key, []).append(value)
    y = {}
    for key, value in zip(zip(lookup.sm[energy].tolist(), lookup.mm[energy].tolist()), lookup.coord_y[energy].tolist()):
        y.setdefault(key, []).append(value)
    centre_x = {k: float(np.median(v)) for k, v in x.items()}
    centre_y = {k: float(np.median(v)) for k, v in y.items()}

    def rank(values):
        levels = np.unique(np.round(list(values), 3))
        return {v: k for k, v in enumerate(levels)}
    column_of = rank(centre_x.values())
    row_of = rank(centre_y.values())
    n_azimuth = setup.n_azimuth
    keys = list(zip(sm.tolist(), mm.tolist()))
    return {"ring": sm // n_azimuth, "azimuth": sm % n_azimuth,
            "row": np.array([row_of.get(round(centre_y.get(k, np.nan), 3), 0) for k in keys]),
            "column": np.array([column_of[round(centre_x[k], 3)] for k in keys]),
            "rings": len(setup.ring_z), "n_azimuth": n_azimuth,
            "rows": max(len(row_of), 1), "columns": max(len(column_of), 1)}


def _median_grid(index_rows, index_columns, values, shape):
    grid = np.full(shape, np.nan)
    finite = np.isfinite(values)
    keys = index_rows[finite] * shape[1] + index_columns[finite]
    for key in np.unique(keys):
        grid.flat[key] = np.median(values[finite][keys == key])
    return grid


def _image(ax, grid, title, label, diverging=False, fig=None):
    finite = grid[np.isfinite(grid)]
    if diverging:
        limit = float(np.percentile(np.abs(finite), 98)) if len(finite) else 1.0
        image = ax.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
    else:
        image = ax.imshow(grid, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=label)
    return image


def overview(setup, channels, delays_ps, title):
    """Median applied delay per SM (rings x azimuths) and per minimodule (rings*rows x azimuths*columns)."""
    place = layout(setup)
    position = np.searchsorted(setup.time_channels, channels)
    value = np.full(len(setup.time_channels), np.nan)
    value[position] = delays_ps
    fig = Figure(figsize=(16, 9))
    top, bottom = fig.subplots(2, 1)
    _image(top, _median_grid(place["ring"], place["azimuth"], value, (place["rings"], place["n_azimuth"])),
           f"{title}: median delay per SM", "ps", fig=fig)
    top.set_xlabel("azimuthal SM index")
    top.set_ylabel("ring")
    rows = place["ring"] * place["rows"] + place["row"]
    columns = place["azimuth"] * place["columns"] + place["column"]
    _image(bottom, _median_grid(rows, columns, value, (place["rings"] * place["rows"],
                                                         place["n_azimuth"] * place["columns"])),
           f"{title}: median delay per minimodule", "ps", fig=fig)
    for ring in range(1, place["rings"]):
        bottom.axhline(ring * place["rows"] - 0.5, color="w", lw=0.8)
    for azimuth in range(1, place["n_azimuth"]):
        bottom.axvline(azimuth * place["columns"] - 0.5, color="w", lw=0.3)
    bottom.set_xlabel("azimuthal SM x transaxial minimodule column")
    bottom.set_ylabel("ring x axial minimodule row")
    fig.tight_layout()
    return fig


def legacy_figure(setup, new, legacy, title):
    """``new``/``legacy``: {channel: (delay, sigma)}; median-aligned comparison over common channels."""
    common = np.array(sorted(set(new) & set(legacy)), np.int64)
    fig = Figure(figsize=(16, 10))
    scatter, histogram, grid = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 1, 2)
    if not len(common):
        scatter.set_title("no common channels")
        return fig
    a = np.array([new[c][0] for c in common])
    b = np.array([legacy[c][0] for c in common])
    difference = a - b
    difference -= np.median(difference)
    scatter.scatter(b / 1e3, (a - np.median(a - b)) / 1e3, s=1, alpha=0.3)
    line = [min(b.min(), a.min()) / 1e3, max(b.max(), a.max()) / 1e3]
    scatter.plot(line, line, "r-", lw=0.8)
    scatter.set_xlabel("legacy delay -skew (ns)")
    scatter.set_ylabel("new delay, median-aligned (ns)")
    scatter.set_title(f"{title}: {len(common)} common channels")
    robust = 1.4826 * float(np.median(np.abs(difference)))
    limit = max(5 * robust, 1.0)
    histogram.hist(np.clip(difference, -limit, limit), bins=120)
    histogram.set_xlabel("new - legacy, median-aligned (ps)")
    histogram.set_title(f"robust sd {robust:.1f} ps")
    place = layout(setup)
    value = np.full(len(setup.time_channels), np.nan)
    value[np.searchsorted(setup.time_channels, common)] = difference
    _image(grid, _median_grid(place["ring"], place["azimuth"], value, (place["rings"], place["n_azimuth"])),
           "median (new - legacy) per SM", "ps", diverging=True, fig=fig)
    grid.set_xlabel("azimuthal SM index")
    grid.set_ylabel("ring")
    fig.tight_layout()
    return fig


def residual_figure(train, heldout, title):
    """``train``/``heldout``: evaluate(..., channel_shifts=True) results on the same channel list.

    The train change is about zero at the fixed point (a convergence check), so it is reported, not drawn;
    the held-out change over its uncertainty should be N(0, 1) when the uncertainties are right.
    """
    fig = Figure(figsize=(14, 6))
    left, right = fig.subplots(1, 2)

    def spread(values):
        values = values[np.isfinite(values)]
        return (1.4826 * np.median(np.abs(values - np.median(values))) if len(values) else np.nan), len(values)

    z = heldout["shift_ps"] / heldout["shift_sigma_ps"]
    z_sd, count = spread(z)
    train_sd, _ = spread(train["shift_ps"])
    left.hist(np.clip(z[np.isfinite(z)], -8, 8), bins=80, histtype="step", density=True,
              label=f"held out: robust sd {z_sd:.2f} ({count} channels)")
    grid = np.linspace(-8, 8, 200)
    left.plot(grid, np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi), "k--", lw=0.8, label="N(0, 1)")
    left.set_xlabel("held-out one-step delay change / its uncertainty")
    left.legend(fontsize=8)
    left.set_title(f"{title}: residual consistency (train change robust sd {train_sd:.2f} ps)")
    shift = heldout["shift_ps"][np.isfinite(heldout["shift_ps"])]
    shift_sd, _ = spread(shift)
    right.hist(np.clip(shift, -500, 500), bins=100, histtype="step")
    right.set_xlabel("held-out one-step delay change (ps)")
    right.set_title(f"robust sd {shift_sd:.1f} ps; median uncertainty "
                    f"{np.nanmedian(heldout['shift_sigma_ps']):.1f} ps")
    fig.tight_layout()
    return fig


def paths(lut_path):
    lut_path = Path(lut_path)
    return {suffix: lut_path.with_name(lut_path.stem + suffix) for suffix in SUFFIXES}


def write_imas_plots(lut_path, setup, *, legacy=None, residuals=None):
    """Write every requested figure; refuses existing outputs before writing anything."""
    targets = paths(lut_path)
    wanted = {"_overview.png": True, "_legacy.png": legacy is not None, "_residuals.png": residuals is not None}
    planned = [targets[s] for s, on in wanted.items() if on] + core_plots.plot_paths(lut_path)
    existing = [str(p) for p in planned if p.exists()]
    if existing:
        raise core_plots.PlotError(f"diagnostic outputs exist: {existing}; choose a new LUT name")
    table = outputs.read_imas_lut(lut_path, setup)
    rows = table.rows[table.rows["apply"]]
    title = Path(lut_path).stem
    result = {"asic": core_plots.write_asic_plots(lut_path, rows["time_channel"], rows["delay_ps"], title=title)}
    overview(setup, rows["time_channel"], rows["delay_ps"], title).savefig(targets["_overview.png"], dpi=110)
    result["files"] = [str(targets["_overview.png"])] + result["asic"]["files"]
    if legacy is not None:
        new = {int(c): (float(d), float(s)) for c, d, s in zip(rows["time_channel"], rows["delay_ps"], rows["sigma_ps"])}
        legacy_figure(setup, new, legacy, title).savefig(targets["_legacy.png"], dpi=110)
        result["files"].append(str(targets["_legacy.png"]))
        result["legacy"] = solve.compare(new, legacy)
    if residuals is not None:
        residual_figure(*residuals, title).savefig(targets["_residuals.png"], dpi=110)
        result["files"].append(str(targets["_residuals.png"]))
    return result


def residuals(lut_path, setup, observations, train_files, heldout_files):
    """Per-channel one-step changes on the train and held-out events with the LUT's delays fixed."""
    table = outputs.read_imas_lut(lut_path, setup)
    rows = table.rows[table.rows["apply"]]
    allowed = rows["time_channel"].astype(np.int64)
    delays = rows["delay_ps"]
    results = []
    for files in (train_files, heldout_files):
        def keep(selected, files=files):
            return (np.isin(selected["file"], files) & np.isin(selected["channel_i"], allowed)
                    & np.isin(selected["channel_j"], allowed))
        data = em.EmData.from_observation_set(observations, allowed, roles=(0, 1, 2), where=keep)
        results.append(em.evaluate(data, delays, solve.EM_OPTIONS, channel_shifts=True))
    return tuple(results)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew plot", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lut", type=Path, help="canonical IMAS LUT CSV")
    parser.add_argument("--config", type=Path, default=Path("configs/imas_2DAQ.yaml"))
    parser.add_argument("--legacy", type=Path, help="legacy ch-skew file to compare against")
    parser.add_argument("--observations", type=Path, help="observation set for train/held-out residuals")
    parser.add_argument("--train-files", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--heldout-files", type=int, nargs="+", default=[2])
    args = parser.parse_args(argv)
    setup = ImasSetup.from_files(args.config)
    try:
        legacy = solve.legacy_table(args.legacy) if args.legacy else None
        shifts = (residuals(args.lut, setup, ObservationSet(args.observations), args.train_files, args.heldout_files)
                  if args.observations else None)
        result = write_imas_plots(args.lut, setup, legacy=legacy, residuals=shifts)
    except (LutError, core_plots.PlotError, OSError) as error:
        print(f"plots failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"files": result["files"], "asic_summary": result["asic"]["summary"],
                      "legacy": result.get("legacy")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
