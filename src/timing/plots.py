"""Per-ASIC diagnostics of a canonical Cornell TIME-channel delay LUT (spec 008 T7b, FR-19).

Usage::

    python -m src.cornell.skew plot LUT.csv [--threshold-ps 1000]

Global TIME-channel IDs decode into PETsys port/slave/chip/channel
(``src.utils.get_electronics_nums``); one ASIC is one (port, slave, chip).
Applied delays are modelled as an ASIC offset plus a channel-index profile
shared by all ASICs (owner observation 2026-10-13: a ~1.4 ns within-ASIC
pattern repeats on every ASIC). Both are robust medians, alternated a few
times because ASICs read different channel subsets; the profile has zero
median over its observed indices. A channel whose residual exceeds the
threshold is listed; statuses and delays are never changed. Four PNG plots
and a per-ASIC summary CSV are written next to the LUT, never overwriting an
existing file. Figures use matplotlib's ``Figure`` directly, so no display or
pyplot state is needed. Diagnostics only, not validation evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from matplotlib.figure import Figure

from src.utils import get_electronics_nums

CHANNELS_PER_ASIC = 64
DEFAULT_THRESHOLD_PS = 1000.0
PLOT_SUFFIXES = ("_asic_within.png", "_asic_heatmap.png", "_asic_overview.png", "_asic_median_slope.png",
                 "_asic_summary.csv")


class PlotError(ValueError):
    """An output exists already or the input cannot be plotted."""


def plot_paths(lut_path):
    """Output files of ``lut_path`` (``out/lut.csv`` -> ``out/lut_asic_within.png`` ...)."""
    lut_path = Path(lut_path)
    return [lut_path.with_name(lut_path.stem + suffix) for suffix in PLOT_SUFFIXES]


def asic_pattern(time_channels, delays_ps, *, min_asics=3, iterations=4):
    """Offsets per ASIC, the shared channel-index profile and residuals of finite delays.

    Returns a dict of per-channel arrays (``time_channel``, ``delay_ps``, ``port``, ``slave``, ``chip``,
    ``channel``, ``asic``, ``relative_ps`` = delay - ASIC offset, ``residual_ps``), per-ASIC arrays
    (``asics``, ``offset_ps``, ``median_ps``, ``slope_ps_per_channel``, ``count``) and ``profile_ps``
    (64 values; NaN where fewer than ``min_asics`` ASICs observe the index, whose residuals are NaN).
    """
    time_channels = np.asarray(time_channels, np.int64)
    delays = np.asarray(delays_ps, float)
    keep = np.isfinite(delays)
    time_channels, delays = time_channels[keep], delays[keep]
    order = np.argsort(time_channels, kind="stable")
    time_channels, delays = time_channels[order], delays[order]
    numbers = np.array([get_electronics_nums(int(c)) for c in time_channels], np.int64).reshape(-1, 4)
    asic_id = time_channels // CHANNELS_PER_ASIC
    channel = numbers[:, 3]
    asics, index = np.unique(asic_id, return_inverse=True)
    profile = np.zeros(CHANNELS_PER_ASIC)
    offset = np.zeros(len(asics))
    observed = np.array([len(np.unique(index[channel == k])) for k in range(CHANNELS_PER_ASIC)])
    shared = observed >= min_asics
    for _ in range(iterations):
        offset = np.array([np.median(delays[index == a] - profile[channel[index == a]]) for a in range(len(asics))])
        relative = delays - offset[index]
        profile = np.array([np.median(relative[channel == k]) if shared[k] else 0.0
                            for k in range(CHANNELS_PER_ASIC)])
        if shared.any():  # gauge: the profile has zero median over its shared indices
            shift = np.median(profile[shared])
            profile[shared] -= shift
    if len(asics):
        offset = np.array([np.median(delays[index == a] - profile[channel[index == a]]) for a in range(len(asics))])
    relative = delays - offset[index]
    residual = np.where(shared[channel], relative - profile[channel], np.nan)
    median = np.array([np.median(delays[index == a]) for a in range(len(asics))])
    slope = np.array([np.polyfit(channel[index == a], delays[index == a], 1)[0]
                      if len(np.unique(channel[index == a])) >= 3 else np.nan for a in range(len(asics))])
    return {"time_channel": time_channels, "delay_ps": delays, "port": numbers[:, 0], "slave": numbers[:, 1],
            "chip": numbers[:, 2], "channel": channel, "asic": asic_id, "asic_index": index,
            "relative_ps": relative, "residual_ps": residual, "asics": asics, "offset_ps": offset,
            "median_ps": median, "slope_ps_per_channel": slope, "count": np.bincount(index, minlength=len(asics)),
            "profile_ps": np.where(shared, profile, np.nan)}


def pattern_outliers(pattern, threshold_ps=DEFAULT_THRESHOLD_PS):
    """Channels whose residual from ASIC offset + channel-index profile exceeds ``threshold_ps``."""
    residual = pattern["residual_ps"]
    rows = np.flatnonzero(np.abs(np.nan_to_num(residual)) > threshold_ps)
    return [{"time_channel": int(pattern["time_channel"][k]), "port": int(pattern["port"][k]),
             "slave": int(pattern["slave"][k]), "chip": int(pattern["chip"][k]),
             "channel": int(pattern["channel"][k]), "delay_ps": float(pattern["delay_ps"][k]),
             "residual_ps": float(residual[k])} for k in rows]


def _limits(values, scale=1.3):
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    low, high = np.percentile(values, [0.5, 99.5]) * scale
    return (low, high) if high > low else None


def _within(pattern, title):
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot()
    for a in range(len(pattern["asics"])):
        rows = pattern["asic_index"] == a
        ax.plot(pattern["channel"][rows], pattern["relative_ps"][rows], color="0.6", lw=0.3, alpha=0.4)
    channel, relative = pattern["channel"], pattern["relative_ps"]
    present = [k for k in range(CHANNELS_PER_ASIC) if np.isfinite(pattern["profile_ps"][k])]
    if present:
        q = np.array([np.percentile(relative[channel == k], [16, 50, 84]) for k in present])
        ax.plot(present, q[:, 1], "r-o", ms=3, label="median over ASICs")
        ax.fill_between(present, q[:, 0], q[:, 2], color="r", alpha=0.2, label="16-84 %")
        ax.legend()
    limits = _limits(relative)
    if limits:
        ax.set_ylim(*limits)
    ax.set_xlabel("channel in ASIC (0-63)")
    ax.set_ylabel("delay - ASIC offset (ps)")
    ax.set_title(f"{title}: within-ASIC channel trend ({len(pattern['asics'])} ASICs)")
    fig.tight_layout()
    return fig


def _heatmap(pattern, title):
    grid = np.full((max(len(pattern["asics"]), 1), CHANNELS_PER_ASIC), np.nan)
    grid[pattern["asic_index"], pattern["channel"]] = pattern["relative_ps"]
    fig = Figure(figsize=(10, max(6.0, len(pattern["asics"]) * 0.06)))
    ax = fig.add_subplot()
    finite = np.abs(grid[np.isfinite(grid)])
    limit = float(np.percentile(finite, 98)) if len(finite) and np.percentile(finite, 98) > 0 else 1.0
    image = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xlabel("channel in ASIC")
    ax.set_ylabel("ASIC (port/slave/chip order)")
    ax.set_title(f"{title}: delay - ASIC offset")
    fig.colorbar(image, ax=ax, label="ps")
    fig.tight_layout()
    return fig


def _port_lines(axes, pattern, top_axis):
    for port in np.unique(pattern["port"]):
        first = pattern["asic_index"][pattern["port"] == port].min()
        for ax in axes:
            ax.axvline(first, color="k", lw=0.5)
        top_axis.text(first, 1.0, f" port {port}", va="top", fontsize=8, transform=top_axis.get_xaxis_transform())


def _overview(pattern, title):
    fig = Figure(figsize=(16, 8))
    top, bottom = fig.subplots(2, 1, sharex=True)
    x = pattern["asic_index"] + pattern["channel"] / CHANNELS_PER_ASIC
    colour = pattern["asic_index"] % 10
    top.scatter(x, pattern["delay_ps"] / 1e3, c=colour, cmap="tab10", s=3, vmin=0, vmax=9)
    top.set_ylabel("delay (ns)")
    top.set_title(f"{title}: delay per channel grouped by ASIC")
    bottom.scatter(x, pattern["relative_ps"], c=colour, cmap="tab10", s=3, vmin=0, vmax=9)
    bottom.set_ylabel("delay - ASIC offset (ps)")
    limits = _limits(pattern["relative_ps"])
    if limits:
        bottom.set_ylim(*limits)
    bottom.set_xlabel("ASIC (port/slave/chip order) + channel / 64")
    _port_lines((top, bottom), pattern, top)
    fig.tight_layout()
    return fig


def _median_slope(pattern, title):
    fig = Figure(figsize=(14, 6))
    top, bottom = fig.subplots(2, 1, sharex=True)
    x = np.arange(len(pattern["asics"]))
    top.plot(x, pattern["median_ps"] / 1e3, "o", ms=3, label="median")
    top.plot(x, pattern["offset_ps"] / 1e3, "x", ms=3, label="offset (pattern removed)")
    top.set_ylabel("ASIC delay (ns)")
    top.legend()
    top.set_title(f"{title}: per-ASIC median and slope")
    bottom.plot(x, pattern["slope_ps_per_channel"], "o", ms=3)
    bottom.set_ylabel("slope (ps / channel)")
    bottom.set_xlabel("ASIC (port/slave/chip order)")
    _port_lines((top, bottom), pattern, top)
    fig.tight_layout()
    return fig


def write_asic_plots(lut_path, time_channels, delays_ps, *, threshold_ps=DEFAULT_THRESHOLD_PS, title=None):
    """Write the four PNG plots and summary CSV for ``lut_path``; returns files, outliers and a short summary."""
    paths = plot_paths(lut_path)
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise PlotError(f"per-ASIC outputs exist: {existing}; choose a new LUT name")
    pattern = asic_pattern(time_channels, delays_ps)
    outliers = pattern_outliers(pattern, threshold_ps)
    title = title or Path(lut_path).stem
    for path, make in zip(paths, (_within, _heatmap, _overview, _median_slope)):
        make(pattern, title).savefig(path, dpi=110)
    flagged = np.bincount(pattern["asic_index"][np.abs(np.nan_to_num(pattern["residual_ps"])) > threshold_ps],
                          minlength=len(pattern["asics"]))
    with open(paths[-1], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("asic", "port", "slave", "chip", "channels", "offset_ps", "median_ps",
                         "slope_ps_per_channel", "pattern_outliers"))
        for a, asic in enumerate(pattern["asics"]):
            port, slave, chip, _ = get_electronics_nums(int(asic) * CHANNELS_PER_ASIC)
            writer.writerow((int(asic), port, slave, chip, int(pattern["count"][a]), f"{pattern['offset_ps'][a]:.1f}",
                             f"{pattern['median_ps'][a]:.1f}", f"{pattern['slope_ps_per_channel'][a]:.3f}",
                             int(flagged[a])))
    residual = pattern["residual_ps"][np.isfinite(pattern["residual_ps"])]
    profile = pattern["profile_ps"][np.isfinite(pattern["profile_ps"])]
    summary = {"channels": int(len(pattern["delay_ps"])), "asics": int(len(pattern["asics"])),
               "profile_range_ps": float(np.ptp(profile)) if len(profile) else None,
               "asic_offset_sd_ps": float(np.std(pattern["offset_ps"])) if len(pattern["asics"]) else None,
               "residual_robust_sd_ps": float(1.4826 * np.median(np.abs(residual))) if len(residual) else None}
    return {"threshold_ps": float(threshold_ps), "files": [str(p) for p in paths], "outliers": outliers,
            "summary": summary,
            "profile_ps": [None if not np.isfinite(v) else round(float(v), 1) for v in pattern["profile_ps"]]}


def main(argv=None):
    from src.cornell.skew.lut import CanonicalLut, LutError

    parser = argparse.ArgumentParser(prog="python -m src.cornell.skew plot", description="Per-ASIC skew plots of a canonical Cornell delay LUT (spec 008).")
    parser.add_argument("lut", type=Path, help="canonical LUT CSV (its .json manifest must sit next to it)")
    parser.add_argument("--threshold-ps", type=float, default=DEFAULT_THRESHOLD_PS,
                        help="list channels whose residual from the per-ASIC pattern exceeds this (ps)")
    args = parser.parse_args(argv)
    try:
        rows = CanonicalLut.read(args.lut).rows
        applied = rows[rows["apply"]]
        result = write_asic_plots(args.lut, applied["time_channel"], applied["delay_ps"],
                                  threshold_ps=args.threshold_ps)
    except (LutError, PlotError, OSError) as error:
        print(f"plots failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in ("files", "summary", "outliers")}, indent=1))
    return 0

