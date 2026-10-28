"""IMAS walk-table diagnostics (spec 012 T8, FR-7). Diagnostics only, not validation evidence.

``python -m src.imas.skew walk-plot --walk-lut W.json [--profiles P.json ...] [--ctr REPORT.json --tables em em2+walk3]
--out DIR`` writes, never overwriting, into ``DIR``:

* ``walk_table.png``: walk2d offset and fit events per bin (charge x share), the doi and kev offset curves with
  their fit events per bin; bins filled from the nearest populated bin are marked;
* ``walk_profiles.png`` (``--profiles``): per walk input, the median residual before and after the correction per
  bin, for each dataset (``residual_profiles``; each file holds ``{"profiles": {label: profile}}``);
* ``walk_ctr.png`` (``--ctr``): model-free FWHM per source position and pooled, for the named tables of a spec 011
  CTR report; acquisitions ``<position>_<nn>`` are grouped by position.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from src.imas.skew.walk import BLOCKS, INPUTS, WalkLut
from src.timing.walk import WalkError

OUTPUTS = ("walk_table.png", "walk_profiles.png", "walk_ctr.png")
LABELS = {"charge": "timing-hit charge (a.u.)", "share": "TIME charge share", "doi_ratio": "ENERGY sum / max",
          "kev": "energy (keV)"}


def _title(lut):
    table = (lut.manifest or {}).get("table") or {}
    return f"{lut.model.schema}" + (f"  sha256 {table['sha256'][:12]}" if table.get("sha256") else "")


def _curve(ax, ax_events, lut, block):
    """Offset steps of a one-input block with its events per bin on a twin axis; filled bins hollow red."""
    edges, offset, events = lut.edges[block][0], lut.offset_ps[block], lut.events[block]
    centre = (edges[:-1] + edges[1:]) / 2
    ax_events.stairs(np.maximum(events, 0), edges, color="0.75", fill=True)
    positive = events[events > 0]
    if len(positive) and positive.max() > 10 * positive.min():     # quantile bins: equal counts, a linear axis
        ax_events.set(yscale="log")
    else:
        ax_events.set_ylim(0, 1.1 * max(float(events.max()), 1.0))
    ax_events.set(ylabel="fit events per bin")
    ax.set_zorder(ax_events.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.stairs(offset, edges, color="C0", lw=1.4, baseline=None)
    ax.plot(centre, offset, "o", ms=3, color="C0")
    filled = sorted(i[0] for i in lut.filled[block])
    if filled:
        ax.plot(centre[filled], offset[filled], "o", ms=7, mfc="none", mec="C3", label="filled from nearest bin")
        ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel=LABELS[BLOCKS[block][0]], ylabel="offset (ps)", title=f"{block}")
    ax.grid(alpha=0.3)


def plot_table(lut, path):
    """One PNG of the table: walk2d offset and events (charge x share), doi and kev offsets with their events."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ec, es = lut.edges["walk2d"]
    offset = lut.offset_ps["walk2d"]
    limit = max(float(np.abs(offset).max()), 1.0)
    mesh = axes[0, 0].pcolormesh(ec, es, offset.T, cmap="coolwarm", vmin=-limit, vmax=limit, shading="flat")
    fig.colorbar(mesh, ax=axes[0, 0], label="offset (ps)")
    counts = axes[0, 1].pcolormesh(ec, es, np.maximum(lut.events["walk2d"], 1).T, norm=LogNorm(), shading="flat")
    fig.colorbar(counts, ax=axes[0, 1], label="fit events per bin")
    centre_c, centre_s = (ec[:-1] + ec[1:]) / 2, (es[:-1] + es[1:]) / 2
    for ax, title in ((axes[0, 0], "walk2d offset"), (axes[0, 1], "walk2d fit events")):
        if lut.filled["walk2d"]:
            index = np.array(lut.filled["walk2d"])
            ax.plot(centre_c[index[:, 0]], centre_s[index[:, 1]], "x", color="k", ms=5, label="filled from nearest bin")
        ax.set(xlabel=LABELS["charge"], ylabel=LABELS["share"], title=title)
        if ec[0] > 0:                           # quantile edges crowd at low charge, where the walk is
            ax.set_xscale("log")
    if lut.filled["walk2d"]:
        axes[0, 0].legend(fontsize=8, loc="lower right")
    for ax, block in ((axes[1, 0], "doi"), (axes[1, 1], "kev")):
        _curve(ax, ax.twinx(), lut, block)
    fig.suptitle(_title(lut))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def residual_profiles(values, before_ps, after_ps, edges, *, core_ps=600.0):
    """Median residual per input bin before and after the correction (plain lists; None for an empty bin).

    ``values`` = {input: per-event array, NaN missing}; per input, the events are those with a finite input and
    finite residuals, the medians use those with |residual| < ``core_ps`` (separately before and after); values
    outside ``edges`` fall into the end bins as in the LUT.
    """
    before, after = np.asarray(before_ps, np.float64), np.asarray(after_ps, np.float64)
    out = {}
    for name, edge in edges.items():
        edge = np.asarray(edge, np.float64)
        v = np.asarray(values[name], np.float64)
        n = len(edge) - 1
        ok = np.isfinite(v) & np.isfinite(before) & np.isfinite(after)
        index = np.clip(np.searchsorted(edge, v, side="right") - 1, 0, n - 1)
        row = {"edges": edge.tolist(), "events": np.bincount(index[ok], minlength=n).astype(int).tolist()}
        for label, residual in (("before", before), ("after", after)):
            keep = ok & (np.abs(residual) < core_ps)
            order = np.argsort(index[keep], kind="stable")
            groups = np.split(residual[keep][order], np.cumsum(np.bincount(index[keep], minlength=n))[:-1])
            row[f"{label}_median_ps"] = [float(np.median(g)) if len(g) else None for g in groups]
        out[name] = row
    return out


def plot_profiles(profiles, path):
    """One PNG, a panel per input: before (dashed, hollow) and after (solid) median residual per bin centre."""
    names = [n for n in INPUTS if any(n in p for p in profiles.values())]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.4), squeeze=False)
    for ax, name in zip(axes[0], names):
        for k, (label, profile) in enumerate(profiles.items()):
            if name not in profile:
                continue
            row = profile[name]
            edges = np.asarray(row["edges"], np.float64)
            centre = (edges[:-1] + edges[1:]) / 2
            for key, style, face in (("before", "--", "none"), ("after", "-", f"C{k}")):
                y = np.array([np.nan if m is None else m for m in row[f"{key}_median_ps"]], np.float64)
                ax.plot(centre, y, style, marker="o", ms=4, mfc=face, color=f"C{k}", lw=1,
                        label=f"{label} {key}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set(xlabel=LABELS[name], ylabel="median residual (ps)", title=f"residual vs {name}")
        if name == "charge" and all(min(p[name]["edges"]) > 0 for p in profiles.values() if name in p):
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fwhm(entry):
    value = ((entry or {}).get("fit") or {}).get("model_free_fwhm_ps")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def plot_ctr(report, tables, path):
    """Model-free FWHM per source position (mean of its acquisitions, each acquisition a small marker) and pooled,
    for each of ``tables`` in a spec 011 CTR report. Returns {"positions": {position: {table: fwhm}}, "pooled":
    {table: fwhm}, "skipped_tables": [...], "warnings": [...]}; a table missing from the report is skipped."""
    results = report["results"]
    shown, warnings = [t for t in tables if t in results], []
    skipped = [t for t in tables if t not in results]
    warnings += [f"table {t!r} not in the report (has {sorted(results)}); skipped" for t in skipped]
    positions, per_acquisition = [], {}
    for table in shown:
        for name, entry in results[table].items():
            if name == "pooled":
                continue
            position = re.sub(r"_\d+$", "", name)
            if position not in positions:
                positions.append(position)
            per_acquisition.setdefault((position, table), []).append(_fwhm(entry))
    summary = {p: {} for p in positions}
    for (position, table), values in per_acquisition.items():
        values = [v for v in values if v is not None]
        summary[position][table] = float(np.mean(values)) if values else None
    pooled = {t: _fwhm(results[t].get("pooled")) for t in shown}
    fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * (len(positions) + 1) + 2), 4.8))
    width = 0.8 / max(len(shown), 1)
    for k, table in enumerate(shown):
        dx = (k - (len(shown) - 1) / 2) * width
        ax.plot([i + dx for i, p in enumerate(positions) if summary[p].get(table) is not None],
                [summary[p][table] for p in positions if summary[p].get(table) is not None],
                "o", ms=9, color=f"C{k}", label=table)
        for i, p in enumerate(positions):
            values = [v for v in per_acquisition.get((p, table), []) if v is not None]
            ax.plot(np.full(len(values), i + dx), values, ".", ms=4, color=f"C{k}", alpha=0.45)
        if pooled[table] is not None:
            ax.plot([len(positions) + dx], [pooled[table]], "s", ms=9, color=f"C{k}")
    ax.axvline(len(positions) - 0.5, color="0.6", lw=0.8)
    ax.set_xticks(range(len(positions) + 1), positions + ["pooled"], rotation=20, ha="right")
    ax.set(ylabel="model-free FWHM (ps)", title="point-source CTR per position (large: mean of acquisitions; "
                                                "small: each acquisition)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"positions": summary, "pooled": pooled, "skipped_tables": skipped, "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew walk-plot", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--walk-lut", type=Path, required=True, help="IMAS walk LUT JSON (with its manifest)")
    parser.add_argument("--profiles", type=Path, nargs="+", default=[], help='JSON files {"profiles": {label: ...}}')
    parser.add_argument("--ctr", type=Path, help="spec 011 IMAS CTR report JSON")
    parser.add_argument("--tables", nargs="+", help="CTR report tables to plot (default: all)")
    parser.add_argument("--out", type=Path, required=True, help="output directory (existing PNGs are never overwritten)")
    args = parser.parse_args(argv)
    wanted = [OUTPUTS[0]] + ([OUTPUTS[1]] if args.profiles else []) + ([OUTPUTS[2]] if args.ctr else [])
    existing = [args.out / name for name in wanted if (args.out / name).exists()]
    if existing:
        print(f"walk-plot failed: output exists: {', '.join(map(str, existing))}", file=sys.stderr)
        return 1
    try:
        lut = WalkLut.read(args.walk_lut)
        profiles = {}
        for file in args.profiles:
            for label, profile in json.loads(file.read_text(encoding="utf-8"))["profiles"].items():
                if label in profiles:
                    raise WalkError(f"profile label {label!r} appears twice")
                profiles[label] = profile
        if args.profiles and not profiles:
            raise WalkError("the profiles files hold no profiles")
        report = json.loads(args.ctr.read_text(encoding="utf-8")) if args.ctr else None
    except (WalkError, OSError, KeyError, ValueError) as error:
        print(f"walk-plot failed: {error}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    plot_table(lut, args.out / OUTPUTS[0])
    if args.profiles:
        plot_profiles(profiles, args.out / OUTPUTS[1])
    if report is not None:
        summary = plot_ctr(report, args.tables or list(report["results"]), args.out / OUTPUTS[2])
        for warning in summary["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)
    for name in wanted:
        print(args.out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
