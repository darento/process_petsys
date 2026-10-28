"""Diagnostics of an IMAS NECR CTR run (spec 013 FR-8); diagnostics, not validation evidence.

Written by ``necr.run_necr`` into ``<output>/plots/`` from the report and the saved histograms:

- ``timing_<frame>.png``: per table, the raw and background-subtracted NEMA timing histograms (headline line);
- ``dt_vs_r_<frame>.png``: per table, Δt against the signed LOR-to-line distance r (headline line);
- ``ctr_vs_activity.png``: background-subtracted and raw NEMA FWHM against activity, every table, with the
  legacy ``.lm`` results when declared;
- ``line_fit.png``: the fitted line's slab points with x(z) and y(z) of every line;
- ``photopeak.png``: the 511 keV photopeak per frame against activity.
"""
from __future__ import annotations

import numpy as np

from src.timing import nema


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _width(entry, kind="background_subtracted"):
    value = entry["nema"][kind]
    return value["fwhm_ps"] if value["status"] == "fitted" else None


def timing_histograms(plt, out, report, saved, frame, k):
    line = report["headline_line"]
    l = list(saved["lines"]).index(line)
    tables = list(saved["tables"])
    figure, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 3.8), squeeze=False)
    for t, (ax, name) in enumerate(zip(axes[0], tables)):
        band = saved["band"][l, t, k]
        entry = report["results"][frame][line][name]
        for hist, kind, label in ((band.sum(axis=0), "raw", "raw"),
                                  (nema.background_subtracted(band), "background_subtracted", "background-subtracted")):
            width = _width(entry, kind)
            ax.step(nema.T_CENTERS, hist, where="mid", lw=0.9,
                    label=f"{label}: FWHM " + (f"{width:.0f} ps" if width else entry["nema"][kind]["status"]))
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Δt (ps)")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel(f"events / {2 * nema.T_RANGE_PS / nema.T_BINS:g} ps")
    figure.suptitle(f"{frame}: NEMA timing histograms, line {line}, same events")
    figure.tight_layout()
    figure.savefig(out / f"timing_{frame}.png", dpi=110)
    plt.close(figure)


def dt_vs_r(plt, out, report, saved, frame, k):
    line = report["headline_line"]
    l = list(saved["lines"]).index(line)
    tables = list(saved["tables"])
    figure, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 3.8), squeeze=False)
    extent = (-nema.R_RANGE_MM, nema.R_RANGE_MM, -nema.T_RANGE_PS, nema.T_RANGE_PS)
    for t, (ax, name) in enumerate(zip(axes[0], tables)):
        ax.imshow(saved["band"][l, t, k].T, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        for r in (-nema.R_SIGNAL_MM, nema.R_SIGNAL_MM):
            ax.axvline(r, color="white", lw=0.6, ls="--")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("r (mm)")
    axes[0][0].set_ylabel("Δt (ps)")
    figure.suptitle(f"{frame}: Δt against r, line {line} (dashed: signal band)")
    figure.tight_layout()
    figure.savefig(out / f"dt_vs_r_{frame}.png", dpi=110)
    plt.close(figure)


def ctr_vs_activity(plt, out, report):
    line = report["headline_line"]
    frames = report["frames"]
    activity = np.array([f["activity_kbq_ml"] for f in frames])
    order = np.argsort(activity)
    figure, ax = plt.subplots(figsize=(7.5, 4.8))
    for name in report["results"]["pooled"][line]:
        values, errors = [], []
        for f in frames:
            entry = report["results"][f["name"]][line][name]
            values.append(_width(entry) or np.nan)
            errors.append(entry["uncertainty"]["background_subtracted fwhm_ps"]["sd_ps"] or 0.0)
        line_plot = ax.errorbar(activity[order], np.array(values)[order], yerr=np.array(errors)[order], marker="o",
                                ms=4, capsize=2, lw=1, label=f"{name} (background-subtracted)")
        raw = [_width(report["results"][f["name"]][line][name], "raw") or np.nan for f in frames]
        ax.plot(activity[order], np.array(raw)[order], ls=":", lw=1, color=line_plot[0].get_color())
    legacy = report.get("legacy_results")
    if legacy:
        named = {f["name"]: f["activity_kbq_ml"] for f in frames}
        points = sorted((named[n], v["fwhm_bg_ps"], v["fwhm_raw_ps"]) for n, v in legacy["frames"].items())
        if points:
            a, bg, raw = map(np.array, zip(*points))
            ax.plot(a, bg, "k*", ms=8, label=f"legacy .lm results ({legacy['source']}), background-subtracted")
            ax.plot(a, raw, "k+", ms=8, label="legacy .lm results, raw")
    ax.set_xlabel("activity concentration (kBq/ml)")
    ax.set_ylabel("NEMA FWHM (ps)")
    ax.set_title(f"CTR against activity, line {line} (dotted: raw)")
    ax.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(out / "ctr_vs_activity.png", dpi=110)
    plt.close(figure)


def line_fit(plt, out, report):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    z = np.linspace(-360.0, 360.0, 2)
    for name, line in report["lines"].items():
        p1, p2 = np.array(line["p1_mm"]), np.array(line["p2_mm"])
        xy = p1[:2] + ((z - p1[2]) / (p2[2] - p1[2]))[:, None] * (p2 - p1)[:2]
        for ax, c in zip(axes, (0, 1)):
            ax.plot(z, xy[:, c], lw=1, label=f"{name} ({line['method']})")
        if line["method"] == "lor_convergence" and line["fit"]["slab_points"]:
            points = np.array(line["fit"]["slab_points"])
            for ax, c in zip(axes, (1, 2)):
                ax.plot(points[:, 0], points[:, c], ".", ms=4, label=f"{name}: slab points")
    for ax, label in zip(axes, ("x", "y")):
        ax.set_xlabel("z (mm)")
        ax.set_ylabel(f"{label} (mm)")
        ax.legend(fontsize=7)
    figure.suptitle("Line source in the map frame")
    figure.tight_layout()
    figure.savefig(out / "line_fit.png", dpi=110)
    plt.close(figure)


def photopeak(plt, out, report):
    frames = report["frames"]
    peaks = [(f["activity_kbq_ml"], report["energy"]["frames"][f["name"]].get("peak_kev")) for f in frames]
    peaks = sorted((a, p) for a, p in peaks if p is not None)
    figure, ax = plt.subplots(figsize=(6.5, 4.2))
    if peaks:
        ax.plot(*zip(*peaks), "o-")
    ax.axhline(511.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("activity concentration (kBq/ml)")
    ax.set_ylabel("photopeak (keV)")
    ax.set_title(f"511 keV photopeak per frame ({len(peaks)} of {len(frames)} fitted)")
    figure.tight_layout()
    figure.savefig(out / "photopeak.png", dpi=110)
    plt.close(figure)


def write_plots(output, report):
    """Every FR-8 PNG into ``output/plots``; returns their paths."""
    plt = _pyplot()
    out = output / "plots"
    out.mkdir(exist_ok=True)
    with np.load(output / "necr_histograms.npz") as saved:
        for k, frame in enumerate(saved["frames"]):
            timing_histograms(plt, out, report, saved, str(frame), k)
            dt_vs_r(plt, out, report, saved, str(frame), k)
    ctr_vs_activity(plt, out, report)
    line_fit(plt, out, report)
    photopeak(plt, out, report)
    return sorted(out.glob("*.png"))
