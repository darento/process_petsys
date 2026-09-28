"""Offline PDF reporting for PETsys LDAT inspections (no GUI imports)."""

from __future__ import annotations

from pathlib import Path
import os

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np

from src.ldat_inspector import (SLAB_EXTENT_MM, TIMESTAMP_SECONDS, Selection, channel_status, factor_origins,
                                fit_on_display_bins, flood_counts, slab_origins, slab_totals, slab_view,
                                slab_rule_text, slab_x_edges, uniformity, unresolved_slab_pairs)


def report_rows(dataset, selection: Selection, target=511.0, tolerance_pct=10.0):
    """One measurement per mapped SuperModule, shared by table and detail pages."""
    rows = uniformity(dataset, selection, target, tolerance_pct)
    for row in rows:
        sm = row["sm"]
        data = dataset.modules.get(sm)
        row["selected"] = int(selection.mask(data).sum()) if data is not None else 0
        row["channels"] = channel_status(dataset, sm)
    return rows


def _text(axis, lines, *, top=0.95, fontsize=9, linespacing=1.55):
    axis.axis("off")
    axis.text(0.05, top, "\n".join(lines), transform=axis.transAxes,
              va="top", family="monospace", fontsize=fontsize, linespacing=linespacing)


def _reasons(counts):
    return ", ".join(f"{reason} {n:,}" for reason, n in sorted(counts.items())) or "none"


def _limits_lines(dataset, selection, limits, slab_flood, only_sm):
    """Limits files, active flood/DOI views and their exclusion counts (FR-16-FR-18)."""
    if dataset.settings.system != "CORNELL":
        return []
    lines = []
    for label, entry in (("COG limits", limits.get("cog")), ("DOI limits", limits.get("doi"))):
        lines.append(f"{label}: {entry.path} ({len(entry):,} keys, {entry.invalid} left == right)"
                     if entry is not None else f"{label}: none")
    if slab_flood:
        totals = slab_totals(dataset, limits["cog"], selection, None if only_sm is None else {only_sm})
        scope = "all SMs" if only_sm is None else f"SM {only_sm}"
        lines.append(f"Flood view: slab-assigned (X from the slab, COG Y decompressed with the COG limits); "
                     f"{scope}, sides passing the cuts: {totals['shown']:,} shown, excluded: "
                     f"{_reasons(totals['excluded'])}; clipped to the row: {totals['clipped']:,}; "
                     f"unresolved-slab pairs rejected at ingest (all files): {unresolved_slab_pairs(dataset):,}")
    else:
        lines.append("Flood view: COG/RTP centroid (spec 001)")
    if dataset.doi_mm:
        lines.append("DOI view: decompressed mm-equivalent depth, (doi - right) * 20 / (left - right) with the DOI "
                     "limits; a linear light-sharing mapping, not an independently validated depth; excluded "
                     f"(all sides, fail every DOI cut): {_reasons(dataset.doi_excluded)}")
    else:
        lines.append("DOI view: light-sharing ratio")
    return lines


def _counts(counts):
    return ", ".join(f"{name} {n:,}" for name, n in counts.items() if n) or "none"


def _origin_lines(dataset, selection, only_sm):
    """keV factor origins of the Cornell calibration (FR-20): sides, slabs, and the fit population."""
    settings = dataset.settings
    if not settings.calibrated or settings.system != "CORNELL":
        return []
    status = dataset.calibration_status
    if status is None:
        return ["keV factor origins: unknown (no _status.txt sidecar next to the calibration; never assumed fitted)"]
    sms = None if only_sm is None else {only_sm}
    sides = {}
    for sm, data in sorted(dataset.modules.items()):
        if sms is None or sm in sms:
            for name, n in factor_origins(dataset, data).items():
                sides[name] = sides.get(name, 0) + n
    scope = "all SMs" if only_sm is None else f"SM {only_sm}"
    return [f"keV factor origins: {status.path}",
            f"Ingest sides by keV factor origin ({scope}): {_counts(sides)}",
            f"Mapped slabs by keV factor origin ({scope}): {_counts(slab_origins(dataset, sms))}",
            "Photopeak/uniformity fits and counts: " + ("fitted keV factors only (borrowed and estimated left out)"
                                                        if selection.fitted_only
                                                        else "all keV factors (borrowed and estimated included)")]


def _provenance(dataset, selection, limits=None, slab_flood=False, only_sm=None):
    limits = {"doi": dataset.doi_limits, **(limits or {})}
    settings = dataset.settings
    paths = ["Input acquisitions:"]
    for file in dataset.files:
        result = f"{file.pairs_accepted:,}/{file.pairs_read:,} accepted coincidence pairs"
        if file.prefix_limited:
            result += " (read prefix; not whole-file rate)"
        elif settings.max_pairs is None and file.success:
            result += " (whole file)"
        if file.errors:
            result += " (rejected: " + ", ".join(
                f"{reason}={count}" for reason, count in file.errors.items()) + ")"
        if not file.success:
            result = f"FAILED: {file.error}"
        paths.append(f"  [{file.index}] {file.path}  {result}")
    return [
        f"System: {settings.system}",
        f"Config: {settings.config_path}",
        f"Map: {dataset.map_path}",
        f"Energy calibration: {settings.calibration_path if settings.calibrated else 'OFF (raw PETsys a.u.)'}",
        f"Ingest cuts: at least {settings.min_channels} energy channels; "
        f"per-channel >= {settings.min_channel_energy:g} a.u.; "
        + ("whole files (no pair limit)" if settings.max_pairs is None
           else f"max {settings.max_pairs:,} coincidence pairs / file"),
        *([slab_rule_text(dataset, None if only_sm is None else {only_sm})
           + ("" if only_sm is None else f" (SM {only_sm})")] if settings.system == "CORNELL" else []),
        f"Display: BOTH detector energies {selection.energy_low:g}..{selection.energy_high:g} "
        f"{'keV' if settings.calibrated else 'a.u.'}; "
        f"DOI {'mm' if dataset.doi_mm else 'ratio'} {selection.doi_low:g}..{selection.doi_high:g}",
        f"Local ROI: X {selection.x_low:g}..{selection.x_high:g} mm, "
        f"Y {selection.y_low:g}..{selection.y_high:g} mm",
        "Channel occupancy is measured before display cuts, not a dead-channel verdict.",
        *( [f"Unavailable calibrated energy: {sum(np.count_nonzero(~np.isfinite(data.energy)) for data in dataset.modules.values()):,} detector sides (missing/invalid calibration factor)."]
          if settings.calibrated else []),
        *_limits_lines(dataset, selection, limits, slab_flood, only_sm),
        *_origin_lines(dataset, selection, only_sm),
        ("No singles data in this LDAT input; decompressed DOI is a linear light-sharing mapping, not a validated depth."
         if dataset.doi_mm else "No singles data in this LDAT input; DOI is a light-sharing ratio, not depth in mm."),
        *(["Cornell one-time-channel slabs retain the legacy random neighbour assignment before keV calibration"
           + ("; so do FR-19 recovered sides with a neighbour tie or no fired neighbour."
              if settings.slab_rule != "legacy" else ".")]
          if settings.system == "CORNELL" and settings.calibrated else []),
        "Timestamps are PETsys picoseconds; within-file spans only, never merged live time.",
        *paths,
    ]


def _overview(pdf, dataset, selection, rows, *, target, tolerance_pct, only_sm, limits=None, slab_flood=False):
    fig = Figure(figsize=(11.7, 8.3), dpi=100)
    axis = fig.add_subplot(111)
    selected = rows if only_sm is None else [row for row in rows if row["sm"] == only_sm]
    ok = sum(row["result"] == "IN TOLERANCE" for row in selected)
    not_ok = sum(row["result"] == "OUT OF TOLERANCE" for row in selected)
    unavailable = sum(row["result"] == "UNAVAILABLE" for row in selected)
    accepted = sum(file.pairs_accepted for file in dataset.files if file.success)
    heading = [
        "LDAT INSPECTOR - OFFLINE ACQUISITION REPORT",
        f"Scope: {'full mapped system' if only_sm is None else 'SuperModule ' + str(only_sm)}",
        "",
        f"Accepted coincidence pairs: {accepted:,}; ingested detector sides: "
        f"{sum(map(len, dataset.modules.values())):,}",
        (f"Photopeak target: {target:.1f} keV; tolerance: +/-{tolerance_pct:g}%"
         if dataset.settings.calibrated else "Photopeak uniformity: unavailable (raw a.u.; no keV calibration)"),
        f"In tolerance: {ok}; out of tolerance: {not_ok}; fit unavailable: {unavailable}",
        "These are observational fits, not clinical PASS/FAIL decisions.",
        "",
    ]
    # Wrap long paths to stay inside the page; matplotlib writes text into the PDF.
    import textwrap
    lines = heading + [piece for line in _provenance(dataset, selection, limits, slab_flood, only_sm)
                       for piece in textwrap.wrap(line, width=118, subsequent_indent="    ")]
    _text(axis, lines, fontsize=8.0)
    pdf.savefig(fig)


def _tables(pdf, rows, calibrated=True):
    for start in range(0, len(rows), 32):
        fig = Figure(figsize=(11.7, 8.3), dpi=100)
        axis = fig.add_subplot(111)
        lines = ["SUPERMODULE SUMMARY - fit uses ROI/DOI sides before display energy cut",
                  "SM   Ingested   Selected  T seen/exp  E seen/exp    Peak keV  Res %    Status",
                  *([] if calibrated else ["Peak/resolution unavailable: raw a.u.; no keV calibration"]),
                 "-" * 91]
        for row in rows[start:start + 32]:
            fit, status = row["fit"], row["channels"]
            mu = f"{fit['mu']:.1f}" if fit["mu"] is not None else "--"
            res = f"{fit['resolution']:.1f}" if fit["resolution"] is not None else "--"
            lines.append(f"{row['sm']:>3} {status['events']:>10,} {row['selected']:>10,} "
                         f"{len(status['active_time']):>3}/{len(status['expected_time']):<3}     "
                         f"{len(status['active_energy']):>3}/{len(status['expected_energy']):<3}      "
                         f"{mu:>7}   {res:>5}    {row['result']}")
        _text(axis, lines, fontsize=8.0)
        pdf.savefig(fig)


def _module_page(pdf, dataset, selection, row, cog=None):
    sm = row["sm"]
    data = dataset.modules.get(sm)
    fig = Figure(figsize=(11.7, 8.3), dpi=100)
    axes = fig.subplots(2, 2)
    ax_info, ax_energy, ax_doi, ax_flood = axes.flat
    fit, status = row["fit"], row["channels"]
    extent = SLAB_EXTENT_MM if cog is not None else 102.0
    info = [f"SUPERMODULE {sm} - {dataset.settings.system}", "",
            f"Ingested detector sides: {status['events']:,}",
            f"Selected after paired energy / ROI / DOI: {row['selected']:,}",
            f"Time channels observed: {len(status['active_time'])}/{len(status['expected_time'])}",
            f"Energy channels observed: {len(status['active_energy'])}/{len(status['expected_energy'])}",
            f"Observation: {status['state']}",
            f"Peak fit: {fit['status']}",
            f"Peak position: {fit['mu']:.1f} keV" if fit["mu"] is not None else "Peak position: unavailable",
            f"Resolution: {fit['resolution']:.1f}%" if fit["resolution"] is not None else "Resolution: unavailable",
            f"Uniformity: {row['result']}",
            *([f"keV factor sides: {_counts(factor_origins(dataset, data))}"]
              if data is not None and dataset.settings.calibrated and dataset.settings.system == "CORNELL"
              and dataset.calibration_status is not None else []),
            "", "Time channels not observed (first 15):",
            str(sorted(status["unobserved_time"])[:15]),
            "Energy channels not observed (first 15):",
            str(sorted(status["unobserved_energy"])[:15]),
            "", "No singles bucket; no dead-channel proof from this sample."]
    if data is not None:
        info.extend(("", "Observed timestamp spans per file (not full-run rates):"))
        for file in dataset.files:
            if not file.success:
                continue
            timestamps = data.timestamp[data.file_index == file.index]
            if timestamps.size >= 2:
                seconds = (float(timestamps.max()) - float(timestamps.min())) * TIMESTAMP_SECONDS
                paired = (data.timestamp[data.file_index == file.index].astype(float)
                          - data.partner_timestamp[data.file_index == file.index].astype(float)) * 1e-3
                info.append(f"  [{file.index}] {seconds:.3f} s / {timestamps.size:,} sides"
                            + (" (prefix)" if file.prefix_limited else ""))
                info.append(f"      median paired hit difference: {np.median(paired):+.2f} ns (not clock drift)")
    _text(ax_info, info, fontsize=7.4, linespacing=1.25)
    if data is not None and len(data):
        spatial = selection.mask(data, energy=False)
        selected = selection.mask(data)
        calibrated = dataset.settings.calibrated
        energy_high = 1500 if calibrated else 300
        _, display_edges, _ = ax_energy.hist(data.energy[spatial], bins=140, range=(0, energy_high), color="#347dc1")
        if calibrated:
            ax_energy.axvspan(350, 700, color="#d84f41", alpha=0.035, zorder=0)
        if fit["status"] == "FIT":
            overlay = fit_on_display_bins(fit, display_edges)
            if overlay is not None:
                ax_energy.plot(overlay["x"], overlay["total"], color="#d84f41", lw=1.5)
        for bound in (selection.energy_low, selection.energy_high):
            ax_energy.axvline(bound, color="#cf6439", ls="--", lw=0.8)
        ax_doi.hist(data.doi[selected], bins=45, color="#8656ac", **({"range": (0, 20)} if dataset.doi_mm else {}))
        x, y, shown = data.x, data.y, selected
        if cog is not None:
            view = slab_view(dataset, data, cog, mask=selected)
            x, y, shown = view["x"], view["y"], selected & np.isfinite(view["y"])
            ax_flood.set_title(f"Selected slab flood (decompressed Y) • excluded: {_reasons(view['excluded'])}",
                               fontsize=9)
        if shown.any():
            counts, xedges, yedges = flood_counts(x[shown], y[shown], 80, extent,
                                                  slab_x_edges(dataset, sm) if cog is not None else None)
            cmap = matplotlib.colormaps["plasma"].copy()
            cmap.set_bad("white")
            ax_flood.pcolormesh(xedges, yedges, counts, cmap=cmap, vmin=0.1)
    ax_energy.set(xlabel="Energy (keV)" if dataset.settings.calibrated else "Raw energy (a.u.)",
                  ylabel="Sides", title="Calibrated energy / photopeak model" if dataset.settings.calibrated
                  else "Raw PETsys energy (no keV fit)")
    ax_doi.set(xlabel="Decompressed DOI (mm-equivalent; linear mapping)" if dataset.doi_mm
               else "DOI light-sharing ratio", ylabel="Sides", title="Selected DOI")
    ax_flood.set(xlabel="Slab X (mm)" if cog is not None else "Local X (mm)",
                 ylabel="Decompressed Y (mm)" if cog is not None else "Local Y (mm)", xlim=(0, extent), ylim=(0, extent))
    if cog is None:
        ax_flood.set_title("Selected flood map")
    fig.subplots_adjust(left=0.07, right=0.96, top=0.94, bottom=0.08, hspace=0.3, wspace=0.24)
    pdf.savefig(fig)


def write_report(path, dataset, selection: Selection, *, sm=None, target=511.0,
                 tolerance_pct=10.0, cog_limits=None, doi_limits=None, slab_flood=False):
    """Write a PDF atomically; every view uses the same report_rows measurements.

    ``cog_limits`` / ``doi_limits`` are the loaded limits files, listed in the
    provenance (the dataset's DOI view file is listed by default). ``slab_flood``
    draws the SM flood pages in the slab-assigned view with ``cog_limits``; the
    DOI pages follow the dataset's DOI view.
    """
    if not any(file.success for file in dataset.files):
        raise ValueError("No successfully processed LDAT files to report")
    if slab_flood and (cog_limits is None or dataset.settings.system != "CORNELL"):
        raise ValueError("The slab flood view needs a Cornell dataset and a COG limits file")
    limits = {"cog": cog_limits, "doi": doi_limits if doi_limits is not None else dataset.doi_limits}
    rows = report_rows(dataset, selection, target, tolerance_pct)
    if sm is not None and sm not in {row["sm"] for row in rows}:
        raise ValueError(f"SuperModule {sm} is not in the selected mapping")
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial = target_path.with_suffix(target_path.suffix + ".partial")
    try:
        with PdfPages(partial) as pdf:
            pdf.infodict()["Title"] = f"LDATInspector {dataset.settings.system} offline report"
            pdf.infodict()["Subject"] = "PETsys coincidence LDAT inspection"
            _overview(pdf, dataset, selection, rows, target=target,
                      tolerance_pct=tolerance_pct, only_sm=sm, limits=limits, slab_flood=slab_flood)
            selected_rows = rows if sm is None else [row for row in rows if row["sm"] == sm]
            _tables(pdf, selected_rows, dataset.settings.calibrated)
            for row in selected_rows:
                _module_page(pdf, dataset, selection, row, cog_limits if slab_flood else None)
        os.replace(partial, target_path)
    finally:
        partial.unlink(missing_ok=True)
    return str(target_path)
