"""Offline PDF reporting for PETsys LDAT inspections (no GUI imports)."""

from __future__ import annotations

from pathlib import Path
import os

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np

from .engine import (FINDINGS_POPULATION, SLAB_EXTENT_MM, TIMESTAMP_SECONDS, FindingThresholds,
                                Selection, channel_status, factor_origins, fit_on_display_bins, flood_counts,
                                minimodule_layout, minimodule_metrics, slab_origins, slab_totals, slab_view,
                                slab_rule_text, slab_x_edges, system_channel_findings, uniformity,
                                unresolved_slab_pairs)


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


# _text starts at 0.95 of the default axes: 698 px down from the top on an 11.7 x 8.3 in,
# 100 dpi page. Pixels per line are at most fontsize x linespacing x 1.42 (measured).
_TEXT_ROOM_PX = 698 - 25


def _text_pages(pdf, groups, *, fontsize, linespacing, continued, min_fontsize=6.0):
    """Save text on as many pages as needed: shrink the font to ``min_fontsize``, then continue.

    ``groups`` are the wrapped pieces of each logical line; a page break never splits one.
    """
    per_line = lambda size: size * linespacing * 1.42  # noqa: E731
    lines = [piece for group in groups for piece in group]
    chunks = [lines]
    if len(lines) * per_line(fontsize) > _TEXT_ROOM_PX:
        if len(lines) * per_line(min_fontsize) <= _TEXT_ROOM_PX:
            fontsize = _TEXT_ROOM_PX / (len(lines) * linespacing * 1.42)
        else:
            fontsize = min_fontsize
            capacity = int(_TEXT_ROOM_PX // per_line(fontsize)) - 1  # leaves room for "(continued)"
            chunks = [[]]
            for group in groups:
                if chunks[-1] and len(chunks[-1]) + len(group) > capacity:
                    chunks.append([])
                chunks[-1].extend(group)
    for page, chunk in enumerate(chunks):
        fig = Figure(figsize=(11.7, 8.3), dpi=100)
        _text(fig.add_subplot(111), ([f"{continued} (continued)"] if page else []) + chunk,
              fontsize=fontsize, linespacing=linespacing)
        pdf.savefig(fig)


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


def _scope(dataset):
    """Whole-file or prefix scope of the ingest (FR-14)."""
    max_pairs = dataset.settings.max_pairs
    return ("whole files (no pair limit)" if max_pairs is None
            else f"prefix, max {max_pairs:,} coincidence pairs / file")


def _calibration(dataset):
    settings = dataset.settings
    return (f"keV with {Path(settings.calibration_path).name}" if settings.calibrated
            else "raw PETsys a.u. (no keV calibration)")


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
        f"per-channel >= {settings.min_channel_energy:g} a.u.; " + _scope(dataset),
        *([slab_rule_text(dataset, None if only_sm is None else {only_sm})
           + ("" if only_sm is None else " (all files; pairs, not per SM)" if settings.slab_rule == "legacy"
              else f" (SM {only_sm})")] if settings.system == "CORNELL" else []),
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


def _overview(pdf, dataset, selection, rows, *, target, tolerance_pct, only_sm, limits=None, slab_flood=False,
              thresholds=FindingThresholds()):
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
        f"Channel findings thresholds: {thresholds.text()}",
        "",
    ]
    # Wrap long paths to stay inside the page; matplotlib writes text into the PDF.
    import textwrap
    groups = [[line] for line in heading] + [textwrap.wrap(line, width=118, subsequent_indent="    ") or [""]
                                             for line in _provenance(dataset, selection, limits, slab_flood, only_sm)]
    _text_pages(pdf, groups, fontsize=8.0, linespacing=1.55, continued=heading[0])


def _port_columns(ports):
    """DAQ port, MASTER/SLAVE and FEB/D port columns (FR-23); '--' without a map entry."""
    if ports is None:
        return f"{'--':>4}  {'--':<6} {'--':>5}"
    port, slave, febd = ports
    role = {0: "MASTER", 1: "SLAVE"}.get(slave, f"ID {slave}")
    return f"{port:>4}  {role:<6} {febd:>5}"


def _tables(pdf, rows, calibrated=True, sm_ports=None):
    sm_ports = sm_ports or {}
    for start in range(0, len(rows), 32):
        fig = Figure(figsize=(11.7, 8.3), dpi=100)
        axis = fig.add_subplot(111)
        lines = ["SUPERMODULE SUMMARY - fit uses ROI/DOI sides before display energy cut; "
                 "ports from the map's mod_feb_map",
                  "SM   DAQ  M/S    FEB/D  Ingested   Selected  T seen/exp  E seen/exp    Peak keV  Res %    Status",
                  *([] if calibrated else ["Peak/resolution unavailable: raw a.u.; no keV calibration"]),
                 "-" * 111]
        for row in rows[start:start + 32]:
            fit, status = row["fit"], row["channels"]
            mu = f"{fit['mu']:.1f}" if fit["mu"] is not None else "--"
            res = f"{fit['resolution']:.1f}" if fit["resolution"] is not None else "--"
            lines.append(f"{row['sm']:>3} {_port_columns(sm_ports.get(row['sm']))}"
                         f" {status['events']:>10,} {row['selected']:>10,} "
                         f"{len(status['active_time']):>3}/{len(status['expected_time']):<3}     "
                         f"{len(status['active_energy']):>3}/{len(status['expected_energy']):<3}      "
                         f"{mu:>7}   {res:>5}    {row['result']}")
        _text(axis, lines, fontsize=8.0)
        pdf.savefig(fig)


def _findings_pages(pdf, dataset, findings, thresholds):
    """Channel Status table (FR-5-FR-7): one row per SuperModule in the report scope."""
    def flags(kind):
        return "insufficient" if kind["insufficient"] else " / ".join(
            str(kind["counts"].get(state, 0)) for state in ("NOT OBSERVED", "LOW", "HIGH"))

    def median(kind):
        return "--" if kind["median"] is None else f"{kind['median']:,.0f}"

    states = {}
    for row in findings:
        states[row["state"]] = states.get(row["state"], 0) + 1
    order = ("NOT OBSERVED", "HIGH", "LOW", "INSUFFICIENT EVENTS", "NO DATA", "OK")
    heading = ["CHANNEL FINDINGS - " + FINDINGS_POPULATION + ", before display cuts",
               f"Scope: {_scope(dataset)}; not a dead/hot hardware verdict",
               f"Thresholds: {thresholds.text()}",
               f"{len(findings)} SuperModules: " + ", ".join(f"{states[s]} {s}" for s in order if s in states),
               "SM     Ingest  mM seen  T chan  T NO/LOW/HIGH  T median  E chan  E NO/LOW/HIGH  E median"
               "  Unexpected  Finding",
               "-" * 118]
    for start in range(0, max(len(findings), 1), 34):
        fig = Figure(figsize=(11.7, 8.3), dpi=100)
        axis = fig.add_subplot(111)
        lines = list(heading)
        for row in findings[start:start + 34]:
            sm, t, e = row["sm"], row["time"], row["energy"]
            data = dataset.modules.get(sm)
            expected = dataset.expected_mm.get(sm, set())
            seen = len(set(np.unique(data.mm).tolist()) & expected) if data is not None else 0
            unexpected = len(t["unexpected"]) + len(e["unexpected"])
            lines.append(f"{sm:>3} {row['events']:>10,}  {seen:>3}/{len(expected):<3}  "
                         f"{len(t['channels']):>6}  {flags(t):>13}  {median(t):>8}  "
                         f"{len(e['channels']):>6}  {flags(e):>13}  {median(e):>8}  "
                         f"{(str(unexpected) + ' ch') if unexpected else '--':>10}  {row['state']}")
        _text(axis, lines, fontsize=7.6)
        pdf.savefig(fig)


def _minimodule_page(pdf, dataset, selection, sm, metrics, populated, findings, thresholds):
    """Per-minimodule counts and photopeak (FR-13) and the SM's flagged channels (FR-5)."""
    import textwrap
    calibrated = dataset.settings.calibrated
    units = "keV" if calibrated else "a.u."
    lines = [f"SUPERMODULE {sm} - MINIMODULES AND CHANNEL FINDINGS",
             f"Scope: {_scope(dataset)}; energy: {_calibration(dataset)}",
             "Ingest: accepted detector sides (ingest population). Selected: sides passing the display cuts "
             f"(both energies {selection.energy_low:g}..{selection.energy_high:g} {units}, DOI, ROI"
             + (", fitted keV factors only" if selection.fitted_only else "") + ").",
             ("Fit: photopeak on ROI/DOI sides with the energy window off (spec 001 fit guards)." if calibrated
              else "Fit: unavailable in raw a.u. (no keV calibration)."),
             "",
             "mM      Ingest    Selected   Fit sides   Centroid keV   Res %   Status",
             "-" * 96]
    mms = sorted(set(populated) | {mm for s, mm in metrics if s == sm})
    fitted = assessed = 0
    for mm in mms:
        row = metrics.get((sm, mm)) or {"ingest": 0, "selected": 0, "fit_sides": 0, "fit": None}
        fit = row["fit"]
        mu = res = "--"
        if not populated.get(mm, True):
            status = "unpopulated (config)"
        else:
            assessed += 1
            if fit is not None and fit["status"] == "FIT":
                mu, res, status = f"{fit['mu']:.1f}", f"{fit['resolution']:.1f}", "FIT"
                fitted += 1
            else:
                status = fit["status"] if fit is not None else "not computed"
        lines.append(f"{mm:>2} {row['ingest']:>11,} {row['selected']:>11,} {row['fit_sides']:>11,}"
                     f"   {mu:>12}   {res:>5}   {status}")
    lines.append(f"Fitted minimodules: {fitted}/{assessed} populated" if calibrated
                 else f"Populated minimodules: {assessed}")
    lines += ["", f"CHANNEL FINDINGS - {findings['state']} ({FINDINGS_POPULATION}, before display cuts)",
              f"Thresholds: {thresholds.text()}"]
    for kind in ("time", "energy"):
        found = findings[kind]
        observed = int((found["hits"] > 0).sum())
        lines.append(f"{kind.upper():<6} {observed}/{len(found['channels'])} observed"
                     + ("" if found["median"] is None else f", median {found['median']:,.0f} hits"))
        if found["insufficient"]:
            lines.append(f"       insufficient: {found['insufficient']}")
            groups = (("0 hits", [ch for ch, n in zip(found["channels"], found["hits"]) if n == 0]),)
        else:
            counts = found["counts"]
            lines.append(f"       {counts['NOT OBSERVED']} not observed / {counts['LOW']} low / "
                         f"{counts['HIGH']} high")
            groups = [(label.lower(), [ch for ch, s in zip(found["channels"], found["states"]) if s == label])
                      for label in ("NOT OBSERVED", "LOW", "HIGH")]
        for label, ids in groups:
            if ids:
                lines += textwrap.wrap(f"{label}: " + ", ".join(map(str, ids)), width=130,
                                       initial_indent="       ", subsequent_indent="         ")
        if found["unexpected"]:
            lines += textwrap.wrap("unexpected hits (unpopulated mM): " + ", ".join(
                f"{ch} ({n:,})" for ch, n in found["unexpected"].items()), width=130,
                initial_indent="       ", subsequent_indent="         ")
    lines.append("Observational: no channel is declared dead or hot from a coincidence sample.")
    lines += _timestamp_lines(dataset, dataset.modules.get(sm))
    groups = [textwrap.wrap(line, width=140, subsequent_indent="    ") or [""] for line in lines]
    _text_pages(pdf, groups, fontsize=7.6, linespacing=1.4, continued=lines[0])


def _timestamp_lines(dataset, data):
    """Observed per-file timestamp spans of one SM, one line per file (moved from the detail page)."""
    if data is None:
        return []
    lines = ["", "Observed timestamp spans per file (not full-run rates; paired hit difference is not clock drift):"]
    for file in dataset.files:
        if not file.success:
            continue
        in_file = data.file_index == file.index
        timestamps = data.timestamp[in_file]
        if timestamps.size >= 2:
            seconds = (float(timestamps.max()) - float(timestamps.min())) * TIMESTAMP_SECONDS
            paired = (timestamps.astype(float) - data.partner_timestamp[in_file].astype(float)) * 1e-3
            lines.append(f"  [{file.index}] {seconds:.3f} s / {timestamps.size:,} sides"
                         + (" (prefix)" if file.prefix_limited else "")
                         + f"; median paired hit difference {np.median(paired):+.2f} ns")
    return lines


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
            "", "No singles bucket; no dead-channel proof from this sample.",
            "Per-file timestamp spans: on the next page."]
    import textwrap  # the info panel shares the page width with the energy plot
    info = [piece for line in info for piece in (textwrap.wrap(line, width=80, subsequent_indent="    ") or [""])]
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
                 tolerance_pct=10.0, cog_limits=None, doi_limits=None, slab_flood=False,
                 thresholds: FindingThresholds = FindingThresholds()):
    """Write a PDF atomically; every view uses the same report_rows measurements.

    Pages: provenance, SuperModule summary, channel findings (``thresholds``),
    then per SM its detail page and its minimodule / channel-findings page.
    Findings and minimodule rows come from the same engine functions as the GUI.

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
    thresholds.validate()
    rows = report_rows(dataset, selection, target, tolerance_pct)
    if sm is not None and sm not in {row["sm"] for row in rows}:
        raise ValueError(f"SuperModule {sm} is not in the selected mapping")
    findings = [row for row in system_channel_findings(dataset, thresholds) if sm is None or row["sm"] == sm]
    by_sm = {row["sm"]: row for row in findings}
    metrics = minimodule_metrics(dataset, selection, sms=None if sm is None else [sm])
    layout = minimodule_layout(dataset)
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial = target_path.with_suffix(target_path.suffix + ".partial")
    try:
        with PdfPages(partial) as pdf:
            pdf.infodict()["Title"] = f"LDATInspector {dataset.settings.system} offline report"
            pdf.infodict()["Subject"] = "PETsys coincidence LDAT inspection"
            _overview(pdf, dataset, selection, rows, target=target,
                      tolerance_pct=tolerance_pct, only_sm=sm, limits=limits, slab_flood=slab_flood,
                      thresholds=thresholds)
            selected_rows = rows if sm is None else [row for row in rows if row["sm"] == sm]
            _tables(pdf, selected_rows, dataset.settings.calibrated, dataset.sm_ports)
            _findings_pages(pdf, dataset, findings, thresholds)
            for row in selected_rows:
                _module_page(pdf, dataset, selection, row, cog_limits if slab_flood else None)
                _minimodule_page(pdf, dataset, selection, row["sm"], metrics,
                                 layout.get(row["sm"], {}).get("populated", {}), by_sm[row["sm"]], thresholds)
        os.replace(partial, target_path)
    finally:
        partial.unlink(missing_ok=True)
    return str(target_path)
