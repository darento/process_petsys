"""Legacy QC output types with explicit scope, units and provenance (spec 003 T10).

Same files as ``scripts_cornell/cornell_system_validation.py``:
``missing_channels_report.pdf`` always; with plots ``photopeak_values.xlsx``,
``photopeak_SM_<sm>.png``, ``channels_present_per_cassette.png``,
``slab_distribution_SM<sm>[_no_data].png``, ``floodmap_SM_<sm>.png`` and
``floodmap_all_SM.png``; with slabs also ``photopeak_<sm>_<mm>.png`` and
``photopeak_distribution.png``. ``qc_summary.json`` is written last; a
directory without it is incomplete. Every file is created exclusively in a new
directory. Changes from the reference:

- Layouts use the selected map's SuperModules instead of SM 0-29 (slab
  distribution), cassettes 0-1 (channel frequency, cassette = SM // 3) and
  SM 0-2 (combined flood). The combined flood places the per-SuperModule
  0.21 mm histograms side by side, highest SM ID on the left as in the
  reference, instead of re-binning three SuperModules at 0.315 mm.
- Unavailable fits show no photopeak line/value; the Excel keeps the five
  reference columns (fitted rows sorted by Mu) and adds status, sample count
  and labelled sample moments, plus a provenance sheet.
- The PDF keeps the reference sections and adds sample scope, populations,
  cuts, units, mapping, expectation rule and source mode.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from src.mapping_generator import ChannelType
from src.utils import get_electronics_nums

from .inputs import InputError
from .qc import (FIT_MIN_PEAK, FIT_STATUSES, FLOOD_ENERGY, FLOOD_RANGE, GENERATOR, LEGACY_HALF_MINIMODULES,
                 MINIMODULE_CB, PEAK_FINDER, PHOTOPEAK_BINS, PHOTOPEAK_RANGE, SLAB_CB)

SUMMARY = "qc_summary.json"
PDF = "missing_channels_report.pdf"
EXCEL = "photopeak_values.xlsx"
EXCEL_COLUMNS = ("SM", "Minimodule", "Mu", "Sigma", "Energy Resolution (%)", "Fit status", "Samples",
                 "Sample mean (a.u.)", "Sample std (a.u.)")
SM_COLORS = ("blue", "green", "orange", "red", "purple", "brown", "pink", "gray", "olive", "cyan")


def default_directory(parent, now=None):
    """Reference ``system_QC_results/<YYYYmmdd-HHMMSS>`` naming under ``parent``."""
    return Path(parent) / (now or datetime.now()).strftime("%Y%m%d-%H%M%S")


def _source_text(result):
    if result.source_mode is None:
        return "not recorded (offline analysis)"
    text = "with source" if result.source_mode.value == "with" else "without source"
    if result.acquisition_time_s is not None:
        text += f", {result.acquisition_time_s:g} s acquisition"
    return text


def expectation_rule(result):
    declared = sorted(result.absent)
    legacy_only = sorted(result.legacy_absent - result.absent)
    current_only = sorted(result.absent - result.legacy_absent)
    return {
        "rule": "selected map minus config unpopulated_minimodules",
        "declared_unpopulated": [list(k) for k in declared],
        "legacy_rule": f"SuperModule (sm + 1) % 3 == 0 populated only in minimodules {list(LEGACY_HALF_MINIMODULES)}",
        "expected_now_skipped_by_legacy": [list(k) for k in legacy_only],
        "skipped_now_expected_by_legacy": [list(k) for k in current_only],
    }


def summary(result, outputs):
    totals = result.totals()
    findings = result.findings
    missing_time, missing_energy = findings.totals()

    def fits(entries, full):
        counts = {status: 0 for status in FIT_STATUSES}
        for entry in entries:
            counts[entry.status] += 1
        listed = [entry for entry in entries if full or not entry.available or entry.message]
        return {"status_counts": counts,
                "entries" if full else "non_fitted": [
                    {"key": list(e.key), "status": e.status, "samples": e.samples, "in_histogram": e.in_histogram,
                     "mu_au": e.mu, "sigma_au": e.sigma, "resolution_percent": e.resolution_percent,
                     "sample_mean_au": e.sample_mean, "sample_std_au": e.sample_std, "message": e.message}
                    for e in listed]}

    return {
        "schema_version": 1,
        "generator": GENERATOR,
        "process": "completed",
        "note": "Process completion is not a detector verdict; findings are observations of the coincidence sample.",
        "source_mode": None if result.source_mode is None else result.source_mode.value,
        "acquisition_time_s": result.acquisition_time_s,
        "options": {"plots": result.plots, "slabs": result.slabs},
        "energy_units": "a.u. (raw PETsys QDC; no keV calibration applied)",
        "calibration": None,
        "cuts": {"en_min_ch_au": result.en_min_ch, "en_min_ch_applies_to": "every hit, at reading",
                 "min_ch": result.min_ch,
                 "min_ch_applies_to": "energy channels per side, before and after selecting the max-energy minimodule"},
        "sampling": {"pair_limit_per_file": result.pair_limit,
                     "limit_semantics": "stop once accepted pairs reach the limit (reference: break when > 1,000,000); "
                                        "a whole-file result only when stopped_at_limit is false",
                     "input_order": [f.path for f in result.files],
                     "random_streams": result.random_streams, "workers": result.workers},   # FR-15 (T33)
        "populations": {
            "records_read": "pairs yielded by the reader (one extra pair is read before a limit stop)",
            "validated_records": "records read; each was validated as read (FR-24); records not read are not",
            "records_in_file": "known only when the whole file was read (compact files have no record count)",
            "occupancy_pairs": "pairs passing the channel cuts, unresolved-slab pairs included",
            "occupancy_hits": "channel hits of the max-energy minimodule of both sides of occupancy pairs",
            "accepted_pairs": "occupancy pairs with both slabs resolved; energy/flood/slab samples use their sides",
        },
        "totals": totals,
        "inputs": [{"path": f.path, "validated_records": f.validated_records, "records_in_file": f.records_in_file,
                    "records_read": f.records_read,
                    "pairs_processed": f.pairs_processed, "occupancy_pairs": f.occupancy_pairs,
                    "occupancy_hits": f.occupancy_hits, "accepted_pairs": f.accepted_pairs,
                    "accepted_sides": f.accepted_sides, "stopped_at_limit": f.stopped_at_limit,
                    "rejected": f.rejected, "slab_flags": f.slab_flags} for f in result.files],
        "expectation": expectation_rule(result),
        "findings": {
            "expected_minimodules": len(findings.expected_minimodules),
            "minimodules_without_hits": [list(k) for k in findings.missing_minimodules],
            "declared_unpopulated_with_hits": [list(k) for k in findings.unexpected_minimodules],
            "missing_time_channels": missing_time, "missing_energy_channels": missing_energy,
            "missing_channels": {str(sm): v for sm, v in findings.missing.items() if v["Time"] or v["Energy"]},
        },
        "histograms": {"photopeak_bins": PHOTOPEAK_BINS, "photopeak_range_au": list(PHOTOPEAK_RANGE),
                       "fit": f"src.fits.fit_gaussian(cb={MINIMODULE_CB} minimodule / {SLAB_CB} slab, "
                              f"min_peak={FIT_MIN_PEAK}, pk_finder='{PEAK_FINDER}')",
                       "resolution": "2.35 * sigma / mu * 100 (fitted entries only)",
                       "flood_window_au": list(FLOOD_ENERGY), "flood_range_mm": list(FLOOD_RANGE),
                       "flood_bins": int(len(result.flood_edges) - 1)},
        "minimodule_fits": fits(result.minimodule_fits, True) if result.plots else None,
        "slab_fits": fits(result.slab_fits, False) if result.slabs else None,
        "slab_counts": [[*key, count] for key, count in result.slab_counts.items()],
        "sources": result.sources,
        "storage_bytes": result.storage_bytes,
        "outputs": outputs,
    }


# PDF -----------------------------------------------------------------------

def write_pdf(result, stream, directory_name):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    findings = result.findings
    totals = result.totals()
    missing_time, missing_energy = findings.totals()
    styles = getSampleStyleSheet()
    title = ParagraphStyle("CustomTitle", parent=styles["Heading1"], fontSize=16,
                           textColor=colors.HexColor("#2c3e50"), spaceAfter=30, alignment=TA_CENTER)
    heading = ParagraphStyle("CustomHeading", parent=styles["Heading2"], fontSize=14,
                             textColor=colors.HexColor("#34495e"), spaceAfter=12, spaceBefore=12)
    sm_heading = ParagraphStyle("SMHeading", parent=styles["Heading3"], fontSize=12,
                                textColor=colors.HexColor("#2980b9"), spaceAfter=8, spaceBefore=8)
    normal = styles["Normal"]
    story = [Paragraph("SYSTEM VALIDATION - MISSING CHANNELS REPORT", title), Spacer(1, 0.3 * inch),
             Paragraph("MINIMODULE STATUS", heading)]
    ok = findings.all_minimodules_observed
    status = "OK" if ok else "NOT OK"
    story.append(Paragraph(f"<font color='{'green' if ok else 'red'}'><b>Status: {status}</b></font>", normal))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        f"Observed in the sampled coincidences ({totals['occupancy_pairs']} pairs passing the channel cuts, "
        f"{totals['occupancy_hits']} channel hits); a channel without hits is not a hardware dead-channel verdict.",
        normal))
    story.append(Spacer(1, 0.1 * inch))
    if ok:
        story.append(Paragraph("All minimodules are present in the events.", normal))
    else:
        story.append(Paragraph("<b>Missing Minimodules:</b>", normal))
        story.append(Spacer(1, 0.1 * inch))
        table = Table([["SM", "Minimodule"]] + [[str(sm), str(mm)] for sm, mm in findings.missing_minimodules],
                      colWidths=[1.5 * inch, 1.5 * inch])
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                                   ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                   ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                   ("FONTSIZE", (0, 0), (-1, 0), 12), ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                                   ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                                   ("GRID", (0, 0), (-1, -1), 1, colors.black)]))
        story.append(table)
    if findings.unexpected_minimodules:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("<b>Declared unpopulated minimodules with hits:</b> " +
                               ", ".join(f"({sm}, {mm})" for sm, mm in findings.unexpected_minimodules), normal))
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("MISSING CHANNELS SUMMARY", heading))
    story.append(Spacer(1, 0.2 * inch))
    if not (missing_time or missing_energy):
        story.append(Paragraph("All expected channels are present in the events for all SMs.", normal))
    else:
        story.append(Paragraph(f"<font color='#e74c3c'><b>Total Missing Time Channels:</b> {missing_time}</font>",
                               normal))
        story.append(Paragraph(f"<font color='#f39c12'><b>Total Missing Energy Channels:</b> {missing_energy}</font>",
                               normal))
        story.append(Spacer(1, 0.3 * inch))
        for sm, missing in sorted(result.findings.missing.items()):
            if not (missing["Time"] or missing["Energy"]):
                continue
            story.append(Paragraph(f"SM {sm}", sm_heading))
            for kind, colour in (("Time", "#e74c3c"), ("Energy", "#f39c12")):
                if not missing[kind]:
                    continue
                story.append(Paragraph(f"<b>{kind} Channels:</b>", normal))
                rows = [["Channel", "ASIC", "(SM, mM)"]]
                rows += [[str(ch), str(get_electronics_nums(ch)), str(result.mapping.modules.get(ch, ("N/A", "N/A")))]
                         for ch in missing[kind]]
                table = Table(rows, colWidths=[1.2 * inch, 2 * inch, 1.5 * inch])
                table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(colour)),
                                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                           ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                           ("FONTSIZE", (0, 0), (-1, -1), 9),
                                           ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                                           ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
                                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
                story.append(table)
                story.append(Spacer(1, 0.15 * inch))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("SAMPLE AND PROVENANCE", heading))
    rule = expectation_rule(result)
    lines = [
        f"Source mode: {_source_text(result)}",
        f"Cuts: en_min_ch {result.en_min_ch:g} a.u. per hit at reading; min_ch {result.min_ch} energy channels per "
        "side before and after max-energy minimodule selection. Energy units: a.u. (no keV calibration).",
        f"Sample bound: at most {result.pair_limit} accepted pairs per file. Records read {totals['records_read']}; "
        f"pairs processed {totals['pairs_processed']}; occupancy pairs {totals['occupancy_pairs']}; accepted "
        f"resolved pairs {totals['accepted_pairs']} ({totals['accepted_sides']} sides); "
        + "; ".join(f"{name.replace('_', ' ')} {count}" for name, count in totals["rejected"].items()) + ".",
        f"Mapping: {result.sources['map']['path']} (sha256 {result.sources['map']['sha256'][:12]}); processing "
        f"config: {result.sources['processing_config']['path']} (sha256 "
        f"{result.sources['processing_config']['sha256'][:12]}).",
        "Expected channels: selected map minus declared unpopulated minimodules "
        f"({len(rule['declared_unpopulated'])} declared). The legacy hardcoded half-SuperModule rule is not applied; "
        f"{len(rule['expected_now_skipped_by_legacy'])} minimodules it would skip are expected here and "
        f"{len(rule['skipped_now_expected_by_legacy'])} it would expect are skipped.",
    ]
    for line in lines:
        story.append(Paragraph(line, normal))
        story.append(Spacer(1, 0.05 * inch))
    files = [["File", "Read", "Accepted", "Stopped at limit"]]
    files += [[Path(f.path).name, str(f.records_read), str(f.accepted_pairs), "yes" if f.stopped_at_limit else "no"]
              for f in result.files]
    table = Table(files, colWidths=[3.2 * inch, 1 * inch, 1 * inch, 1.2 * inch])
    table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
    story.append(table)
    story.append(Spacer(1, 0.4 * inch))
    footer = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
    story.append(Paragraph(f"Report generated: {directory_name}", footer))
    SimpleDocTemplate(stream, pagesize=letter).build(story)


# Excel ---------------------------------------------------------------------

def write_excel(result, stream):
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Photopeak Values"
    sheet.append(list(EXCEL_COLUMNS))
    fitted = sorted((e for e in result.minimodule_fits if e.available), key=lambda e: e.mu)
    other = [e for e in result.minimodule_fits if not e.available]
    for entry in fitted + other:
        sheet.append([entry.key[0], entry.key[1], entry.mu, entry.sigma, entry.resolution_percent, entry.status,
                      entry.samples, entry.sample_mean, entry.sample_std])
    totals = result.totals()
    provenance = workbook.create_sheet("Provenance")
    for row in (("Units", "a.u. (raw; no keV calibration)"),
                ("Mu/Sigma", "Gaussian photopeak fit; blank when the fit is unavailable"),
                ("Sample mean/std", "population moments of the sampled sides, never a photopeak"),
                ("Fit", f"fit_gaussian cb={MINIMODULE_CB}, min_peak={FIT_MIN_PEAK}, {PHOTOPEAK_BINS} bins "
                        f"{PHOTOPEAK_RANGE[0]}-{PHOTOPEAK_RANGE[1]} a.u."),
                ("Source mode", _source_text(result)),
                ("Pair limit per file", result.pair_limit),
                ("Accepted resolved pairs", totals["accepted_pairs"]),
                ("Accepted sides", totals["accepted_sides"]),
                ("en_min_ch (a.u.)", result.en_min_ch), ("min_ch", result.min_ch),
                ("Map", result.sources["map"]["path"]), ("Map sha256", result.sources["map"]["sha256"])):
        provenance.append(list(row))
    for f in result.files:
        provenance.append(["Input", f.path, f.records_read, f.accepted_pairs,
                           "stopped at limit" if f.stopped_at_limit else "whole file"])
    workbook.save(stream)


# Plots ---------------------------------------------------------------------

def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    return plt


def _save(figure, path, written):
    with open(path, "xb") as out:
        figure.savefig(out, format="png")
    written.append(path)


def _grid(count):
    if count <= 16:
        return 4, 4
    side = math.ceil(math.sqrt(count))
    return side, side


def plot_photopeaks(entries, histograms_edges, directory, written, *, per_slab):
    """Reference ``extract_photopeak_slab``/``_mM`` figures, one per minimodule/SuperModule."""
    plt = _pyplot()
    groups = {}
    for entry in entries:
        groups.setdefault(entry.key[:2] if per_slab else entry.key[0], []).append(entry)
    centres = (histograms_edges[:-1] + histograms_edges[1:]) / 2
    width = histograms_edges[1] - histograms_edges[0]
    for group, members in groups.items():
        slots = max(entry.key[-1] for entry in members) + 1
        rows, cols = _grid(slots)
        figure, axes = plt.subplots(rows, cols, figsize=(15, 10))
        try:
            axes = np.asarray(axes).flatten()
            figure.suptitle(f"Minimodule: {group}" if per_slab else f"SM: {group}")
            for entry in members:
                ax = axes[entry.key[-1]]
                ax.bar(centres, entry.histogram, width=width, align="center", color="blue", alpha=0.7)
                label = "Slab" if per_slab else "Minimodule"
                if entry.available:
                    ax.axvline(entry.mu, color="r", linestyle="dashed", linewidth=2)
                    ax.legend([f"{label}: {entry.key}\nRes: {round(entry.resolution_percent, 2)}%\n"
                               f"mu: {round(entry.mu, 1)} a.u."], fontsize=8)
                else:
                    ax.legend([f"{label}: {entry.key}\nFit unavailable ({entry.status})\nn: {entry.samples}"],
                              fontsize=8)
            name = f"photopeak_{group[0]}_{group[1]}.png" if per_slab else f"photopeak_SM_{group}.png"
            _save(figure, directory / name, written)
        finally:
            plt.close(figure)


def plot_slab_mu_distribution(result, path, written):
    plt = _pyplot()
    mus = [e.mu for e in result.slab_fits if e.available]
    unavailable = sum(not e.available for e in result.slab_fits)
    figure = plt.figure()
    try:
        ax = figure.gca()
        ax.hist(mus, bins=PHOTOPEAK_BINS, range=PHOTOPEAK_RANGE, alpha=0.6)
        mean, std = float(np.mean(mus)), float(np.std(mus))
        ax.axvline(mean, color="r", linestyle="dashed", linewidth=2)
        ax.fill_betweenx([0, ax.get_ylim()[1]], mean - std, mean + std, color="r", alpha=0.2)
        ax.set_title(f"Histogram of Photopeak Distributions ({unavailable} slab fits unavailable)")
        ax.set_xlabel("Mu (a.u.)")
        ax.set_ylabel("Frequency")
        ax.text(0.6, 0.7, f"Mean: {mean}\nStd Dev: {std}", transform=ax.transAxes)
        _save(figure, path, written)
    finally:
        plt.close(figure)


def cassettes(modules):
    """Reference cassette convention (3 SuperModules each), over the selected map's SuperModules."""
    groups = {}
    for sm in sorted(modules):
        groups.setdefault(sm // 3, []).append(sm)
    return groups


def plot_channel_frequency(result, types, path, written):
    plt = _pyplot()
    groups = cassettes(result.expected)
    figure, axes = plt.subplots(nrows=len(groups), ncols=1, figsize=(15, 5 * len(groups)), squeeze=False)
    try:
        for ax, (cassette, members) in zip(axes[:, 0], groups.items()):
            data, with_data = {}, []
            for sm in members:
                if result.occupancy.get(sm):
                    with_data.append(sm)
                    for channel, count in result.occupancy[sm].items():
                        data[channel] = data.get(channel, 0) + count
            if not data:
                ax.set_title(f"No channel data for Cassette {cassette} (Expected SMs: {members})")
                ax.text(0.5, 0.5, "No channel data", ha="center", va="center", transform=ax.transAxes)
                continue
            channels = sorted(data)
            colours = ["red" if ChannelType.ENERGY in types.get(ch, ()) else
                       "green" if ChannelType.TIME in types.get(ch, ()) else "gray" for ch in channels]
            x = np.arange(len(channels))
            ax.bar(x, [data[ch] for ch in channels], color=colours, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([str(ch) for ch in channels], rotation=90, ha="center", fontsize=8)
            ax.set_title(f"Channels in Cassette {cassette} (SMs: {', '.join(map(str, with_data))}) | "
                         "Energy: red, Time: green")
            ax.set_xlabel("Channel number")
            ax.set_ylabel("Hits in occupancy pairs")
        figure.tight_layout()
        _save(figure, path, written)
    finally:
        plt.close(figure)


def plot_slab_distribution(result, directory, written):
    plt = _pyplot()
    for sm in sorted(result.expected):
        keys = sorted((k for k in result.slab_counts if k[0] == sm), key=lambda k: (k[1], k[2]))
        figure, ax = plt.subplots(figsize=(12, 7))
        try:
            if not keys:
                ax.set_title(f"SM {sm} (No Data)")
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel("Slab ID (mM_slab)")
                ax.set_ylabel("Count")
                ax.set_xticks([])
                _save(figure, directory / f"slab_distribution_SM{sm}_no_data.png", written)
                continue
            x = np.arange(len(keys))
            ax.bar(x, [result.slab_counts[k] for k in keys], color=SM_COLORS[sm % len(SM_COLORS)], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{k[1]}_{k[2]}" for k in keys], rotation=90, ha="center", fontsize=8)
            ax.set_title(f"Slab Count Distribution for SM {sm}")
            ax.set_xlabel("Slab ID (mM_slab)")
            ax.set_ylabel("Accepted sides")
            figure.tight_layout()
            _save(figure, directory / f"slab_distribution_SM{sm}.png", written)
        finally:
            plt.close(figure)


def _log_norm(counts):
    from matplotlib.colors import LogNorm
    nonzero = counts[counts > 0]
    return LogNorm(vmin=nonzero.min(), vmax=counts.max()) if nonzero.size else LogNorm(vmin=1e-1, vmax=1)


def plot_floods(result, directory, written):
    plt = _pyplot()
    import matplotlib.colors as mcolors

    edges = result.flood_edges
    for sm, counts in result.floods.items():
        figure, ax = plt.subplots(figsize=(8, 6))
        try:
            mesh = ax.pcolormesh(edges, edges, counts.T, cmap=plt.cm.viridis, norm=_log_norm(counts))
            ax.set_title(f"Floodmap for SM {sm} ({FLOOD_ENERGY[0]}-{FLOOD_ENERGY[1]} a.u.)")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            figure.colorbar(mesh, ax=ax, label="Counts")
            _save(figure, directory / f"floodmap_SM_{sm}.png", written)
        finally:
            plt.close(figure)
    if not result.floods:
        return
    order = sorted(result.floods, reverse=True)             # reference: highest SM on the left
    combined = np.concatenate([result.floods[sm] for sm in order], axis=0)
    span = FLOOD_RANGE[1] - FLOOD_RANGE[0]
    x_edges = np.concatenate([edges[:-1] + i * span for i in range(len(order))] + [[edges[-1] + (len(order) - 1) * span]])
    colours = plt.cm.plasma(np.linspace(0, 1, 256)) ** 0.5
    figure, ax = plt.subplots(figsize=(max(18, 6 * len(order)), 6))
    try:
        masked = np.ma.masked_less_equal(combined.T, 0)
        mesh = ax.pcolormesh(x_edges, edges, masked, cmap=mcolors.ListedColormap(colours), norm=_log_norm(combined))
        ax.set_title("Floodmap for all SMs (left to right: SM " + ", ".join(map(str, order)) + ")")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        figure.colorbar(mesh, ax=ax, label="Counts")
        _save(figure, directory / "floodmap_all_SM.png", written)
    finally:
        plt.close(figure)


# Directory -----------------------------------------------------------------

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(result, destination, *, in_place=False, title=None):
    """Create ``destination`` exclusively and write the reference output types, summary last.

    ``in_place`` (T29): ``destination`` is the caller's existing folder; every file is still created
    exclusively, so a name collision fails. ``title``: the PDF title (default: the directory name).
    Returns the written paths; the last one is the summary.
    """
    destination = Path(destination)
    if in_place:
        if destination.is_symlink() or not destination.is_dir():
            raise InputError(f"QC results folder is not a plain directory: {destination}")
    else:
        if not destination.parent.is_dir():
            raise InputError(f"QC results parent does not exist: {destination.parent}")
        destination.mkdir()                 # never reuses an existing results directory
    written = []
    with open(destination / PDF, "xb") as stream:
        write_pdf(result, stream, title or destination.name)
    written.append(destination / PDF)
    if result.plots:
        edges = np.histogram(np.zeros(0), bins=PHOTOPEAK_BINS, range=PHOTOPEAK_RANGE)[1]
        if result.slabs:
            plot_photopeaks(result.slab_fits, edges, destination, written, per_slab=True)
        plot_photopeaks(result.minimodule_fits, edges, destination, written, per_slab=False)
        with open(destination / EXCEL, "xb") as stream:
            write_excel(result, stream)
        written.append(destination / EXCEL)
        plot_channel_frequency(result, result.mapping.types, destination / "channels_present_per_cassette.png",
                               written)
        plot_slab_distribution(result, destination, written)
        plot_floods(result, destination, written)
        if any(e.available for e in result.slab_fits):
            plot_slab_mu_distribution(result, destination / "photopeak_distribution.png", written)
    outputs = [{"path": p.name, "sha256": _sha256(p)} for p in written]
    content = summary(result, outputs)
    content["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    content["results_directory"] = str(destination)
    with open(destination / SUMMARY, "x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    written.append(destination / SUMMARY)
    return tuple(written)
