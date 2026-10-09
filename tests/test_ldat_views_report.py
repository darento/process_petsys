"""Channel findings and per-minimodule tables in LDATInspector's PDF reports (spec 002 T12; spec 007 T22).

The PDF tables equal the engine values; scope, calibration and populations are labelled.

Moved from ``scripts/ldat_views_check.py``: each section function is the script's, copied unchanged
except for underscore prefixes, ``destroy(app)``, a ``tmp`` folder argument and the removed ``--png``
figures. A module fixture runs a section once and records its ``check`` calls (``ldat_helpers.Checks``);
each check is one test, so a GUI section's steps still run in order on one hidden window.
"""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pytest

from ldat_helpers import SELECTION, Checks, build, metrics_dataset
from src.ldat_inspector.engine import (FileResult, FindingThresholds, Selection, SideTable, channel_findings,
                                       load_setup, merge_results, minimodule_layout, minimodule_metrics,
                                       system_channel_findings)

pytestmark = pytest.mark.fr("002-FR-15")
fr = pytest.mark.fr


def pdf_pages(path):
    from pypdf import PdfReader
    import re
    return [re.sub(r"\s+", " ", page.extract_text()) for page in PdfReader(str(path)).pages]


def findings_line(dataset, row):
    """The expected channel-findings row of one SM, from the engine's findings (whitespace-normalised)."""
    def flags(kind):
        return "insufficient" if kind["insufficient"] else "/".join(
            str(kind["counts"].get(s, 0)) for s in ("NOT OBSERVED", "LOW", "HIGH"))

    def median(kind):
        return "--" if kind["median"] is None else f"{kind['median']:,.0f}"

    sm, t, e = row["sm"], row["time"], row["energy"]
    data = dataset.modules.get(sm)
    expected = dataset.expected_mm.get(sm, set())
    seen = len(set(np.unique(data.mm).tolist()) & expected) if data is not None else 0
    unexpected = len(t["unexpected"]) + len(e["unexpected"])
    return " ".join(map(str, (sm, f"{row['events']:,}", f"{seen}/{len(expected)}", len(t["channels"]), flags(t),
                              median(t), len(e["channels"]), flags(e), median(e),
                              f"{unexpected} ch" if unexpected else "--", row["state"])))


def mm_line(mm, row, populated):
    fit = row["fit"]
    if not populated:
        tail = "-- -- unpopulated (config)"
    elif fit["status"] == "FIT":
        tail = f"{fit['mu']:.1f} {fit['resolution']:.1f} FIT"
    else:
        tail = f"-- -- {fit['status']}"
    return f"{mm} {row['ingest']:,} {row['selected']:,} {row['fit_sides']:,} {tail}"


def report_checks(check, tmp):
    """T12: channel findings and per-minimodule tables in the PDF reports equal the engine values."""
    import re
    import time

    from src.ldat_inspector.report import write_report

    thresholds = FindingThresholds(low_frac=0.04)
    for system in ("CORNELL", "IMAS"):
        dataset, assignment, truth, _ = build(system)
        pdf = tmp / f"{system}_findings.pdf"
        t0 = time.perf_counter()
        write_report(str(pdf), dataset, Selection(0.0, 10.0), thresholds=thresholds)
        pages = pdf_pages(pdf)
        elapsed = time.perf_counter() - t0
        rows = system_channel_findings(dataset, thresholds)
        findings_pages = [p for p in pages if p.startswith("CHANNEL FINDINGS")]
        text = re.sub(r"\s*/\s*", "/", " ".join(findings_pages))
        missing = [row["sm"] for row in rows if findings_line(dataset, row) not in text]
        states = {case: channel_findings(dataset, sm, thresholds)["state"] for case, sm in assignment.items()}
        n = len(rows)
        want_pages = 1 + -(-n // 32) + -(-n // 34) + 2 * n
        check(f"{system} system PDF: channel findings rows for all {n} SMs equal the engine under the "
              "report's thresholds (low 0.04 turns the LOW SM OK)",
              not missing and len(findings_pages) == -(-n // 34) and states["low"] == "OK"
              and f"{n} SuperModules:" in text and len(pages) == want_pages,
              f"missing {missing[:5]}; {len(findings_pages)} findings pages; {len(pages)}/{want_pages} pages; "
              f"{elapsed:.1f} s")
        check(f"{system} system PDF: population, scope and thresholds on the findings and provenance pages",
              "channel hit counts in the coincidence ingest population, before display cuts" in findings_pages[0]
              and "Scope: whole files (no pair limit); not a dead/hot hardware verdict" in findings_pages[0]
              and "Thresholds: low < 0.04 × median, high > 3 × median" in findings_pages[0]
              and "Channel findings thresholds: low < 0.04 × median" in pages[0]
              and "whole files (no pair limit)" in pages[0], findings_pages[0][:200])
        mixed = assignment["mixed"]
        found = channel_findings(dataset, mixed, thresholds)
        page = next(p for p in pages if p.startswith(f"SUPERMODULE {mixed} - MINIMODULES"))
        ids = lambda kind, state: ", ".join(str(ch) for ch, s in zip(found[kind]["channels"], found[kind]["states"])  # noqa: E731
                                            if s == state)
        check(f"{system} SM {mixed} page lists its flagged channels by state; raw fits unavailable",
              "CHANNEL FINDINGS - NOT OBSERVED" in page
              and f"not observed: {ids('time', 'NOT OBSERVED')}" in page
              and f"high: {ids('energy', 'HIGH')}" in page
              and not ids("energy", "LOW") and " low: " not in page  # 14 hits is not below 0.04 x 100
              and "Fit: unavailable in raw a.u. (no keV calibration)." in page
              and "unavailable: raw a.u. (no keV calibration)" in page and " FIT" not in page, page[:300])
        if "half-populated" in assignment:
            sm = assignment["half-populated"]
            page = next(p for p in pages if p.startswith(f"SUPERMODULE {sm} - MINIMODULES"))
            check(f"{system} half-populated SM {sm}: 8 unpopulated rows and its unexpected hits on its page",
                  page.count("unpopulated (config)") == 8 and "unexpected hits (unpopulated mM):" in page
                  and "Populated minimodules: 8" in page, page[:200])

    # per-minimodule tables: calibrated, SM scope and system scope, prefix labels
    calibrated, _ = metrics_dataset(calibrated=True)
    encal = tmp / "synthetic.encal"
    encal.write_text("", encoding="utf-8")
    calibrated = replace(calibrated, settings=replace(calibrated.settings, calibration_path=str(encal),
                                                      max_pairs=5_000))
    metrics = minimodule_metrics(calibrated, SELECTION)
    layout = minimodule_layout(calibrated)
    for scope, sm in (("SM 1", 1), ("system", None)):
        pdf = tmp / f"mm_{scope}.pdf"
        write_report(str(pdf), calibrated, SELECTION, sm=sm)
        pages = pdf_pages(pdf)
        wrong, statuses = [], Counter()
        for s in ((1,) if sm is not None else (0, 1, 2)):
            page = next(p for p in pages if p.startswith(f"SUPERMODULE {s} - MINIMODULES"))
            populated = layout[s]["populated"]
            for mm in sorted(populated):
                row = metrics[(s, mm)] if populated[mm] else {"ingest": 0, "selected": 0, "fit_sides": 0,
                                                             "fit": None}
                if mm_line(mm, row, populated[mm]) not in page:
                    wrong.append((s, mm))
                statuses[row["fit"]["status"] if populated[mm] else "unpopulated"] += 1
        fitted = sum(1 for (s, _), m in metrics.items() if s == 1 and m["fit"]["status"] == "FIT")
        sm1 = next(p for p in pages if p.startswith("SUPERMODULE 1 - MINIMODULES"))
        check(f"calibrated {scope} PDF: every minimodule row (ingest, selected, fit sides, centroid, "
              "resolution, status) equals minimodule_metrics",
              not wrong and statuses["FIT"] >= (12 if sm is not None else 35)
              and (sm is not None or statuses["unpopulated"] == 8)
              and f"Fitted minimodules: {fitted}/16 populated" in sm1
              and "insufficient" in mm_line(3, metrics[(1, 3)], True) and mm_line(3, metrics[(1, 3)], True) in sm1,
              f"wrong {wrong[:4]}; {dict(statuses)}")
        check(f"calibrated {scope} PDF: prefix scope, calibration and populations labelled",
              "prefix, max 5,000 coincidence pairs / file" in pages[0]
              and "Scope: prefix, max 5,000 coincidence pairs / file; energy: keV with synthetic.encal" in sm1
              and "Selected: sides passing the display cuts (both energies 400..650 keV, DOI, ROI)" in sm1
              and "energy window off" in sm1
              and "Scope: prefix, max 5,000 coincidence pairs / file; not a dead/hot" in " ".join(pages), sm1[:300])
    raw, _ = metrics_dataset(calibrated=False)
    pdf = tmp / "mm_raw.pdf"
    write_report(str(pdf), raw, SELECTION, sm=0)
    page = next(p for p in pdf_pages(pdf) if p.startswith("SUPERMODULE 0 - MINIMODULES"))
    raw_metrics = minimodule_metrics(raw, SELECTION)
    check("raw SM 0 PDF: minimodule counts kept, every fit unavailable (raw a.u.)",
          all(mm_line(mm, raw_metrics[(0, mm)], True) in page for mm in range(16))
          and page.count("unavailable: raw a.u. (no keV calibration)") == 16 and "Fitted minimodules" not in page
          and "energy: raw PETsys a.u. (no keV calibration)" in page, page[:300])
    # SM report with ten long-path input files (owner report 2026-09-28: six real files overflowed): the info text must not run into the plots
    import matplotlib.backends.backend_agg
    import src.ldat_inspector.report as report
    _, columns = metrics_dataset(calibrated=False)
    n = len(columns["sm"])
    long_dir = "C:/Users/someone/Desktop/data/Cornell/full_system/" + "x" * 60  # real paths wrap to 3 lines
    files = [FileResult(i, f"{long_dir}/synthetic_{i}.ldat", n // 2 + 500, n // 2,
                        errors={"min channels": 300, "ValueError": 10, "unresolved Cornell slab": 190},
                        table=SideTable.from_pairs(
        i, raw_energy=columns["raw_energy"], calibration_key=np.zeros(n, np.int32), x=columns["x"], y=columns["y"],
        doi=columns["doi"], timestamp=np.arange(n, dtype=np.int64) * 10_000_000, sm=columns["sm"].astype(np.int16),
        mm=columns["mm"].astype(np.int8), random_slab=np.zeros(n, bool))) for i in range(10)]
    six = merge_results(raw.settings, files, load_setup(raw.settings))
    overlaps, real_pages = [], report.PdfPages

    class OverlapPages(real_pages):
        def savefig(self, figure=None, **kwargs):
            if figure is not None and len(figure.axes) == 1:  # text pages: every line inside the page
                matplotlib.backends.backend_agg.FigureCanvasAgg(figure).draw()
                page = figure.bbox
                for t in figure.axes[0].texts:
                    lines = t.get_text().split("\n")
                    if lines[0].endswith("(continued)") and len(lines) > 1 and lines[1].startswith("    "):
                        overlaps.append("a wrapped entry split by a page break: " + lines[1][:40])
                    box = t.get_window_extent(figure.canvas.get_renderer())
                    if box.y0 < page.y0 or box.y1 > page.y1 or box.x1 > page.x1:
                        overlaps.append("text outside page: " + t.get_text()[:40])
            if figure is not None and len(figure.axes) == 4:  # the SM detail page
                matplotlib.backends.backend_agg.FigureCanvasAgg(figure).draw()
                renderer = figure.canvas.get_renderer()
                info = [t.get_window_extent(renderer) for t in figure.axes[0].texts]
                for axis in figure.axes[1:]:
                    box = axis.get_tightbbox(renderer)
                    overlaps.extend(axis.get_title() for b in info if b.overlaps(box))
            return super().savefig(figure, **kwargs)

    report.PdfPages = OverlapPages
    try:
        write_report(str(tmp / "ten_files.pdf"), six, SELECTION, sm=0)
    finally:
        report.PdfPages = real_pages
    text = " ".join(pdf_pages(tmp / "ten_files.pdf")).replace(" ", "")
    check("ten-file SM report: no text off the page or over a plot; every input file and per-file span reported",
          not overlaps and all(f"{long_dir}/synthetic_{i}.ldat" in text for i in range(10))  # never split by a page
          and "Observedtimestampspansperfile" in text and text.count("medianpairedhitdifference") == 10,
          f"overlaps {overlaps}")

    from src.ldat_inspector.report import _provenance
    legacy =[line for line in _provenance(build("CORNELL")[0], SELECTION, only_sm=0) if line.startswith("Slab rule")]
    check("SM-scope provenance: the legacy unresolved-slab count is labelled all files, not per SM",
          len(legacy) == 1 and legacy[0].endswith("rejected at ingest (all files; pairs, not per SM)"), f"{legacy}")
    try:
        write_report(str(tmp / "bad.pdf"), raw, SELECTION, sm=0, thresholds=FindingThresholds(high_frac=1.0))
        refused = False
    except ValueError:
        refused = True
    check("invalid report thresholds are refused before writing", refused and not (tmp / "bad.pdf").exists())


REPORT = {
    "cornell-findings-rows":
        "CORNELL system PDF: channel findings rows for all 30 SMs equal the engine under the report's thresholds "
        "(low 0.04 turns the LOW SM OK)",
    "cornell-findings-labels":
        "CORNELL system PDF: population, scope and thresholds on the findings and provenance pages",
    "cornell-sm-page": "CORNELL SM 0 page lists its flagged channels by state; raw fits unavailable",
    "cornell-half-populated":
        "CORNELL half-populated SM 2: 8 unpopulated rows and its unexpected hits on its page",
    "imas-findings-rows":
        "IMAS system PDF: channel findings rows for all 120 SMs equal the engine under the report's thresholds "
        "(low 0.04 turns the LOW SM OK)",
    "imas-findings-labels":
        "IMAS system PDF: population, scope and thresholds on the findings and provenance pages",
    "imas-sm-page": "IMAS SM 0 page lists its flagged channels by state; raw fits unavailable",
    "calibrated-sm1-rows":
        "calibrated SM 1 PDF: every minimodule row (ingest, selected, fit sides, centroid, resolution, status) "
        "equals minimodule_metrics",
    "calibrated-sm1-labels": "calibrated SM 1 PDF: prefix scope, calibration and populations labelled",
    "calibrated-system-rows":
        "calibrated system PDF: every minimodule row (ingest, selected, fit sides, centroid, resolution, status) "
        "equals minimodule_metrics",
    "calibrated-system-labels": "calibrated system PDF: prefix scope, calibration and populations labelled",
    "raw-sm0": "raw SM 0 PDF: minimodule counts kept, every fit unavailable (raw a.u.)",
    "ten-files":
        "ten-file SM report: no text off the page or over a plot; every input file and per-file span reported",
    "sm-scope-provenance":
        "SM-scope provenance: the legacy unresolved-slab count is labelled all files, not per SM",
    "invalid-thresholds": "invalid report thresholds are refused before writing",
}


@pytest.fixture(scope="module")
def report():
    with tempfile.TemporaryDirectory(prefix="ldat_") as temporary:
        yield Checks(REPORT).run(report_checks, Path(temporary))


@pytest.mark.slow  # the fixture runs the section: 55 s serial
@fr("002-FR-5", "002-FR-13", "002-FR-14")
@pytest.mark.parametrize("check", REPORT)
def test_report(report, check):
    report.verdict(check)
