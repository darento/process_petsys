"""keV factor origins in LDATInspector (spec 002 T17; spec 007 T22).

From the calibration's _status.txt: per-SM side and slab counts (unpopulated vs no fit), the fitted-only
cut, no/malformed sidecars, the GUI switches, note and column, and the PDF provenance. The owner's resolved
calibration check is ``real_data`` (spec 007 Clarify, T21/T22).

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

from ldat_helpers import CONFIGS, Checks, destroy, require_display, time_channels_by_position
from src.ldat_inspector.engine import (FileResult, Selection, SideTable, Settings, load_setup, merge_results,
                                       minimodule_metrics, unpopulated_minimodules)
from src.mapping_generator import ChannelType

pytestmark = pytest.mark.fr("002-FR-15")
fr = pytest.mark.fr


# (SM, minimodule, position, slab): (status text, sides); every other mapped slab is "fit".
ORIGIN_CASES = {
    (0, 4, 0, 0): ("borrowed from slab 1", 12), (0, 4, 0, 1): ("fit", 20),
    (0, 4, 1, 2): ("fit; check: higher-energy peak near 99.1", 8), (0, 4, 1, 3): ("fit", 20),
    (0, 4, 2, 4): ("estimated from neighbour slabs [5]", 10),
    (0, 4, 2, 5): ("estimated from minimodule median (11 slabs)", 6),
    (0, 4, 3, 6): ("no fit: 150 events (< 200)", 4), (0, 4, 3, 7): ("fit", 20),
    (0, 4, 4, 8): (None, 4),  # in the .encal, missing from the status file: unknown
    (1, 0, 0, 0): ("borrowed from slab 1", 6), (1, 0, 0, 1): ("fit", 10),
}


# Even side counts per key keep both sides of every pair on one key (pairs are rows 2k, 2k + 1).
ORIGIN_SIDES = {0: Counter({"fitted": 60, "fitted (check)": 8, "borrowed": 12, "estimated (neighbours)": 10,
                            "estimated (median)": 6, "no fit": 4, "unknown": 4}),
                1: Counter({"fitted": 10, "borrowed": 6})}


def origin_fixture(tmp, *, sidecar=True, status_text=None):
    """Calibrated Cornell sides on the ORIGIN_CASES keys, with an .encal (+ status) for every mapped slab."""
    settings = Settings(str(CONFIGS["CORNELL"]), "", "CORNELL", max_pairs=None, calibrated=False)
    setup = load_setup(settings)
    unpopulated = unpopulated_minimodules(setup.config)
    special = {(time_channels_by_position(setup, sm, mm)[p], slab): value
               for (sm, mm, p, slab), value in ORIGIN_CASES.items()}
    encal, status, sides = ["ID(t_ch, slab)\tmu\tsigma\n"], ["ID(t_ch, slab)\tstatus\n"], []
    for ch in sorted(setup.channel_types):
        if ChannelType.TIME not in setup.channel_types[ch] or ch not in setup.coordinates:
            continue
        sm, mm = setup.channel_modules[ch]
        for slab in (2 * setup.coordinates[ch][2], 2 * setup.coordinates[ch][2] + 1):
            text, n = special.get((ch, slab), ("no fit: no resolved events" if mm in unpopulated.get(sm, ())
                                               else "fit", 0))
            mu = 0.0 if text and text.startswith("no fit") else 100.0
            encal.append(f"({ch}, {slab})\t{mu}\t5.0\n")
            if text is not None:
                status.append(f"({ch}, {slab})\t{text}\n")
            sides += [((ch << 5) | slab, sm, mm)] * n
    name = "origin_fixture.encal"
    (tmp / name).write_text("".join(encal), encoding="utf-8")
    sidecar_path = tmp / "origin_fixture_status.txt"
    if sidecar:
        sidecar_path.write_text(status_text if status_text is not None else "".join(status), encoding="utf-8")
    else:
        sidecar_path.unlink(missing_ok=True)
    n = len(sides)
    column = lambda i, dtype: np.array([s[i] for s in sides], dtype=dtype)  # noqa: E731
    table = SideTable.from_pairs(0, raw_energy=np.full(n, 100.0), calibration_key=column(0, np.int64),
                                 x=np.full(n, 50.0), y=np.full(n, 50.0), doi=np.full(n, 0.5),
                                 timestamp=np.arange(n, dtype=np.int64), sm=column(1, np.int16),
                                 mm=column(2, np.int8), random_slab=np.zeros(n, bool))
    calibrated = replace(settings, calibration_path=str(tmp / name), calibrated=True)
    return merge_results(calibrated, [FileResult(0, "synthetic", n // 2, n // 2, table=table)])


def origin_checks(check, tmp):

    from src.ldat_inspector.engine import (FACTOR_ORIGINS, apply_calibration, apply_doi_view, factor_origins,
                                    load_calibration_status, slab_origins)

    dataset = origin_fixture(tmp)
    got = {sm: factor_origins(dataset, data) for sm, data in dataset.modules.items()}
    total = sum(got.values(), Counter())
    check("keV factor origins: exact detector-side counts per SM and in total from the status sidecar",
          got == ORIGIN_SIDES and total == sum(ORIGIN_SIDES.values(), Counter()) and total["unknown"] == 4,
          f"{ {sm: dict(c) for sm, c in got.items()} }")
    slabs0, slabs2, every = slab_origins(dataset, {0}), slab_origins(dataset, {2}), slab_origins(dataset)
    check("slab origins: SM 0 has 256 slabs (250 fitted, one of each other origin); the half-populated SM 2 "
          "has 128 unpopulated slabs, not 'no fit'",
          slabs0 == Counter({"fitted": 250, "fitted (check)": 1, "borrowed": 1, "estimated (neighbours)": 1,
                             "estimated (median)": 1, "no fit": 1, "unknown": 1})
          and slabs2 == Counter({"fitted": 128, "unpopulated": 128})
          and sum(every.values()) == 7680 and every["unpopulated"] == 1280 and every["borrowed"] == 2,
          f"SM 0 {dict(slabs0)}; SM 2 {dict(slabs2)}; all {dict(every)}")
    data = dataset.modules[0]
    keep = Selection(400.0, 650.0, fitted_only=True).mask(data)
    plain = Selection(400.0, 650.0).mask(data)
    codes = [FACTOR_ORIGINS[c] for c in data.origin[keep]]
    check("fitted-only cut keeps exactly the fitted and fitted (check) sides; 'no fit' sides have no keV",
          set(codes) <= {"fitted", "fitted (check)"} and int(keep.sum()) == 68 and int(plain.sum()) == 100
          and np.array_equal(keep, plain & (data.origin <= 1)), f"{int(keep.sum())} of {int(plain.sum())}")
    metrics = minimodule_metrics(dataset, Selection(400.0, 650.0, fitted_only=True), fits=False, sms=[0])
    check("per-minimodule selected counts use the fitted-only population", metrics[(0, 4)]["selected"] == 68,
          f"{metrics[(0, 4)]['selected']}")
    same = apply_doi_view(dataset, None)
    raw = apply_calibration(dataset, dataset.settings.calibration_path, False)
    back = apply_calibration(raw, dataset.settings.calibration_path, True)
    check("origins follow the calibration: kept by the DOI view, none in raw a.u., restored when keV returns",
          np.array_equal(same.modules[0].origin, data.origin) and raw.modules[0].origin is None
          and factor_origins(raw, raw.modules[0]) == Counter() and slab_origins(raw) == Counter()
          and raw.calibration_status is None and factor_origins(back, back.modules[0]) == ORIGIN_SIDES[0])

    plain_dataset = origin_fixture(tmp, sidecar=False)
    unknown = factor_origins(plain_dataset, plain_dataset.modules[0])
    check("without a sidecar every side and slab is 'unknown' (never fitted); fitted-only keeps nothing",
          unknown == Counter({"unknown": 104}) and plain_dataset.calibration_status is None
          and slab_origins(plain_dataset, {0}) == Counter({"unknown": 256})
          and not Selection(400.0, 650.0, fitted_only=True).mask(plain_dataset.modules[0]).any(),
          f"{dict(unknown)}")
    imas = replace(dataset, settings=replace(dataset.settings, system="IMAS"))
    check("IMAS calibrations have no slab factor origins (nothing reported)",
          factor_origins(imas, imas.modules[0]) == Counter() and slab_origins(imas) == Counter())

    bad = {"unrecognised status": "ID(t_ch, slab)\tstatus\n(256, 0)\tguessed\n",
           "missing tab": "(256, 0) fit\n", "slab 16": "(256, 16)\tfit\n",
           "duplicate": "(256, 0)\tfit\n(256, 0)\tfit\n", "header only": "ID(t_ch, slab)\tstatus\n"}
    accepted = []
    for label, text in bad.items():
        try:
            origin_fixture(tmp, status_text=text)
            accepted.append(label)
        except ValueError:
            pass
    check(f"{len(bad)} malformed sidecars reject the calibration (no guessed origin)", not accepted, f"{accepted}")
    origin_fixture(tmp)  # the malformed cases above overwrote the sidecar
    loaded = load_calibration_status(str(tmp / "origin_fixture.encal"))
    check("a missing sidecar is None, not an empty status", loaded is not None and len(loaded) > 0
          and load_calibration_status(str(tmp / "absent.encal")) is None)


def origin_gui_checks(check, tmp):
    import time
    from tkinter import messagebox, ttk
    from unittest.mock import patch

    import exe_programs.ldat_inspector_gui as gui
    from src.ldat_inspector.engine import factor_origins, uniformity

    dataset = origin_fixture(tmp)
    app = gui.LDATWorkbench()
    app.withdraw()
    errors = []

    def load(ds):
        app.dataset = ds
        app._update_after_processing()
        app.update()

    try:
        with patch.object(messagebox, "showerror", side_effect=lambda *a, **k: errors.append(a)):
            app.tabs.set(gui.SM_TAB)
            load(dataset)
            cell = [str(v) for v in app.channel_tree.item("0", "values")]
            summary = app.channel_summary.cget("text")
            check("Channel Status: borrowed / estimated keV factor sides per SM, totals and mapped slabs in the summary",
                  cell[10] == "12 / 16" and [str(v) for v in app.channel_tree.item("1", "values")][10] == "6 / 0"
                  and "ingest sides: fitted 70 • check 8 • borrowed 18" in summary
                  and "unknown 4" in summary and "unpopulated 1,280" in summary, f"{cell[10]}; {summary}")
            data = dataset.modules[0]
            spatial = app._selection().mask(data, energy=False) & np.isfinite(data.energy)
            note = app.sm_summary.get("1.0", "end")
            want = factor_origins(dataset, data, spatial)
            rows = [f"  {gui.ORIGIN_SHORT[name]:<16}{want[name]:>11,}" for name in gui.ORIGIN_SHORT if want.get(name)]
            check("SuperModule summary: keV factor origins of the energy plot's sides; the fitted-only switch is available",
                  "KEV FACTORS (energy plot sides)\n" + "\n".join(rows) in note
                  and [r.split() for r in rows[:3]] == [["fitted", "60"], ["check", "8"], ["borrowed", "12"]]
                  and not any("keV factor" in t.get_text() for t in app.energy_ax.texts)
                  and app.fitted_check.cget("state") == "normal" and app.fitted_note.cget("text") == "",
                  note.replace("\n", " / "))
            app.fitted_only.set(True)
            app._refresh_all()
            info = app.explorer_info.cget("text")
            check("'Fitted keV factors only' restricts the selection (68 of SM 0's sides) and says so on the plot",
                  app._selection().fitted_only and "68 after paired energy" in info
                  and "fitted keV factors only" in app.sm_summary.get("1.0", "end")
                  and app.energy_ax.get_title().endswith("• fitted keV factors"), info)
            app.fitted_only.set(False)
            app._refresh_all()
            app._uniformity()
            app.update()
            windows = [w for w in app.winfo_children() if isinstance(w, gui.ctk.CTkToplevel)]
            trees = [w for w in windows[-1].winfo_children() if isinstance(w, ttk.Treeview)] if windows else []
            rows = {str(trees[0].item(i)["values"][0]): trees[0].item(i)["values"] for i in trees[0].get_children()} \
                if trees else {}
            want = {row["sm"]: row for row in uniformity(dataset, replace(app._selection(), fitted_only=True))}
            check("Photopeak Uniformity uses fitted factors only by default (its own switch)",
                  app.uniformity_fitted_only.get() and rows and int(rows["0"][1]) == want[0]["events"]
                  and want[0]["events"] < len(data), f"SM 0 sides {rows.get('0', ['?', '?'])[1]} vs {len(data)}")
            for window in windows:
                window.destroy()

            load(origin_fixture(tmp, sidecar=False))
            note = app.sm_summary.get("1.0", "end")
            check("no sidecar: 'unknown' in Channel Status and the SM summary; fitted-only unavailable",
                  [str(v) for v in app.channel_tree.item("0", "values")][10] == "unknown"
                  and "origin unknown: no _status.txt" in note and app.fitted_check.cget("state") == "disabled"
                  and "no _status.txt" in app.fitted_note.cget("text") and not app._selection().fitted_only
                  and "keV factor origin unknown" in app.channel_summary.cget("text"), note)
            load(replace(dataset, settings=replace(dataset.settings, calibrated=False)))
            check("raw a.u.: no factor column values, no origin note, fitted-only unavailable",
                  [str(v) for v in app.channel_tree.item("0", "values")][10] == "—"
                  and "KEV FACTORS" not in app.sm_summary.get("1.0", "end")
                  and app.fitted_check.cget("state") == "disabled" and "raw a.u." in app.fitted_note.cget("text"))
        check("no error dialogs during the keV factor origin checks", not errors, f"{errors[:1]}")
    finally:
        destroy(app)


def origin_report_checks(check, tmp):
    import re

    from pypdf import PdfReader

    from src.ldat_inspector.report import write_report

    dataset = origin_fixture(tmp)

    def text_of(path):
        return re.sub(r"\s+", " ", " ".join(page.extract_text() for page in PdfReader(str(path)).pages))

    pdf = tmp / "origins.pdf"
    write_report(str(pdf), dataset, Selection(400.0, 650.0, fitted_only=True), sm=0)
    text = text_of(pdf)
    check("PDF provenance: status path, sides and slabs by origin in scope, and the fitted-only fit population",
          "origin_fixture_status.txt" in text.replace(" ", "")
          and "Ingest sides by keV factor origin (SM 0): fitted 60, fitted (check) 8, borrowed 12, "
              "estimated (neighbours) 10, estimated (median) 6, no fit 4, unknown 4" in text
          and "Mapped slabs by keV factor origin (SM 0): fitted 250" in text
          and "fitted keV factors only (borrowed and estimated left out)" in text
          and "keV factor sides: fitted 60" in text, text[:200])
    plain = tmp / "unknown.pdf"
    write_report(str(plain), origin_fixture(tmp, sidecar=False), Selection(400.0, 650.0))
    text = text_of(plain)
    check("PDF without a sidecar reads 'keV factor origins: unknown … never assumed fitted'",
          "keV factor origins: unknown" in text and "never assumed fitted" in text and "fitted 60" not in text)


ORIGINS = {
    "side-counts":
        "keV factor origins: exact detector-side counts per SM and in total from the status sidecar",
    "slab-counts":
        "slab origins: SM 0 has 256 slabs (250 fitted, one of each other origin); the half-populated SM 2 has 128 "
        "unpopulated slabs, not 'no fit'",
    "fitted-only":
        "fitted-only cut keeps exactly the fitted and fitted (check) sides; 'no fit' sides have no keV",
    "mm-selected": "per-minimodule selected counts use the fitted-only population",
    "follow-calibration":
        "origins follow the calibration: kept by the DOI view, none in raw a.u., restored when keV returns",
    "no-sidecar":
        "without a sidecar every side and slab is 'unknown' (never fitted); fitted-only keeps nothing",
    "imas": "IMAS calibrations have no slab factor origins (nothing reported)",
    "malformed-sidecars": "5 malformed sidecars reject the calibration (no guessed origin)",
    "missing-sidecar": "a missing sidecar is None, not an empty status",
}
ORIGINS_GUI = {
    "channel-status":
        "Channel Status: borrowed / estimated keV factor sides per SM, totals and mapped slabs in the summary",
    "sm-summary":
        "SuperModule summary: keV factor origins of the energy plot's sides; the fitted-only switch is available",
    "fitted-only":
        "'Fitted keV factors only' restricts the selection (68 of SM 0's sides) and says so on the plot",
    "uniformity": "Photopeak Uniformity uses fitted factors only by default (its own switch)",
    "no-sidecar": "no sidecar: 'unknown' in Channel Status and the SM summary; fitted-only unavailable",
    "raw": "raw a.u.: no factor column values, no origin note, fitted-only unavailable",
    "no-error-dialogs": "no error dialogs during the keV factor origin checks",
}
ORIGINS_REPORT = {
    "provenance":
        "PDF provenance: status path, sides and slabs by origin in scope, and the fitted-only fit population",
    "no-sidecar": "PDF without a sidecar reads 'keV factor origins: unknown … never assumed fitted'",
}


@pytest.fixture(scope="module")
def origins():
    with tempfile.TemporaryDirectory(prefix="ldat_") as temporary:
        yield Checks(ORIGINS).run(origin_checks, Path(temporary))


@fr("002-FR-20")
@pytest.mark.parametrize("check", ORIGINS)
def test_origins(origins, check):
    origins.verdict(check)


@pytest.fixture(scope="module")
def origins_gui():
    require_display()
    with tempfile.TemporaryDirectory(prefix="ldat_") as temporary:
        yield Checks(ORIGINS_GUI).run(origin_gui_checks, Path(temporary))


@pytest.mark.gui
@fr("002-FR-20")
@pytest.mark.parametrize("check", ORIGINS_GUI)
def test_origins_gui(origins_gui, check):
    origins_gui.verdict(check)


@pytest.fixture(scope="module")
def origins_report():
    with tempfile.TemporaryDirectory(prefix="ldat_") as temporary:
        yield Checks(ORIGINS_REPORT).run(origin_report_checks, Path(temporary))


@pytest.mark.slow  # the fixture runs the section: 8.8 s serial
@fr("002-FR-14", "002-FR-20")
@pytest.mark.parametrize("check", ORIGINS_REPORT)
def test_origins_report(origins_report, check):
    origins_report.verdict(check)


# --- owner resolved calibration --------------------------------------------------------------------------------
# Calibration: PETSYS_CAL_DIR/<RESOLVED_ENCAL> and its _status.txt sidecar, the January 2026-01-19 compact
# per-slab calibration (resolved, with the 0.2 a.u. per-channel cut). Map: January, through CONFIGS["CORNELL"].

RESOLVED_ENCAL = ("20260119_2NaSourcesAxialSeparated_vBiasCompDiscCalibAdjusted2hits_300s_"
                  "coincCompact11s_resolved.encal")


@pytest.mark.real_data
@fr("002-FR-20", "007-FR-4")
def test_real_resolved_calibration_no_fit_slabs_are_the_unpopulated_ones(real_cal_file):
    """Sidecar counts; the 1,280 'no fit' slabs are exactly the unpopulated ones."""
    from src.ldat_inspector.engine import FACTOR_ORIGINS, load_calibration_status, slab_origins

    encal = real_cal_file(RESOLVED_ENCAL)
    real_cal_file(RESOLVED_ENCAL.replace(".encal", "_status.txt"))
    status = load_calibration_status(str(encal))
    in_file = Counter(FACTOR_ORIGINS[c] for c in status.codes)
    with tempfile.TemporaryDirectory(prefix="ldat_") as temporary:
        dataset = origin_fixture(Path(temporary))
    mapped = slab_origins(replace(dataset, calibration_status=status))
    assert (in_file == Counter({"fitted": 5356, "no fit": 1280, "borrowed": 791, "fitted (check)": 114,
                                "estimated (neighbours)": 80, "estimated (median)": 59})
            and mapped == Counter({"fitted": 5356, "unpopulated": 1280, "borrowed": 791, "fitted (check)": 114,
                                   "estimated (neighbours)": 80, "estimated (median)": 59})), \
        f"file {dict(in_file)}; mapped {dict(mapped)}"
