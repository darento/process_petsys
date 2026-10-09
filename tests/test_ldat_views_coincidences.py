"""The SM x SM pair matrix, paired time differences and the Coincidences tab of LDATInspector (spec 002 T11;
spec 007 T22).

Exact counts and values on a hand-built pair list, then the tab.

Moved from ``scripts/ldat_views_check.py``: each section function is the script's, copied unchanged
except for underscore prefixes, ``destroy(app)``, a ``tmp`` folder argument and the removed ``--png``
figures. A module fixture runs a section once and records its ``check`` calls (``ldat_helpers.Checks``);
each check is one test, so a GUI section's steps still run in order on one hidden window.
"""

from dataclasses import replace

import numpy as np
import pytest

from ldat_helpers import CONFIGS, Checks, destroy, require_display, side
from src.ldat_inspector.engine import (FileResult, Selection, SideTable, Settings, merge_results, pair_dt,
                                       pair_mask, pair_matrix)

pytestmark = pytest.mark.fr("002-FR-15")
fr = pytest.mark.fr


PAIR_SELECTION = Selection(400.0, 650.0, 0.1, 0.9, 10.0, 90.0, 0.0, 102.0)


T0_PS = 290_000_000_000_000  # ~290 s in ps: large stamps, so the difference must be taken in integers


def pair_dataset():
    """Cornell pairs with known sides. Returns (dataset, pairs); each pair is a dict of both sides.

    - SM 0 - SM 1: 101 passing pairs, t0 - t1 = 3 + (k - 50) * 0.5 ns; every other pair
      stored with the SM 1 side first;
    - SM 0 - SM 2: 20 passing, 5 failing the partner energy, 5 failing DOI on the SM 2
      side only, 5 failing ROI X on the SM 0 side only;
    - SM 3 - SM 3: 10 passing pairs inside one SM; SM 4 - SM 5: 7 passing pairs.
    """
    good = dict(e=511.0, doi=0.5, x=50.0, y=50.0)
    pairs = []

    def add(sm_a, sm_b, dt_ns, a=None, b=None, swap=False):
        t_a = T0_PS + 1_000_000 * len(pairs)
        t_b = t_a - int(round(dt_ns * 1000))
        side_a = {**good, **(a or {}), "sm": sm_a, "t": t_a}
        side_b = {**good, **(b or {}), "sm": sm_b, "t": t_b}
        pairs.append((side_b, side_a) if swap else (side_a, side_b))

    for k in range(101):
        add(0, 1, 3 + (k - 50) * 0.5, swap=k % 2 == 1)
    for k in range(20):
        add(0, 2, -8.0 + k)
    for k in range(5):
        add(0, 2, 1.0, b={"e": 300.0})
        add(0, 2, 1.0, b={"doi": 0.95})
        add(0, 2, 1.0, a={"x": 95.0})
    for k in range(10):
        add(3, 3, 0.25 * k)
    for k in range(7):
        add(4, 5, 12.0)
    sides = [side for pair in pairs for side in pair]
    n = len(sides)
    column = lambda key, dtype=float: np.array([s[key] for s in sides], dtype=dtype)  # noqa: E731
    table = SideTable.from_pairs(0, raw_energy=column("e"), calibration_key=np.zeros(n, np.int32),
                                 x=column("x"), y=column("y"), doi=column("doi"),
                                 timestamp=column("t", np.int64), sm=column("sm", np.int16),
                                 mm=np.zeros(n, np.int8), random_slab=np.zeros(n, bool))
    settings = Settings(str(CONFIGS["CORNELL"]), "", "CORNELL", max_pairs=None, calibrated=False)
    dataset = merge_results(settings, [FileResult(0, "synthetic", n // 2, n // 2, table=table)])
    return dataset, pairs


def passes(side, other, s):
    return (s.doi_low <= side["doi"] <= s.doi_high and s.x_low <= side["x"] <= s.x_high
            and s.y_low <= side["y"] <= s.y_high
            and s.energy_low <= side["e"] <= s.energy_high and s.energy_low <= other["e"] <= s.energy_high)


def coincidence_checks(check):
    dataset, pairs = pair_dataset()
    s = PAIR_SELECTION
    passing = [(a, b) for a, b in pairs if passes(a, b, s) and passes(b, a, s)]
    sms = sorted(dataset.expected_time)
    want = np.zeros((len(sms), len(sms)), np.int64)
    for a, b in passing:  # written out pair by pair, independently of the bincount
        i, j = sms.index(a["sm"]), sms.index(b["sm"])
        want[i, j] += 1
        if i != j:
            want[j, i] += 1
    matrix = pair_matrix(dataset, s)
    check("coincidence matrix: exact symmetric SM x SM counts under the cuts, each pair counted once",
          matrix["sms"] == sms and np.array_equal(matrix["counts"], want)
          and np.array_equal(matrix["counts"], matrix["counts"].T)
          and matrix["pairs"] == len(passing) == 138 and matrix["ingest_pairs"] == len(pairs) == 153
          and want[0, 1] == 101 and want[0, 2] == 20 and want[3, 3] == 10,
          f"{matrix['pairs']} of {matrix['ingest_pairs']} pairs; 0-1 {matrix['counts'][0, 1]}, "
          f"0-2 {matrix['counts'][0, 2]}, 3-3 {matrix['counts'][3, 3]}")
    loose = pair_matrix(dataset, Selection(0.0, 1500.0, 0.0, 1.0, 0.0, 102.0, 0.0, 102.0))
    check("without cuts every accepted pair is in the matrix once (the failing 0-2 pairs return)",
          loose["pairs"] == 153 and loose["counts"][0, 2] == 35
          and int(np.triu(loose["counts"]).sum()) == 153, f"0-2 {loose['counts'][0, 2]}")

    def expected_dt(sm_a, sm_b):
        out = []
        for a, b in passing:
            for x, y in ((a, b), (b, a)):
                if x["sm"] == sm_a and (sm_b is None or y["sm"] == sm_b):
                    out.append((x["t"] - y["t"]) / 1000)
                    break  # a same-SM pair once
        return np.sort(np.array(out))

    ab, ba = pair_dt(dataset, 0, 1, selection=s), pair_dt(dataset, 1, 0, selection=s)
    want01 = expected_dt(0, 1)
    check("Δt = t_a − t_b: exact values, sign fixed by the ordered pair (swapped storage does not matter)",
          np.array_equal(np.sort(ab["dt_ns"]), want01) and np.array_equal(np.sort(-ba["dt_ns"]), want01)
          and ab["count"] == ba["count"] == 101 and ab["median"] == 3.0 and ba["median"] == -3.0,
          f"median {ab['median']} / {ba['median']}")
    p16, p84 = np.percentile(want01, [16, 84])
    check("Δt median and central-68 % width match the fixture (p84 − p16)",
          ab["p16"] == p16 and ab["p84"] == p84 and ab["width68"] == p84 - p16 and abs(ab["width68"] - 34.0) < 1e-9,
          f"p16 {ab['p16']}, p84 {ab['p84']}, width {ab['width68']}")
    everyone = pair_dt(dataset, 0, None, selection=s)
    within = pair_dt(dataset, 3, 3, selection=s)
    within_all = pair_dt(dataset, 3, None, selection=s)
    none = pair_dt(dataset, 7, 8, selection=s)
    check("Δt against all partners; a same-SM pair counted once; no pairs gives no statistics",
          everyone["count"] == 121 and np.array_equal(np.sort(everyone["dt_ns"]), expected_dt(0, None))
          and within["count"] == within_all["count"] == 10
          and np.array_equal(np.sort(within["dt_ns"]), expected_dt(3, 3))
          and none["count"] == 0 and none["median"] is None and none["width68"] is None,
          f"SM 0 vs all {everyone['count']}, SM 3 within {within['count']}")
    mask = pair_mask(dataset, s)
    check("the precomputed pair mask gives the same matrix and Δt (as the GUI reuses it)",
          np.array_equal(pair_matrix(dataset, s, mask=mask)["counts"], matrix["counts"])
          and np.array_equal(pair_dt(dataset, 0, 1, mask=mask)["dt_ns"], ab["dt_ns"])
          and mask.sum() == 2 * 138)


def coincidence_gui_checks(check):
    import threading
    import time
    from tkinter import messagebox
    from unittest.mock import patch

    from matplotlib.backend_bases import MouseEvent

    import exe_programs.ldat_inspector_gui as gui

    dataset, _ = pair_dataset()
    # the fixture energies stand for keV, so loading keeps a keV window (config energy_range)
    dataset = replace(dataset, settings=replace(dataset.settings, calibrated=True))
    app = gui.LDATWorkbench()
    app.withdraw()
    errors = []

    def pump(until, timeout=20):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            app.update()
            if until():
                return True
            time.sleep(0.01)
        return False

    def click(col, row):
        app.update()
        app.coinc_canvas.draw()
        x, y = app._matrix_ax.transData.transform((float(col), float(row)))
        app.coinc_canvas.callbacks.process("button_press_event",
                                           MouseEvent("button_press_event", app.coinc_canvas, x, y, button=1))

    try:
        real, threads = gui.pair_matrix, set()

        def recorded(*args, **kwargs):
            threads.add(threading.get_ident())
            return real(*args, **kwargs)

        with patch.object(messagebox, "showerror", side_effect=lambda *a, **k: errors.append(a)), \
                patch.object(gui, "pair_matrix", recorded):
            for var, value in zip((app.energy_low, app.energy_high, app.doi_low, app.doi_high,
                                   app.x_low, app.x_high, app.y_low, app.y_high),
                                  ("400", "650", "0.1", "0.9", "10", "90", "0", "102")):
                var.set(value)
            app.tabs.set(gui.COINC_TAB)
            app.dataset = dataset
            app._update_after_processing()
            selection = app._selection()  # loading applies the config's energy_range
            first = [t.get_text() for t in app.coinc_fig.axes[0].texts] if app.coinc_fig.axes else []
            ready = pump(lambda: app._matrix_ax is not None)
            want = pair_matrix(dataset, selection)
            drawn = np.ma.filled(app._matrix_ax.images[0].get_array().astype(float), 0)
            title = app._matrix_ax.get_title()
            check("Coincidences tab: 'computing' first, then the matrix from a background job, equal to the engine",
                  any("Computing" in t for t in first) and ready and np.array_equal(drawn, want["counts"])
                  and threading.get_ident() not in threads and threads
                  and f"{want['pairs']:,} of {want['ingest_pairs']:,} pairs, each counted once" in title
                  and "both sides pass the display cuts" in title, title)
            check("default Δt is the first SM against all partners",
                  app.pair_info.cget("text").startswith("SM 0 ↔ all partners: 121 pairs")
                  and len([p for p in app._matrix_ax.patches if p.get_gid() == "pick"]) == 1,
                  app.pair_info.cget("text"))
            sms = want["sms"]
            click(sms.index(1), sms.index(0))  # row SM 0, column SM 1
            text = app.pair_info.cget("text")
            heights = sum(p.get_height() for p in app._dt_ax.patches if hasattr(p, "get_height"))  # bars, not the 68 % span
            picks = [p for p in app._matrix_ax.patches if p.get_gid() == "pick"]
            check("clicking a cell selects that SM pair: Δt histogram of all its pairs, median and 68 % width",
                  app.pair_a.get() == "SM 0" and app.pair_b.get() == "SM 1"
                  and text == "SM 0 ↔ SM 1: 101 pairs • median +3.00 ns • central 68 % width 34.00 ns"
                  and heights == 101 and len(picks) == 2, text)
            click(sms.index(0), sms.index(1))
            reverse = app.pair_info.cget("text")
            click(sms.index(3), sms.index(3))
            same = app._dt_ax.get_title()
            check("the mirrored cell flips the sign; a diagonal cell counts same-SM pairs once, sign arbitrary",
                  reverse.startswith("SM 1 ↔ SM 0: 101 pairs • median -3.00 ns")
                  and app.pair_info.cget("text").startswith("SM 3 ↔ SM 3: 10 pairs") and "sign arbitrary" in same,
                  reverse)
            labels = [w.cget("text") for w in app.tabs.tab(gui.COINC_TAB).winfo_children()
                      if isinstance(w, gui.ctk.CTkLabel)]
            check("labels state the observational limits (geometry, time of flight; not clock or CTR)",
                  gui.DT_LABEL in labels and "not a clock or CTR calibration" in same
                  and "time of flight" in same, f"{labels}")
            app.pair_b.set(gui.ALL_PARTNERS)
            app.pair_a.set("SM 7")
            app._draw_pair_dt()
            check("an SM without passing pairs reads 'no pairs pass the cuts' (no statistics)",
                  app.pair_info.cget("text") == "SM 7 ↔ all partners: no pairs pass the cuts")
            jobs = len(threads)
            app.doi_high.set("1.0")  # the failing-DOI 0-2 pairs now pass
            app._refresh_all()
            pump(lambda: app._matrix_ax is not None and len(app._pair_cache) == 2)
            loose = pair_matrix(dataset, app._selection())
            drawn = np.ma.filled(app._matrix_ax.images[0].get_array().astype(float), 0)
            app.doi_high.set("0.9")
            app._refresh_all()
            cached = app._matrix_ax is not None  # the first selection is still cached: drawn at once
            check("a cut change recomputes the matrix; the previous selection is cached",
                  np.array_equal(drawn, loose["counts"]) and loose["counts"][0, 2] == 25 and cached,
                  f"0-2 {loose['counts'][0, 2]}, cached {cached}")
            app._show_tab(gui.SM_TAB)
            stale_before = gui.COINC_TAB in app._stale
            app._step_sm(1)
            check("stepping SMs does not make the Coincidences tab stale",
                  not stale_before and gui.COINC_TAB not in app._stale)
            app._invalidate_data()
            check("invalidating inputs clears the Coincidences state",
                  not app._pair_cache and app._pair_current is None and app._matrix_ax is None)
        check("no error dialogs during the Coincidences checks", not errors, f"{errors[:1]}")
    finally:
        destroy(app)


COINCIDENCES = {
    "matrix": "coincidence matrix: exact symmetric SM x SM counts under the cuts, each pair counted once",
    "no-cuts": "without cuts every accepted pair is in the matrix once (the failing 0-2 pairs return)",
    "dt-sign":
        "Δt = t_a − t_b: exact values, sign fixed by the ordered pair (swapped storage does not matter)",
    "dt-width": "Δt median and central-68 % width match the fixture (p84 − p16)",
    "dt-partners": "Δt against all partners; a same-SM pair counted once; no pairs gives no statistics",
    "pair-mask": "the precomputed pair mask gives the same matrix and Δt (as the GUI reuses it)",
}
COINCIDENCES_GUI = {
    "computing-first":
        "Coincidences tab: 'computing' first, then the matrix from a background job, equal to the engine",
    "default-dt": "default Δt is the first SM against all partners",
    "click-cell": "clicking a cell selects that SM pair: Δt histogram of all its pairs, median and 68 % width",
    "mirrored-and-diagonal":
        "the mirrored cell flips the sign; a diagonal cell counts same-SM pairs once, sign arbitrary",
    "labels": "labels state the observational limits (geometry, time of flight; not clock or CTR)",
    "no-pairs": "an SM without passing pairs reads 'no pairs pass the cuts' (no statistics)",
    "cut-change": "a cut change recomputes the matrix; the previous selection is cached",
    "stepping-not-stale": "stepping SMs does not make the Coincidences tab stale",
    "invalidate": "invalidating inputs clears the Coincidences state",
    "no-error-dialogs": "no error dialogs during the Coincidences checks",
}


@pytest.fixture(scope="module")
def coincidences():
    return Checks(COINCIDENCES).run(coincidence_checks)


@fr("002-FR-11", "002-FR-12")
@pytest.mark.parametrize("check", COINCIDENCES)
def test_coincidences(coincidences, check):
    coincidences.verdict(check)


@pytest.fixture(scope="module")
def coincidences_gui():
    require_display()
    return Checks(COINCIDENCES_GUI).run(coincidence_gui_checks)


@pytest.mark.gui
@fr("002-FR-12")
@pytest.mark.parametrize("check", COINCIDENCES_GUI)
def test_coincidences_gui(coincidences_gui, check):
    coincidences_gui.verdict(check)
