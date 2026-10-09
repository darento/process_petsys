"""The SuperModule tab of LDATInspector (spec 002 T10; spec 007 T22).

Prev/Next, wheel and Page Up/Down stepping (clamped, debounced), visible-tab-only redraw, the summary panel
against spec 001's Status-tab content, and the per-minimodule table from a background job (latest request
wins, caches reused), calibrated and raw.

Moved from ``scripts/ldat_views_check.py``: each section function is the script's, copied unchanged
except for underscore prefixes, ``destroy(app)``, a ``tmp`` folder argument and the removed ``--png``
figures. A module fixture runs a section once and records its ``check`` calls (``ldat_helpers.Checks``);
each check is one test, so a GUI section's steps still run in order on one hidden window.
"""

from collections import Counter

import pytest

from ldat_helpers import EXPECTED_ROW, Checks, build, destroy, metrics_dataset, require_display
from src.ldat_inspector.engine import FINDING_COLOURS, RAW_FIT, minimodule_layout

pytestmark = pytest.mark.fr("002-FR-15")
fr = pytest.mark.fr


def mm_rows(app):
    """Per-minimodule table as {mm: (text cells..., tag)}; Treeview hands numbers back as numbers."""
    tree = app.mm_tree
    return {int(tree.item(i, "values")[0]): tuple(str(v) for v in tree.item(i, "values")[1:])
            + tuple(tree.item(i, "tags")) for i in tree.get_children()}


def want_mm_row(row, populated):
    if not populated:
        return f"{row['ingest']:,}", f"{row['selected']:,}", "—", "—", "unpopulated (config)", "unpopulated"
    fit = row["fit"]
    if fit["status"] == "FIT":
        return f"{row['ingest']:,}", f"{row['selected']:,}", f"{fit['mu']:.1f}", f"{fit['resolution']:.1f}", "FIT", "fit"
    return f"{row['ingest']:,}", f"{row['selected']:,}", "—", "—", fit["status"], "unavailable"


def supermodule_gui_checks(check):
    """T10: stepping, visible-tab-only redraw, summary panel and per-minimodule table."""
    import threading
    import time
    from tkinter import messagebox
    from unittest.mock import patch

    import exe_programs.ldat_inspector_gui as gui
    from src.ldat_inspector.engine import channel_status

    app = gui.LDATWorkbench()
    app.withdraw()
    errors, calls = [], Counter()
    sm_tab = gui.SM_TAB
    for name, draw in list(app._drawers.items()):
        def counted(draw=draw, name=name):
            calls[name] += 1
            draw()
        app._drawers[name] = counted

    def pump(until, timeout=20):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            app.update()
            if until():
                return True
            time.sleep(0.01)
        return False

    def shown(sm):
        return app.explorer_info.cget("text").startswith((f"SM {sm} ", f"SM {sm}:"))

    def select(sm):  # as the combobox command does
        app.module_var.set(f"SM {sm}")
        app._sm_changed()

    try:
        with patch.object(messagebox, "showerror", side_effect=lambda *a, **k: errors.append(a)):
            check("tabs are Channel Status, SuperModule, System Overview, Coincidences, Timestamps; the Status tab is retired",
                  app.tabs._name_list == list(gui.TABS) and "SuperModule Status" not in app.tabs._name_list
                  and not hasattr(app, "detail"), f"{app.tabs._name_list}")
            dataset, assignment, truth, setup = build("CORNELL")
            app.tabs.set(sm_tab)
            app.dataset = dataset
            app._update_after_processing()
            app.update()
            ids = sorted(dataset.expected_time)
            check("loading shows the first SM; Prev is disabled at the start",
                  app._sm_ids == ids and app._selected_sm() == ids[0] and shown(ids[0])
                  and app.sm_position.cget("text") == f"1 / {len(ids)}"
                  and str(app.prev_button.cget("state")) == "disabled"
                  and str(app.next_button.cget("state")) == "normal", app.explorer_info.cget("text"))

            # --- Prev / Next buttons, clamped
            calls.clear()
            app.prev_button.invoke()  # disabled: no step, no redraw
            at_start = app._selected_sm() == ids[0] and not app._step_sm(-1) and calls[sm_tab] == 0
            app.next_button.invoke()
            moved = app._selected_sm() == ids[1] and shown(ids[1]) and app.sm_position.cget("text") == "2 / 30"
            app.prev_button.invoke()
            back = app._selected_sm() == ids[0] and shown(ids[0])
            select(ids[-1])
            app.next_button.invoke()
            at_end = (app._selected_sm() == ids[-1] and not app._step_sm(1)
                      and str(app.next_button.cget("state")) == "disabled"
                      and app.sm_position.cget("text") == f"{len(ids)} / {len(ids)}")
            check("Prev/Next step one SM and redraw at once; both are clamped at the ends",
                  at_start and moved and back and at_end and calls[sm_tab] == 3, f"{dict(calls)}")

            # --- mouse wheel over the SM box and the toolbar background, debounced. Tk drops
            # generated events for unmapped windows: map it, fully transparent, for these two checks.
            app.attributes("-alpha", 0.0)
            app.deiconify()
            app.update()
            select(ids[0])
            calls.clear()
            app.module_combo._entry.event_generate("<MouseWheel>", delta=120)  # up at the first SM: clamped
            for _ in range(3):
                app.module_combo._entry.event_generate("<MouseWheel>", delta=-120)
            immediate = (app._selected_sm(), calls[sm_tab], app.sm_position.cget("text"))
            pump(lambda: False, 0.35)
            burst = (app._selected_sm(), calls[sm_tab], shown(ids[3]))
            app.sm_toolbar._canvas.event_generate("<MouseWheel>", delta=120)
            pump(lambda: False, 0.35)
            check("wheel over the SM box or toolbar steps (down = next, clamped); a burst redraws once after 150 ms",
                  immediate == (ids[3], 0, "4 / 30") and burst == (ids[3], 1, True)
                  and app._selected_sm() == ids[2] and calls[sm_tab] == 2, f"{immediate} {burst} {dict(calls)}")

            # --- Page Up / Page Down, only while the SuperModule tab is visible
            widget = app.canvas.get_tk_widget()
            widget.focus_force()
            select(ids[-2])
            calls.clear()
            widget.event_generate("<Next>")
            widget.event_generate("<Next>")  # clamped at the last SM
            pump(lambda: False, 0.35)
            down = app._selected_sm() == ids[-1] and calls[sm_tab] == 1 and shown(ids[-1])
            widget.event_generate("<Prior>")
            pump(lambda: False, 0.35)
            up = app._selected_sm() == ids[-2] and calls[sm_tab] == 2
            app.tabs.set("Channel Status")
            app.channel_tree.focus_force()
            app.channel_tree.event_generate("<Next>")
            pump(lambda: False, 0.35)
            ignored = app._selected_sm() == ids[-2] and calls[sm_tab] == 2
            check("Page Down / Page Up step (clamped, debounced) only while the SuperModule tab is visible",
                  down and up and ignored, f"down {down}, up {up}, ignored {ignored}, {dict(calls)}")
            app.withdraw()
            app.attributes("-alpha", 1.0)

            # --- only the visible tab is redrawn
            app._show_tab(sm_tab)  # nothing changed since it was drawn: no redraw
            calls.clear()
            for _ in range(3):
                app.prev_button.invoke()
            stepping = dict(calls)
            app._refresh_all()  # a cut change
            after_cut = dict(calls)
            app.tabs._segmented_button_callback("Timestamps")  # a click on the tab button
            app._show_tab("System Overview")
            pump(lambda: app._overview_grid is not None)
            overview_draws = calls["System Overview"]
            app._show_tab(sm_tab)  # still current: no redraw
            app.next_button.invoke()  # a step does not make the other tabs stale
            app._show_tab("Timestamps")
            app._show_tab("System Overview")
            final = dict(calls)
            check("stepping redraws only the SuperModule tab; a cut change redraws the visible tab, "
                  "others once when shown",
                  stepping == {sm_tab: 3} and after_cut == {sm_tab: 4}
                  and final == {sm_tab: 5, "Timestamps": 1, "System Overview": overview_draws}
                  and overview_draws >= 1, f"stepping {stepping}, after cut {after_cut}, final {final}")

            # --- summary panel: spec 001's Status-tab content, and the findings
            app._show_tab(sm_tab)
            lost = []
            for case, sm in assignment.items():
                app._open_sm(sm)
                pump(lambda: app.mm_tree.get_children())
                text = app.sm_summary.get("1.0", "end")
                status = channel_status(dataset, sm)
                need = [f"TIME   {len(status['active_time'])}/{len(status['expected_time'])} observed",
                        f"ENERGY {len(status['active_energy'])}/{len(status['expected_energy'])} observed",
                        f"Ingest sides    {status['events']:>11,}"]
                need += [str(ch) for ch in sorted(status["unobserved_time"] | status["unobserved_energy"])]
                missing = [s for s in need if s not in text]
                table = mm_rows(app)
                data = dataset.modules.get(sm)
                occupancy = Counter(data.mm.tolist()) if data is not None else Counter()
                missing += [f"mM{mm}" for mm, n in occupancy.items() if table.get(mm, ("",))[0] != f"{n:,}"]
                state = EXPECTED_ROW[case]
                if app.sm_state.cget("text") != f"SM {sm} • CORNELL • {state}" \
                        or app.sm_state.cget("fg_color") != FINDING_COLOURS[state]:
                    missing.append(f"state {app.sm_state.cget('text')}")
                if missing:
                    lost.append(f"{case}: {missing[:3]}")
            check("summary keeps the Status-tab content for every fixture SM: sides, observed x/y, "
                  "unobserved IDs, mM occupancy, finding colour", not lost, "; ".join(lost))
            mixed = assignment["mixed"]
            t_ch, e_ch = sorted(dataset.expected_time[mixed]), sorted(dataset.expected_energy[mixed])
            app._open_sm(mixed)
            text = app.sm_summary.get("1.0", "end")
            app._open_sm(assignment["half-populated"])
            half = app.sm_summary.get("1.0", "end")
            app._open_sm(assignment["few sides"])
            few = app.sm_summary.get("1.0", "end")
            check("summary lists flagged channels by state, stray hits and insufficient reasons",
                  f"not observed: {t_ch[0]}" in text and f"low: {e_ch[0]}" in text and f"high: {e_ch[3]}" in text
                  and "1 not observed / 0 low / 0 high" in text and "Minimodules seen 1/16 expected" in text
                  and half.count("(5,000)") == 4 and "unexpected hits" in half and "Minimodules seen 1/8" in half
                  and "insufficient: 98 ingest sides < 100" in few
                  and f"0 hits: {sorted(dataset.expected_time[assignment['few sides']])[0]}" in few, text)
            app._open_sm(assignment["low"])
            app.finding_low.set("0.04")
            app._apply_thresholds()
            relaxed = app.sm_state.cget("text")
            app.finding_low.set("0.15")
            app._apply_thresholds()
            check("threshold Apply updates the visible SuperModule summary",
                  relaxed.endswith("• OK") and app.sm_state.cget("text").endswith("• LOW"), relaxed)

            # --- per-minimodule table (calibrated): a background job, latest request wins, caches reused
            mdata, _ = metrics_dataset(calibrated=True)
            for var, value in zip((app.energy_low, app.energy_high, app.doi_low, app.doi_high,
                                   app.x_low, app.x_high, app.y_low, app.y_high),
                                  ("400", "650", "0.1", "0.9", "10", "90", "0", "102")):
                var.set(value)
            real = gui.minimodule_metrics
            jobs, threads = [], set()

            def recorded(*args, delay=0.0, **kwargs):
                threads.add(threading.get_ident())
                time.sleep(delay)
                jobs.append(kwargs.get("sms"))
                return real(*args, **kwargs)

            with patch.object(gui, "minimodule_metrics", recorded):
                app.dataset = mdata
                app._update_after_processing()
                computing = "computing" in app.mm_title.cget("text") and not app.mm_tree.get_children()
                pump(lambda: app.mm_tree.get_children())
                selection = app._selection()
                full = real(mdata, selection)
                rows = mm_rows(app)
                want = {mm: want_mm_row(full[(0, mm)], True) for mm in mdata.expected_mm[0]}
                total_selected = sum(full[(0, mm)]["selected"] for mm in mdata.expected_mm[0])
                summary = app.sm_summary.get("1.0", "end")
                check("SM 0 table: 'computing' first, then ingest, selected, centroid, resolution and fit per mM",
                      computing and rows == want and jobs == [[0]] and threading.get_ident() not in threads
                      and f"Selected sides  {total_selected:>11,}" in summary
                      and f"{total_selected:,} after paired" in app.explorer_info.cget("text")
                      and "16/16 fitted" in app.mm_title.cget("text"),
                      f"{app.mm_title.cget('text')}; jobs {jobs}")
                select(1)
                pump(lambda: app.mm_tree.get_children())
                rows = mm_rows(app)
                check("a failed or empty fit is unavailable with its status (never a value)",
                      rows.get(3) == want_mm_row(full[(1, 3)], True) and rows[3][-1] == "unavailable"
                      and rows.get(7, ())[:2] == ("0", "0") and rows[7][-1] == "unavailable",
                      f"{rows.get(3)} {rows.get(7)}")

            jobs.clear()
            slow = lambda *a, **k: recorded(*a, delay=0.4, **k)  # noqa: E731
            with patch.object(gui, "minimodule_metrics", slow):
                app._sm_cache = []
                select(0)  # starts a job for SM 0
                app.next_button.invoke()  # SM 1 wanted ...
                app.next_button.invoke()  # ... then SM 2: only the latest runs next
                pump(lambda: jobs == [[0], [2]] and app.mm_tree.get_children(), 10)
                rows = mm_rows(app)
                populated = minimodule_layout(mdata)[2]["populated"]
                want2 = {mm: want_mm_row(full.get((2, mm), {"ingest": 0, "selected": 0}), populated[mm])
                         for mm in populated}
                check("stepping during a job: the latest SM runs next, skipped SMs are never computed; "
                      "half-populated rows read 'unpopulated (config)'",
                      jobs == [[0], [2]] and rows == want2
                      and sum(1 for r in rows.values() if r[-1] == "unpopulated") == 8, f"jobs {jobs}")
                select(0)
                cached = len(app.mm_tree.get_children()) == 16
                app.overview_mode.set("Photopeak centroid (keV)")
                with patch.object(gui, "minimodule_metrics", recorded):
                    app._show_tab("System Overview")
                    pump(lambda: any(e["fits"] for e in app._overview_cache))
                jobs.clear()
                app._sm_cache = []
                app._show_tab(sm_tab)
                select(4)
                reused = len(app.mm_tree.get_children()) == 16 and not jobs
                check("revisited SMs and System Overview fits are reused without a new job",
                      cached and reused, f"cached {cached}, reused {reused}, jobs {jobs}")

            raw, _ = metrics_dataset(calibrated=False)
            app.dataset = raw
            app._update_after_processing()
            pump(lambda: app.mm_tree.get_children())
            rows = mm_rows(app)
            check("raw mode: fit columns unavailable (raw a.u.), counts kept",
                  all(r[2:5] == ("—", "—", RAW_FIT["status"]) for r in rows.values()) and len(rows) == 16
                  and "raw a.u." in app.mm_title.cget("text")
                  and all(rows[mm][0] == want[mm][0] for mm in rows), app.mm_title.cget("text"))
            app._invalidate_data()
            check("invalidating inputs clears the summary, table and stepping controls",
                  not app.mm_tree.get_children() and not app.sm_summary.get("1.0", "end").strip()
                  and app.sm_position.cget("text") == "—" and not app._sm_ids and not app._sm_cache
                  and str(app.prev_button.cget("state")) == str(app.next_button.cget("state")) == "disabled")
        check("no error dialogs during the SuperModule checks", not errors, f"{errors[:1]}")
    finally:
        destroy(app)


SUPERMODULE_GUI = {
    "tabs":
        "tabs are Channel Status, SuperModule, System Overview, Coincidences, Timestamps; the Status tab is "
        "retired",
    "loading": "loading shows the first SM; Prev is disabled at the start",
    "prev-next": "Prev/Next step one SM and redraw at once; both are clamped at the ends",
    "wheel":
        "wheel over the SM box or toolbar steps (down = next, clamped); a burst redraws once after 150 ms",
    "page-keys": "Page Down / Page Up step (clamped, debounced) only while the SuperModule tab is visible",
    "visible-tab-redraw":
        "stepping redraws only the SuperModule tab; a cut change redraws the visible tab, others once when shown",
    "summary-status-content":
        "summary keeps the Status-tab content for every fixture SM: sides, observed x/y, unobserved IDs, mM "
        "occupancy, finding colour",
    "summary-flags": "summary lists flagged channels by state, stray hits and insufficient reasons",
    "threshold-apply": "threshold Apply updates the visible SuperModule summary",
    "sm0-table": "SM 0 table: 'computing' first, then ingest, selected, centroid, resolution and fit per mM",
    "failed-fit": "a failed or empty fit is unavailable with its status (never a value)",
    "stepping-during-job":
        "stepping during a job: the latest SM runs next, skipped SMs are never computed; half-populated rows read "
        "'unpopulated (config)'",
    "reuse": "revisited SMs and System Overview fits are reused without a new job",
    "raw": "raw mode: fit columns unavailable (raw a.u.), counts kept",
    "invalidate": "invalidating inputs clears the summary, table and stepping controls",
    "no-error-dialogs": "no error dialogs during the SuperModule checks",
}


@pytest.fixture(scope="module")
def supermodule_gui():
    require_display()
    return Checks(SUPERMODULE_GUI).run(supermodule_gui_checks)


@pytest.mark.gui
@pytest.mark.slow  # the fixture runs the section: 10.6 s serial
@fr("002-FR-10", "002-FR-13")
@pytest.mark.parametrize("check", SUPERMODULE_GUI)
def test_supermodule_gui(supermodule_gui, check):
    supermodule_gui.verdict(check)
