"""Channel findings and the Channel Status tab of LDATInspector (spec 002 T6, T7; spec 007 T22).

T6: synthetic datasets on the real IMAS and Cornell maps, each SM with hit counts chosen by hand so every
per-channel state, the row priority and each threshold change is known in advance. Cornell also covers a
half-populated SM whose unpopulated minimodules have hits (B3): they must not be assessed. T7: the Channel
Status tab in a withdrawn LDATWorkbench on the same Cornell fixtures: row tags and text, threshold Apply,
the channel map and bars drawn for a clicked row, and "Open in SuperModule".

Moved from ``scripts/ldat_views_check.py``: each section function is the script's, copied unchanged
except for underscore prefixes, ``destroy(app)``, a ``tmp`` folder argument and the removed ``--png``
figures. A module fixture runs a section once and records its ``check`` calls (``ldat_helpers.Checks``);
each check is one test, so a GUI section's steps still run in order on one hidden window.
"""

from dataclasses import replace

import numpy as np
import pytest

from ldat_helpers import BASE, EXPECTED_ROW, Checks, build, destroy, require_display
from src.ldat_inspector.engine import (FINDING_COLOURS, FINDING_PRIORITY, FindingThresholds, channel_findings,
                                       channel_geometry, system_channel_findings)

pytestmark = pytest.mark.fr("002-FR-15")
fr = pytest.mark.fr


def expected_states(hits, median, sufficient, thresholds):
    if not sufficient:
        return ["INSUFFICIENT EVENTS"] * len(hits)
    out = []
    for n in hits:  # written out independently of np.select in the engine
        if n == 0:
            out.append("NOT OBSERVED")
        elif n > thresholds.high_frac * median:
            out.append("HIGH")
        elif n < thresholds.low_frac * median:
            out.append("LOW")
        else:
            out.append("OK")
    return out


def findings_checks(check):
    default = FindingThresholds()
    check("provisional defaults 0.15 / 3.0 / 20 hits / 100 sides",
          (default.low_frac, default.high_frac, default.min_median, default.min_events) == (0.15, 3.0, 20.0, 100)
          and "0.15" in default.text() and "3 × median" in default.text())
    check("every state, including NO DATA, has a RAWInspector colour",
          set(FINDING_COLOURS) == set(FINDING_PRIORITY) | {"NO DATA"})

    for system in ("IMAS", "CORNELL"):
        dataset, assignment, truth, setup = build(system)
        mismatches, rows = [], {}
        for case, sm in assignment.items():
            found = channel_findings(dataset, sm)
            rows[case] = found["state"]
            events = len(dataset.modules[sm]) if sm in dataset.modules else 0
            for kind, by_channel in zip(("time", "energy"), truth[sm]):
                channels = sorted(by_channel)
                hits = [by_channel[ch] for ch in channels]
                median = float(np.median(hits))
                sufficient = events >= default.min_events and median >= default.min_median
                want = expected_states(hits, median, sufficient, default)
                got = found[kind]
                if got["channels"] != channels or got["hits"].tolist() != hits or got["states"] != want \
                        or got["median"] != median:
                    mismatches.append(f"{case}/{kind}")
        check(f"{system}: exact per-channel states and medians for {len(assignment)} fixture SMs",
              not mismatches, ", ".join(mismatches))
        check(f"{system}: row priority NOT OBSERVED > HIGH > LOW > INSUFFICIENT > OK; NO DATA without sides",
              rows == {case: EXPECTED_ROW[case] for case in assignment}, f"{rows}")

        mixed = channel_findings(dataset, assignment["mixed"])
        check(f"{system}: boundary hits (15, 300) are OK; 14 is LOW, 301 HIGH, 0 NOT OBSERVED",
              mixed["energy"]["states"][:4] == ["LOW", "OK", "OK", "HIGH"]
              and mixed["time"]["states"][0] == "NOT OBSERVED"
              and mixed["time"]["counts"]["NOT OBSERVED"] == 1 and mixed["energy"]["counts"]["HIGH"] == 1,
              f"{mixed['energy']['states'][:4]}")
        few = channel_findings(dataset, assignment["few sides"])
        low_median = channel_findings(dataset, assignment["low median"])
        check(f"{system}: low statistics read insufficient events, not not-observed (FR-7), with the reason",
              few["time"]["states"][0] == "INSUFFICIENT EVENTS" and "98 ingest sides" in few["time"]["insufficient"]
              and low_median["time"]["states"][0] == "INSUFFICIENT EVENTS"
              and "median 19" in low_median["time"]["insufficient"] and low_median["energy"]["insufficient"] is None,
              f"{few['time']['insufficient']}; {low_median['time']['insufficient']}")

        changes = {
            "low_frac 0.04 clears LOW (5 >= 4)":
                (assignment["low"], replace(default, low_frac=0.04), "OK"),
            "high_frac 10 clears HIGH (1000 is not > 1000), LOW remains":
                (assignment["high"], replace(default, high_frac=10.0), "LOW"),
            "min_median 101 makes an OK SM insufficient":
                (assignment["ok"], replace(default, min_median=101.0), "INSUFFICIENT EVENTS"),
            "min_median 19 assesses the low-median SM":
                (assignment["low median"], replace(default, min_median=19.0), "NOT OBSERVED"),
            "min_events 98 assesses the 98-side SM":
                (assignment["few sides"], replace(default, min_events=98), "NOT OBSERVED"),
        }
        got = {label: channel_findings(dataset, sm, t)["state"] for label, (sm, t, _) in changes.items()}
        check(f"{system}: threshold changes move the row states as expected",
              all(got[label] == want for label, (_, _, want) in changes.items()), f"{got}")

        rows_all = system_channel_findings(dataset)
        mapped = sorted(set(dataset.expected_time) | set(dataset.expected_energy))
        check(f"{system}: system findings give one row per mapped SM in order",
              [row["sm"] for row in rows_all] == mapped
              and all(row["state"] == "NO DATA" for row in rows_all if row["sm"] not in assignment.values()),
              f"{len(rows_all)} rows")

        if "half-populated" in assignment:
            sm = assignment["half-populated"]
            found = channel_findings(dataset, sm)
            stray = {**found["time"]["unexpected"], **found["energy"]["unexpected"]}
            populated = {mm for ch, (s, mm) in setup.channel_modules.items()
                         if s == sm and (ch in found["time"]["channels"] or ch in found["energy"]["channels"])}
            check(f"{system}: half-populated SM {sm} assesses only populated minimodules; stray hits listed",
                  populated == dataset.expected_mm[sm] and len(found["time"]["channels"]) == 64
                  and len(found["energy"]["channels"]) == 64 and len(stray) == 4
                  and all(n == 5000 for n in stray.values()) and found["state"] == "OK"
                  and found["time"]["median"] == BASE,
                  f"{len(stray)} stray channels, row {found['state']}")

    invalid = {"low_frac 1": dict(low_frac=1.0), "low_frac -0.1": dict(low_frac=-0.1),
               "high_frac 1": dict(high_frac=1.0), "min_median -1": dict(min_median=-1.0),
               "low_frac NaN": dict(low_frac=float("nan")), "min_events 0": dict(min_events=0),
               "min_events 2.5": dict(min_events=2.5)}
    accepted = []
    for label, change in invalid.items():
        try:
            FindingThresholds(**change).validate()
            accepted.append(label)
        except ValueError:
            pass
    check("invalid thresholds are rejected", not accepted, f"accepted {accepted}")


def geometry_checks(check):
    for system in ("IMAS", "CORNELL"):
        dataset, assignment, _, setup = build(system)
        problems = []
        for sm in sorted(dataset.expected_time):
            g = channel_geometry(dataset, sm)
            if [ch for ch, *_ in g["time"]] != sorted(dataset.expected_time[sm], key=lambda c: (
                    setup.channel_modules[c][1], setup.coordinates[c][0])):
                problems.append(f"SM {sm} time order")
            if sorted(ch for ch, *_ in g["energy"]) != sorted(dataset.expected_energy[sm]):
                problems.append(f"SM {sm} energy set")
            for ch, x, y0, y1, mm in g["time"]:
                bx0, bx1, by0, by1 = g["minimodules"][mm]["box"]
                if x != setup.coordinates[ch][0] or not (bx0 < x < bx1) or (y0, y1) != (by0, by1):
                    problems.append(f"SM {sm} ch {ch}")
            for ch, y, x0, x1, mm in g["energy"]:
                bx0, bx1, by0, by1 = g["minimodules"][mm]["box"]
                if y != setup.coordinates[ch][1] or not (by0 < y < by1) or (x0, x1) != (bx0, bx1):
                    problems.append(f"SM {sm} ch {ch}")
            populated = {mm for mm, info in g["minimodules"].items() if info["populated"]}
            if populated != dataset.expected_mm[sm] or len(g["minimodules"]) != 16:
                problems.append(f"SM {sm} minimodules")
        check(f"{system}: channel geometry puts every expected channel at its map X (time) / Y (energy) "
              "inside its minimodule box", not problems, ", ".join(problems[:5]))


def gui_checks(check):
    """Hidden-window Channel Status checks on the Cornell fixtures."""
    from tkinter import messagebox
    from unittest.mock import patch

    from matplotlib.collections import LineCollection

    from exe_programs.ldat_inspector_gui import LDATWorkbench, STATE_EDGES

    dataset, assignment, truth, setup = build("CORNELL")
    app = LDATWorkbench()
    app.withdraw()
    errors = []
    try:
        with patch.object(messagebox, "showerror", side_effect=lambda *a, **k: errors.append(a)):
            app.dataset = dataset
            app._update_after_processing()
            app.update()
            tree = app.channel_tree
            # Treeview hands numeric cells back as numbers; compare the displayed text
            rows = {int(iid): {"tags": tree.item(iid)["tags"], "values": [str(v) for v in tree.item(iid)["values"]]}
                    for iid in tree.get_children()}
            wrong = [case for case, sm in assignment.items()
                     if rows[sm]["tags"] != [EXPECTED_ROW[case]] or rows[sm]["values"][-1] != EXPECTED_ROW[case]]
            check("Channel Status: one row per mapped SM with the RAWInspector state tag and text",
                  len(rows) == 30 and not wrong
                  and all(str(tree.tag_configure(state, "background")) == colour
                          for state, colour in FINDING_COLOURS.items()), f"wrong: {wrong}")
            mixed = rows[assignment["mixed"]]["values"]
            half = rows[assignment["half-populated"]]["values"]
            few = rows[assignment["few sides"]]["values"]
            check("row columns: sides, minimodules, flag counts, medians, unexpected hits",
                  mixed[1] == "400" and mixed[2] == "1/16" and mixed[4] == "1 / 0 / 0" and mixed[5] == "100"
                  and mixed[7] == "0 / 1 / 1" and half[2] == "1/8" and half[9] == "4 ch"
                  and few[3] == "insufficient" and few[4] == "—",
                  f"mixed {mixed}; half {half[2]}, {half[9]}; few {few[3]}")
            summary = app.channel_summary.cget("text")
            check("summary shows totals, population wording and thresholds",
                  "30 SuperModules" in summary and "NOT OBSERVED" in summary and "coincidence ingest" in summary
                  and "0.15" in summary and "not a dead/hot hardware verdict" in summary, summary)

            before = rows[assignment["low"]]["tags"]
            app.finding_low.set("0.04")
            app._apply_thresholds()
            after = tree.item(str(assignment["low"]))["tags"]
            app.finding_high.set("0.9")
            app._apply_thresholds()
            rejected = len(errors) == 1 and app.thresholds.high_frac == 3.0 \
                and tree.item(str(assignment["low"]))["tags"] == ["OK"]
            errors.clear()
            check("threshold Apply updates rows; an invalid threshold is rejected and keeps the rows",
                  before == ["LOW"] and after == ["OK"] and rejected and "0.04" in app.channel_summary.cget("text"),
                  f"{before} -> {after}; rejected {rejected}")
            app.finding_low.set("0.15")
            app.finding_high.set("3")
            app._apply_thresholds()

            sm = assignment["mixed"]
            app.tabs.set("Channel Status")
            tree.selection_set(str(sm))
            app.update()
            drawn = app.channel_axes
            t_map, t_bars = drawn["time"]
            e_map, e_bars = drawn["energy"]

            def collections(axis):
                return [c for c in axis.collections if isinstance(c, LineCollection)]

            t_lines, e_lines = collections(t_map)[-1], collections(e_map)[-1]
            t_x = sorted(seg[0][0] for seg in t_lines.get_segments())
            flagged = collections(t_map)[0] if len(collections(t_map)) == 2 else None
            geometry = channel_geometry(dataset, sm)
            check("clicking a row draws that SM's channel map: time vertical at fine X, energy horizontal at fine Y",
                  app._channel_sm == sm and app.tabs.get() == "Channel Status"
                  and len(t_lines.get_segments()) == 128 and len(e_lines.get_segments()) == 128
                  and t_x == sorted(x for _, x, *_ in geometry["time"])
                  and all(seg[0][0] == seg[1][0] for seg in t_lines.get_segments())
                  and all(seg[0][1] == seg[1][1] for seg in e_lines.get_segments())
                  and flagged is not None and len(flagged.get_segments()) == 1,
                  f"{len(t_lines.get_segments())} time / {len(e_lines.get_segments())} energy segments")
            t_heights = [p.get_height() for p in t_bars.patches]
            e_colours = [p.get_facecolor() for p in e_bars.patches]
            from matplotlib.colors import to_rgba
            lines = {round(line.get_ydata()[0], 6) for line in t_bars.get_lines() if len(set(line.get_ydata())) == 1}
            first_e = [ch for ch, *_ in geometry["energy"]]
            want_e = [STATE_EDGES[dict(zip(channel_findings(dataset, sm)["energy"]["channels"],
                                           channel_findings(dataset, sm)["energy"]["states"]))[ch]]
                      for ch in first_e]
            check("bars per channel grouped by minimodule, coloured by state, with median and threshold lines",
                  len(t_heights) == 128 and sorted(t_heights) == sorted(truth[sm][0].values())
                  and [to_rgba(c) for c in want_e] == [tuple(c) for c in e_colours]
                  and {100.0, 15.0, 300.0} <= lines and "mM" in " ".join(t.get_text() for t in t_bars.texts),
                  f"threshold lines {sorted(lines)}")
            flags_text = "; ".join(app.channel_flags.get("1.0", "end").splitlines()[4:])  # after the header
            check("flagged channel list names each flagged channel", flags_text.count("NOT OBSERVED") == 1
                  and flags_text.count("LOW") == 1 and flags_text.count("HIGH") == 1, flags_text)
            half_sm = assignment["half-populated"]
            tree.selection_set(str(half_sm))
            app.update()
            hatched = [p for p in app.channel_axes["time"][0].patches if p.get_hatch()]
            check("half-populated SM: unpopulated minimodules hatched, stray hits listed",
                  len(hatched) == 8 and app.channel_flags.get("1.0", "end").count("unexpected") == 4)
            app._open_in_supermodule()
            app.update()
            check('"Open in SuperModule" selects and draws the SM in the SuperModule tab',
                  app.module_var.get() == f"SM {app._channel_sm}" and app.tabs.get() == "SuperModule"
                  and app.explorer_info.cget("text").startswith(f"SM {app._channel_sm} "))
            app._invalidate_data()
            check("invalidating inputs clears the Channel Status tab",
                  not tree.get_children() and app._channel_sm is None
                  and str(app.open_sm_button.cget("state")) == "disabled")
        check("no error dialogs during the Channel Status checks", not errors, f"{errors[:1]}")
    finally:
        destroy(app)


FINDINGS = {
    "defaults": "provisional defaults 0.15 / 3.0 / 20 hits / 100 sides",
    "colours": "every state, including NO DATA, has a RAWInspector colour",
    "imas-states": "IMAS: exact per-channel states and medians for 8 fixture SMs",
    "imas-priority": "IMAS: row priority NOT OBSERVED > HIGH > LOW > INSUFFICIENT > OK; NO DATA without sides",
    "imas-boundaries": "IMAS: boundary hits (15, 300) are OK; 14 is LOW, 301 HIGH, 0 NOT OBSERVED",
    "imas-insufficient":
        "IMAS: low statistics read insufficient events, not not-observed (FR-7), with the reason",
    "imas-threshold-changes": "IMAS: threshold changes move the row states as expected",
    "imas-system-rows": "IMAS: system findings give one row per mapped SM in order",
    "cornell-states": "CORNELL: exact per-channel states and medians for 9 fixture SMs",
    "cornell-priority":
        "CORNELL: row priority NOT OBSERVED > HIGH > LOW > INSUFFICIENT > OK; NO DATA without sides",
    "cornell-boundaries": "CORNELL: boundary hits (15, 300) are OK; 14 is LOW, 301 HIGH, 0 NOT OBSERVED",
    "cornell-insufficient":
        "CORNELL: low statistics read insufficient events, not not-observed (FR-7), with the reason",
    "cornell-threshold-changes": "CORNELL: threshold changes move the row states as expected",
    "cornell-system-rows": "CORNELL: system findings give one row per mapped SM in order",
    "cornell-half-populated":
        "CORNELL: half-populated SM 2 assesses only populated minimodules; stray hits listed",
    "invalid-thresholds": "invalid thresholds are rejected",
}
GEOMETRY = {
    "imas":
        "IMAS: channel geometry puts every expected channel at its map X (time) / Y (energy) inside its "
        "minimodule box",
    "cornell":
        "CORNELL: channel geometry puts every expected channel at its map X (time) / Y (energy) inside its "
        "minimodule box",
}
STATUS_GUI = {
    "rows": "Channel Status: one row per mapped SM with the RAWInspector state tag and text",
    "row-columns": "row columns: sides, minimodules, flag counts, medians, unexpected hits",
    "summary": "summary shows totals, population wording and thresholds",
    "threshold-apply": "threshold Apply updates rows; an invalid threshold is rejected and keeps the rows",
    "channel-map":
        "clicking a row draws that SM's channel map: time vertical at fine X, energy horizontal at fine Y",
    "bars": "bars per channel grouped by minimodule, coloured by state, with median and threshold lines",
    "flagged-list": "flagged channel list names each flagged channel",
    "half-populated": "half-populated SM: unpopulated minimodules hatched, stray hits listed",
    "open-in-supermodule": '"Open in SuperModule" selects and draws the SM in the SuperModule tab',
    "invalidate": "invalidating inputs clears the Channel Status tab",
    "no-error-dialogs": "no error dialogs during the Channel Status checks",
}


@pytest.fixture(scope="module")
def findings():
    return Checks(FINDINGS).run(findings_checks)


@fr("002-FR-5", "002-FR-7")
@pytest.mark.parametrize("check", FINDINGS)
def test_findings(findings, check):
    findings.verdict(check)


@pytest.fixture(scope="module")
def geometry():
    return Checks(GEOMETRY).run(geometry_checks)


@fr("002-FR-6")
@pytest.mark.parametrize("check", GEOMETRY)
def test_geometry(geometry, check):
    geometry.verdict(check)


@pytest.fixture(scope="module")
def status_gui():
    require_display()
    return Checks(STATUS_GUI).run(gui_checks)


@pytest.mark.gui
@fr("002-FR-5", "002-FR-6", "002-FR-7")
@pytest.mark.parametrize("check", STATUS_GUI)
def test_status_gui(status_gui, check):
    status_gui.verdict(check)
