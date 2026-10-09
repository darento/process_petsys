"""Cornell COG/DOI limits files in LDATInspector (spec 002 T14, T15; spec 007 T22).

T14: slab X against get_slab_cornell, clipped decompressed Y and decompressed DOI on known sides,
per-reason exclusions, malformed files and a file swap without the reader. T15: the limits pickers,
flood-view and DOI-unit selectors in the GUI (unavailable states, cut reset, exclusions on the plots, no
LDAT reread), the mm DOI dataset view from a background job, and PDF provenance. The owner-file checks
are ``real_data`` (list-mode copies) or retired (repo-root copies); spec 007 Clarify, T21/T22.

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
from src.ldat_inspector.engine import (FileResult, Selection, SideTable, Settings, decompressed_doi,
                                       flood_counts, load_limits, load_setup, merge_results, slab_totals,
                                       slab_view, slab_x_edges, unresolved_slab_pairs)
from src.mapping_generator import ChannelType

pytestmark = pytest.mark.fr("002-FR-15")
fr = pytest.mark.fr


def limits_dataset(sides):
    """One Cornell SM-0 table from [(calibration key, y, doi, mm)]; sides keep their order."""
    n = len(sides)
    column = lambda i, dtype=float: np.array([s[i] for s in sides], dtype=dtype)  # noqa: E731
    table = SideTable.from_pairs(0, raw_energy=np.full(n, 100.0), calibration_key=column(0, np.int64),
                                 x=np.zeros(n), y=column(1), doi=column(2), timestamp=np.arange(n, dtype=np.int64),
                                 sm=np.zeros(n, np.int16), mm=column(3, np.int8), random_slab=np.zeros(n, bool))
    settings = Settings(str(CONFIGS["CORNELL"]), "", "CORNELL", max_pairs=None, calibrated=False)
    return merge_results(settings, [FileResult(0, "synthetic", n // 2, n // 2, table=table)])


def limits_checks(check, tmp):
    import random

    import src.ldat_inspector.fastread as fastread
    import src.ldat_inspector.engine as engine
    from src.utils import get_slab_cornell

    setup = load_setup(Settings(str(CONFIGS["CORNELL"]), "", "CORNELL", max_pairs=None, calibrated=False))
    row1 = time_channels_by_position(setup, 0, 4)  # minimodule 4: row 1 (Y offset 2 x 25.6)
    row3 = time_channels_by_position(setup, 0, 13)  # minimodule 13: row 3 (Y offset 0)

    # Slab X against src/utils.py:get_slab_cornell on hand-built hits: every position alone
    # (edges resolved, inner positions a seeded coin flip), with p - 1 and with p + 1.
    random.seed(7)
    cases, mismatches = [], []
    for p, ch in sorted(row1.items()):
        for neighbour in (None, p - 1, p + 1):
            if neighbour is not None and neighbour not in row1:
                continue
            hits = [[0, 10.0, ch]] + ([[0, 5.0, row1[neighbour]]] if neighbour is not None else [])
            slab, flag, x = get_slab_cornell(hits, setup.channel_types, setup.coordinates)
            cases.append((p, neighbour, slab, flag, x, (ch << 5) | slab))
    dataset = limits_dataset([(key, 50.0, 3.0, 4) for *_, key in cases])
    cog_file = tmp / "cog.txt"
    cog_file.write_text("".join(f"({key >> 5}, {key & 31})\t40.0\t56.0\n" for key in sorted({c[-1] for c in cases})),
                        encoding="utf-8")
    view = slab_view(dataset, dataset.modules[0], load_limits(str(cog_file)))
    for (p, neighbour, slab, flag, x, _), got in zip(cases, view["x"]):
        want = setup.coordinates[row1[p]][0] + (0.8 if slab % 2 else -0.8)  # written out: B1 convention
        if got != x or abs(got - want) > 1e-12:
            mismatches.append(f"p{p}/{neighbour}: {got} vs {x}")
    flags = Counter(flag for _, _, _, flag, _, _ in cases)
    edges = slab_x_edges(dataset, 0)
    columns = np.digitize(view["x"], edges)
    by_slab = {}
    for (*_, key), column in zip(cases, columns):
        by_slab.setdefault(key, set()).add(int(column))
    check("slab-view columns: 64 per full Cornell SM (one per mapped slab X); each slab in its own column",
          len(edges) == 65 and all(len(c) == 1 for c in by_slab.values())
          and len({next(iter(c)) for c in by_slab.values()}) == len(by_slab) == 16
          and bool(np.all((view["x"] > edges[0]) & (view["x"] < edges[-1]))), f"{len(edges) - 1} columns")
    check("slab X equals get_slab_cornell's X for every position alone and with each neighbour (B1 convention)",
          not mismatches and len(cases) == 22 and flags == Counter({3: 12, 1: 6, 0: 4}),
          f"{len(cases)} cases, flags {dict(flags)}; {mismatches[:3]}")

    # Decompressed Y and DOI on known sides.
    a, b, c, d = row1[2], row1[5], row1[6], row3[1]
    cog_text = (f"({a}, 4)\t40.0\t56.0\n({a}, 5)\t41.0\t57.0\n({b}, 10)\t60.0\t60.0\n"
                f"({d}, 3)\t10.0\t20.0\n")
    doi_text = f"({a}, 4)\t2.0\t7.0\n({a}, 5)\t2.5\t7.5\n({b}, 10)\t3.0\t3.0\n({d}, 3)\t1.0\t6.0\n"
    sides = [  # (key, y, doi, mm), expected (Y, DOI mm); None = excluded
        (((a << 5) | 4, 44.0, 4.5, 4), (6.4 + 51.2, 10.0)),
        (((a << 5) | 4, 38.0, 7.0, 4), (0.0 + 51.2, 0.0)),  # Y clipped low; DOI at 0 kept
        (((a << 5) | 4, 60.0, 2.0, 4), (25.6 + 51.2, 20.0)),  # Y clipped high; DOI at 20 kept
        (((a << 5) | 5, 49.0, 7.6, 4), (12.8 + 51.2, None)),  # DOI -0.4 mm: out of range
        (((a << 5) | 5, 57.0, 1.9, 4), (25.6 + 51.2, None)),  # Y at the limit (not clipped); DOI 22.4 mm
        (((c << 5) | 12, 50.0, 3.0, 4), (None, None)),  # no entry in either file
        (((b << 5) | 10, 62.0, 3.5, 4), (None, None)),  # left == right in both files (would divide by 0)
        (((a << 5) | 6, 50.0, 3.0, 4), (None, None)),  # channel present, slab not
        (((d << 5) | 3, 12.5, 3.5, 13), (6.4, 10.0)),  # row 3: no offset
        (((d << 5) | 3, 20.0, 1.0, 13), (25.6, 20.0)),
    ]
    dataset = limits_dataset([s for s, _ in sides])
    data = dataset.modules[0]
    (tmp / "cog_a.txt").write_text(cog_text, encoding="utf-8")
    (tmp / "doi_a.txt").write_text(doi_text, encoding="utf-8")
    cog, doi = load_limits(str(tmp / "cog_a.txt")), load_limits(str(tmp / "doi_a.txt"))
    view, depth = slab_view(dataset, data, cog), decompressed_doi(dataset, data, doi)

    def same(got, want):
        return all((w is None and np.isnan(g)) or (w is not None and abs(g - w) < 1e-9) for g, w in zip(got, want))

    check("decompressed Y: (y − left)·25.6/(right − left) clipped to 0–25.6 plus (3 − mm // 4)·25.6",
          same(view["y"], [w[0] for _, w in sides]) and view["clipped"] == 2,
          f"{np.round(view['y'], 4).tolist()}, clipped {view['clipped']}")
    check("decompressed DOI: (doi − right)·20/(left − right); 0 and 20 mm kept, outside is NaN",
          same(depth["doi_mm"], [w[1] for _, w in sides]), f"{np.round(depth['doi_mm'], 4).tolist()}")
    check("exclusions counted per reason, never replaced by a value",
          view["excluded"] == Counter({"missing key": 2, "invalid limits": 1})
          and depth["excluded"] == Counter({"missing key": 2, "invalid limits": 1, "out of range": 2})
          and int(np.isnan(view["y"]).sum()) == sum(view["excluded"].values())
          and int(np.isnan(depth["doi_mm"]).sum()) == sum(depth["excluded"].values())
          and cog.invalid == doi.invalid == 1,
          f"Y {dict(view['excluded'])}; DOI {dict(depth['excluded'])}")

    # Swapping a file recomputes from stored columns only: any reader call fails.
    stored = {name: values.copy() for name, values in dataset.table.columns.items()}
    (tmp / "cog_b.txt").write_text(cog_text.replace(f"({a}, 4)\t40.0\t56.0", f"({a}, 4)\t42.0\t58.0"),
                                   encoding="utf-8")
    calls = []

    def reader(*args, **kwargs):
        calls.append(args)
        raise AssertionError("reader called")

    patched = {(engine, "iter_pairs"), (engine, "process_file"), (engine, "process_file_reference"),
               (fastread, "process_file_fast")}
    originals = {(module, name): getattr(module, name) for module, name in patched}
    try:
        for module, name in patched:
            setattr(module, name, reader)
        swapped = slab_view(dataset, data, load_limits(str(tmp / "cog_b.txt")))
        back = slab_view(dataset, data, cog)
    finally:
        for (module, name), value in originals.items():
            setattr(module, name, value)
    check("swapping the COG file recomputes without the reader; the stored columns are unchanged",
          not calls and abs(swapped["y"][0] - (2.0 * 25.6 / 16 + 51.2)) < 1e-9
          and np.array_equal(back["y"], view["y"], equal_nan=True)
          and np.array_equal(swapped["y"][3:], view["y"][3:], equal_nan=True)
          and all(np.array_equal(stored[k], v) for k, v in dataset.table.columns.items()),
          f"row 0: {view['y'][0]:.3f} -> {swapped['y'][0]:.3f} mm; reader calls {len(calls)}")

    imas = replace(dataset, settings=replace(dataset.settings, system="IMAS"))
    refused = []
    for label, call in (("slab view", lambda: slab_view(imas, data, cog)),
                        ("DOI", lambda: decompressed_doi(imas, data, doi))):
        try:
            call()
        except ValueError:
            refused.append(label)
    unresolved = replace(dataset, files=[replace(dataset.files[0], errors=Counter({"unresolved Cornell slab": 9,
                                                                                   "min channels": 4}))])
    check("IMAS datasets are refused; unresolved-slab pairs come from the ingest counters",
          refused == ["slab view", "DOI"] and unresolved_slab_pairs(unresolved) == 9
          and unresolved_slab_pairs(dataset) == 0, f"refused {refused}")

    malformed = {
        "header line": "time_ch\tslab\tleft\tright\n(256, 0)\t1\t2\n",
        "missing column": "(256, 0)\t1.0\n",
        "not a number": "(256, 0)\t1.0\tabc\n",
        "NaN limit": "(256, 0)\tnan\t2.0\n",
        "infinite limit": "(256, 0)\t1.0\tinf\n",
        "slab 16": "(256, 16)\t1.0\t2.0\n",
        "negative channel": "(-1, 0)\t1.0\t2.0\n",
        "spaces not tabs": "(256, 0) 1.0 2.0\n",
        "duplicate key": "(256, 0)\t1.0\t2.0\n(256, 0)\t1.0\t2.0\n",
        "bad line after good ones": "(256, 0)\t1.0\t2.0\n(256, 1)\t1.0\t2.0\n(256 1)\t1.0\t2.0\n",
        "empty": "",
        "blank lines only": "\n\n",
    }
    accepted = []
    for label, text in malformed.items():
        path = tmp / "bad.txt"
        path.write_text(text, encoding="utf-8")
        try:
            load_limits(str(path))
            accepted.append(label)
        except ValueError:
            pass
    try:
        load_limits(str(tmp / "absent.txt"))
        accepted.append("absent file")
    except FileNotFoundError:
        pass
    (tmp / "crlf.txt").write_bytes(b"(256, 1)\t1.0\t2.0\r\n\r\n(256, 0)\t3.0\t4.0\r\n")
    crlf = load_limits(str(tmp / "crlf.txt"))
    check(f"{len(malformed) + 1} malformed or absent files are rejected whole; CRLF and blank lines load",
          not accepted and len(crlf) == 2 and crlf.lookup([256 << 5])[0][0] == 3.0, f"accepted {accepted}")


def slab_fixture(tmp):
    """Cornell SM 0 sides on minimodules 4 (row 1) and 13 (row 3) plus limits files in ``tmp``.

    20 sides per (time channel, own slab) key inside its COG and DOI limits; then 5 sides on
    a key missing from both files, 3 sides above their key's COG right limit (clipped) and
    4 sides with DOI ratio 8 (-4 mm: out of range). Returns (dataset, {name: path}).
    """
    setup = load_setup(Settings(str(CONFIGS["CORNELL"]), "", "CORNELL", max_pairs=None, calibrated=False))
    rng = np.random.default_rng(3)
    sides, cog, doi = [], [], []
    for mm in (4, 13):
        row_low = (3 - mm // 4) * 25.6
        for p, ch in sorted(time_channels_by_position(setup, 0, mm).items()):
            for slab in (2 * p, 2 * p + 1):
                left, right = row_low + 4.0, row_low + 21.6
                cog.append(f"({ch}, {slab})\t{left:.1f}\t{right:.1f}\n")
                doi.append(f"({ch}, {slab})\t2.0\t7.0\n")
                for y, d in zip(rng.uniform(left, right, 20), rng.uniform(2.0, 7.0, 20)):
                    sides.append(((ch << 5) | slab, setup.coordinates[ch][0], y, d, mm))
    first = sides[0]
    other = time_channels_by_position(setup, 0, 5)[0]  # minimodule 5: in neither file
    sides += [((other << 5) | 0, 50.0, 40.0, 4.0, 5)] * 5
    sides += [(first[0], first[1], 51.2 + 21.6 + 2.0, 4.0, first[4])] * 3  # 2 mm above its right limit
    sides += [(first[0], first[1], first[2], 8.0, first[4])] * 4
    n = len(sides)
    column = lambda i, dtype=float: np.array([s[i] for s in sides], dtype=dtype)  # noqa: E731
    table = SideTable.from_pairs(0, raw_energy=np.full(n, 100.0), calibration_key=column(0, np.int64),
                                 x=column(1), y=column(2), doi=column(3), timestamp=np.arange(n, dtype=np.int64),
                                 sm=np.zeros(n, np.int16), mm=column(4, np.int8), random_slab=np.zeros(n, bool))
    settings = Settings(str(CONFIGS["CORNELL"]), "", "CORNELL", max_pairs=None, calibrated=False)
    result = FileResult(0, "synthetic", n // 2, n // 2, table=table, errors=Counter({"unresolved Cornell slab": 7}))
    dataset = merge_results(settings, [result])
    first_key = f"({first[0] >> 5}, {first[0] & 31})\t"
    shifted = [line if not line.startswith(first_key) else
               first_key + "\t".join(f"{float(v) + 2.0:.1f}" for v in line.strip().split("\t")[1:]) + "\n"
               for line in cog]
    paths = {"cog": tmp / "cog_fixture.txt", "cog_b": tmp / "cog_fixture_b.txt", "doi": tmp / "doi_fixture.txt",
             "bad": tmp / "bad_limits.txt"}
    paths["cog"].write_text("".join(cog), encoding="utf-8")
    paths["cog_b"].write_text("".join(shifted), encoding="utf-8")
    paths["doi"].write_text("".join(doi), encoding="utf-8")
    paths["bad"].write_text("(256, 0)\t1.0\n", encoding="utf-8")
    return dataset, paths


def limits_gui_checks(check, tmp):
    import threading
    import time
    from tkinter import messagebox
    from unittest.mock import patch

    from matplotlib.colors import to_hex

    import exe_programs.ldat_inspector_gui as gui
    import src.ldat_inspector.engine as engine

    dataset, paths = slab_fixture(tmp)
    cog, cog_b, doi = (load_limits(str(paths[k])) for k in ("cog", "cog_b", "doi"))
    app = gui.LDATWorkbench()
    app.withdraw()
    errors, reads, answers, threads = [], [], [], set()

    def pump(until, timeout=20):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            app.update()
            if until():
                return True
            time.sleep(0.01)
        return False

    def pick(path):
        with patch.object(gui.filedialog, "askopenfilename", return_value=str(path)):
            app._select_limits("doi" if "doi" in Path(path).name else "cog")

    def reader(*args, **kwargs):
        reads.append(args)
        raise AssertionError("LDAT reread")

    real_view = gui.apply_doi_view

    def recorded(*args, **kwargs):
        threads.add(threading.get_ident())
        return real_view(*args, **kwargs)

    def flood_total():
        mesh = app.flood_ax.collections[0].get_array() if app.flood_ax.collections else np.ma.zeros(0)
        return int(np.ma.filled(mesh, 0).sum())

    def states():
        return ([c.cget("state") for c in app._flood_combos], app.flood_note.cget("text"),
                app.doi_unit_combo.cget("state"), app.doi_unit_note.cget("text"))

    try:
        with patch.object(messagebox, "showerror", side_effect=lambda *a, **k: errors.append(a)), \
                patch.object(messagebox, "askyesno", side_effect=lambda *a, **k: answers.append(a) or True), \
                patch.object(gui, "process_file", reader), patch.object(engine, "iter_pairs", reader), \
                patch.object(gui, "apply_doi_view", recorded):
            imas = states()
            app.system.set("CORNELL")
            no_file = states()
            check("selectors unavailable: IMAS reads 'Cornell only', Cornell without files 'no … limits file'",
                  imas == (["disabled"] * 2, "n/a: Cornell only", "disabled", "n/a: Cornell only")
                  and no_file == (["disabled"] * 2, "n/a: no COG limits file", "disabled", "n/a: no DOI limits file")
                  and app.flood_view.get() == gui.FLOOD_VIEWS[0] and app.doi_unit.get() == gui.DOI_UNITS[0],
                  f"{imas}; {no_file}")
            app.tabs.set(gui.SM_TAB)
            app.dataset = dataset
            app._update_after_processing()
            selection = app._selection()
            data = dataset.modules[0]
            chosen = selection.mask(data)
            pick(paths["cog"])
            check("COG picker: file loaded, short name shown, flood selector available (SM tab and overview)",
                  app.limits["cog"] is not None and len(app.limits["cog"]) == 32
                  and app.limit_labels["cog"].cget("text") == "cog_fixture.txt"
                  and states()[:2] == (["readonly"] * 2, ""), f"{states()}")
            cog_before = flood_total()
            app.flood_view.set(gui.FLOOD_VIEWS[1])
            app._flood_view_changed()
            view = slab_view(dataset, data, cog, mask=chosen)
            shown = int((chosen & np.isfinite(view["y"])).sum())
            title = " / ".join([app.flood_ax.get_title(), *(t.get_text() for t in app.flood_ax.texts)])
            check("slab flood: the engine's slab X / decompressed Y, 0–102.4 mm, exclusions and clipping on the plot",
                  flood_total() == shown == len(data) - 5 and cog_before == len(data)
                  and f"{shown:,} of {len(data):,} sides; 1 column/slab" in title
                  and "excluded: missing key 5" in title and "clipped to row 3" in title
                  and "legacy rule: 7 unresolved pairs rejected" in title
                  and app.flood_ax.get_xlim() == (0.0, 102.4) and "Slab X" in app.flood_ax.get_xlabel(),
                  title.replace("\n", " / "))
            check("slab view: no COG ROI rectangle or region selector (the ROI stays on COG/RTP coordinates)",
                  not app.flood_ax.patches and app._selector is None
                  and any("COG/RTP coordinates" in t.get_text() for t in app.flood_ax.texts))
            app.overview_mode.set(gui.FLOOD_MODE)
            app._show_tab("System Overview")
            suptitle = app.overview_fig._suptitle.get_text() if app.overview_fig._suptitle else ""
            check("System Overview flood mode follows the selector, with the same exclusions",
                  suptitle.startswith("Slab flood maps (decompressed Y, cog_fixture.txt)") and "missing key 5" in suptitle
                  and "legacy rule: 7 unresolved pairs rejected" in suptitle
                  and f"{shown:,} sides passing the cuts" in suptitle, suptitle)
            app._show_tab(gui.SM_TAB)
            kept = app.limits["cog"]
            pick(paths["bad"])
            check("a malformed limits file is rejected with a dialog; the previous file is kept",
                  len(errors) == 1 and "bad_limits.txt" in str(errors[0]) and app.limits["cog"] is kept, f"{errors}")
            errors.clear()
            pick(paths["cog_b"])
            swapped = slab_view(dataset, data, cog_b, mask=chosen)
            mesh = np.ma.filled(app.flood_ax.collections[0].get_array(), 0)
            want, *_ = flood_counts(swapped["x"][chosen & np.isfinite(swapped["y"])],
                                    swapped["y"][chosen & np.isfinite(swapped["y"])], int(app.bins.get()), 102.4,
                                    slab_x_edges(dataset, 0))
            check("swapping the COG file redraws from the new limits without rereading LDAT",
                  np.array_equal(mesh, np.ma.filled(want, 0)) and not np.array_equal(swapped["y"], view["y"], equal_nan=True)
                  and not reads and app.dataset is dataset, f"reads {len(reads)}")
            app._limits_info("cog")
            check("removing the COG file (label click, confirm) returns to COG / RTP and restores the ROI tools",
                  app.limits["cog"] is None and app.flood_view.get() == gui.FLOOD_VIEWS[0]
                  and states()[:2] == (["disabled"] * 2, "n/a: no COG limits file") and app._selector is not None
                  and any(to_hex(p.get_edgecolor()) == "#4ec06c" for p in app.flood_ax.patches)
                  and flood_total() == len(data)
                  and "cog_fixture_b.txt" in answers[-1][1], f"{states()}")

            pick(paths["doi"])
            check("DOI picker: file loaded, unit selector available, ratio still the view",
                  app.limits["doi"] is not None and states()[2:] == ("readonly", "") and not app.dataset.doi_mm)
            app.doi_unit.set(gui.DOI_UNITS[1])
            app._doi_unit_changed()
            cut = (app.doi_low.get(), app.doi_high.get())
            ready = pump(lambda: app.dataset is not dataset and not app._busy)
            mm = app.dataset
            want = decompressed_doi(dataset, data, doi)
            check("mm DOI: the cut resets to 0–20 mm and a background job applies the engine's decompressed view",
                  cut == ("0", "20") and ready and mm.doi_mm and threads and threading.get_ident() not in threads
                  and np.array_equal(mm.table.doi, want["doi_mm"], equal_nan=True)
                  and mm.doi_excluded == Counter({"missing key": 5, "out of range": 4})
                  and "decompressed mm" in app.doi_group_title.cget("text"), f"cut {cut}")
            base = app._selection().mask(mm.modules[0], doi=False)
            doi_title = " / ".join([app.doi_ax.get_title(), *(t.get_text() for t in app.doi_ax.texts)])
            heights = sum(p.get_height() for p in app.doi_ax.patches)
            check("mm DOI plot: 0–20 mm axis, mapping label and file, excluded counts per reason",
                  "mm-equivalent" in app.doi_ax.get_xlabel() and "not a validated depth" in doi_title
                  and "doi_fixture.txt" in doi_title and app.doi_ax.get_xlim() == (0.0, 20.0)
                  and "9 excluded: missing key 5, out of range 4" in doi_title
                  and heights == int((base & np.isfinite(mm.modules[0].doi)).sum()) == len(data) - 9,
                  doi_title.replace("\n", " / "))
            selected = int(app._selection().mask(mm.modules[0]).sum())
            check("the mm cut applies to every view: excluded sides fail it (selected count)",
                  f"{selected:,} after paired energy + DOI + ROI" in app.explorer_info.cget("text")
                  and selected == len(data) - 9, app.explorer_info.cget("text"))
            app.doi_high.set("10")
            app._refresh_all()
            half = int(app._selection().mask(mm.modules[0]).sum())
            check("an mm DOI cut (0–10 mm) keeps exactly the sides with decompressed DOI in [0, 10]",
                  half == int(((want["doi_mm"] >= 0) & (want["doi_mm"] <= 10)).sum()) and 0 < half < selected,
                  f"{half} sides")
            app.doi_unit.set(gui.DOI_UNITS[0])
            app._doi_unit_changed()
            cut = (app.doi_low.get(), app.doi_high.get())
            pump(lambda: not app.dataset.doi_mm and not app._busy)
            check("back to Ratio: cut 0–15 and the stored ratio is the view again (no reread)",
                  cut == ("0", "15") and not app.dataset.doi_mm
                  and app.dataset.table.doi is app.dataset.table.columns["doi"] and not reads, f"cut {cut}")
            app.doi_unit.set(gui.DOI_UNITS[1])
            app._doi_unit_changed()
            pump(lambda: app.dataset.doi_mm and not app._busy)
            app._limits_info("doi")
            fallback = (app.doi_unit.get(), app.doi_low.get(), app.doi_high.get())
            pump(lambda: not app.dataset.doi_mm and not app._busy)
            check("removing the DOI file in mm mode falls back to Ratio (cut 0–15) and the ratio view",
                  fallback == (gui.DOI_UNITS[0], "0", "15") and not app.dataset.doi_mm
                  and states()[2:] == ("disabled", "n/a: no DOI limits file"), f"{fallback}")

            pick(paths["cog"])
            pick(paths["doi"])
            app.flood_view.set(gui.FLOOD_VIEWS[1])
            app._flood_view_changed()
            app.doi_unit.set(gui.DOI_UNITS[1])
            app._doi_unit_changed()
            pump(lambda: app.dataset.doi_mm and not app._busy)
            import re
            from pypdf import PdfReader
            pdf = tmp / "gui_limits.pdf"
            with patch.object(gui.filedialog, "asksaveasfilename", return_value=str(pdf)):
                app._report(0)
            done = pump(lambda: not app._report_busy, timeout=60)
            text = re.sub(r"\s+", " ", " ".join(p.extract_text() for p in PdfReader(str(pdf)).pages)) if done else ""
            check("the GUI's SuperModule report uses the loaded limits files and the active views",
                  done and "Flood view: slab-assigned" in text and "DOI view: decompressed mm-equivalent" in text
                  and "cog_fixture.txt" in text.replace(" ", "") and "doi_fixture.txt" in text.replace(" ", ""),
                  f"report written {done}")
            imas_dataset = replace(dataset, settings=replace(dataset.settings, system="IMAS"))
            app.dataset = imas_dataset
            app._update_after_processing()
            check("an IMAS dataset makes both selectors unavailable even with limits files loaded",
                  states() == (["disabled"] * 2, "n/a: Cornell only", "disabled", "n/a: Cornell only")
                  and not app._slab_active(), f"{states()}")
        check("no unexpected error dialogs and no LDAT reads during the limits checks", not errors and not reads,
              f"{errors[:1]} reads {len(reads)}")
    finally:
        destroy(app)


def limits_report_checks(check, tmp):
    import re

    from pypdf import PdfReader

    from src.ldat_inspector.engine import apply_doi_view
    from src.ldat_inspector.report import write_report

    dataset, paths = slab_fixture(tmp)
    cog, doi = load_limits(str(paths["cog"])), load_limits(str(paths["doi"]))
    mm = apply_doi_view(dataset, doi)
    selection = Selection(0.0, 300.0, 0.0, 20.0, 0.0, 102.0, 0.0, 102.0)
    pdf = tmp / "limits.pdf"
    write_report(str(pdf), mm, selection, sm=0, cog_limits=cog, doi_limits=doi, slab_flood=True)
    text = re.sub(r"\s+", " ", " ".join(page.extract_text() for page in PdfReader(str(pdf)).pages))
    flat = re.sub(r"\s+", "", text)
    totals = slab_totals(mm, cog, selection, {0})
    check("PDF provenance lists both limits paths, the active views and their exclusion counts",
          re.sub(r"\s+", "", str(paths["cog"])) in flat and re.sub(r"\s+", "", str(paths["doi"])) in flat
          and "Flood view: slab-assigned" in text and "excluded: none; clipped to the row: 3" in text
          and "unresolved-slab pairs rejected at ingest (all files): 7" in text
          and "DOI view: decompressed mm-equivalent" in text and "missing key 5, out of range 4" in text
          and "DOI mm 0..20" in text and "not a validated depth" in text
          and totals["excluded"] == Counter() and totals["clipped"] == 3,  # missing-key sides fail the mm DOI cut
          f"{totals}")
    check("PDF SM page draws the slab flood and the mm DOI with their labels",
          "Selected slab flood (decompressed Y)" in text and "Decompressed DOI (mm-equivalent" in text
          and "Slab X (mm)" in text)
    plain = tmp / "plain.pdf"
    write_report(str(plain), dataset, selection, sm=0, cog_limits=cog)
    text = re.sub(r"\s+", " ", " ".join(page.extract_text() for page in PdfReader(str(plain)).pages))
    check("a PDF in the default views names the loaded COG file, 'DOI limits: none', COG/RTP and ratio",
          "Flood view: COG/RTP centroid" in text and "DOI limits: none" in text and "DOI view: light-sharing ratio" in text
          and "cog_fixture.txt" in re.sub(r"\s+", "", text))
    refused = False
    try:
        write_report(str(tmp / "refused.pdf"), dataset, selection, sm=0, slab_flood=True)
    except ValueError:
        refused = True
    check("a slab-flood report without a COG file is refused", refused)


LIMITS = {
    "slab-columns":
        "slab-view columns: 64 per full Cornell SM (one per mapped slab X); each slab in its own column",
    "slab-x":
        "slab X equals get_slab_cornell's X for every position alone and with each neighbour (B1 convention)",
    "decompressed-y":
        "decompressed Y: (y − left)·25.6/(right − left) clipped to 0–25.6 plus (3 − mm // 4)·25.6",
    "decompressed-doi": "decompressed DOI: (doi − right)·20/(left − right); 0 and 20 mm kept, outside is NaN",
    "exclusions": "exclusions counted per reason, never replaced by a value",
    "cog-swap": "swapping the COG file recomputes without the reader; the stored columns are unchanged",
    "imas-refused": "IMAS datasets are refused; unresolved-slab pairs come from the ingest counters",
    "malformed-files": "13 malformed or absent files are rejected whole; CRLF and blank lines load",
}
LIMITS_GUI = {
    "selectors-unavailable":
        "selectors unavailable: IMAS reads 'Cornell only', Cornell without files 'no … limits file'",
    "cog-picker": "COG picker: file loaded, short name shown, flood selector available (SM tab and overview)",
    "slab-flood":
        "slab flood: the engine's slab X / decompressed Y, 0–102.4 mm, exclusions and clipping on the plot",
    "slab-view-no-roi":
        "slab view: no COG ROI rectangle or region selector (the ROI stays on COG/RTP coordinates)",
    "overview-flood": "System Overview flood mode follows the selector, with the same exclusions",
    "malformed-rejected": "a malformed limits file is rejected with a dialog; the previous file is kept",
    "cog-swap": "swapping the COG file redraws from the new limits without rereading LDAT",
    "cog-removed":
        "removing the COG file (label click, confirm) returns to COG / RTP and restores the ROI tools",
    "doi-picker": "DOI picker: file loaded, unit selector available, ratio still the view",
    "mm-doi": "mm DOI: the cut resets to 0–20 mm and a background job applies the engine's decompressed view",
    "mm-doi-plot": "mm DOI plot: 0–20 mm axis, mapping label and file, excluded counts per reason",
    "mm-cut-selection": "the mm cut applies to every view: excluded sides fail it (selected count)",
    "mm-cut-0-10": "an mm DOI cut (0–10 mm) keeps exactly the sides with decompressed DOI in [0, 10]",
    "back-to-ratio": "back to Ratio: cut 0–15 and the stored ratio is the view again (no reread)",
    "doi-removed": "removing the DOI file in mm mode falls back to Ratio (cut 0–15) and the ratio view",
    "gui-report": "the GUI's SuperModule report uses the loaded limits files and the active views",
    "imas-unavailable": "an IMAS dataset makes both selectors unavailable even with limits files loaded",
    "no-errors-no-reads": "no unexpected error dialogs and no LDAT reads during the limits checks",
}
LIMITS_REPORT = {
    "provenance": "PDF provenance lists both limits paths, the active views and their exclusion counts",
    "sm-page": "PDF SM page draws the slab flood and the mm DOI with their labels",
    "default-views":
        "a PDF in the default views names the loaded COG file, 'DOI limits: none', COG/RTP and ratio",
    "slab-flood-without-cog": "a slab-flood report without a COG file is refused",
}


@pytest.fixture(scope="module")
def limits():
    with tempfile.TemporaryDirectory(prefix="ldat_") as temporary:
        yield Checks(LIMITS).run(limits_checks, Path(temporary))


@fr("002-FR-16", "002-FR-17", "002-FR-18")
@pytest.mark.parametrize("check", LIMITS)
def test_limits(limits, check):
    limits.verdict(check)


@pytest.fixture(scope="module")
def limits_gui():
    require_display()
    with tempfile.TemporaryDirectory(prefix="ldat_") as temporary:
        yield Checks(LIMITS_GUI).run(limits_gui_checks, Path(temporary))


@pytest.mark.gui
@pytest.mark.slow  # the fixture runs the section: 5.0 s serial
@fr("002-FR-14", "002-FR-16", "002-FR-17", "002-FR-18")
@pytest.mark.parametrize("check", LIMITS_GUI)
def test_limits_gui(limits_gui, check):
    limits_gui.verdict(check)


@pytest.fixture(scope="module")
def limits_report():
    with tempfile.TemporaryDirectory(prefix="ldat_") as temporary:
        yield Checks(LIMITS_REPORT).run(limits_report_checks, Path(temporary))


@fr("002-FR-14", "002-FR-16", "002-FR-17", "002-FR-18")
@pytest.mark.parametrize("check", LIMITS_REPORT)
def test_limits_report(limits_report, check):
    limits_report.verdict(check)


# --- owner limits files --------------------------------------------------------------------------------------
# Data: PETSYS_DATA_DIR/Cornell/files_for_listmode, the list-mode copies of the Cornell COG/DOI limits files.
# Map: January, through CONFIGS["CORNELL"]. The script's repo-root copies (gitignored) are retired (spec 007
# Clarify, T21/T22).

LIST_MODE_LIMITS = {"COG": "Cornell/files_for_listmode/cog_limits_fullSystem.txt",
                    "DOI": "Cornell/files_for_listmode/doi_limits_full_system.txt"}


@pytest.mark.real_data
@fr("002-FR-16", "002-FR-17", "002-FR-18", "007-FR-4")
def test_list_mode_limits_copies_load(real_data_file):
    """6,375 keys each, every key its time channel's own slabs; the DOI copy's single left == right entry
    is counted."""
    setup = load_setup(Settings(str(CONFIGS["CORNELL"]), "", "CORNELL", max_pairs=None, calibrated=False))
    loaded = {}
    for label, relative in LIST_MODE_LIMITS.items():
        limits = load_limits(str(real_data_file(relative)))
        channels, slabs = limits.keys >> 5, limits.keys & 31
        own = all(ChannelType.TIME in setup.channel_types.get(int(ch), ()) and int(s) // 2 == setup.coordinates[int(ch)][2]
                  for ch, s in zip(channels, slabs))
        loaded[label] = (len(limits), limits.invalid, own)
    assert loaded == {"COG": (6375, 0, True), "DOI": (6375, 1, True)}, loaded
