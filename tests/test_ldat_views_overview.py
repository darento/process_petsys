"""Minimodule layout, metrics and the System Overview tab of LDATInspector (spec 002 T8, T9, T18;
spec 007 T22).

T8: minimodule grids derived from the real maps and from a synthetic 2 x 3 map, SuperModule placement, and
per-minimodule counts and fits compared with a direct per-side recount and ``fit_peak`` on the same masks.
T9: the composite System Overview image (engine), then the tab: all four metrics, unpopulated vs zero vs
unavailable rendering, a click on the canvas resolving to (SM, mM), "Open SM", and a slow background job
that must not block ``_poll_events``. T18 (FR-21): the Cornell composite in the real-system orientation.

Moved from ``scripts/ldat_views_check.py``: each section function is the script's, copied unchanged
except for underscore prefixes, ``destroy(app)``, a ``tmp`` folder argument and the removed ``--png``
figures. A module fixture runs a section once and records its ``check`` calls (``ldat_helpers.Checks``);
each check is one test, so a GUI section's steps still run in order on one hidden window.
"""

from collections import Counter
import os

import numpy as np
import pytest

from ldat_helpers import CONFIGS, SELECTION, Checks, build, destroy, metrics_dataset, require_display
from src.ldat_inspector.engine import (OVERVIEW_METRICS, RAW_FIT, TILE_EMPTY, TILE_UNAVAILABLE,
                                       TILE_UNPOPULATED, TILE_VALUE, Dataset, Selection, Settings,
                                       channel_geometry, fit_peak, flood_counts, minimodule_layout,
                                       minimodule_metrics, overview_grid, supermodule_layout)
from src.mapping_generator import ChannelType

pytestmark = pytest.mark.fr("002-FR-15")
fr = pytest.mark.fr


def legacy_layout(dataset):
    """Spec 001 GUI ``_layout``, kept here to prove the engine placement is unchanged."""
    config = dataset.config
    if dataset.settings.system == "CORNELL":
        rings = len(config.get("ring_z") or [0, 1, 2])
        ncols = max(len(config.get("ring_yx") or {}), 1)
        return rings, ncols, lambda sm: (sm % rings, sm // rings)
    ncols = max(len(config.get("ring_yx") or {}), 1)
    rings = max(len(config.get("ring_z") or []), 1)
    return rings, ncols, lambda sm: (sm // ncols, sm % ncols)


def synthetic_map_dataset(grid, per_mm=4, pitch=3.2, spacing=14.0, swap=None):
    """A one-SM dataset whose minimodules sit on a ``grid`` = (rows, cols) of centres.

    Minimodule numbers run right-to-left, top-to-bottom, like the real maps;
    ``swap`` places two minimodules at the same centre (not a grid).
    """
    rows, cols = grid
    coordinates, modules, types, expected_mm = {}, {}, {}, {0: set()}
    expected_t, expected_e = {0: set()}, {0: set()}
    truth = {r * cols + (cols - 1 - c): (r, c) for r in range(rows) for c in range(cols)}
    centres = {mm: ((c + 0.5) * spacing, (rows - r - 0.5) * spacing) for mm, (r, c) in truth.items()}
    if swap:
        centres[swap[1]] = centres[swap[0]]
    ch = 0
    offsets = (np.arange(per_mm) - (per_mm - 1) / 2) * pitch
    for mm, (cx, cy) in sorted(centres.items()):
        for dx in offsets:
            coordinates[ch], modules[ch], types[ch] = (cx + dx, cy, 0), (0, mm), [ChannelType.TIME]
            expected_t[0].add(ch)
            ch += 1
        for dy in offsets:
            coordinates[ch], modules[ch], types[ch] = (cx, cy + dy, 0), (0, mm), [ChannelType.ENERGY]
            expected_e[0].add(ch)
            ch += 1
        expected_mm[0].add(mm)
    settings = Settings(str(CONFIGS["IMAS"]), "", "IMAS", calibrated=False)
    dataset = Dataset(settings, [], {}, expected_t, expected_e, {}, {}, "synthetic", {}, {}, expected_mm,
                      None, coordinates, modules, types)
    return dataset, truth


def cornell_global(config, sm, x, y):
    """(Z, θ in degrees, tangential offset) of a local point, as scripts_cornell's sm_map_gen/local_to_global.

    local_to_global: x -= sm_z + 48; y -= 48; the point (r, y, -x) is rotated about Z by
    θ = arctan2(*ring_yx[cassette]), so Z = sm_z + 48 - x and the tangential offset is y - 48.
    """
    n = len(config["ring_z"])
    sm_z = config["ring_z"][sm % n]
    theta = np.degrees(np.arctan2(*config["ring_yx"][sm // n])) % 360
    return sm_z + 48 - x, theta, y - 48


def orientation_checks(check, dataset):
    """FR-21 (T18): the Cornell composite follows the real-system geometry of the scripts."""
    from src.ldat_inspector.engine import supermodule_axis_labels
    config = dataset.config
    grid = overview_grid(dataset, None, "Ingest sides")
    layout = minimodule_layout(dataset)
    place = {pixel: cornell_global(config, sm, *layout[sm]["centres"][mm]) for pixel, (sm, mm) in grid["cells"].items()}
    by_col, by_row = {}, {}
    for (r, c), value in place.items():
        by_col.setdefault(c, []).append((r, value))
        by_row.setdefault(r, []).append((c, value))
    down = all(a[1][0] > b[1][0] for col in by_col.values() for a, b in zip(sorted(col), sorted(col)[1:]))
    same_column = all(np.ptp([v[1] for _, v in col]) == 0 and np.ptp([v[2] for _, v in col]) < 0.5
                      for col in by_col.values())
    right = all((a[1][1], a[1][2]) < (b[1][1], b[1][2]) for row in by_row.values()
                for a, b in zip(sorted(row), sorted(row)[1:]))
    same_row = all(np.ptp([v[0] for _, v in row]) < 0.5 for row in by_row.values())
    origins = grid["origins"]
    check("CORNELL (FR-21): a lower pixel row always has a smaller Z and a column further right a larger "
          "(θ, tangential offset), from the scripts' local_to_global",
          down and right and same_column and same_row and len(place) == 480
          and origins[2] == (0, 0) and origins[0] == (10, 0) and origins[29] == (0, 45) and origins[27] == (10, 45),
          f"down {down}, right {right}, columns {same_column}, rows {same_row}; SM 2/0/29 at "
          f"{origins[2]}/{origins[0]}/{origins[29]}")
    rows, cols, row_title, col_title = supermodule_axis_labels(dataset)
    check("CORNELL (FR-21): axis labels give Z (+ at top) and the cassette angle θ from the config",
          rows == ["Z +102 mm", "Z +0 mm", "Z -102 mm"] and cols[:3] == ["θ 0°", "θ 36°", "θ 72°"]
          and cols[-1] == "θ 324°" and "ring_z" in row_title and "ring_yx" in col_title, f"{rows} {cols}")


def layout_checks(check):
    for system in ("IMAS", "CORNELL"):
        dataset, _, _, _ = build(system)
        layout = minimodule_layout(dataset)
        mapped = sorted(set(dataset.expected_time))
        shapes = {tuple(entry["shape"]) for entry in layout.values()}
        unique = all(len(set(entry["cells"].values())) == 16 for entry in layout.values())
        grid = np.full((4, 4), -1)
        for mm, (r, c) in layout[mapped[0]]["cells"].items():
            grid[r, c] = mm
        populated = {sm: sum(entry["populated"].values()) for sm, entry in layout.items()}
        want = {sm: len(dataset.expected_mm[sm]) for sm in mapped}
        check(f"{system}: every SM's minimodule grid derived from the map is 4 x 4 with unique cells",
              sorted(layout) == mapped and shapes == {(4, 4)} and unique and populated == want,
              f"SM {mapped[0]} rows top->bottom: {grid.tolist()}")
        centres = layout[mapped[0]]["centres"]
        geometry = channel_geometry(dataset, mapped[0])["minimodules"]
        inside = all(geometry[mm]["box"][0] < x < geometry[mm]["box"][1]
                     and geometry[mm]["box"][2] < y < geometry[mm]["box"][3] for mm, (x, y) in centres.items())
        corner = layout[mapped[0]]["cells"][max(centres, key=lambda mm: centres[mm][0] + centres[mm][1])]
        if system == "IMAS":
            check(f"{system}: row 0 is the largest Y and column 0 the smallest X (flood-map orientation)",
                  inside and corner == (0, 3), f"largest X + Y minimodule at cell {corner}")
            rows, cols, cells = supermodule_layout(dataset)
            legacy_rows, legacy_cols, locate = legacy_layout(dataset)
            check(f"{system}: SuperModule placement is unique and matches spec 001",
                  (rows, cols) == (legacy_rows, legacy_cols) and len(cells) == len(mapped)
                  and len(set(cells.values())) == len(cells)
                  and all(cells[sm] == locate(sm) and cells[sm][0] < rows and cells[sm][1] < cols for sm in cells),
                  f"{rows} x {cols} for {len(cells)} SMs")
        else:  # FR-21: rows run down with local X (Z decreasing), columns right with local Y (θ increasing)
            check(f"{system}: row 0 is the smallest local X and column 0 the smallest local Y (FR-21)",
                  inside and corner == (3, 3), f"largest X + Y minimodule at cell {corner}; grid {grid.tolist()}")
            orientation_checks(check, dataset)

    for grid in ((2, 3), (3, 1), (1, 5)):
        dataset, truth = synthetic_map_dataset(grid)
        layout = minimodule_layout(dataset)[0]
        check(f"synthetic {grid[0]} x {grid[1]} map gives its own grid (layout derived, not assumed)",
              tuple(layout["shape"]) == grid and layout["cells"] == truth,
              f"shape {layout['shape']}")
    dataset, _ = synthetic_map_dataset((2, 2), swap=(0, 1))
    try:
        minimodule_layout(dataset)
        check("two minimodules at one centre are rejected, not silently merged", False)
    except ValueError as exc:
        check("two minimodules at one centre are rejected, not silently merged", True, str(exc))


def metrics_checks(check):
    selection = Selection(400.0, 650.0, 0.1, 0.9, 10.0, 90.0, 0.0, 102.0)
    dataset, columns = metrics_dataset(calibrated=True)
    metrics = minimodule_metrics(dataset, selection)
    # independent recount, one side at a time, from the pair-ordered input
    e, doi, x, y = columns["raw_energy"], columns["doi"], columns["x"], columns["y"]
    ingest, selected, spatial = Counter(), Counter(), {}
    for i in range(len(e)):
        key = (int(columns["sm"][i]), int(columns["mm"][i]))
        partner = i ^ 1
        roi = 0.1 <= doi[i] <= 0.9 and 10 <= x[i] <= 90 and 0 <= y[i] <= 102
        ingest[key] += 1
        if roi:
            spatial.setdefault(key, []).append(e[i])
            if 400 <= e[i] <= 650 and 400 <= e[partner] <= 650:
                selected[key] += 1
    # every expected minimodule of every mapped SM gets a row, including SMs without sides
    expected_keys = {(sm, mm) for sm, mms in dataset.expected_mm.items() for mm in mms}
    counts_ok = set(metrics) == expected_keys and all(
        metrics[k]["ingest"] == ingest[k] and metrics[k]["selected"] == selected[k]
        and metrics[k]["fit_sides"] == len(spatial.get(k, ())) for k in expected_keys)
    check("per-minimodule ingest, selected and fit-population counts match a per-side recount",
          counts_ok and metrics[(1, 7)]["ingest"] == 0 and len(expected_keys) == 20 * 16 + 10 * 8
          and all(metrics[(4, mm)]["ingest"] == 0 for mm in range(16)),
          f"{len(metrics)} minimodules; SM 1 mM 3: {metrics[(1, 3)]['ingest']} sides")
    mismatched = []
    for key in sorted(expected_keys):
        want = fit_peak(np.array(spatial.get(key, [])))
        got = metrics[key]["fit"]
        if got["status"] != want["status"] or got["mu"] != want["mu"] or got["resolution"] != want["resolution"]:
            mismatched.append(key)
    statuses = Counter(m["fit"]["status"] for m in metrics.values())
    check("per-minimodule fits equal fit_peak on the same ROI/DOI masks (energy window off)",
          not mismatched and statuses["FIT"] >= 35 and metrics[(1, 3)]["fit"]["status"] != "FIT"
          and metrics[(1, 7)]["fit"]["status"] != "FIT",
          f"{dict(statuses)}; mismatched {mismatched[:3]}")
    raw, _ = metrics_dataset(calibrated=False)
    raw_metrics = minimodule_metrics(raw, selection)
    check("raw mode marks every minimodule fit unavailable, keeping the counts",
          all(m["fit"] is RAW_FIT for m in raw_metrics.values())
          and all(raw_metrics[k]["selected"] == metrics[k]["selected"] for k in expected_keys))
    calls = []
    stopped = minimodule_metrics(dataset, selection, cancelled=lambda: calls.append(1) or len(calls) > 1)
    no_fits = minimodule_metrics(dataset, selection, fits=False)
    check("cancelling returns None; fits=False gives counts without fits",
          stopped is None and len(calls) == 2 and all(m["fit"] is None for m in no_fits.values())
          and all(no_fits[k]["ingest"] == metrics[k]["ingest"] for k in expected_keys))
    some = minimodule_metrics(dataset, selection, sms=[1, 2, 99])
    check("sms=[...] gives exactly those SuperModules' rows, equal to the full computation",
          set(some) == {k for k in expected_keys if k[0] in (1, 2)}
          and all(some[k]["ingest"] == metrics[k]["ingest"] and some[k]["selected"] == metrics[k]["selected"]
                  and some[k]["fit"]["status"] == metrics[k]["fit"]["status"]
                  and some[k]["fit"]["mu"] == metrics[k]["fit"]["mu"] for k in some), f"{len(some)} rows")


def overview_checks(check):
    dataset, _ = metrics_dataset(calibrated=True)
    metrics = minimodule_metrics(dataset, SELECTION)
    grids = {name: overview_grid(dataset, metrics, name) for name in OVERVIEW_METRICS}
    ingest = grids["Ingest sides"]
    kind = ingest["kind"]
    check("overview: Cornell composite is 3 rings x 10 cassettes of 4 x 4 tiles with one-pixel gaps",
          kind.shape == (14, 49) and len(ingest["cells"]) == 480
          and int((kind == TILE_EMPTY).sum()) == 14 * 49 - 480
          and all(kind[r, c] == TILE_EMPTY for r in (4, 9) for c in range(49))
          and all(kind[r, c] == TILE_EMPTY for c in range(4, 49, 5) for r in range(14)),
          f"{kind.shape}, {len(ingest['cells'])} tiles")
    pixel_of = {cell: pixel for pixel, cell in ingest["cells"].items()}
    # FR-21 (T18; was (5, 8) with SM 0 at the top): mM 0 has the largest local X and Y, so the bottom-right tile
    check("overview: SM 4 mM 0 sits at pixel (8, 8) (Z 0, θ 36°, bottom-right tile); SM 2 mM 3 at the top-left",
          pixel_of[(4, 0)] == (8, 8) and ingest["cells"][(0, 3)] == (2, 3) and ingest["origins"][4] == (5, 5)
          and ingest["origins"][0] == (10, 0))
    counts_ok = all(
        (kind[pixel] == TILE_UNPOPULATED and sm in (2, 5, 8, 11, 14, 17, 20, 23, 26, 29)
         and mm not in dataset.expected_mm[sm])
        or (kind[pixel] == TILE_VALUE and ingest["values"][pixel] == metrics[(sm, mm)]["ingest"]
            and grids["Selected sides"]["values"][pixel] == metrics[(sm, mm)]["selected"])
        for pixel, (sm, mm) in ingest["cells"].items())
    check("overview: count tiles equal the metrics; zero-side minimodules are values (0), unpopulated are not",
          counts_ok and int((kind == TILE_UNPOPULATED).sum()) == 80
          and kind[pixel_of[(1, 7)]] == TILE_VALUE and ingest["values"][pixel_of[(1, 7)]] == 0
          and kind[pixel_of[(4, 0)]] == TILE_VALUE and ingest["values"][pixel_of[(4, 0)]] == 0)
    centroid, resolution = grids["Photopeak centroid (keV)"], grids["Energy resolution (%)"]
    fits_ok = all(
        (centroid["kind"][pixel] == TILE_VALUE and centroid["values"][pixel] == metrics[key]["fit"]["mu"]
         and resolution["values"][pixel] == metrics[key]["fit"]["resolution"])
        if metrics.get(key, {}).get("fit", {}).get("status") == "FIT"
        else centroid["kind"][pixel] in (TILE_UNAVAILABLE, TILE_UNPOPULATED)
        for pixel, key in centroid["cells"].items())
    check("overview: fit tiles equal the fits; failed fits are unavailable (never a low value), with the reason",
          fits_ok and centroid["kind"][pixel_of[(1, 3)]] == TILE_UNAVAILABLE
          and centroid["reason"][(1, 3)] == metrics[(1, 3)]["fit"]["status"]
          and centroid["kind"][pixel_of[(1, 7)]] == TILE_UNAVAILABLE,
          f"{int((centroid['kind'] == TILE_VALUE).sum())} fitted tiles")
    raw, _ = metrics_dataset(calibrated=False)
    raw_metrics = minimodule_metrics(raw, SELECTION)
    raw_centroid = overview_grid(raw, raw_metrics, "Photopeak centroid (keV)")
    raw_ingest = overview_grid(raw, raw_metrics, "Ingest sides")
    check("overview: raw mode makes every fit tile unavailable and keeps the count tiles",
          int((raw_centroid["kind"] == TILE_VALUE).sum()) == 0
          and int((raw_centroid["kind"] == TILE_UNAVAILABLE).sum()) == 400
          and set(raw_centroid["reason"].values()) == {RAW_FIT["status"]}
          and np.array_equal(raw_ingest["values"], ingest["values"], equal_nan=True))
    imas, _, _, _ = build("IMAS")
    imas_grid = overview_grid(imas, minimodule_metrics(imas, Selection(), fits=False), "Ingest sides")
    check("overview: IMAS composite is 5 rings x 24 of 4 x 4 tiles",
          imas_grid["kind"].shape == (24, 119) and len(imas_grid["cells"]) == 120 * 16)


# Wall-clock bounds are asserted in serial runs only: parallel workers stretch them (spec 007 Clarify, T20).
SERIAL = "PYTEST_XDIST_WORKER" not in os.environ


def overview_gui_checks(check):
    import threading
    import time
    from tkinter import messagebox
    from unittest.mock import patch

    from matplotlib.backend_bases import MouseEvent
    from matplotlib.colors import to_rgba

    import exe_programs.ldat_inspector_gui as gui

    dataset, _ = metrics_dataset(calibrated=True)
    app = gui.LDATWorkbench()
    app.withdraw()
    errors, stamps = [], []
    original_poll = app._poll_events

    def poll():
        stamps.append(time.monotonic())
        original_poll()

    app._poll_events = poll  # the next after() call picks this up

    def pump(until, timeout=30):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            app.update()
            if until():
                return True
            time.sleep(0.01)
        return False

    try:
        with patch.object(messagebox, "showerror", side_effect=lambda *a, **k: errors.append(a)):
            for var, value in zip((app.energy_low, app.energy_high, app.doi_low, app.doi_high,
                                   app.x_low, app.x_high, app.y_low, app.y_high),
                                  ("400", "650", "0.1", "0.9", "10", "90", "0", "102")):
                var.set(value)
            app.dataset = dataset
            app.tabs.set("System Overview")
            app._update_after_processing()
            first = [t.get_text() for t in app.overview_fig.axes[0].texts] if app.overview_fig.axes else []
            ready = pump(lambda: app._overview_grid is not None)
            check("overview tab shows 'computing' first, then the tiles from the background job",
                  any("Computing" in t for t in first) and ready, f"{first}")
            # loading data applies the config's energy_range to the energy window; DOI/ROI stay as set
            selection = app._selection()
            expected = minimodule_metrics(dataset, selection)
            drawn = {}
            for mode in OVERVIEW_METRICS:
                app.overview_mode.set(mode)
                app._draw_overview()
                pump(lambda: app._overview_grid is not None and app._overview_grid["mode"] == mode)
                want = overview_grid(dataset, expected, mode)
                image = app._overview_ax.images[1].get_array()
                drawn[mode] = (np.array_equal(np.ma.filled(image.astype(float), np.nan), want["values"],
                                              equal_nan=True)
                               and OVERVIEW_METRICS[mode][1] in app._overview_ax.get_title())
            check("all four metrics draw the expected per-minimodule values and label",
                  all(drawn.values()), f"{drawn}")

            grid = app._overview_grid  # resolution view
            background = app._overview_ax.images[0].get_array()
            pixel_of = {cell: pixel for pixel, cell in grid["cells"].items()}
            unpop, unavailable = pixel_of[(2, 2)], pixel_of[(1, 3)]
            hatched = [pch for pch in app._overview_ax.patches if pch.get_hatch()]
            legend = [t.get_text() for t in app._overview_ax.get_legend().get_texts()]
            check("unpopulated tiles are hatched light grey, unavailable fits dark grey, both outside the colormap",
                  tuple(background[unpop]) == to_rgba(gui.TILE_COLOURS[TILE_UNPOPULATED])
                  and tuple(background[unavailable]) == to_rgba(gui.TILE_COLOURS[TILE_UNAVAILABLE])
                  and np.ma.is_masked(app._overview_ax.images[1].get_array()[unavailable])
                  and len(hatched) == 80 and "fit unavailable" in legend and "unpopulated (config)" in legend,
                  f"legend {legend}")
            app.overview_mode.set("Ingest sides")
            app._draw_overview()
            image = app._overview_ax.images[1]
            zero = pixel_of[(4, 0)]
            check("zero-side minimodules use the colormap minimum (a value, not a grey tile)",
                  image.get_array()[zero] == 0 and image.norm.vmin == 0
                  and "0 sides (colormap minimum)" in [t.get_text() for t in app._overview_ax.get_legend().get_texts()])

            app.update()
            app.overview_canvas.draw()
            x, y = app._overview_ax.transData.transform((8.0, 8.0))  # column 8, row 8: SM 4 mM 0 (FR-21)
            event = MouseEvent("button_press_event", app.overview_canvas, x, y, button=1)
            app.overview_canvas.callbacks.process("button_press_event", event)
            info = app.overview_info.cget("text")
            x, y = app._overview_ax.transData.transform((2.0, 12.0))  # SM 0 (origin (10, 0)), cell (2, 2): mM 5
            event = MouseEvent("button_press_event", app.overview_canvas, x, y, button=1)
            app.overview_canvas.callbacks.process("button_press_event", event)
            info0 = app.overview_info.cget("text")
            want0 = expected[(0, 5)]
            x, y = app._overview_ax.transData.transform((4.0, 0.0))  # the gap column between cassettes
            app.overview_canvas.callbacks.process(
                "button_press_event", MouseEvent("button_press_event", app.overview_canvas, x, y, button=1))
            check("a click on the canvas resolves the pixel to (SM, mM) and shows its identity; gaps are ignored",
                  # FR-23 (T20): the SM's map address sits between SM and mM
                  info.startswith("SM 4 · DAQ port 0 · SLAVE · FEB/D port 3 · mM 0") and "ingest 0 sides" in info
                  and info0.startswith("SM 0 · DAQ port 0 · MASTER · FEB/D port 1 · mM 5 ·")
                  and app.overview_info.cget("text") == info0
                  and app._overview_pick == (0, 5), f"{info} | {info0}")
            app.overview_mode.set("Photopeak centroid (keV)")
            app._draw_overview()
            app._select_overview_tile(0, 5)
            text = app.overview_info.cget("text")
            check("tile text gives counts and, after a fit view, the photopeak",
                  f"ingest {want0['ingest']:,} sides" in text and f"selected {want0['selected']:,}" in text
                  and (f"{want0['fit']['mu']:.1f} keV" in text), text)
            ticks = ([t.get_text() for t in app._overview_ax.get_yticklabels()],
                     [t.get_text() for t in app._overview_ax.get_xticklabels()])
            check("System Overview ticks and titles give Z (+ at top) and θ, with the in-SM directions (FR-21)",
                  ticks[0] == ["Z +102 mm", "Z +0 mm", "Z -102 mm"] and ticks[1][:2] == ["θ 0°", "θ 36°"]
                  and "Axial Z" in app._overview_ax.get_ylabel()
                  and "→ local Y (θ), ↓ local X (−Z)" in app._overview_ax.get_xlabel(), f"{ticks}")
            app.overview_mode.set(gui.FLOOD_MODE)
            app._draw_overview()
            flood_axes = app.overview_fig.axes
            sm0 = flood_axes[2 * 10 + 0]  # SuperModule row 2 (Z -102), column 0 (θ 0°)
            data = app.dataset.modules[0]
            shown = app._selection().mask(data)
            counts, _, _ = flood_counts(data.x[shown], data.y[shown], 28, 102.0)
            mesh = sm0.collections[0]
            check("Cornell flood thumbnails: local Y horizontal, local X increasing downward (FR-21)",
                  sm0.get_ylim() == (102.0, 0.0) and sm0.get_xlim() == (0.0, 102.0)
                  and np.array_equal(np.ma.filled(mesh.get_array().reshape(28, 28), -1),
                                     np.ma.filled(counts.T, -1))
                  and not np.array_equal(np.ma.filled(counts, -1), np.ma.filled(counts.T, -1))
                  and "columns θ; in each SM → local Y, ↓ local X" in app.overview_fig._suptitle.get_text(),
                  f"ylim {sm0.get_ylim()}")
            # owner review 2026-09-28: one absolute colour scale for all SM thumbnails (source position)
            peak = 0
            for sm, entry in app.dataset.modules.items():
                keep = app._selection().mask(entry)
                if keep.any():
                    peak = max(peak, int(flood_counts(entry.x[keep], entry.y[keep], 28, 102.0)[0].max()))
            bars = [axis for axis in app.overview_fig.axes if axis.get_ylabel().startswith("sides per bin")]
            meshes = [axis.collections[0] for axis in flood_axes if axis.collections and axis not in bars]
            check("flood thumbnails share one absolute colour scale (0 to the system's peak bin) with one colour bar",
                  len(meshes) >= 3 and {(m.norm.vmin, m.norm.vmax) for m in meshes} == {(0.0, float(peak))}
                  and len(bars) == 1 and "one colour scale for all SMs" in app.overview_fig._suptitle.get_text(),
                  f"{sorted({(m.norm.vmin, m.norm.vmax) for m in meshes})[:3]}; peak {peak}; {len(bars)} colour bars")
            app.overview_mode.set("Photopeak centroid (keV)")
            app._draw_overview()
            app._overview_open_sm()
            check('"Open SM" selects and draws the clicked SuperModule in the SuperModule tab',
                  app.module_var.get() == "SM 0" and app.tabs.get() == "SuperModule"
                  and app.explorer_info.cget("text").startswith("SM 0 "))

            # a slow background job: the Tk thread keeps polling while it runs
            real = gui.minimodule_metrics
            seen = {"threads": set(), "cancelled": 0}

            def slow(*args, cancelled=None, **kwargs):
                seen["threads"].add(threading.get_ident())
                for _ in range(40):
                    if cancelled is not None and cancelled():
                        seen["cancelled"] += 1
                        return None
                    time.sleep(0.05)
                return real(*args, cancelled=cancelled, **kwargs)

            with patch.object(gui, "minimodule_metrics", slow):
                app.tabs.set("System Overview")
                app._overview_cache = []
                app.overview_mode.set("Photopeak centroid (keV)")
                stamps.clear()
                app._draw_overview()
                pump(lambda: False, 0.4)
                app.energy_low.set("420")  # supersedes the running job
                app._draw_overview()
                done = pump(lambda: app._overview_grid is not None
                            and app._overview_cache and app._overview_cache[0]["selection"].energy_low == 420, 20)
            gaps = np.diff(stamps)
            check("a slow background job never blocks _poll_events; a superseded job stops",
                  done and gaps.size > 20 and (gaps.max() < 0.4 or not SERIAL)
                  and threading.get_ident() not in seen["threads"]
                  and seen["cancelled"] == 1 and len(app._overview_cache) == 1,
                  f"{gaps.size} polls, longest gap {gaps.max():.2f} s, cancelled {seen['cancelled']}")
            app._invalidate_data()
            check("invalidating inputs clears the overview cache and selection",
                  not app._overview_cache and app._overview_grid is None
                  and str(app.overview_open.cget("state")) == "disabled")
        check("no error dialogs during the System Overview checks", not errors, f"{errors[:1]}")
    finally:
        destroy(app)


LAYOUT = {
    "imas-grid": "IMAS: every SM's minimodule grid derived from the map is 4 x 4 with unique cells",
    "imas-orientation": "IMAS: row 0 is the largest Y and column 0 the smallest X (flood-map orientation)",
    "imas-placement": "IMAS: SuperModule placement is unique and matches spec 001",
    "cornell-grid": "CORNELL: every SM's minimodule grid derived from the map is 4 x 4 with unique cells",
    "cornell-orientation": "CORNELL: row 0 is the smallest local X and column 0 the smallest local Y (FR-21)",
    "cornell-real-geometry":
        "CORNELL (FR-21): a lower pixel row always has a smaller Z and a column further right a larger (θ, "
        "tangential offset), from the scripts' local_to_global",
    "cornell-axis-labels":
        "CORNELL (FR-21): axis labels give Z (+ at top) and the cassette angle θ from the config",
    "synthetic-2x3": "synthetic 2 x 3 map gives its own grid (layout derived, not assumed)",
    "synthetic-3x1": "synthetic 3 x 1 map gives its own grid (layout derived, not assumed)",
    "synthetic-1x5": "synthetic 1 x 5 map gives its own grid (layout derived, not assumed)",
    "shared-centre-rejected": "two minimodules at one centre are rejected, not silently merged",
}
METRICS = {
    "counts": "per-minimodule ingest, selected and fit-population counts match a per-side recount",
    "fits": "per-minimodule fits equal fit_peak on the same ROI/DOI masks (energy window off)",
    "raw": "raw mode marks every minimodule fit unavailable, keeping the counts",
    "cancel-and-no-fits": "cancelling returns None; fits=False gives counts without fits",
    "sms-subset": "sms=[...] gives exactly those SuperModules' rows, equal to the full computation",
}
OVERVIEW = {
    "cornell-composite":
        "overview: Cornell composite is 3 rings x 10 cassettes of 4 x 4 tiles with one-pixel gaps",
    "tile-positions":
        "overview: SM 4 mM 0 sits at pixel (8, 8) (Z 0, θ 36°, bottom-right tile); SM 2 mM 3 at the top-left",
    "count-tiles":
        "overview: count tiles equal the metrics; zero-side minimodules are values (0), unpopulated are not",
    "fit-tiles":
        "overview: fit tiles equal the fits; failed fits are unavailable (never a low value), with the reason",
    "raw": "overview: raw mode makes every fit tile unavailable and keeps the count tiles",
    "imas-composite": "overview: IMAS composite is 5 rings x 24 of 4 x 4 tiles",
}
OVERVIEW_GUI = {
    "computing-first": "overview tab shows 'computing' first, then the tiles from the background job",
    "four-metrics": "all four metrics draw the expected per-minimodule values and label",
    "unpopulated-and-unavailable":
        "unpopulated tiles are hatched light grey, unavailable fits dark grey, both outside the colormap",
    "zero-sides": "zero-side minimodules use the colormap minimum (a value, not a grey tile)",
    "click": "a click on the canvas resolves the pixel to (SM, mM) and shows its identity; gaps are ignored",
    "tile-text": "tile text gives counts and, after a fit view, the photopeak",
    "ticks": "System Overview ticks and titles give Z (+ at top) and θ, with the in-SM directions (FR-21)",
    "flood-orientation": "Cornell flood thumbnails: local Y horizontal, local X increasing downward (FR-21)",
    "flood-colour-scale":
        "flood thumbnails share one absolute colour scale (0 to the system's peak bin) with one colour bar",
    "open-sm": '"Open SM" selects and draws the clicked SuperModule in the SuperModule tab',
    "slow-job": "a slow background job never blocks _poll_events; a superseded job stops",
    "invalidate": "invalidating inputs clears the overview cache and selection",
    "no-error-dialogs": "no error dialogs during the System Overview checks",
}


@pytest.fixture(scope="module")
def layout():
    return Checks(LAYOUT).run(layout_checks)


@fr("002-FR-8", "002-FR-21")
@pytest.mark.parametrize("check", LAYOUT)
def test_layout(layout, check):
    layout.verdict(check)


@pytest.fixture(scope="module")
def metrics():
    return Checks(METRICS).run(metrics_checks)


@fr("002-FR-8", "002-FR-13")
@pytest.mark.parametrize("check", METRICS)
def test_metrics(metrics, check):
    metrics.verdict(check)


@pytest.fixture(scope="module")
def overview():
    return Checks(OVERVIEW).run(overview_checks)


@fr("002-FR-8", "002-FR-21")
@pytest.mark.parametrize("check", OVERVIEW)
def test_overview(overview, check):
    overview.verdict(check)


@pytest.fixture(scope="module")
def overview_gui():
    require_display()
    return Checks(OVERVIEW_GUI).run(overview_gui_checks)


@pytest.mark.gui
@pytest.mark.slow  # the fixture runs the section: 6.4-6.6 s serial
@fr("002-FR-8", "002-FR-9", "002-FR-21", "002-FR-23")
@pytest.mark.parametrize("check", OVERVIEW_GUI)
def test_overview_gui(overview_gui, check):
    overview_gui.verdict(check)
