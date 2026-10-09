"""SuperModule hardware address from the map's mod_feb_map (spec 002 FR-23, T20; spec 007 T19).

Moved from ``scripts/ldat_ports_check.py``: each ``check`` is one test. The Cornell dataset is
``ldat_helpers.build`` on the January config, compared with the map that config selects (the
script named ``maps/cornell_map_full_system.yaml``, the September layout since 2026-10-07).
"""

import time

import pytest
import yaml

from helpers import REPO
from ldat_helpers import CONFIGS, build, destroy
import src.ldat_inspector.report as report
from src.ldat_inspector.engine import Selection, read_sm_ports, sm_port_text

pytestmark = pytest.mark.fr("002-FR-23")


def selected_map(system):
    return REPO / yaml.safe_load(CONFIGS[system].read_text(encoding="utf-8"))["map_file"]


@pytest.fixture(scope="module")
def datasets():
    return {system: build(system)[0] for system in ("CORNELL", "IMAS")}


@pytest.fixture(scope="module")
def cornell(datasets):
    return datasets["CORNELL"]


# --- synthetic map entries ---------------------------------------------------

def test_valid_entries_read_short_non_integer_bool_and_empty_left_out(tmp_path):
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump({"mod_feb_map": {0: [0, 0, 1], 1: [4, 1, 3], 2: [1, 2], 3: ["a", 0, 1],
                                                    4: [1, True, 1], 5: None, "6": [2, 0, 5]}}))
    assert read_sm_ports(path) == {0: (0, 0, 1), 1: (4, 1, 3), 6: (2, 0, 5)}


def test_map_without_mod_feb_map_and_missing_file_give_no_addresses(tmp_path):
    (tmp_path / "none.yaml").write_text("x_pitch: 3.2\n")
    assert read_sm_ports(tmp_path / "none.yaml") == {} and read_sm_ports(tmp_path / "missing.yaml") == {}


def test_labels_slave_master_slave_id_and_unavailable():
    texts = [sm_port_text(p) for p in ((4, 1, 1), (0, 0, 5), (3, 2, 1), None)]
    assert texts == ["DAQ port 4 · SLAVE · FEB/D port 1", "DAQ port 0 · MASTER · FEB/D port 5",
                     "DAQ port 3 · slave ID 2 · FEB/D port 1", "ports unavailable"]


# --- the selected maps -------------------------------------------------------

@pytest.mark.parametrize("system", ["CORNELL", "IMAS"])
def test_dataset_ports_equal_the_selected_map_and_cover_every_mapped_sm(datasets, system):
    dataset = datasets[system]
    raw = yaml.safe_load(selected_map(system).read_text())["mod_feb_map"]
    want = {int(sm): tuple(v) for sm, v in raw.items()}
    assert dataset.sm_ports == want and set(dataset.expected_time) <= set(want), \
        f"{len(dataset.sm_ports)} vs {len(want)}"


def test_cornell_sm_15_and_18_addresses(cornell):
    assert cornell.sm_ports.get(15) == (4, 1, 1) and cornell.sm_ports.get(18) == (4, 0, 1)


# --- System Overview tile line in a hidden LDATWorkbench ---------------------

@pytest.fixture(scope="module")
def overview(cornell):
    import tkinter as tk

    import exe_programs.ldat_inspector_gui as gui

    try:
        app = gui.LDATWorkbench()
    except tk.TclError as exc:
        pytest.skip(f"gui: no display available ({exc})")
    app.withdraw()
    try:
        app.dataset = cornell
        app.tabs.set("System Overview")
        app._update_after_processing()
        start = time.monotonic()
        while time.monotonic() - start < 60 and app._overview_grid is None:
            app.update()
            time.sleep(0.01)
        assert app._overview_grid is not None, "System Overview grid not drawn in 60 s"
        cells = set(app._overview_grid["cells"].values())
        yield app, gui, cells, next(mm for sm, mm in sorted(cells) if sm == 15)
    finally:
        destroy(app)


@pytest.mark.gui
def test_clicked_tile_line_shows_sm_15_address(overview):
    app, _, _, sm15 = overview
    app._select_overview_tile(15, sm15)
    text = app.overview_info.cget("text")
    assert text.startswith(f"SM 15 · DAQ port 4 · SLAVE · FEB/D port 1 · mM {sm15}"), text


@pytest.mark.gui
def test_unpopulated_tile_line_also_shows_the_address(overview, cornell):
    app, gui, cells, _ = overview
    grid = app._overview_grid
    unpop = [(sm, mm) for sm, mm in sorted(cells)
             if grid["kind"][next(p for p, c in grid["cells"].items() if c == (sm, mm))] == gui.TILE_UNPOPULATED]
    assert unpop, "the January config declares half-populated SuperModules"
    sm, mm = unpop[0]
    app._select_overview_tile(sm, mm)
    text = app.overview_info.cget("text")
    assert sm_port_text(cornell.sm_ports[sm]) in text and "unpopulated" in text, text


@pytest.mark.gui
def test_sm_without_a_map_entry_shows_ports_unavailable(overview, cornell):
    app, _, _, sm15 = overview
    saved = cornell.sm_ports.pop(15)
    try:
        app._select_overview_tile(15, sm15)
        text = app.overview_info.cget("text")
    finally:
        cornell.sm_ports[15] = saved
    assert "SM 15 · ports unavailable · mM" in text, text


# --- report summary table ----------------------------------------------------

@pytest.fixture(scope="module")
def summary(cornell, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ldat-ports")
    captured = []
    original = report._text

    def spy(axis, lines, **kwargs):
        captured.append(list(lines))
        return original(axis, lines, **kwargs)

    report._text = spy
    try:
        report.write_report(tmp / "sm15.pdf", cornell, Selection(), sm=15)
        rows = report.report_rows(cornell, Selection())
        ports = dict(cornell.sm_ports)
        del ports[18]
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(tmp / "table.pdf") as pdf:
            report._tables(pdf, rows, False, ports)
    finally:
        report._text = original
    return [page for page in captured if page[0].startswith("SUPERMODULE SUMMARY")]


def test_supermodule_report_table_header_and_sm_15_row(summary):
    header = summary[0][1]
    line15 = next(line for line in summary[0] if line.split()[:1] == ["15"])
    assert (header.startswith("SM   DAQ  M/S    FEB/D  Ingested")
            and line15.split()[:4] == ["15", "4", "SLAVE", "1"]), f"{header!r} {line15!r}"


def test_full_table_sm_without_entry_shows_dashes_and_sm_0_master(summary):
    full = [line for page in summary[1:] for line in page]
    line18 = next(line for line in full if line.split()[:1] == ["18"])
    line0 = next(line for line in full if line.split()[:1] == ["0"])
    assert line18.split()[:4] == ["18", "--", "--", "--"] and line0.split()[:4] == ["0", "0", "MASTER", "1"], \
        f"{line18!r} {line0!r}"


def test_columns_keep_the_pre_fr23_alignment(summary):
    header = summary[0][1]
    line15 = next(line for line in summary[0] if line.split()[:1] == ["15"])
    # Before FR-23 'Ingested' ended one character before the right-aligned count; keep that.
    events_end = 3 + 1 + len(report._port_columns((4, 1, 1))) + 1 + 10
    assert (header.index("Ingested") + len("Ingested") == events_end - 1
            and line15[:events_end].endswith(line15[:events_end].split()[-1])), f"{header!r}\n{line15!r}"
