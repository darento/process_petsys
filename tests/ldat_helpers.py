"""Shared LDATInspector test builders (spec 007 FR-6, T18, T20).

Copied from the LDAT scripts: ``fixture_files``/``write_pairs`` (``ldat_inspector_check``),
``CONFIGS``/``RECOVERY_CASES``/``fixture_pairs`` and the side/LDAT writers (``ldat_scale_check``),
``build`` (``ldat_views_check``), the slow reader, workbench and close-child body
(``ldat_processing_check``). Cornell fixtures use the tracked January config (spec 007 Clarify, T18).
GUI helpers import Tk lazily, so the module imports without a display.

Import as ``from ldat_helpers import ...``; ``pyproject.toml`` puts ``tests/`` on the path.
"""

from collections import Counter
import struct
import time

import numpy as np
import yaml

import helpers
from helpers import DATA, REPO
from src.ldat_inspector.engine import FileResult, Settings, SideTable, load_setup, merge_results, unpopulated_minimodules
from src.mapping_generator import ChannelType, map_factory

CONFIGS = {"IMAS": REPO / "configs" / "imas_1DAQ.yaml",
           "CORNELL": DATA / "configs" / "cornell_january.yaml"}


# --- ldat_inspector_check: two-module fixture with a calibration -----------

def fixture_files(root, system):
    """Config (absolute map path), calibration and one ``{"t", "e"}`` channel pair on SM 0 and SM 1.

    Was ``configs/cornell_1cassettes.yaml`` for Cornell (untracked); now ``CONFIGS["CORNELL"]``.
    """
    config = yaml.safe_load(CONFIGS[system].read_text(encoding="utf-8"))
    config["map_file"] = str(REPO / config["map_file"])
    config_file = root / "config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    # Calibration is built from real mapped channel IDs, not assumed IDs.
    _, modules, types, _ = map_factory(config["map_file"])
    groups = {}
    for ch, (sm, mm) in modules.items():
        if sm not in (0, 1):
            continue
        group = groups.setdefault(sm, {}).setdefault(mm, {})
        if ChannelType.TIME in types[ch]:
            group.setdefault("t", ch)
        if ChannelType.ENERGY in types[ch]:
            group.setdefault("e", ch)
    selected = []
    for sm in (0, 1):
        mm = next(mm for mm, channels in groups[sm].items() if "t" in channels and "e" in channels)
        selected.append(groups[sm][mm])
    calibration = root / "calibration.txt"
    if system == "IMAS":
        calibration.write_text("ID\tmu\n" + "".join(f"{part['t']}\t100\n" for part in selected), encoding="utf-8")
    else:
        calibration.write_text("ID(t_ch, slab)\tmu\n" + "".join(
            f"({part['t']}, {slab})\t100\n" for part in selected for slab in range(16)), encoding="utf-8")
    return config_file, calibration, selected


def write_pairs(path, channels, count, *, energy=100.0, truncate=False, second_offset_ps=0):
    with open(path, "wb") as handle:
        for i in range(count):
            handle.write(struct.pack("2B", 2, 2))
            for side_index, part in enumerate(channels):
                for ch in (part["t"], part["e"]):
                    handle.write(struct.pack("qfi", 1_000_000_000_000 + i * 10_000_000_000
                                             + (second_offset_ps if side_index else 0), energy, ch))
        if truncate:
            handle.write(b"\x02\x02\x00")


# --- ldat_scale_check: native-layout sides ---------------------------------

def module_channels(setup, sm, mm):
    """Time channels by slab position (Cornell) or X order (IMAS); energy channels."""
    time, energy = [], []
    for ch, key in setup.channel_modules.items():
        if key != (sm, mm) or ch not in setup.coordinates:
            continue
        if ChannelType.TIME in setup.channel_types[ch]:
            time.append(ch)
        if ChannelType.ENERGY in setup.channel_types[ch]:
            energy.append(ch)
    time.sort(key=lambda ch: (setup.coordinates[ch][2], setup.coordinates[ch][0]))
    energy.sort(key=lambda ch: setup.coordinates[ch][1])
    assert len(time) >= 8 and len(energy) >= 4, (sm, mm, len(time), len(energy))
    return time, energy


def side(setup, sm, mm, time, energy, stamp, extra=()):
    """Hits (timestamp, energy, channel) for one detector side."""
    t_ch, e_ch = module_channels(setup, sm, mm)
    hits = [(stamp + 10 * pos, value, t_ch[pos]) for pos, value in time.items()]
    hits += [(stamp + 1000 + i, value, e_ch[i]) for i, value in enumerate(energy)]
    return hits + [(stamp + 2000, value, ch) for ch, value in extra]


def unmapped_channel(setup):
    return next(ch for ch in range(1, 10_000_000) if ch not in setup.channel_types)


def fixture_pairs(system, setup):
    """(name, count, expected outcome, builder) for the main mixed fixture file.

    Settings: >= 4 energy channels, >= 0.2 a.u. per channel. Det1 is on SM 0,
    det2 on SM 1, both minimodule 0 unless stated.
    """
    full = [30.0, 20.0, 10.0, 5.0]
    bad = unmapped_channel(setup)

    def pair(det1_time, det1_energy=full, det1_extra=(), det2_time=None,
             det2_energy=full, det2_extra=(), det1_mm1=None):
        def build(i):
            stamp = 1_000_000_000_000 + i * 5_000_000
            det1 = side(setup, 0, 0, det1_time, det1_energy, stamp, det1_extra)
            if det1_mm1 is not None:
                det1 += side(setup, 0, 1, *det1_mm1, stamp + 7)
            det2 = side(setup, 1, 0, det2_time or {3: 12.0, 2: 5.0}, det2_energy,
                         stamp + 1_234 + i % 7, det2_extra)
            return det1, det2
        return build

    fixtures = [
        ("adjacent neighbour p-1", 3, "accepted", pair({3: 12.0, 2: 5.0})),
        ("adjacent neighbour p+1", 3, "accepted", pair({3: 12.0, 4: 5.0})),
        ("one time channel, middle", 3, "accepted", pair({3: 12.0})),
        ("one time channel, edge 0", 3, "accepted", pair({0: 12.0})),
        ("two time channels, edge 7", 3, "accepted", pair({7: 12.0, 6: 5.0})),
        ("second minimodule weaker", 3, "accepted",
         pair({3: 12.0, 2: 5.0}, det1_mm1=({3: 20.0}, [10.0, 5.0]))),
        ("second minimodule stronger, too few channels", 3, "ValueError",
         pair({3: 12.0, 2: 5.0}, det1_mm1=({3: 20.0}, [40.0, 30.0]))),
        ("unmapped channel on det1", 3, "KeyError", pair({3: 12.0, 2: 5.0}, det1_extra=((bad, 50.0),))),
        ("unmapped channel on det2", 3, "KeyError", pair({3: 12.0, 2: 5.0}, det2_extra=((bad, 50.0),))),
        ("three energy channels on det1", 3, "min channels", pair({3: 12.0, 2: 5.0}, det1_energy=full[:3])),
        ("per-channel cut leaves three", 3, "min channels", pair({3: 12.0, 2: 5.0}, det1_energy=[30.0, 20.0, 10.0, 0.1])),
        ("energy channels only (sum rows/cols boundary)", 3, "min channels", pair({})),
    ]
    if system == "CORNELL":
        fixtures += [
            ("non-adjacent top two", 3, "unresolved Cornell slab", pair({3: 12.0, 6: 6.0})),
            ("non-adjacent top two, neighbour fired", 3, "unresolved Cornell slab",
             pair({3: 12.0, 6: 6.0, 2: 3.0})),
        ]
    return fixtures


def write_ldat(path, pairs, *, tail=b""):
    """``helpers.write_ldat``, then ``tail`` (e.g. a truncated record)."""
    helpers.write_ldat(path, pairs)
    with open(path, "ab") as handle:
        handle.write(tail)
    return path


# (name, count, det1 time hits by slab position, legacy slab or None if rejected,
#  recovered slab(s), random under recovery). Det2 is {3: 12, 2: 5} -> slab 6.
RECOVERY_CASES = [
    ("only p-1 fired", 4, {3: 12.0, 6: 6.0, 2: 3.0}, None, {6}, False),
    ("only p+1 fired", 4, {3: 12.0, 6: 6.0, 4: 3.0}, None, {7}, False),
    ("both fired, p-1 stronger", 4, {3: 12.0, 6: 6.0, 2: 4.0, 4: 3.0}, None, {6}, False),
    ("both fired, p+1 stronger", 4, {3: 12.0, 6: 6.0, 2: 3.0, 4: 4.0}, None, {7}, False),
    ("both fired, exact tie", 24, {3: 12.0, 6: 6.0, 2: 3.0, 4: 3.0}, None, {6, 7}, True),
    ("neither neighbour fired", 24, {3: 12.0, 6: 6.0}, None, {6, 7}, True),
    ("neighbour ties the non-adjacent second, listed after it", 4, {3: 12.0, 6: 6.0, 2: 6.0}, None, {6}, False),
    ("neighbour ties the non-adjacent second, listed before it", 4, {3: 12.0, 2: 6.0, 6: 6.0}, 6, {6}, False),
    ("p = 1 with edge neighbour 0 fired", 4, {1: 12.0, 5: 6.0, 0: 3.0}, None, {2}, False),
    ("p = 6 with edge neighbour 7 fired", 4, {6: 12.0, 2: 6.0, 7: 3.0}, None, {13}, False),
    ("edge p = 0, non-adjacent second (edge rule)", 4, {0: 12.0, 5: 6.0}, 1, {1}, False),
    ("edge p = 7, non-adjacent second (edge rule)", 4, {7: 12.0, 2: 6.0, 6: 3.0}, 14, {14}, False),
]


# --- hidden LDATWorkbench ---------------------------------------------------

def slow_reader(path, settings, index=0):
    """Stand-in worker that runs until cancelled (at most 60 s); spawn workers import it from here."""
    import src.ldat_inspector.fastread as ldat_fastread

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if ldat_fastread.cancelled():
            return FileResult(index, str(path), error="Cancelled")
        time.sleep(0.05)
    return FileResult(index, str(path), error="slow reader timed out")


def pump(app, until, timeout):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        app.update()
        if until():
            return True
        time.sleep(0.02)
    return False


def workbench(settings, paths):
    """A withdrawn LDATWorkbench set up for raw IMAS processing of ``paths``."""
    from exe_programs.ldat_inspector_gui import LDATWorkbench

    app = LDATWorkbench()
    app.withdraw()
    app.config_path = settings.config_path
    app.system.set("IMAS")
    app.calibrated.set(False)
    app.min_channels.set("4")
    app.min_channel_energy.set("0.2")
    app.files = [str(p) for p in paths]
    return app


def console(app):
    return app.console.get("1.0", "end")


def close_child(config, file):
    """Child-process body: start a slow run, close the window mid-run, exit."""
    settings = Settings(config, "", "IMAS", max_pairs=None, calibrated=False)
    app = workbench(settings, [file])
    app._reader = slow_reader
    app.whole_files.set(True)
    app._start_processing()
    pump(app, lambda: app._pool is not None and "worker processes" in console(app), 30)
    pump(app, lambda: False, 1.5)  # workers running
    print("CLOSING", flush=True)
    app._close()


def destroy(app):
    """Cancel the window's pending ``after`` jobs, then destroy it. The test process keeps pumping
    later windows, which would run a destroyed window's jobs (Tcl "invalid command name")."""
    for job in app.tk.splitlist(app.tk.call("after", "info")):
        app.tk.call("after", "cancel", job)
    app.destroy()


# --- ldat_views_check: channel-finding dataset -----------------------------

BASE = 100  # hits per channel unless a case says otherwise; default thresholds give 15 and 300


def cases(expected_t, expected_e):
    """case -> (ingest sides, time hit overrides, energy hit overrides, base time, base energy).

    Overrides are {index into the SM's sorted channels: hits}.
    """
    half = len(expected_t) // 2 + 1
    return {
        # boundaries: 15 = 0.15 x 100 is not LOW, 300 = 3 x 100 is not HIGH
        "mixed": (400, {0: 0}, {0: 14, 1: 15, 2: 300, 3: 301}, BASE, BASE),
        "high": (400, {3: 1}, {5: 1000}, BASE, BASE),
        "low": (400, {7: 5}, {}, BASE, BASE),
        "ok": (400, {}, {}, BASE, BASE),
        "few sides": (98, {0: 0}, {}, BASE, BASE),
        "low median": (400, {0: 0}, {}, 19, BASE),
        # more than half the time channels at 0 hits: median 0, so time is insufficient
        "zero median": (400, {i: 0 for i in range(half)}, {0: 0}, BASE, BASE),
        "no data": (0, {}, {}, 0, 0),
    }


def is_time(setup, ch):
    return ChannelType.TIME in setup.channel_types[ch]


def build(system):
    """(dataset, {case: sm}, {sm: (time hits by channel, energy hits by channel)}, setup)."""
    settings = Settings(str(CONFIGS[system]), "", system, max_pairs=None, calibrated=False)
    setup = load_setup(settings)
    empty = merge_results(settings, [], setup)
    half = unpopulated_minimodules(setup.config)
    full_sms = [sm for sm in sorted(empty.expected_time) if sm not in half]
    cases_ = cases(sorted(empty.expected_time[full_sms[0]]), sorted(empty.expected_energy[full_sms[0]]))
    assignment = dict(zip(cases_, full_sms))
    t_counts, e_counts, sides, truth = {}, {}, [], {}
    for case, sm in assignment.items():
        n_sides, t_over, e_over, t_base, e_base = cases_[case]
        t_ch, e_ch = sorted(empty.expected_time[sm]), sorted(empty.expected_energy[sm])
        t = {ch: t_over.get(i, t_base) for i, ch in enumerate(t_ch)}
        e = {ch: e_over.get(i, e_base) for i, ch in enumerate(e_ch)}
        truth[sm] = (t, e)
        t_counts[sm] = Counter({ch: n for ch, n in t.items() if n})
        e_counts[sm] = Counter({ch: n for ch, n in e.items() if n})
        sides += [sm] * n_sides
    if half:
        sm = min(half)
        assignment["half-populated"] = sm
        t = {ch: BASE for ch in empty.expected_time[sm]}
        e = {ch: BASE for ch in empty.expected_energy[sm]}
        truth[sm] = (t, e)
        stray = sorted(ch for ch, (s, mm) in setup.channel_modules.items() if s == sm and mm in half[sm])
        t_counts[sm] = Counter(t)
        e_counts[sm] = Counter(e)
        for ch in stray[:4]:  # hits on unpopulated-minimodule channels, far above the median
            (t_counts if is_time(setup, ch) else e_counts)[sm][ch] = 5000
        sides += [sm] * 400
    n = len(sides)
    assert n % 2 == 0
    sm_col = np.array(sides, dtype=np.int16)
    table = SideTable.from_pairs(0, raw_energy=np.ones(n), calibration_key=np.zeros(n, np.int32),
                                 x=np.zeros(n), y=np.zeros(n), doi=np.zeros(n),
                                 timestamp=np.arange(n, dtype=np.int64), sm=sm_col,
                                 mm=np.zeros(n, np.int8), random_slab=np.zeros(n, bool))
    result = FileResult(0, "synthetic", n // 2, n // 2, table=table, time_counts=t_counts,
                        energy_counts=e_counts)
    return merge_results(settings, [result], setup), assignment, truth, setup
