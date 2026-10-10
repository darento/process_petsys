"""Shared builders and fakes for the PETsys Manager tests (spec 007 FR-6, spec 003 checks).

Copied from the spec 003 check scripts so no test imports a script or another test file.
Reference oracles (``scripts_cornell``) stay out: their outputs become golden files (T8).

- Profile and process fakes: ``FixtureProbe``, ``profile_fixture``, ``FakeChild``,
  ``FakeBackend``, ``DirectDummyChild``, ``FakeDaemon``, ``FakeResources``, ``ToolWorld``.
- Wire encoders: ``compact_record``, ``fixed_record``, ``encode``, ``encode_fixed``, ``encode_compact``.
- Fixture builders: ``CalibrationFixtures`` (a root-based builder), and the ``TestCase`` mixins
  ``SettingsFixtures``, ``ListmodeFixtures``, ``QCFixtures`` and ``CLIFixtures``. ``ListmodeFixtures.at(root)`` and
  ``QCFixtures.at(root)`` build the same fixtures outside a test class.
- ``PrivateOutput``: per-class private folder under the user temp directory (short paths:
  Windows MAX_PATH applies to the literal fixture paths), removed after the class unless
  ``PETSYS_KEEP_FIXTURES`` is set.
"""

from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from threading import Lock, Timer

import numpy as np
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")

from helpers import REPO
from src.cornell import cli
from src.cornell import listmode as lm
from src.cornell import qc
from src.cornell.inputs import load_calibration, load_limits, load_processing_config
from src.detector_features_fixed import calculate_DOI_vectorized
from src.mapping_generator import ChannelType
from src.petsys_manager.commands import build_internal
from src.petsys_manager.artifacts import RunStore
from src.petsys_manager.contracts import (Action, Artifact, CommandResult, DataFormat, Identity, InputDescriptor,
                                          Population, ResultStatus)
from src.petsys_manager.settings import (LMMetadata, MachineProfile, SystemProbe, ToolCapabilities,
                                         preflight)
from src.read_fixed import read_fixed_file_numpy
from src.utils_fixed import get_maxEnergy_sm_mM_vectorized

KEEP_ENV = "PETSYS_KEEP_FIXTURES"


# Private output -------------------------------------------------------------------------

def private_parent():
    """The scripts' private temp parent: %LOCALAPPDATA%/Temp/process_petsys or ~/.cache/process_petsys."""
    parent = (Path(os.environ["LOCALAPPDATA"]) / "Temp/process_petsys" if os.name == "nt"
              else Path.home() / ".cache/process_petsys")
    parent.mkdir(parents=True, exist_ok=True)
    return parent


class PrivateOutput:
    """``TestCase`` mixin: ``cls.output`` is a fresh private folder for the class."""

    fixture_prefix = "petsys-manager-"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output = Path(tempfile.mkdtemp(prefix=cls.fixture_prefix, dir=private_parent()))

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get(KEEP_ENV):
            shutil.rmtree(cls.output, ignore_errors=True)
        super().tearDownClass()


# Machine profile and process fakes (petsys_manager_check, _daqd_check) -------------------

class FixtureProbe(SystemProbe):
    platform_name = "Linux"

    def executable(self, path):
        return path.is_file()  # Test fixture markers, not runnable hardware tools.

    def device(self, path):
        return path.is_file() and path.name in ("card0", "card1")


def profile_fixture(root):
    """A fixture machine profile rooted at ``root`` (created): marker tools, cards and private files."""
    root.mkdir(parents=True)
    for directory in ("tools", "configs", "maps", "private", "data", "encal", "reports", "lm"):
        (root / directory).mkdir()
    for tool in ("daqd", "init_system", "acquire_sipm_data", "set_bias", "convert_raw_to_coincidence", "convert_raw_to_group"):
        (root / "tools" / tool).write_text("fixture marker, never executable", encoding="utf-8")
    for name in ("card0", "card1", "selected.ini", "raw data ; & [literal].rawf", "raw data ; & [literal].idxf",
                 "input.ldat", "cog.txt", "doi.txt", "cal.encal", "pairs.txt", "regions.tsv"):
        (root / "private" / name).write_text("private fixture", encoding="utf-8")
    (root / "maps/selected.yaml").write_text("fixture_map: true\n", encoding="utf-8")
    (root / "configs/selected.yaml").write_text(
        "map_file: maps/selected.yaml\nmin_ch: 4\nen_min_ch: 0.2\nenergy_range: [357, 665]\n"
        "unpopulated_minimodules:\n  2: [2, 3]\n", encoding="utf-8")
    metadata = LMMetadata(isotope="Na22", acquisition_time_s=11., measurement_time_s=11.,
        detector_size_x_mm=50., detector_size_y_mm=50., module_number=2, ring_number=1,
        ring_distance_mm=100., detector_pixels_x=100, detector_pixels_y=100,
        timestamp_unit="input_numeric")
    return MachineProfile(petsys_folder="tools", ini_file="private/selected.ini",
        yaml_file="configs/selected.yaml", data_dir="data", calibration_dir="encal", report_dir="reports",
        lm_dir="lm", cog_limits_file="private/cog.txt", doi_limits_file="private/doi.txt",
        calibration_file="private/cal.encal", pair_map_file="private/pairs.txt", region_map_file="private/regions.tsv",
        cards=(str(root / "private/card0"), str(root / "private/card1")),
        capabilities=ToolCapabilities(), lm_metadata=metadata)


class SettingsFixtures:
    """``TestCase`` mixin (petsys_manager_check ``SettingsChecks`` setup): a fresh profile per test."""

    def setUp(self):
        super().setUp()
        self.root = self.output / self._testMethodName
        self.profile = profile_fixture(self.root)
        self.probe = FixtureProbe()
        self.inputs = (InputDescriptor(Path("private/input.ldat"), "compact", "coincidence"),)

    def check_action(self, action, profile=None, options=None, inputs=None, **kwargs):
        return preflight(profile or self.profile, action, options, self.inputs if inputs is None else inputs,
                         repo_root=self.root, probe=kwargs.pop("probe", self.probe), **kwargs)

    def assertReady(self, report):
        self.assertTrue(report.ready, report.issues)
        self.assertIsNotNone(report.settings)


class FakeChild:
    selectable_pipes = False
    pid = 12345

    def __init__(self, code=0, *, term_exits=True, descendants=False, linger_checks=0, stdout=b"hello\n",
                 stderr=b""):
        self.code = code
        self.term_exits = term_exits
        self.descendants = descendants
        self.linger_checks = linger_checks   # group checks a just-exiting helper stays visible for
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.signals = []
        self.waited = False

    def poll(self):
        return self.code

    def group_alive(self):
        if self.code is not None and self.linger_checks > 0:
            self.linger_checks -= 1
            return True
        return self.code is None or self.descendants

    def terminate(self):
        self.signals.append("TERM")
        if self.term_exits and self.code is None:
            self.code = -15

    def kill(self):
        self.signals.append("KILL")
        self.descendants = False
        if self.code is None:
            self.code = -9

    def wait(self, timeout):
        self.waited = True
        if self.code is None:
            raise subprocess.TimeoutExpired("fake-owned-command", timeout)
        return self.code


class SplitWriterChild(FakeChild):
    """A converter writing one split every ``interval_s`` from launch, exiting 0 one interval after the last."""

    def __init__(self, writes, interval_s):
        super().__init__(None, stdout=b"")
        self.writes, self.interval_s, self.due = list(writes), interval_s, time.monotonic()

    def poll(self):
        while self.code is None and time.monotonic() >= self.due:
            if self.writes:
                self.writes.pop(0)()
                self.due += self.interval_s
            else:
                self.code = 0
        return self.code


class FakeBackend:
    def __init__(self, child=None, error=None):
        self.child = FakeChild() if child is None else child
        self.error = error
        self.launched = []

    def launch(self, command):
        self.launched.append(command)
        if self.error:
            raise self.error
        return self.child


class DirectDummyChild:
    """Check-only native direct child; does NOT claim Windows group support."""
    selectable_pipes = False

    def __init__(self, process):
        self.process = process
        self.pid = process.pid
        self.stdout = process.stdout
        self.stderr = process.stderr

    def poll(self):
        return self.process.poll()

    def group_alive(self):
        return self.poll() is None

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()

    def wait(self, timeout):
        return self.process.wait(timeout=timeout)


class FakeDaemon(FakeChild):
    """Stays alive until signalled or told to die; records every signal it receives."""

    def __init__(self, *, pid=4242, term_exits=True):
        super().__init__(None, term_exits=term_exits, stdout=b"", stderr=b"")
        self.pid = pid

    def die(self, code):
        self.code = code


class FakeResources:
    """In-memory socket/shm table. Has no delete operation: the service cannot remove anything."""

    def __init__(self, existing=(), *, serve_after=0, name="/daqd_shm", peer="owned"):
        self.paths = set(existing)
        self.serve_after = serve_after
        self.name = name
        self.peer = peer
        self.queries = 0
        self.daemon = None
        self.lock = Lock()

    def existing(self, config):
        with self.lock:
            return tuple(p for p in (config.socket_path, config.shared_memory_path) if p in self.paths)

    def is_socket(self, path):
        with self.lock:
            return path in self.paths

    def query(self, socket_path, timeout, abort):
        with self.lock:
            self.queries += 1
            if self.daemon is not None and self.daemon.code is not None:
                raise OSError("connection refused")
            if self.serve_after is None or self.queries <= self.serve_after:
                raise OSError("No reply from DAQD yet")
            peer = self.daemon.pid if self.peer == "owned" else self.peer
            return self.name, peer

    def appear(self, config):
        with self.lock:
            self.paths.update((config.socket_path, config.shared_memory_path))


# Wire encoders (petsys_manager_numeric_check, _calibration_check, _listmode_check) -------

HIT = struct.Struct("<qfi")
PADDING = HIT.pack(0, 0.0, -1)


def fixed_record(sides, limit=4):
    payload = bytes(len(side) for side in sides)
    for side in sides:
        payload += b"".join(HIT.pack(*hit) for hit in side)
        payload += b"\0" * (16 * (limit - len(side)))
    return payload


def compact_record(sides):
    return bytes(len(side) for side in sides) + b"".join(HIT.pack(*hit) for side in sides for hit in side)


def fixture_map():
    return {"FEM": "FEM256", "FEBD": "FEBD1k", "channels": 256,
            "mM_channels": 8, "sum_rows_cols": True, "x_pitch": 3.2, "y_pitch": 3.2,
            "mM_disposition": [2, 1], "Time": list(range(128)), "Energy": list(range(128, 256)),
            "mod_feb_map": {7: [0, 0, 0], 41: [2, 0, 0]}}


def encode(path, records, data_format, *, hit_limit=16):
    """Coincidence records ((side1 hits), (side2 hits)) or groups (side hits,)."""
    with open(path, "xb") as out:
        if data_format == DataFormat.FIXED:
            out.write(struct.pack("<i", hit_limit))
        for record in records:
            out.write(bytes(len(side) for side in record))
            for side in record:
                out.write(b"".join(HIT.pack(*hit) for hit in side))
                if data_format == DataFormat.FIXED:
                    out.write(PADDING * (hit_limit - len(side)))


def encode_fixed(path, records, *, group=False, hit_limit=16):
    with open(path, "xb") as out:
        out.write(struct.pack("<i", hit_limit))
        for record in records:
            sides = [record] if group else list(record)
            out.write(bytes(len(side) for side in sides))
            for side in sides:
                out.write(b"".join(HIT.pack(*hit) for hit in side) + PADDING * (hit_limit - len(side)))


def encode_compact(path, records):
    """Compact coincidence: per record two uint8 hit counts, then the hits (no padding)."""
    with open(path, "xb") as out:
        for side1, side2 in records:
            out.write(bytes((len(side1), len(side2))))
            out.write(b"".join(HIT.pack(*hit) for hit in (*side1, *side2)))


# Cornell side geometry and calibration fixtures (petsys_manager_calibration_check) ------

EN_MIN = 0.2


class Geometry:
    """Channel layout of the selected fixture map, per (SuperModule, minimodule)."""

    def __init__(self, mapping):
        self.time, self.energy = {}, {}
        for channel, module in mapping.modules.items():
            local = mapping.local[channel]
            target = self.time if ChannelType.TIME in mapping.types[channel] else self.energy
            target.setdefault(module, {})[int(local[2])] = (channel, local[1])

    def side(self, rng, module, position, direction, energy, *, energy_channels=5, unresolved=False,
             timestamp=1000):
        times = self.time[module]
        hits = [(timestamp, 12.0, times[position][0]),
                (timestamp + 1, 5.0, times[position + (3 * direction if unresolved else direction)][0])]
        chosen = rng.choice(8, size=energy_channels, replace=False)
        fractions = rng.dirichlet(np.ones(energy_channels))
        for i, (slot, fraction) in enumerate(zip(chosen, fractions)):
            hits.append((timestamp + 2 + i, float(energy * fraction), self.energy[module][int(slot)][0]))
        return hits

    def key(self, module, position, direction):
        return self.time[module][position][0], 2 * position + (1 if direction > 0 else 0)


class SlabGeometry:
    """Time/energy channels of the fixture map per (SuperModule, minimodule)."""

    def __init__(self, mapping):
        self.time, self.energy = defaultdict(dict), defaultdict(dict)
        for channel, module in mapping.modules.items():
            local = mapping.local[channel]
            target = self.time if ChannelType.TIME in mapping.types[channel] else self.energy
            target[module][int(local[2])] = (channel, float(local[1]))

    def side(self, rng, module, slab, energy, *, kind="resolved", depth=0.0, small=False, other=None, t=1000,
             energy_channels=5):
        """One detector side whose reference slab is ``slab``; kinds: resolved, one, nonadjacent, notime."""
        position, half = divmod(slab, 2)
        times = self.time[module]
        hits = []
        if kind != "notime":
            hits.append((t, float(12.0 + rng.random()), times[position][0]))
        if kind == "resolved":
            second = (1 if position == 0 else 6 if position == 7 else position - 1 if half == 0 else position + 1)
            hits.append((t + 1, float(5.0 + rng.random()), times[second][0]))
        elif kind == "nonadjacent":
            hits.append((t + 1, 5.0, times[(position + 3) % 8][0]))
        slots = rng.choice(8, size=energy_channels, replace=False)
        weights = np.exp(depth * (slots - 3.5) / 3.5) * rng.uniform(0.7, 1.3, size=energy_channels)
        fractions = weights / weights.sum()
        hits += [(t + 2 + i, float(energy * f), self.energy[module][int(s)][0]) for i, (s, f) in
                 enumerate(zip(slots, fractions))]
        if small:   # below en_min_ch: dropped before every filter
            hits.append((t + 9, 0.1, self.energy[module][int((slots[0] + 1) % 8)][0]))
        if other is not None:   # light shared with another minimodule of the SuperModule
            share, module2 = other
            hits += [(t + 10 + i, float(share), self.energy[module2][i][0]) for i in range(3)]
        return hits


class CalibrationFixtures:
    """Fixture map, processing config and LDAT writers under ``root`` (created) for calibration."""

    def __init__(self, root):
        self.root = root
        (root / "maps").mkdir(parents=True)
        (root / "configs").mkdir()
        self.map_path = root / "maps/selected.yaml"
        self.map_path.write_text(yaml.safe_dump(fixture_map()), encoding="utf-8")
        self.config_path = root / "configs/processing.yaml"
        self.config_path.write_text(yaml.safe_dump({"map_file": "maps/selected.yaml", "min_ch": 4,
                                                    "en_min_ch": EN_MIN, "energy_range": [357, 665]}),
                                    encoding="utf-8")
        self.config = load_processing_config(self.config_path, processing_root=root, action="calibrate")
        self.mapping = self.config.mapping
        self.geometry = SlabGeometry(self.mapping)

    def limits(self, *, missing=(), name="cog_limits.txt"):
        lines = []
        for module, times in self.geometry.time.items():
            ys = sorted(y for _, y in self.geometry.energy[module].values())
            centre, span = (ys[0] + ys[-1]) / 2, ys[-1] - ys[0]
            for position, (channel, _) in times.items():
                for slab in (2 * position, 2 * position + 1):
                    if (module, slab) not in missing:
                        lines.append(f"({channel}, {slab})\t{centre - 0.3 * span:.4f}\t{centre + 0.3 * span:.4f}\n")
        path = self.root / name
        path.write_text("".join(lines), encoding="utf-8")
        return load_limits(path, self.mapping, kind="cog")

    def write(self, name, records, data_format):
        path = self.root / name
        encode(path, records, data_format)
        return InputDescriptor(path, data_format, Population.COINCIDENCE)

    def write_group(self, name, sides):
        path = self.root / name
        encode(path, [(side,) for side in sides], DataFormat.FIXED)
        return InputDescriptor(path, DataFormat.FIXED, Population.GROUP)


A, B, C = (7, 0), (7, 1), (41, 3)


def sides_for(geometry, specs, seed, *, depth_spread=0.0):
    """specs: (module, slab, count, peak, kind) with kind peak|double|flat|one|nonadjacent|tie."""
    rng = np.random.default_rng(seed)
    sides = []
    for module, slab, count, peak, kind in specs:
        for index in range(count):
            depth = rng.uniform(-depth_spread, depth_spread) if depth_spread else rng.uniform(-1, 1)
            if kind == "flat" or (kind != "tie" and rng.random() < 0.25):
                energy = rng.uniform(15, 200)       # continuum under the photopeak
            elif kind == "double":
                energy = rng.normal(peak if rng.random() < 0.55 else 1.6 * peak, 0.05 * peak)
            else:
                energy = rng.normal(peak, 0.07 * peak)
            side_kind = kind if kind in ("one", "nonadjacent") else "resolved"
            other = None
            if kind in ("few", "split"):   # 3 energy channels: fails the side / minimodule filter
                share = (max(energy, 1.0) / 4, (module[0], (module[1] + 4) % 16)) if kind == "split" else None
                sides.append(geometry.side(rng, module, slab, max(energy, 1.0), energy_channels=3, other=share))
                continue
            if kind == "tie":   # exactly equal minimodule energies: the reference's set order decides
                side = geometry.side(rng, module, slab, 50.0, kind="resolved")
                side = side[:2] + [(1002 + i, 10.0, geometry.energy[module][i][0]) for i in range(5)]
                side += [(1010 + i, 10.0, geometry.energy[(module[0], module[1] + 2)][i][0]) for i in range(5)]
                sides.append(side)
                continue
            if index % 9 == 0:
                other = (max(energy, 1.0) / 20, (module[0], (module[1] + 4) % 16))
            sides.append(geometry.side(rng, module, slab, max(energy, 1.0), kind=side_kind, depth=depth,
                                       small=index % 7 == 0, other=other))
    order = rng.permutation(len(sides))
    return [sides[i] for i in order]


P1_SPECS = ([(A, 1, 420, 90.0, "peak"), (A, 14, 420, 95.0, "peak")] +
            [(A, slab, 420, 80.0 + 3 * slab, "peak") for slab in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13)] +
            [(A, 7, 40, 90.0, "one"), (A, 9, 60, 90.0, "nonadjacent"), (A, 0, 50, 90.0, "one"),
             (A, 4, 40, 90.0, "few"), (A, 5, 40, 90.0, "split")] +
            [(B, slab, 420, 100.0, "peak") for slab in (4, 5, 6)] +
            [(B, 9, 150, 100.0, "peak"), (B, 10, 600, 70.0, "double"), (B, 12, 420, 0.0, "flat"),
             (B, 2, 30, 0.0, "tie")] +
            [(C, slab, 420, 110.0, "peak") for slab in (2, 3, 8, 9, 12)] + [(C, 13, 100, 110.0, "peak")])
P3_SPECS = ([(A, slab, 1500, 85.0 + 2 * slab, "peak") for slab in (1, 2, 3, 4, 5, 6)] +
            [(B, slab, 900, 100.0, "peak") for slab in (4, 5, 6, 7)])


def pairs(sides):
    return [(sides[i], sides[i + 1]) for i in range(0, len(sides) - 1, 2)]


# Listmode fixtures (petsys_manager_listmode_check) --------------------------------------

METADATA = LMMetadata("F18", 300.5, 299.0, 102.4, 96.0, 30, 2, 410.5, 90, 96, "ps")
# module, time position, direction, energy mean/sigma (a.u.), weight
LISTMODE_SPECS = [((7, 0), 2, +1, 100.0, 9.0, 30), ((7, 0), 4, -1, 90.0, 8.0, 30), ((7, 1), 3, +1, 110.0, 10.0, 15),
                  ((21, 3), 3, +1, 80.0, 7.0, 15), ((21, 5), 1, +1, 95.0, 9.0, 8), ((7, 2), 2, +1, 100.0, 9.0, 4),
                  ((21, 6), 4, -1, 100.0, 9.0, 6), ((21, 9), 4, +1, 85.0, 8.0, 15)]
NO_COG = (7, 2)           # COG limits absent -> no_position_region
NARROW_COG = (7, 1)       # decompressed Y leaves [0, 25.6] -> y_out_of_range
NO_CAL = (21, 5)          # no calibration factors -> missing_calibration
NO_DOI = (21, 3)          # no DOI limits -> missing_doi_limits
NO_REGION = (21, 6)       # absent from the region map -> unmapped_region
PAIRS = {(0, 10): 1, (0, 12): 2, (1, 10): 3, (0, 0): 5, (10, 12): 6, (2, 10): 7}   # (0, 11) etc. -> no_pair


class ListmodeFixtures:
    """Seeded fixed-LDAT inputs and position maps for listmode. ``TestCase`` mixin (after
    ``PrivateOutput``: root = output / test name) or standalone via ``ListmodeFixtures.at(root)``."""

    @classmethod
    def at(cls, root):
        builder = cls()
        builder.setup_listmode(root)
        return builder

    def setUp(self):
        super().setUp()
        self.setup_listmode(self.output / self._testMethodName)

    def setup_listmode(self, root):
        self.root = root
        (self.root / "maps").mkdir(parents=True)
        (self.root / "configs").mkdir()
        self.map_path = self.root / "maps/selected.yaml"
        value = fixture_map()
        value["mod_feb_map"] = {7: [0, 0, 0], 21: [2, 0, 0]}   # reference debug counts SMs 0-29 only
        self.map_path.write_text(yaml.safe_dump(value), encoding="utf-8")
        self.config = self.processing({"en_min_ch": 0.2, "energy_range": [357, 665]})
        self.mapping = self.config.mapping
        self.geometry = Geometry(self.mapping)
        self.lm_parent = self.root / "lm"
        self.lm_parent.mkdir()

    def processing(self, extra, name="processing.yaml"):
        path = self.root / "configs" / name
        path.write_text(yaml.safe_dump({"map_file": "maps/selected.yaml", "min_ch": 4, **extra}), encoding="utf-8")
        return load_processing_config(path, processing_root=self.root, action="listmode")

    def slab_keys(self, module, position):
        channel = self.geometry.time[module][position][0]
        return [(channel, 2 * position), (channel, 2 * position + 1)]

    def all_keys(self, *, skip=()):
        keys = []
        for module, position, *_ in LISTMODE_SPECS:
            if module not in skip:
                keys += [k for k in self.slab_keys(module, position) if k not in keys]
        return keys

    def cog_limits(self):
        lines = []
        for module, position, *_ in LISTMODE_SPECS:
            if module == NO_COG:
                continue
            ys = sorted(y for _, y in self.geometry.energy[module].values())
            span, centre = ys[-1] - ys[0], (ys[0] + ys[-1]) / 2
            low, high = ((centre - 0.05 * span, centre + 0.05 * span) if module == NARROW_COG
                         else (ys[0] + 0.02 * span, ys[-1] - 0.02 * span))
            lines += [f"{key}\t{low}\t{high}\n" for key in self.slab_keys(module, position)]
        path = self.root / "cog_limits.txt"
        path.write_text("".join(sorted(set(lines))), encoding="utf-8")
        return load_limits(path, self.mapping, kind="cog")

    def doi_limits(self, descriptor):
        maps = lm.CalibrationMaps.from_mapping(self.mapping)
        ratios = []
        for chunk in read_fixed_file_numpy(str(descriptor.path), 1000, group_events=False):
            for hits, header in ((chunk["side1"], chunk["header"][:, 0]), (chunk["side2"], chunk["header"][:, 1])):
                side = {"header": header, "hits": hits}
                _, max_mm = get_maxEnergy_sm_mM_vectorized(side, maps.minimodules, maps.energy_mask)
                ratios.append(calculate_DOI_vectorized(side, max_mm, maps.minimodules, maps.local,
                                                       (maps.time_mask, maps.energy_mask), sum_rows_cols=True))
        low, high = np.percentile(np.concatenate(ratios), [4, 96])
        path = self.root / "doi_limits.txt"
        path.write_text("".join(f"{key}\t{low}\t{high}\n" for key in sorted(self.all_keys(skip=(NO_DOI,)))),
                        encoding="utf-8")
        return load_limits(path, self.mapping, kind="doi")

    def calibration(self, *, regions=5, name="position_5regions.encal"):
        lines = [f"# Position-dependent energy calibration ({regions} regions per slab)\n",
                 "ID(time_ch, slab, region)\tmu\tsigma\n"]
        mean = {module: m for module, _, _, m, _, _ in LISTMODE_SPECS}
        for module, position, *_ in LISTMODE_SPECS:
            if module == NO_CAL:
                continue
            for channel, slab in self.slab_keys(module, position):
                for region in range(regions):
                    lines.append(f"{(channel, slab, region)}\t{mean[module] * (1 + 0.01 * region):.3f}\t9.000\n")
        path = self.root / name
        rows = sorted(set(lines[2:]), key=lambda line: ast.literal_eval(line.split("\t")[0]))
        path.write_text("".join(lines[:2] + rows), encoding="utf-8")
        return load_calibration(path, self.mapping, expected_regions=regions)

    def region_map(self, *, skip=(NO_REGION,), name="region_sm_mm_map.tsv"):
        rows = []
        for sm, mm in sorted(set(self.mapping.modules.values())):
            if (sm, mm) in skip:
                continue
            region = (mm // 4) + (10 if sm == 21 else 0)
            rows.append(f"{(sm, mm)}\t{region}\t{(0.0, 51.2) if sm == 21 else (0, 0)}\n")
        path = self.root / name
        path.write_text("".join(rows), encoding="utf-8")
        return lm.load_region_map(path, self.mapping)

    def pair_map(self, pairs=PAIRS, name="pairs_map_cornell.txt"):
        path = self.root / name
        path.write_text("".join(f"{p} {a} {b}\n" for (a, b), p in pairs.items()), encoding="utf-8")
        return lm.load_pair_map(path)

    def side(self, rng, i, t0, random_slabs=True):
        module, position, direction, mean, sigma, _ = LISTMODE_SPECS[rng.choice(len(LISTMODE_SPECS), p=self.weights)]
        energy = max(rng.normal(mean, sigma), 1.0)
        kind = i % 53
        if kind == 7:
            energy = 230.0 + rng.random()                       # outside the keV window
        hits = self.geometry.side(rng, module, position, direction, energy,
                                  energy_channels=3 if kind in (5, 9) else 5, unresolved=kind == 11, timestamp=t0)
        if kind == 9:                                            # 5 energy channels, only 3 in the max minimodule
            other = (module[0], (module[1] + 1) % 16)
            hits += [(t0 + 9, 0.5, self.geometry.energy[other][s][0]) for s in (0, 1)]
        if random_slabs and kind in (13, 17, 19):                                 # one time channel: reference np.random slab
            hits = [hits[0]] + hits[2:]
        return hits

    def records(self, count, seed=7, random_slabs=True):
        rng = np.random.default_rng(seed)
        weights = np.array([spec[-1] for spec in LISTMODE_SPECS], dtype=float)
        self.weights = weights / weights.sum()
        records = []
        for i in range(count):
            t0 = 1_000_000 + 997 * i
            delta = 40_000 if i % 61 == 3 else int(rng.integers(-300, 300))   # 40000 ps wraps the int16 dt
            records.append((self.side(rng, 2 * i, t0, random_slabs), self.side(rng, 2 * i + 1, t0 + delta, random_slabs)))
        return records

    def ldat(self, name, records, directory=None):
        path = (directory or self.root) / name
        encode_fixed(path, records)
        return InputDescriptor(path, "fixed", "coincidence")

    def inputs(self, names=("acq_coinc_2.ldat", "acq_coinc_10.ldat"), count=2500, random_slabs=True):
        descriptors = [self.ldat(name, self.records(count, seed=7 + i, random_slabs=random_slabs))
                       for i, name in enumerate(names)]
        return descriptors, dict(calibration=self.calibration(), cog_limits=self.cog_limits(),
                                 doi_limits=self.doi_limits(descriptors[0]), pairs=self.pair_map(),
                                 regions=self.region_map(), metadata=METADATA)

    def compact_twins(self, descriptors, *, count=2500, random_slabs=True):
        """Compact files of the same records as ``inputs()`` (same names, other folder)."""
        folder = self.root / "compact"
        folder.mkdir(exist_ok=True)
        twins = []
        for i, descriptor in enumerate(descriptors):
            path = folder / descriptor.path.name
            encode_compact(path, self.records(count, seed=7 + i, random_slabs=random_slabs))
            twins.append(InputDescriptor(path, "compact", "coincidence"))
        return twins

    def generate(self, descriptors, maps, destination=None, *, config=None, seed=3, **options):
        np.random.seed(seed)
        return lm.generate_listmode(descriptors, config or self.config, maps["calibration"], maps["cog_limits"],
                                    maps["doi_limits"], maps["pairs"], maps["regions"], maps["metadata"],
                                    destination or self.lm_parent / "job", **options)


# QC fixtures (petsys_manager_qc_check) --------------------------------------------------

# module, time position, direction, energy mean/sigma (a.u.), weight
QC_SPECS = [((1, 0), 2, +1, 100.0, 9.0, 30), ((1, 0), 4, -1, 90.0, 8.0, 30), ((1, 1), 3, +1, 110.0, 10.0, 20),
            ((2, 0), 3, +1, 80.0, 7.0, 15), ((2, 4), 1, +1, 95.0, 9.0, 0.4), ((2, 5), 2, +1, 100.0, 9.0, 6)]


class QCFixtures:
    """Seeded compact coincidence inputs for QC. ``TestCase`` mixin (after ``PrivateOutput``:
    root = output / test name) or standalone via ``QCFixtures.at(root)``."""

    @classmethod
    def at(cls, root):
        builder = cls()
        builder.setup_qc(root)
        return builder

    def setUp(self):
        super().setUp()
        self.setup_qc(self.output / self._testMethodName)

    def setup_qc(self, root):
        self.root = root
        (self.root / "maps").mkdir(parents=True)
        (self.root / "configs").mkdir()
        self.map_path = self.root / "maps/selected.yaml"
        self.write_map({1: [0, 0, 0], 2: [2, 0, 0]})
        self.config = self.processing({"en_min_ch": 0.2})
        self.mapping = self.config.mapping
        self.geometry = Geometry(self.mapping)
        self.results = self.root / "system_QC_results"
        self.results.mkdir()

    def write_map(self, modules):
        value = fixture_map()
        value["mod_feb_map"] = modules
        self.map_path.write_text(yaml.safe_dump(value), encoding="utf-8")

    def processing(self, extra, name="processing.yaml"):
        path = self.root / "configs" / name
        path.write_text(yaml.safe_dump({"map_file": "maps/selected.yaml", "min_ch": 4, **extra}), encoding="utf-8")
        return load_processing_config(path, processing_root=self.root, action="qc_analyze")

    def manual_side(self, module, position, direction, energies, *, unresolved=False, extra=(), t=1000):
        times = self.geometry.time[module]
        hits = [(t, 12.0, times[position][0]), (t + 1, 5.0, times[position + (3 * direction if unresolved
                                                                                  else direction)][0])]
        hits += [(t + 2 + i, energy, self.geometry.energy[module][i][0]) for i, energy in enumerate(energies)]
        for other, slot, energy in extra:
            hits.append((t + 9, energy, self.geometry.energy[other][slot][0]))
        return hits

    def manual_records(self):
        a = lambda **kw: self.manual_side((1, 0), 2, +1, [20.0] * 5, **kw)
        b = self.manual_side((2, 0), 3, +1, [16.0] * 5)
        return [
            (a(extra=[((1, 0), 5, 0.1)]), b),                                     # resolved; 0.1 a.u. hit cut
            (b, a()),                                                             # resolved, reversed
            (self.manual_side((1, 1), 3, +1, [22.0] * 5, unresolved=True), b),    # occupancy only
            (self.manual_side((1, 0), 2, +1, [30.0] * 3), b),                     # 3 energy channels
            (self.manual_side((1, 0), 2, +1, [30.0] * 3, extra=[((1, 1), 0, 0.5), ((1, 1), 1, 0.5)]), b),
        ]

    def side(self, rng, i, t0):
        module, position, direction, mean, sigma, _ = QC_SPECS[rng.choice(len(QC_SPECS), p=self.weights)]
        energy = max(rng.normal(mean, sigma), 1.0)
        kind = i % 53
        if kind == 7:
            energy = 300.0 + rng.random()                         # outside photopeak range / flood window
        hits = self.geometry.side(rng, module, position, direction, energy,
                                  energy_channels=3 if kind in (5, 9) else 5, unresolved=kind == 11, timestamp=t0)
        if kind == 9:
            other = (module[0], (module[1] + 1) % 16)
            hits += [(t0 + 9, 0.5, self.geometry.energy[other][s][0]) for s in (0, 1)]
        if kind in (13, 17, 19):                                  # one time channel: reference random slab
            hits = [hits[0]] + hits[2:]
        if kind == 23:
            hits.append((t0 + 11, 0.1, self.geometry.energy[module][7][0]))   # below en_min_ch
        return hits

    def records(self, count, seed=7):
        rng = np.random.default_rng(seed)
        weights = np.array([spec[-1] for spec in QC_SPECS], dtype=float)
        self.weights = weights / weights.sum()
        return [(self.side(rng, 2 * i, 1_000_000 + 997 * i), self.side(rng, 2 * i + 1, 1_000_100 + 997 * i))
                for i in range(count)]

    def compact(self, name, records):
        path = self.root / name
        with open(path, "xb") as out:
            for record in records:
                out.write(compact_record(record))
        return InputDescriptor(path, "compact", "coincidence")

    def inputs(self, count=1500, names=("qc_with_source_coincCompact_1.ldat", "qc_with_source_coincCompact_2.ldat")):
        return [self.compact(name, self.records(count, seed=7 + i)) for i, name in enumerate(names)]

    def qc_run(self, descriptors, config=None, *, seed=5, **options):
        random.seed(seed)
        return qc.run_qc(descriptors, config or self.config, **options)


# Processing CLI fixtures (petsys_manager_cli_check) -------------------------------------

LITERAL = "cli ; & $HOME %PATH% [x] (y) 'q' #é"
ACTIONS = {"calibrate": Action.CALIBRATE, "listmode": Action.LISTMODE, "qc": Action.QC_ANALYZE}


def entry(descriptor):
    return {"path": str(descriptor.path), "format": descriptor.format.value,
            "population": descriptor.population.value}


class CLIFixtures(PrivateOutput):
    """``TestCase`` mixin: requests for the three processing actions on fixtures whose every
    path contains spaces and shell metacharacters, and launchers for ``src.cornell.cli``."""

    fixture_prefix = "petsys-manager-cli-"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        profile_root = cls.output / "profile fixture"
        profile = profile_fixture(profile_root)
        report = preflight(profile, Action.LISTMODE, None,
                           (InputDescriptor(Path("private/input.ldat"), "compact", "coincidence"),),
                           repo_root=profile_root, probe=FixtureProbe())
        assert report.ready, report.issues
        cls.settings = report.settings

    def setUp(self):
        super().setUp()
        self.work = self.output / self._testMethodName.replace("test_cli_", "")[:12] / LITERAL
        self.work.mkdir(parents=True)
        self.count = 0
        self.helpers = 0

    def make(self, fixtures):
        """A ``ListmodeFixtures``/``QCFixtures`` builder rooted at a short folder under ``self.work``."""
        self.helpers += 1
        return fixtures.at(self.work / f"h{self.helpers}")

    def calibration_request(self, *, records=None, plot=True, positions=5, data_format="fixed",
                            names=("cal_run_1.ldat", "cal_run_2.ldat")):
        """T20 fixtures: seeded Cornell sides (P1_SPECS for positions 1, P3_SPECS otherwise) in two files."""
        self.helpers += 1
        fx = CalibrationFixtures(self.work / f"cal fixtures {self.helpers}")
        sides = sides_for(fx.geometry, P1_SPECS if positions == 1 else P3_SPECS, seed=5, depth_spread=1.0)
        records_all = pairs(sides)[:records] if records else pairs(sides)
        half = len(records_all) // 2
        descriptors = [fx.write(name, part, DataFormat(data_format))
                       for name, part in zip(names, (records_all[:half], records_all[half:]))]
        limits = fx.limits()
        out = self.work / f"encal {self.helpers} ; & $x"
        out.mkdir()
        request = {"schema_version": 1, "action": "calibrate", "processing_root": str(fx.root),
                   "processing_config": str(fx.config_path), "inputs": [entry(d) for d in descriptors],
                   "files": {"cog_limits": str(limits.path) if positions > 1 else None},
                   "options": {"positions": positions, "event_limit": 10_000_000, "limit_mode": "reference",
                               "target_per_key": None, "memory_budget_mb": None, "workers": 1,
                               "batch_records": 5000},
                   "outputs": {"encal": str(out / "pos ; & 5regions [x].encal"),
                               "sidecar": str(out / "pos ; & 5regions [x].encal.json"),
                               "status": str(out / "pos ; & 5regions [x]_status.txt"),
                               "plot": str(out / "pos ; & 5regions [x].png") if plot else None}}
        return fx, descriptors, limits, request

    def listmode_request(self, *, count=1500, debug=True):
        h = self.make(ListmodeFixtures)
        descriptors, maps = h.inputs(count=count, random_slabs=False)
        request = {"schema_version": 1, "action": "listmode", "processing_root": str(h.root),
                   "processing_config": str(h.root / "configs/processing.yaml"),
                   "inputs": [entry(d) for d in descriptors],
                   "files": {"calibration": str(maps["calibration"].path), "calibration_sidecar": None,
                             "cog_limits": str(maps["cog_limits"].path), "doi_limits": str(maps["doi_limits"].path),
                             "pair_map": str(maps["pairs"].path), "region_map": str(maps["regions"].path)},
                   "options": {"num_regions": 5, "region_boundaries": None,
                               "metadata": {name: getattr(METADATA, name) for name in METADATA.__dataclass_fields__},
                               "batch_records": 1000, "debug": debug, "resume": False, "hit_limit": None,
                               "lm_seed": None, "workers": 1, "in_place": False},
                   "outputs": {"directory": str(h.lm_parent / "lm job ; & $x [1]")}}
        return h, descriptors, maps, request

    def qc_request(self, *, count=1500, plots=True, slabs=True, names=None):
        h = self.make(QCFixtures)
        descriptors = h.inputs(count=count, **({"names": names} if names else {}))
        request = {"schema_version": 1, "action": "qc", "processing_root": str(h.root),
                   "processing_config": str(h.root / "configs/processing.yaml"),
                   "inputs": [entry(d) for d in descriptors], "files": {},
                   "options": {"plots": plots, "slabs": slabs, "source_mode": "with", "acquisition_time_s": 60,
                               "pair_limit": qc.PAIR_LIMIT, "in_place": False, "report_title": None,
                               "qc_seed": None, "workers": 1},
                   "outputs": {"directory": str(h.results / "QC ; & $x [run]")}}
        return h, descriptors, request

    def write_request(self, request, name=None):
        self.count += 1
        path = self.work / f"request {name or self.count} ; & $x.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        return path, self.work / f"result {name or self.count} ; & $x.json"

    def launch(self, action, request_path, result_path):
        spec = build_internal(self.settings, Identity("cli run", "stage", "attempt-1"), ACTIONS[action],
                              request_path, result_path, checkout_root=REPO)
        self.assertEqual(spec.argv[:5], (sys.executable, "-u", "-m", "src.cornell.cli", action))
        self.assertEqual(spec.argv[5:], ("--request", str(request_path), "--result", str(result_path)))
        env = dict(spec.environment)
        for name in ("MPLBACKEND", "DISPLAY", "WAYLAND_DISPLAY"):
            env.pop(name, None)
        return subprocess.run(list(spec.argv), cwd=spec.cwd, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=1800)

    def run_cli(self, action, request, name=None):
        request_path, result_path = self.write_request(request, name)
        proc = self.launch(action, request_path, result_path)
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        return proc, result, result_path

    def in_process(self, action, request, *, cancel_event=None):
        request_path, result_path = self.write_request(request)
        stdout, stderr = io.StringIO(), io.StringIO()
        code = cli.main([action, "--request", str(request_path), "--result", str(result_path)],
                        cancel_event=cancel_event, stdout=stdout, stderr=stderr)
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        return code, result, stdout.getvalue(), stderr.getvalue()

    def assertSucceeded(self, proc, result, result_path):
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["errors"], [])
        self.assertEqual(cli.read_result(result_path), result)
        for item in result["outputs"]:
            path = Path(item["path"])
            self.assertTrue(path.is_absolute() and path.is_file(), path)
            self.assertEqual(path.stat().st_size, item["size_bytes"])
        events = [cli.parse_event(line) for line in proc.stdout.splitlines()]
        self.assertTrue(all(events), proc.stdout[-2000:])
        self.assertEqual([e["sequence"] for e in events], list(range(1, len(events) + 1)))
        kinds = [e["kind"] for e in events]
        self.assertEqual((kinds[0], kinds[-1]), ("started", "finished"))
        self.assertIsInstance(events[0]["inputs"], int)              # a count: a path list would be split
        self.assertLess(max(len(line) for line in proc.stdout.splitlines()), 4096)   # the runner's line bound
        self.assertIn("output", kinds)
        progress = [e for e in events if e["kind"] == "progress"]
        # A worker's byte tick (spec 005 FR-1) has bytes_read but no record count.
        self.assertTrue(progress and all(e["records_read"] > 0 if e["records_read"] is not None else "bytes_read" in e
                                         for e in progress if e.get("phase") != "fits"))
        fits = [e for e in progress if e.get("phase") == "fits"]                 # T25.4: keys, not records
        self.assertTrue(all(isinstance(e["keys_done"], int) and isinstance(e["keys_total"], int) for e in fits))
        self.assertEqual(events[-1]["result"], str(result_path))
        self.assertEqual(events[-1]["status"], "succeeded")
        return events

    def assertFailedClosed(self, proc_or_code, result, *, code, kind, absent=()):
        returncode = getattr(proc_or_code, "returncode", proc_or_code)
        self.assertEqual(returncode, code)
        self.assertIsNotNone(result)
        self.assertNotEqual(result["status"], "succeeded")
        self.assertEqual(result["outputs"], [])
        self.assertEqual(result["summary"], None)
        self.assertEqual(result["error_kind"], kind)
        self.assertTrue(result["errors"])
        for path in absent:
            self.assertFalse(os.path.lexists(path), path)


# Workflow tool backend (petsys_manager_workflow_check) ----------------------------------

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_output(path, content, kind):
    path = Path(path)
    path.write_bytes(content)
    return {"kind": kind, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


class ToolWorld:
    """Check-only backend. ``faults``: stage id -> fault name; ``sources``: LDAT bytes the converter copies.
    ``check`` provides ``root`` and, for stop faults, ``coordinator``. ``converter_interval_s``: the converter
    writes one split per interval instead of all at launch."""

    def __init__(self, check, *, faults=None, sources=(), real_cli=False, converter_interval_s=None):
        self.check = check
        self.faults = dict(faults or {})
        self.sources = tuple(sources)
        self.real_cli = real_cli
        self.converter_interval_s = converter_interval_s
        self.launched = []

    def stop_active(self):
        self.check.coordinator._active.stop()

    def launch(self, command):
        argv = command.argv
        stage = command.identity.stage_id
        cli = argv[1:4] == ("-u", "-m", "src.cornell.cli")
        tool = f"cli:{argv[4]}" if cli else Path(argv[0]).name
        self.launched.append((stage, tool, argv))
        if tool == "set_bias":
            return FakeChild(0, stdout=b"")
        fault = self.faults.get(stage)
        if fault == "spawn":
            raise OSError("injected spawn failure")
        if fault == "stop":
            Timer(0.1, self.stop_active).start()
            return FakeChild(None, stdout=b"")
        if fault == "nonzero":
            return FakeChild(7, stdout=b"", stderr=b"injected nonzero exit\n")
        if fault not in ("missing", "stale"):
            if cli and self.real_cli and fault is None:
                return DirectDummyChild(subprocess.Popen(argv, shell=False, cwd=command.cwd,
                                                         env=dict(command.environment), stdin=subprocess.DEVNULL,
                                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE))
            if tool.startswith("convert_raw_to_") and self.converter_interval_s and fault is None:
                return SplitWriterChild(self.converter_writes(argv, fault), self.converter_interval_s)
            stdout = self.produce(tool, argv, fault)
            if stdout:
                return FakeChild(0, stdout=stdout)
        elif cli and fault == "stale":
            self.foreign_result(argv)
        return FakeChild(0, stdout=b"")

    def produce(self, tool, argv, fault):
        if tool == "acquire_sipm_data":
            prefix = argv[argv.index("-o") + 1]
            for suffix in (".rawf", ".idxf"):
                Path(prefix + suffix).write_bytes(b"" if fault == "invalid" else b"\x01" * 64)
        elif tool.startswith("convert_raw_to_"):
            for write in self.converter_writes(argv, fault):
                write()
        else:
            return self.fake_cli(tool[4:], argv, fault)

    def converter_writes(self, argv, fault):
        """One callable per split: writes the split and the converter's index."""
        prefix = argv[argv.index("-o") + 1]
        names = ([f"{prefix}_{i}.ldat" for i in range(1, len(self.sources) + 1)] + [f"{prefix}_9.ldat"]
                 if "--splitTime" in argv else [f"{prefix}.ldat"])

        def write(index, name):
            content = self.sources[min(index, len(self.sources) - 1)].read_bytes() if index < len(self.sources) \
                else b""
            if fault == "invalid":
                content = content[:-5]
            Path(name).write_bytes(content)
            Path(name[:-5] + ".lidx").write_bytes(b"\x02" * 8)      # the converter's index (T34)
        return [lambda index=index, name=name: write(index, name) for index, name in enumerate(names)]

    def fake_cli(self, action, argv, fault):
        """Writes the outputs and result; returns its stdout: one byte progress event (spec 005 T4)."""
        request_path, result_path = Path(argv[6]), Path(argv[8])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        out, outputs = request["outputs"], []
        if action == "calibrate":
            outputs += [write_output(out["encal"], b"# Position-dependent energy calibration\n", "encal"),
                        write_output(out["sidecar"], b"{}\n", "calibration_sidecar"),
                        write_output(out["status"], b"ID(time_ch, slab, region)\tstatus\n", "calibration_status"),
                        write_output(out["plot"], b"\x89PNG fake\n", "calibration_plot")]
        elif action == "listmode":
            directory = Path(out["directory"])
            if not request["options"]["in_place"]:      # T29: in place = the existing stage folder
                directory.mkdir()
            outputs += [write_output(directory / "fake.lm", b"\0" * 176, "listmode"),
                        write_output(directory / "fake.lm.json", b"{}\n", "listmode_provenance"),
                        write_output(directory / "lm-job.json", b"{}\n", "listmode_job")]
        else:
            directory = Path(out["directory"])
            if not request["options"]["in_place"]:      # T29: in place = the existing stage folder
                directory.mkdir()
            outputs += [write_output(directory / "missing_channels_report.pdf", b"%PDF fake\n", "qc_report"),
                        write_output(directory / "qc_summary.json", b"{}\n", "qc_summary")]
        if fault == "invalid":
            outputs[0]["sha256"] = "0" * 64
        if fault == "outside":
            outputs.append(write_output(self.check.root / f"outside-{os.getpid()}-{len(self.launched)}.bin", b"x",
                                        "qc_report"))
        summary = {"findings": {"minimodules_without_hits": 3, "missing_time_channels": 1}} if action == "qc" else {}
        result = {"schema_version": 1, "action": action, "status": "succeeded", "exit_code": 0,
                  "request": {"path": str(request_path), "sha256": sha256(request_path)},
                  "outputs": outputs, "summary": summary, "errors": []}
        result_path.write_text(json.dumps(result), encoding="utf-8")
        event = {"sequence": 1, "action": action, "kind": "progress", "file_index": 0, "files": 2, "path": "a.ldat",
                 "records_read": 10, "phase": "read", "phases": ["read", "pass 2", "fits"], "bytes_read": 1200,
                 "bytes_total": 4800}
        return (cli.EVENT_PREFIX + json.dumps(event) + "\n").encode("utf-8")

    def foreign_result(self, argv):
        """An old successful result of another request where this stage expects its own."""
        request_path, result_path = Path(argv[6]), Path(argv[8])
        old = self.check.root / f"old-output-{len(self.launched)}.bin"
        result = {"schema_version": 1, "action": argv[4], "status": "succeeded", "exit_code": 0,
                  "request": {"path": str(request_path), "sha256": "f" * 64},
                  "outputs": [write_output(old, b"old", "encal")], "summary": {}, "errors": []}
        result_path.write_text(json.dumps(result), encoding="utf-8")


# Spec 005: recorded runs for offers, Recent Runs and run-folder picking ------------------------------------

COMPACT = (DataFormat.COMPACT, Population.COINCIDENCE)


def record_stage(store, stage_id, files, status=ResultStatus.SUCCEEDED):
    """Finish one stage attempt with ``files``: (relative name, kind, bytes)."""
    attempt = store.reserve_attempt(stage_id, attempt_id="attempt-1")
    artifacts = []
    for name, kind, content in files:
        path = attempt.directory / name
        path.write_bytes(content)
        descriptor = InputDescriptor(path, *COMPACT, validated=True) if kind == "ldat" else None
        artifacts.append(Artifact(path, kind, descriptor))
    ok = status == ResultStatus.SUCCEEDED
    store.finish_attempt(attempt, CommandResult(attempt.identity, status, 0 if ok else 1, "done", artifacts,
                                                outputs_validated=ok))
    return attempt.directory


def recorded_run(destination, stages, results, name="run_2026-10-10_1200"):
    """A finished run: ``results`` maps each stage to its files, or to (files, status)."""
    destination.mkdir(parents=True, exist_ok=True)
    store = RunStore.reserve(destination, {"action": "test"}, name=name, stages=stages)
    status = ResultStatus.SUCCEEDED
    for stage_id in stages:
        files, outcome = results[stage_id] if isinstance(results[stage_id], tuple) else (results[stage_id],
                                                                                         ResultStatus.SUCCEEDED)
        record_stage(store, stage_id, files, outcome)
        if outcome != ResultStatus.SUCCEEDED:
            status = outcome
            break
    store.finish(status, "" if status == ResultStatus.SUCCEEDED else "stopped")
    return store.root


SPLITS = [(f"x_coincCompact_{n}.ldat", "ldat", bytes([n]) * (100 + n)) for n in (0, 1, 2)]
CALIBRATION = [("x.encal", "encal", b"encal"), ("x_status.tsv", "calibration_status", b"s"),
               ("x_plot.png", "calibration_plot", b"png")]
