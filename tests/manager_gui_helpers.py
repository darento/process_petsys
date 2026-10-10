"""Hidden-window bases and fakes for the PETsys Manager GUI tests (spec 003 T13-T16; spec 007 T15).

Moved from scripts/petsys_manager_gui_check.py. Kept apart from ``manager_helpers`` so that
module still imports without Tk.

- ``GUIBase``: withdrawn ``PETsysManager`` windows on fixture profiles (``world``) with a
  ``FixtureProbe``; every Tcl call is checked to be on the main thread (``TkGuard``) and any
  process launch by the shell fails the test. Skipped without a display.
- T14 fakes: ``Hardware`` drives the real ``DaqdService``, ``WorkflowCoordinator`` and
  ``AcquisitionService`` with fake children (``GatedChild``, ``Daemon``, ``Resources``).
- T15: ``Converter`` (fake ``convert_raw_to_*`` writing synthetic LDAT), ``encode``,
  ``synthetic_sides`` and ``ConversionBase``.
No hardware, PETsys tools or real machine settings.
"""

import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import tkinter
import unittest
from dataclasses import replace
from unittest.mock import patch

import customtkinter as ctk
import yaml

from exe_programs import petsys_manager_gui as gui
from manager_helpers import FakeChild, FakeDaemon, FakeResources, PrivateOutput, fixture_map
from src.cornell.inputs import load_processing_config
from src.petsys_manager import acquisition, runner, workflow
from src.petsys_manager.acquisition import DaqdPolicy, DaqdService
from src.petsys_manager.contracts import Action, DataFormat, Identity, Population, RunEvent
from src.petsys_manager.runner import CommandRunner, RunnerPolicy
from src.petsys_manager.session import ManagerSession
from src.petsys_manager.settings import (AcquisitionSafety, LMMetadata, MachineProfile, ProcessingLimits,
                                         SystemProbe, ToolCapabilities, save_profile)
from src.petsys_manager.workflow import WorkflowCoordinator, format_elapsed


PY = sys.executable
TOOLS = ("daqd", "init_system", "acquire_sipm_data", "set_bias", "convert_raw_to_coincidence",
         "convert_raw_to_group")
STAMP = re.compile(r"\d\d:\d\d:\d\d ")    # FR-1: the GUI log prefixes each line with HH:MM:SS


class FixtureProbe(SystemProbe):
    platform_name = "Linux"

    def __init__(self):
        self.gate = None  # threading.Event: block the next probe call until set
        self.blocked = threading.Event()
        self.threads = set()

    def _wait(self):
        self.threads.add(threading.current_thread().name)
        gate = self.gate
        if gate is not None:
            self.blocked.set()
            gate.wait(10)

    def file(self, path):
        self._wait()
        return path.is_file()

    def executable(self, path):
        self._wait()
        return path.is_file()  # fixture markers, never executed

    def device(self, path):
        self._wait()
        return path.is_file() and path.name.startswith("card")

    def writable_directory(self, path):
        self._wait()
        return path.is_dir()


class TkGuard:
    """Proxy for a root's Tcl interpreter: records calls and rejects any off the main thread."""

    def __init__(self, interpreter):
        self._tk = interpreter
        self.calls = 0
        self.violations = []

    def __getattr__(self, name):
        attribute = getattr(self._tk, name)
        if not callable(attribute):
            return attribute

        def guarded(*args, **kwargs):
            if threading.current_thread() is not threading.main_thread():
                self.violations.append((threading.current_thread().name, name, args[:2]))
                raise RuntimeError(f"Tk {name} called off the main thread")
            self.calls += 1
            return attribute(*args, **kwargs)
        return guarded


def world(root, *, shm="/dev/shm/fixture_marker_shm", safety=None):
    """Complete fixture checkout/profile: every path exists; contents are markers only."""
    if root.exists():
        shutil.rmtree(root)
    for directory in ("tools", "dev", "cfg", "maps", "data", "data2", "encal", "rep", "lm", "lim"):
        (root / directory).mkdir(parents=True)
    for tool in TOOLS:
        (root / "tools" / tool).write_text("marker", encoding="utf-8")
    for name in ("card0", "card1"):
        (root / "dev" / name).write_text("marker", encoding="utf-8")
    (root / "cfg" / "daq.ini").write_text("[marker]\n", encoding="utf-8")
    (root / "maps" / "map.yaml").write_text("marker: 1\n", encoding="utf-8")
    (root / "cfg" / "proc.yaml").write_text(
        "map_file: maps/map.yaml\nmin_ch: 1\nen_min_ch: 5\nenergy_range: [400, 650]\n", encoding="utf-8")
    for name in ("cog.txt", "doi.txt", "pairs.txt", "regions.txt", "cal.encal"):
        (root / "lim" / name).write_text("marker", encoding="utf-8")
    (root / "data" / "run ; & $x.rawf").write_bytes(b"\0" * 8)
    (root / "data" / "run ; & $x.idxf").write_bytes(b"\0" * 8)  # converters read the index too
    metadata = LMMetadata("F18", 60.0, 60.0, 50.0, 50.0, 8, 1, 2.0, 8, 8, "ps")
    return MachineProfile(
        petsys_folder=str(root / "tools"), ini_file=str(root / "cfg" / "daq.ini"),
        yaml_file=str(root / "cfg" / "proc.yaml"), data_dir=str(root / "data"),
        calibration_dir=str(root / "encal"), report_dir=str(root / "rep"), lm_dir=str(root / "lm"),
        cog_limits_file=str(root / "lim" / "cog.txt"), doi_limits_file=str(root / "lim" / "doi.txt"),
        calibration_file=str(root / "lim" / "cal.encal"), pair_map_file=str(root / "lim" / "pairs.txt"),
        region_map_file=str(root / "lim" / "regions.txt"),
        cards=(str(root / "dev" / "card0"), str(root / "dev" / "card1")),
        shared_memory_path=shm,
        safety=safety or AcquisitionSafety(startup_timeout_s=50.0, max_loss_percent=4.0, max_attempts=2),
        limits=ProcessingLimits(workers=2, log_tail_lines=1000),
        capabilities=ToolCapabilities(installed_version="fixture-1"),
        lm_metadata=metadata)


def pump(root, until=None, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        root.update()
        if until is not None and until():
            return True
        time.sleep(0.01)
    return until is None


def texts(widget):
    """Every label/button/checkbox text and entry value below a widget."""
    found = []
    for child in widget.winfo_children():
        for option in ("text",):
            try:
                found.append(str(child.cget(option)))
            except (ValueError, tkinter.TclError, AttributeError):
                pass
        if isinstance(child, ctk.CTkEntry):
            found.append(child.get())
        found.extend(texts(child))
    return found


def child(code, *, cwd, timeout=120):
    run = subprocess.run([PY, "-X", "utf8", "-c", code], cwd=cwd, capture_output=True, text=True,
                         encoding="utf-8", timeout=timeout)
    if run.returncode:
        raise AssertionError(f"child failed {run.returncode}:\n{run.stdout}\n{run.stderr}")
    return json.loads(run.stdout.strip().splitlines()[-1])


SETTLE_S = 15.0   # readiness wait: returns once settled; parallel full runs can take over 5 s


class GUIBase(PrivateOutput, unittest.TestCase):
    """Withdrawn Manager windows on fixture profiles; launching anything from the shell fails the test.
    Each subclass sets ``fixture_prefix``; without a display the class is skipped."""

    fixture_prefix = "pm-gui-"
    launches = []

    @classmethod
    def setUpClass(cls):
        try:
            tkinter.Tk().destroy()
        except tkinter.TclError as exc:
            raise unittest.SkipTest(f"gui: no display available ({exc})")
        super().setUpClass()

    def setUp(self):
        self.base = self.output / self._testMethodName[5:20]
        self.base.mkdir()
        GUIBase.launches = []
        record = lambda name: (lambda *a, **k: GUIBase.launches.append(name) or (_ for _ in ()).throw(
            AssertionError(f"{name} launched by the GUI shell")))
        self.patches = [patch.object(subprocess.Popen, "__init__", record("subprocess.Popen")),
                        patch.object(os, "system", record("os.system")),
                        patch.object(multiprocessing.Process, "start", record("multiprocessing")),
                        patch.object(runner.CommandRunner, "__init__", record("CommandRunner")),
                        patch.object(acquisition.DaqdService, "__init__", record("DaqdService")),
                        patch.object(acquisition.AcquisitionService, "__init__", record("AcquisitionService")),
                        patch.object(workflow.WorkflowCoordinator, "__init__", record("WorkflowCoordinator"))]
        if hasattr(os, "posix_spawn"):
            self.patches.append(patch.object(os, "posix_spawn", record("posix_spawn")))
        self.apps = []

    def tearDown(self):
        for app in self.apps:
            if not app.closed:
                app.shutdown_timeout_s = 30.0
                app.on_close()
                pump(app.root, lambda: app.closed, 40)
        self.assertEqual(GUIBase.launches, [])

    def open(self, profile_path, *, probe=None, repo_root=None, settle=True, **services):
        root = ctk.CTk()
        root.withdraw()
        guard = TkGuard(root.tk)
        root.tk = guard  # widgets created below inherit the guarded interpreter
        session = ManagerSession(profile_path, probe=probe or FixtureProbe(), repo_root=repo_root or self.base,
                                 **services)
        roots = []
        original = tkinter.Tk.__init__
        for item in self.patches:
            item.start()
        try:
            with patch.object(tkinter.Tk, "__init__", lambda *a, **k: roots.append(a) or original(*a, **k)):
                app = gui.PETsysManager(root, session)
        finally:
            for item in self.patches:
                item.stop()
        self.assertEqual(roots, [], "the shell must not create another Tk root")
        app.guard = guard
        finalize = app._finalize_close

        def closed():  # the test process keeps pumping other roots: drop customtkinter's pending jobs
            finalize()
            for job in root.tk.splitlist(root.tk.call("after", "info")):
                root.tk.call("after", "cancel", job)
        app._finalize_close = closed
        self.apps.append(app)
        if settle:
            self.settle(app)
        return app

    def settle(self, app, timeout=SETTLE_S):
        self.assertTrue(pump(app.root, lambda: app._check_id is None and app._awaiting is not None
                             and app.shown_generation == app._awaiting, timeout), "readiness never settled")

    def edit(self, app, variable, value):
        variable.set(value)
        self.settle(app)

    def reasons(self, app):
        return {key: label.cget("text") for key, label in app.readiness.items()}

    def log(self, app):
        return app.log_text.get("1.0", "end-1c")


def wait_threads(prefixes, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not [t for t in threading.enumerate() if t.name.startswith(prefixes) and t.is_alive()]:
            return True
        time.sleep(0.02)
    return False


# T14: DAQD / initialization / acquisition fakes ------------------------------------------------

FAST = RunnerPolicy(poll_interval_s=0.005, terminate_grace_s=0.3, reap_timeout_s=2.0, drain_timeout_s=0.5)
DAQD_POLICY = DaqdPolicy(startup_timeout_s=20.0, probe_interval_s=0.01, probe_timeout_s=0.2, terminate_grace_s=0.3)
SAFETY = AcquisitionSafety(startup_timeout_s=10.0, growth_window_s=0.3, poll_interval_s=0.05, min_growth_bytes=1,
                           max_attempts=3, retry_delay_s=0.0, terminate_grace_s=0.3)
SOCKET, SHM = "/tmp/d.sock", "/dev/shm/daqd_shm"


class RunPanelBase(GUIBase):
    """The run panel above the tabs, fed workflow events on a fake GUI clock (spec 005)."""

    def begin(self, action=Action.CALIBRATE, stages=("calibration",), options=None, run_root=None):
        app = self.open(save_profile(world(self.base / "w"), self.base / "profile.yaml"), repo_root=self.base / "w")
        self.now = 100.0
        app.clock = lambda: self.now
        app.session.start_workflow = lambda *args, **kwargs: 1     # nothing launches; events are fed below
        app._start(app.session.profile, action, options, app.cal_status, "Starting...")
        self.feed(app, "workflow_started", "workflow", run_root=str(run_root or self.base / "run"), stages=list(stages))
        return app

    def feed(self, app, kind, stage, **payload):
        app._workflow_event(RunEvent(Identity("run-1", stage, "attempt-1"), 0, kind, "", payload))

    def at(self, app, now):
        """Advance the GUI clock and wait for the panel to show that instant."""
        self.now = now
        self.assertTrue(pump(app.root, lambda: f"Elapsed {format_elapsed(now - 100.0)}" in app.run_panel.timing.cget("text")),
                        app.run_panel.timing.cget("text"))
        panel = app.run_panel
        counter = panel.counter.cget("text")
        self.assertFalse([word for word in ("event", "pair", "single") if word in counter.lower()], counter)
        return panel.bar.cget("mode"), counter, panel.timing.cget("text")


class GatedChild(FakeChild):
    """Runs until its gate opens, then exits with ``final``."""

    def __init__(self, gate, final=0):
        super().__init__(None, stdout=b"")
        self.gate, self.final = gate, final

    def poll(self):
        if self.code is None and self.gate.is_set():
            self.code = self.final
        return self.code

    def group_alive(self):
        return self.poll() is None or self.descendants

    def wait(self, timeout):
        self.poll()
        return super().wait(timeout)


class Daemon(FakeDaemon):
    def __init__(self, hardware):
        super().__init__()
        self.hardware = hardware

    def terminate(self):
        self.hardware.order.append("daqd TERM")
        super().terminate()


class Resources(FakeResources):
    """In-memory socket/shm table (no delete operation); ``serve`` gates the protocol reply."""

    def __init__(self, existing=()):
        super().__init__(existing)
        self.serve = threading.Event()
        self.serve.set()

    def query(self, socket_path, timeout, abort):
        if not self.serve.is_set():
            raise OSError("No reply from DAQD yet")
        return super().query(socket_path, timeout, abort)


class Hardware:
    """Fake children for the REAL DaqdService, WorkflowCoordinator and AcquisitionService.

    ``acquire`` lists per-attempt behaviours: ok, nodata, fail, block (runs until TERM)
    grow (writes the .rawf for ~0.8 s, then stalls until TERM) or pulse (writes ~0.5 s, pauses ~0.6 s,
    writes ~0.9 s, then stalls until TERM). ``bias``: ok or fail;
    ``bias_gate`` holds set_bias until set. No hardware, sockets or shared memory.
    """

    def __init__(self, *, acquire=("ok",), existing=()):
        self.acquire = list(acquire)
        self.init_code = 0
        self.init_stdout = b""
        self.daqd_stderr = b""      # the daemon's own output (T35)
        self.bias = "ok"
        self.bias_gate = threading.Event()
        self.bias_gate.set()
        self.resources = Resources(existing)
        self.order, self.children = [], []
        self.daemon = None
        self.lock = threading.Lock()

    def launch_daqd(self, command):
        with self.lock:
            self.order.append("daqd")
            self.daemon = Daemon(self)
            self.daemon.stderr = io.BytesIO(self.daqd_stderr)
            self.children.append(self.daemon)
            self.resources.daemon = self.daemon
            self.resources.appear(type("C", (), {"socket_path": SOCKET, "shared_memory_path": SHM})())
            return self.daemon

    def launch(self, command):
        tool = Path(command.argv[0]).name
        with self.lock:
            self.order.append(tool)
            if tool == "init_system":
                child = FakeChild(self.init_code, stdout=self.init_stdout)
            elif tool == "set_bias":
                child = GatedChild(self.bias_gate, 0 if self.bias == "ok" else 1)
            elif tool == "acquire_sipm_data":
                child = self._acquisition(command.argv, self.acquire.pop(0) if len(self.acquire) > 1
                                          else self.acquire[0])
            else:
                raise OSError(f"unexpected tool {tool}")
            self.children.append(child)
            return child

    def _acquisition(self, argv, behaviour):
        prefix = argv[argv.index("-o") + 1]
        if behaviour == "ok":
            for suffix in (".rawf", ".idxf"):
                Path(prefix + suffix).write_bytes(b"\x01" * 64)
            return FakeChild(0, stdout=b"")
        if behaviour == "nodata":
            return FakeChild(0, stdout=b"")
        if behaviour == "fail":
            return FakeChild(3, stdout=b"", stderr=b"injected acquisition failure\n")
        child = FakeChild(None, stdout=b"")
        if behaviour == "block":
            Path(prefix + ".rawf").write_bytes(b"\x01" * 64)
        else:
            def grow():
                with open(prefix + ".rawf", "ab") as out:
                    for step in range(16 if behaviour == "grow" else 40):
                        if child.code is not None:
                            return
                        if behaviour == "grow" or not 10 <= step < 22:
                            out.write(b"\x01" * 100_000)
                            out.flush()
                        time.sleep(0.05)
            threading.Thread(target=grow, daemon=True).start()
        return child

    def acquisitions(self):
        return [item for item in self.order if item == "acquire_sipm_data"]

    def daqd_factory(self, **sinks):
        backend = type("B", (), {"launch": lambda _, command: self.launch_daqd(command)})()
        return DaqdService(runner=CommandRunner(policy=FAST, backend=backend),
                           init_runner=CommandRunner(policy=FAST, backend=self), resources=self.resources,
                           policy=DAQD_POLICY, **sinks)

    def coordinator_factory(self, **sinks):
        return WorkflowCoordinator(backend=self, policy=FAST, **sinks)


# T15: conversion fakes and the conversion window base --------------------------------------------

HIT = struct.Struct("<qfi")
PADDING = HIT.pack(0, 0.0, -1)
FIXED_COINC, FIXED_GROUP = (DataFormat.FIXED, Population.COINCIDENCE), (DataFormat.FIXED, Population.GROUP)
COMPACT_COINC = (DataFormat.COMPACT, Population.COINCIDENCE)


def synthetic_sides(channels, count, *, hits=2):
    """Deterministic detector sides on mapped channels: (timestamp, energy, channel) hits."""
    return [[(1000 + index, 10.0 + slot, channels[(index + slot) % len(channels)]) for slot in range(hits)]
            for index in range(count)]


def encode(path, sides, data_format, population, *, hit_limit=16):
    """Hand-encoded LDAT: fixed (4-byte hit limit, padded slots) or compact; coincidence pairs consecutive sides."""
    records = [[side] for side in sides] if population == Population.GROUP else \
        [sides[index:index + 2] for index in range(0, len(sides) - 1, 2)]
    with open(path, "xb") as out:
        if data_format == DataFormat.FIXED:
            out.write(struct.pack("<i", hit_limit))
        for record in records:
            out.write(bytes(len(side) for side in record))
            for side in record:
                out.write(b"".join(HIT.pack(*hit) for hit in side))
                if data_format == DataFormat.FIXED:
                    out.write(PADDING * (hit_limit - len(side)))
    return len(records)


class Converter:
    """Fake convert_raw_to_* children for the REAL WorkflowCoordinator/CommandRunner.

    Writes synthetic LDAT at the exact ``-o`` prefix in the requested format/population
    and hit limit. With --splitTime: splits ``first_split`` and the next, plus one empty
    split. ``behaviour``: ok, block (runs until TERM), wrong (fixed bytes under the compact
    request), fail (exit 4).
    """

    def __init__(self, channels, *, first_split=3, records=40):
        self.channels, self.first_split, self.records = channels, first_split, records
        self.behaviour = "ok"
        self.launched, self.children = [], []
        self.lock = threading.Lock()

    def launch(self, command):
        argv = command.argv
        tool = Path(argv[0]).name
        with self.lock:
            self.launched.append(argv)
            if tool not in ("convert_raw_to_coincidence", "convert_raw_to_group"):
                raise OSError(f"unexpected tool {tool}")
            child = self._run(argv, tool)
            self.children.append(child)
            return child

    def _run(self, argv, tool):
        if self.behaviour == "fail":
            return FakeChild(4, stdout=b"", stderr=b"injected converter failure\n")
        if self.behaviour == "block":
            return FakeChild(None, stdout=b"")
        prefix = argv[argv.index("-o") + 1]
        data_format = DataFormat.FIXED if "--writeBinaryFixed" in argv else DataFormat.COMPACT
        population = Population.GROUP if tool == "convert_raw_to_group" else Population.COINCIDENCE
        hit_limit = int(argv[argv.index("--writeMultipleHits") + 1])
        if self.behaviour == "wrong":    # fixed bytes under the compact request
            data_format = DataFormat.FIXED
        if "--splitTime" in argv:
            names = [f"{prefix}_{self.first_split + offset}.ldat" for offset in range(2)]
            empty = f"{prefix}_{self.first_split + 2}.ldat"
        else:
            names, empty = [f"{prefix}.ldat"], None
        for offset, name in enumerate(names):
            encode(name, synthetic_sides(self.channels, self.records * 2 + offset * 4), data_format, population,
                   hit_limit=hit_limit)
            Path(name[:-5] + ".lidx").write_bytes(b"\x02" * 8)          # the converter's index (T34)
        if empty:
            Path(empty).write_bytes(b"")
            Path(empty[:-5] + ".lidx").write_bytes(b"")
        return FakeChild(0, stdout=b"converted\n")


class ConversionBase(GUIBase):
    def conversion_world(self):
        root = self.base / "w"
        profile = world(root, shm=SHM)
        (root / "maps" / "map.yaml").write_text(yaml.safe_dump(fixture_map()), encoding="utf-8")
        mapping = load_processing_config(root / "cfg" / "proc.yaml", processing_root=root,
                                         action="calibrate").mapping
        self.channels = sorted(mapping.modules)
        return root, profile

    def converter_app(self, converter=None, **profile_changes):
        root, profile = self.conversion_world()
        self.converter = converter or Converter(self.channels)
        hardware = Hardware()
        app = self.open(save_profile(replace(profile, **profile_changes), self.base / "p.yaml"), repo_root=root,
                        daqd_factory=hardware.daqd_factory,
                        coordinator_factory=lambda **sinks: WorkflowCoordinator(backend=self.converter, policy=FAST,
                                                                                **sinks))
        app.prompts = []
        app.ask_yes_no = lambda title, message: app.prompts.append(message) or app.answer
        app.answer = True
        return root, app

    def raw(self, root, name):
        path = root / "data" / name
        path.write_bytes(b"\x01" * 64)
        path.with_suffix(".idxf").write_bytes(b"0\t64\t0\t10\t0.0\t0.0\n")
        return path

    def state(self, app, key):
        return app.buttons[key].cget("state")

    def wait(self, app, predicate, timeout=10.0, message="condition not reached"):
        self.assertTrue(pump(app.root, predicate, timeout), message)

    def run_conversion(self, app, key):
        self.assertEqual(self.state(app, key), "normal", self.reasons(app)[key])
        app.buttons[key].invoke()
        self.assertIsNotNone(app._token)
        self.wait(app, lambda: app._token is None, 20, "conversion did not finish")
        return app.convert_status.cget("text")

    def reason(self, app, key):
        return self.reasons(app)[key]

    def settle_edit(self, app, variable, value):
        variable.set(value)
        self.wait(app, lambda: app._check_id is None and app._awaiting is not None
                  and app.shown_generation == app._awaiting, SETTLE_S, "readiness never settled")

