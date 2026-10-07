"""DAQD ownership, readiness and initialization (spec 003 T6, T35; spec 007 T5).

Moved from scripts/petsys_manager_daqd_check.py. No PETsys tools, hardware, sockets or
shared memory: fake daemon children and an in-memory resource table.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import time
import unittest

import pytest

from helpers import REPO
from manager_helpers import (FakeBackend, FakeChild, FakeDaemon, FakeResources, FixtureProbe, PrivateOutput,
                             profile_fixture)
from src.petsys_manager.acquisition import (DaqdConfig, DaqdPolicy, DaqdResources, DaqdService,
                                            DaqdState)
from src.petsys_manager.contracts import Action
from src.petsys_manager.runner import CommandRunner, RunnerPolicy
from src.petsys_manager.settings import preflight

FAST = RunnerPolicy(poll_interval_s=0.005, terminate_grace_s=0.3, reap_timeout_s=1.0, drain_timeout_s=0.5)
POLICY = DaqdPolicy(startup_timeout_s=1.0, probe_interval_s=0.01, probe_timeout_s=0.05, terminate_grace_s=0.3)


class FreshChildBackend(FakeBackend):
    """A new fake child per launch (the runner closes each child's pipes)."""

    def __init__(self, make_child):
        super().__init__(make_child())
        self.make_child = make_child

    def launch(self, command):
        child = super().launch(command)
        self.child = self.make_child()
        return child


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.mark.fr("003-FR-5", "003-FR-6", "003-FR-7", "003-FR-16")  # spec 003 T6
class DaqdChecks(PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-daqd-"

    def setUp(self):
        self.root = self.output / self._testMethodName
        self.profile = profile_fixture(self.root)
        self.settings = self.preflight(Action.INITIALIZE)
        self.statuses = []

    def preflight(self, action, profile=None):
        report = preflight(profile or self.profile, action, repo_root=self.root, probe=FixtureProbe())
        self.assertTrue(report.ready, report.issues)
        return report.settings

    def service(self, *, daemon=None, resources=None, init_child=None, launch_error=None, appear=True,
                **sinks):
        self.daemon = FakeDaemon() if daemon is None else daemon
        self.resources = FakeResources() if resources is None else resources
        self.resources.daemon = self.daemon
        backend = FakeBackend(self.daemon, error=launch_error)
        original = backend.launch
        def launch(command):
            child = original(command)
            if appear:
                self.resources.appear(service.daqd_config(self.settings))
            return child
        backend.launch = launch
        self.backend = backend
        self.init_backend = FreshChildBackend(init_child or (lambda: FakeChild(0)))
        service = DaqdService(runner=CommandRunner(policy=FAST, backend=backend),
                              init_runner=CommandRunner(policy=FAST, backend=self.init_backend),
                              resources=self.resources, policy=POLICY, status_sink=self.statuses.append,
                              **sinks)
        return service

    def ready(self, service):
        status = service.start(self.settings)
        self.assertEqual(status.state, DaqdState.STARTING, status.message)
        self.assertTrue(wait_for(lambda: service.status().state == DaqdState.READY), service.status())
        return service

    def assertStopped(self, service, state):
        self.assertTrue(wait_for(lambda: service.status().state == state), service.status())

    def test_daqd_existing_socket_blocks_start_without_deletion(self):
        config = self.service().daqd_config(self.settings)
        service = self.service(resources=FakeResources({config.socket_path}))
        status = service.start(self.settings)
        self.assertEqual(status.state, DaqdState.OFF)
        self.assertIn(config.socket_path, status.message)
        self.assertEqual(status.stale_resources, (config.socket_path,))
        self.assertEqual(self.backend.launched, [])
        self.assertEqual(self.resources.paths, {config.socket_path})

    def test_daqd_existing_shared_memory_blocks_start(self):
        config = self.service().daqd_config(self.settings)
        service = self.service(resources=FakeResources({config.shared_memory_path}))
        status = service.start(self.settings)
        self.assertEqual(status.state, DaqdState.OFF)
        self.assertEqual(status.stale_resources, (config.shared_memory_path,))
        self.assertEqual(self.backend.launched, [])

    def test_daqd_real_resource_probe_is_read_only(self):
        directory = self.root / "resources"
        directory.mkdir()
        sock, shm = directory / "d.sock", directory / "daqd_shm"
        sock.write_bytes(b"stale socket")
        shm.write_bytes(b"foreign shared memory")
        config = DaqdConfig(("daqd",), str(directory), str(sock), str(shm))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (sock, shm)}
        self.assertEqual(DaqdResources().existing(config), (str(sock), str(shm)))
        self.assertFalse(DaqdResources().is_socket(str(sock)))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (sock, shm)})
        self.assertEqual(config.expected_shm_name, "/daqd_shm")

    def test_daqd_launch_error_fails_and_blocks_initialization(self):
        service = self.service(launch_error=OSError("no daqd binary"))
        service.start(self.settings)
        self.assertStopped(service, DaqdState.FAILED)
        self.assertIn("launch_error", service.status().message)
        outcome = service.initialize(self.settings)
        self.assertFalse(outcome.initialized)
        self.assertEqual(self.init_backend.launched, [])

    def test_daqd_exit_during_startup_fails(self):
        service = self.service(resources=FakeResources(serve_after=None))
        service.start(self.settings)
        self.assertTrue(wait_for(lambda: service.status().pid == self.daemon.pid))
        self.daemon.die(255)
        self.assertStopped(service, DaqdState.FAILED)
        status = service.status()
        self.assertIn("exit code 255", status.message)
        self.assertFalse(status.initialized)
        self.assertIsNone(status.pid)

    def test_daqd_socket_without_protocol_reply_is_not_ready(self):
        service = self.service(resources=FakeResources(serve_after=None))
        service.start(self.settings)
        self.assertTrue(wait_for(lambda: self.resources.queries >= 3))
        self.assertEqual(service.status().state, DaqdState.STARTING)  # socket file exists, no reply
        self.assertFalse(service.initialize(self.settings).initialized)
        self.assertStopped(service, DaqdState.FAILED)
        self.assertIn("not ready after", service.status().message)
        self.assertEqual(self.daemon.signals[0], "TERM")

    def test_daqd_ready_after_protocol_reply(self):
        service = self.ready(self.service(resources=FakeResources(serve_after=3)))
        self.assertGreaterEqual(self.resources.queries, 4)
        status = service.status()
        self.assertEqual((status.pid, status.initialized), (self.daemon.pid, False))
        self.assertIn("answered the shared-memory query", status.message)
        self.assertEqual(self.daemon.signals, [])

    def test_daqd_wrong_shared_memory_or_foreign_peer_fails(self):
        for resources, text in ((FakeResources(name="/other_shm"), "reported shared memory"),
                                (FakeResources(peer=999), "served by pid 999")):
            service = self.service(resources=resources)
            service.start(self.settings)
            self.assertStopped(service, DaqdState.FAILED)
            self.assertIn(text, service.status().message)
            self.assertEqual(self.daemon.signals[0], "TERM")

    def test_daqd_initialize_requires_ready_owned_daemon(self):
        service = self.service()
        self.assertIn("off", service.initialize(self.settings).message)
        self.assertEqual(service.acquisition_ready(self.settings), (False, "DAQD is off"))
        self.assertEqual(self.init_backend.launched, [])

    def test_daqd_failed_initialization_never_unlocks_acquisition(self):
        service = self.ready(self.service(init_child=lambda: FakeChild(1)))
        outcome = service.initialize(self.settings)
        self.assertFalse(outcome.initialized)
        self.assertEqual(outcome.result.exit_code, 1)
        self.assertFalse(service.status().initialized)
        self.assertFalse(service.acquisition_ready(self.settings)[0])

    def test_daqd_successful_initialization_unlocks_acquisition(self):
        service = self.ready(self.service())
        outcome = service.initialize(self.settings)
        self.assertTrue(outcome.initialized, outcome.message)
        self.assertEqual(len(self.init_backend.launched), 1)
        self.assertEqual(self.init_backend.launched[0].argv, (str(self.root / "tools/init_system"),))
        self.assertEqual(service.acquisition_ready(self.settings), (True, "Initialized"))
        self.assertTrue(service.status().initialized)

    @pytest.mark.fr("003-FR-1")
    def test_daqd_daemon_output_goes_to_its_own_sink(self):
        """T35: the daemon's lines go to daemon_log_sink, init_system's to log_sink; without a
        daemon sink both go to log_sink (earlier behaviour)."""
        daemon = FakeDaemon()
        daemon.stderr = io.BytesIO(b"Got a new client: 8\nCNT  1 2 3\n")
        log, own = [], []
        service = self.ready(self.service(daemon=daemon, init_child=lambda: FakeChild(0, stdout=b"active units\n"),
                                          log_sink=log.append, daemon_log_sink=own.append))
        self.assertTrue(service.initialize(self.settings).initialized)
        self.assertTrue(wait_for(lambda: len(own) == 2), own)
        self.assertEqual(own, ["[stderr] Got a new client: 8", "[stderr] CNT  1 2 3"])
        self.assertEqual(log, ["[stdout] active units"])
        daemon, shared = FakeDaemon(), []
        daemon.stderr = io.BytesIO(b"CNT  4 5 6\n")
        service = self.ready(self.service(daemon=daemon, log_sink=shared.append))
        self.assertTrue(wait_for(lambda: shared == ["[stderr] CNT  4 5 6"]), shared)

    def test_daqd_death_after_initialization_invalidates_it(self):
        service = self.ready(self.service())
        self.assertTrue(service.initialize(self.settings).initialized)
        self.daemon.die(-11)
        self.assertStopped(service, DaqdState.FAILED)
        status = service.status()
        self.assertFalse(status.initialized)
        self.assertFalse(service.acquisition_ready(self.settings)[0])
        config = service.daqd_config(self.settings)
        self.assertEqual(status.stale_resources, (config.socket_path, config.shared_memory_path))
        self.assertEqual(self.resources.paths, {config.socket_path, config.shared_memory_path})
        self.assertEqual(service.start(self.settings).state, DaqdState.FAILED)  # blocked by leftovers
        self.assertEqual(len(self.backend.launched), 1)

    def test_daqd_death_during_initialization_is_not_initialized(self):
        daemon = FakeDaemon()
        class DyingInit(FakeChild):
            def poll(inner):
                daemon.die(1)
                return 0
        service = self.ready(self.service(daemon=daemon, init_child=lambda: DyingInit(0)))
        outcome = service.initialize(self.settings)
        self.assertFalse(outcome.initialized)
        self.assertStopped(service, DaqdState.FAILED)
        self.assertFalse(service.acquisition_ready(self.settings)[0])

    def test_daqd_changed_ini_or_hardware_invalidates_initialization(self):
        service = self.ready(self.service())
        self.assertTrue(service.initialize(self.settings).initialized)
        ini = Path(self.settings.paths["ini_file"])
        original = ini.read_bytes()
        ini.write_bytes(original + b"\n# edited\n")
        ready, reason = service.acquisition_ready(self.settings)
        self.assertFalse(ready)
        self.assertIn("changed", reason)
        ini.write_bytes(original)
        self.assertFalse(service.acquisition_ready(self.settings)[0])  # sticky: initialize again
        self.assertTrue(service.initialize(self.settings).initialized)
        other = self.preflight(Action.INITIALIZE, replace(self.profile, cards=(self.profile.cards[1],)))
        self.assertFalse(service.acquisition_ready(other)[0])
        refused = service.initialize(other)
        self.assertFalse(refused.initialized)
        self.assertIn("differ from the running DAQD", refused.message)
        self.assertEqual(len(self.init_backend.launched), 2)

    def test_daqd_stop_signals_only_owned_child_and_keeps_resources(self):
        service = self.ready(self.service())
        config = service.daqd_config(self.settings)
        self.assertEqual(service.stop().state, DaqdState.STOPPING)
        self.assertStopped(service, DaqdState.OFF)
        self.assertEqual(self.daemon.signals, ["TERM"])
        status = service.status()
        self.assertEqual(status.stale_resources, (config.socket_path, config.shared_memory_path))
        self.assertIn("not removed", status.message)
        self.assertEqual(service.stop().state, DaqdState.OFF)
        self.assertEqual(self.daemon.signals, ["TERM"])
        self.assertEqual(len(self.backend.launched), 1)

    def test_daqd_term_resistant_daemon_is_killed_after_grace(self):
        service = self.ready(self.service(daemon=FakeDaemon(term_exits=False)))
        service.stop()
        self.assertStopped(service, DaqdState.OFF)
        self.assertEqual(self.daemon.signals[:2], ["TERM", "KILL"])

    def test_daqd_close_waits_for_reaping(self):
        service = self.ready(self.service())
        status = service.close(5)
        self.assertEqual(status.state, DaqdState.OFF)
        self.assertTrue(self.daemon.waited)
        self.assertEqual(service.close(1).state, DaqdState.OFF)

    def test_daqd_restart_generations_and_status_revisions(self):
        service = self.ready(self.service())
        self.assertEqual(service.start(self.settings).state, DaqdState.READY)  # second start refused
        self.assertEqual(len(self.backend.launched), 1)
        service.close(5)
        self.resources.paths.clear()  # the fake daemon "cleaned up" its own resources on TERM
        self.daemon = FakeDaemon(pid=4343)
        self.resources.daemon = self.daemon
        self.backend.child = self.daemon
        self.assertEqual(service.start(self.settings).generation, 2)
        self.assertTrue(wait_for(lambda: service.status().state == DaqdState.READY))
        self.assertEqual(service.status().pid, 4343)
        revisions = [status.revision for status in self.statuses]
        self.assertEqual(len(revisions), len(set(revisions)))
        self.assertEqual(max(revisions), service.status().revision)
        self.assertEqual(service.status().revision, service.status().revision)  # reads do not bump
        service.close(5)

    def test_daqd_module_never_removes_or_signals_directly(self):
        source = (REPO / "src/petsys_manager/acquisition.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                 for node in ast.walk(tree) if isinstance(node, ast.Call)}
        self.assertFalse(calls & {"unlink", "remove", "rmtree", "rmdir", "kill", "killpg", "shm_unlink", "system"})
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                   for alias in node.names}
        self.assertFalse({"tkinter", "customtkinter", "subprocess"} & imports)

    def test_daqd_import_has_no_side_effects(self):
        code = ("import sys, threading; before = threading.active_count(); "
                "import src.petsys_manager.acquisition as a; "
                "assert threading.active_count() == before; "
                "assert 'src.petsys_manager.settings' not in sys.modules; "
                "assert 'tkinter' not in sys.modules; print('ok')")
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.stdout.strip(), "ok", result.stderr)

    def test_daqd_policy_bounds(self):
        for kwargs in ({"startup_timeout_s": 0}, {"probe_interval_s": float("nan")},
                       {"terminate_grace_s": True}, {"probe_timeout_s": 10 ** 1000}):
            with self.assertRaises(ValueError):
                DaqdPolicy(**kwargs)

