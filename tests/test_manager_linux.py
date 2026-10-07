"""Linux-only dummy-process checks: process groups, DAQD, acquisition, PETsys Python (spec 003 T3, T6, T7, T22; spec 007 T14).

Moved from scripts/petsys_manager_linux_check.py (--all). Every class is marked ``linux``: elsewhere
it is skipped with a reason (the script exited 2, PENDING). Never PETsys tools or hardware.
The test harness, not the runtime runner, temporarily becomes a subreaper to verify adopted
dummy grandchildren are reaped rather than relying on PID 1. Fixtures are private (~/.cache/process_petsys).
"""

import ctypes
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from threading import Event, Thread, Timer
import time
from types import SimpleNamespace
import unittest

import pytest

from helpers import REPO
from src.petsys_manager.contracts import CommandSpec, Identity, OutputValidation, ResultStatus
from src.petsys_manager.acquisition import AcquisitionService, DaqdPolicy, DaqdService, DaqdState
from src.petsys_manager.artifacts import RunStore, read_manifest
from src.petsys_manager.runner import CommandRunner, RunnerPolicy


GRANDCHILD = """
import os, pathlib, signal, sys, time
if sys.argv[2] == 'resistant':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
with pathlib.Path(sys.argv[1]).open('x') as f:
    f.write(str(os.getpid()))
print('grandchild ready', flush=True)
while True:
    time.sleep(.1)
"""

PARENT = """
import json, os, pathlib, signal, subprocess, sys, time
ready, child_ready, mode = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
child = subprocess.Popen([sys.executable, '-u', '-c', sys.argv[4], str(child_ready),
                         'ordinary' if mode == 'cooperative' else 'resistant'])
if mode == 'cooperative':
    def stop(sig, frame):
        child.terminate()
        child.wait(timeout=3)
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
elif mode == 'both_resistant':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
deadline = time.monotonic() + 5
while not child_ready.exists():
    if time.monotonic() > deadline:
        raise RuntimeError('grandchild failed to start')
    time.sleep(.01)
with ready.open('x') as f:
    json.dump({'parent': os.getpid(), 'child': child.pid, 'pgid': os.getpgrp(),
               'sid': os.getsid(0)}, f)
print('parent ready', flush=True)
if mode == 'leader_exit':
    sys.exit(0)
while True:
    time.sleep(.1)
"""


DUMMY_DAQD = """
import os, signal, socket, struct, sys, time
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
sock_path, shm_path, mode = args['--socket-name'], args['--shm'], args['--mode']
stop = []
signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == 'resistant' else (lambda sig, frame: stop.append(sig)))
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    server.bind(sock_path)  # like daqd: refuses an existing socket path
except OSError as exc:
    print('bind failed', exc, file=sys.stderr, flush=True)
    sys.exit(255)
server.listen(5)
def cleanup(code):
    server.close()
    os.unlink(sock_path)
    if os.path.exists(shm_path):
        os.unlink(shm_path)
    sys.exit(code)
try:
    os.close(os.open(shm_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600))
except FileExistsError:
    server.close()
    os.unlink(sock_path)
    sys.exit(255)
print('socket bound; opening cards', flush=True)
if mode == 'exit_early':
    cleanup(3)
time.sleep(float(args['--delay']))
name = ('/wrong_shm' if mode == 'wrong_name' else '/' + os.path.basename(shm_path)).encode()
server.settimeout(0.05)
while not stop:
    if mode == 'never_serve':
        time.sleep(0.05)
        continue
    try:
        client, _ = server.accept()
    except socket.timeout:
        continue
    with client:
        try:  # like daqd (MSG_NOSIGNAL): an abandoned client is an error, not a crash
            header = client.recv(4)
            if len(header) == 4 and struct.unpack('@HH', header)[0] == 2:
                client.sendall(struct.pack('@HQQQ', struct.calcsize('@HQQQ') + len(name), 1, 0, 0) + name)
        except OSError as exc:
            print('client error', exc, file=sys.stderr, flush=True)
cleanup(0)
"""

DUMMY_INIT = """
import socket, struct, sys
if sys.argv[2] == 'fail':
    print('init failed', file=sys.stderr)
    sys.exit(1)
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
    sock.settimeout(5)
    sock.connect(sys.argv[1])
    sock.sendall(struct.pack('@HH', 2, 4))
    print('initialized via', sock.recv(64)[32:].decode(), flush=True)
"""


DUMMY_ACQUIRE = """
import os, sys, time
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
prefix, mode = args['-o'], args['--mode']
with open(prefix + '.pid', 'w') as out:
    out.write(str(os.getpid()))
for suffix in ('.idxf', '.tmpf', '.modf'):
    open(prefix + suffix, 'w').close()
with open(prefix + '.rawf', 'wb') as raw:
    if mode == 'stall':
        while True:
            time.sleep(0.05)
    deadline = time.monotonic() + float(args['--time'])
    while time.monotonic() < deadline:
        raw.write(b'R' * (100 if mode == 'slow' else 200000))
        raw.flush()
        time.sleep(0.05)
with open(prefix + '.idxf', 'w') as index:
    index.write('0 1 0 1 0 0\\n')
loss = float(args['--loss'])
print('writeRaw:: some events were lost for 0 (  0.0%%) frames; all events were lost for 1 (%5.1f%%) frames'
      % loss, file=sys.stderr, flush=True)
"""


@pytest.mark.linux
@pytest.mark.fr("003-FR-4", "003-FR-5", "003-FR-7", "003-FR-10", "003-FR-16")  # spec 003 T3
class LinuxGroupChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        if cls.libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot read test-only subreaper state")
        cls.previous_subreaper = previous.value
        if cls.libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot enable test-only subreaper")
        private = Path.home() / ".cache/process_petsys"
        private.mkdir(parents=True, exist_ok=True)
        cls.output = Path(tempfile.mkdtemp(prefix="petsys-manager-linux-", dir=private))

    @classmethod
    def tearDownClass(cls):
        if cls.libc.prctl(36, cls.previous_subreaper, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot restore test-only subreaper state")

    def setUp(self):
        self.root = self.output / self._testMethodName
        self.root.mkdir()
        self.identity = Identity("dummy-linux-run", self._testMethodName, "attempt-1")
        self.policy = RunnerPolicy(terminate_grace_s=.15, reap_timeout_s=3., drain_timeout_s=1.,
                                   poll_interval_s=.01, log_tail_lines=20)

    def reap_grandchild(self, pid, *, expect_kill):
        deadline = time.monotonic() + 5
        while True:
            try:
                waited, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                # Cooperative dummy parent already waited for its own child.
                self.assertFalse(expect_kill, "Resistant grandchild was not adopted by the harness")
                self.assertFalse(Path(f"/proc/{pid}").exists())
                return
            if waited:
                if expect_kill:
                    self.assertTrue(os.WIFSIGNALED(status))
                    self.assertEqual(os.WTERMSIG(status), signal.SIGKILL)
                return
            if time.monotonic() > deadline:
                self.fail(f"Dummy grandchild {pid} was not terminated/reaped")
            time.sleep(.01)

    def group_case(self, mode):
        marker = self.root / "owned-pids.json"
        child_ready = self.root / "owned-grandchild.txt"
        command = CommandSpec((sys.executable, "-u", "-c", PARENT, str(marker), str(child_ready), mode, GRANDCHILD),
                              self.root, self.identity, dict(os.environ))
        cancel = Event()
        stop_monitor = Event()
        ready = Event()
        events = []
        def monitor():
            deadline = time.monotonic() + 6
            while not stop_monitor.is_set():
                try:
                    metadata = json.loads(marker.read_text())
                    if metadata:
                        ready.set()
                        if mode != "leader_exit":
                            cancel.set()
                        return
                except (OSError, ValueError):
                    pass  # Parent may still be writing its independent fixture.
                if time.monotonic() > deadline:
                    cancel.set()
                    return
                stop_monitor.wait(.01)
        unrelated = subprocess.Popen((sys.executable, "-c", "import time; time.sleep(60)"),
            start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        monitoring = Thread(target=monitor, daemon=True)
        watchdog = Timer(12, cancel.set)
        monitoring.start()
        watchdog.start()
        metadata = None
        group_cleaned = False
        try:
            result = CommandRunner(policy=self.policy).run(command, cancellation=cancel, event_sink=events.append,
                validate_outputs=lambda command: OutputValidation(True, "fixture validator"))
            self.assertTrue(ready.is_set(), "Dummy parent/grandchild readiness timed out")
            metadata = json.loads(marker.read_text())
            started = next(event for event in events if event.kind == "started")
            self.assertEqual(metadata["parent"], started.payload["pid"])
            self.assertEqual(metadata["pgid"], metadata["parent"])
            self.assertEqual(metadata["sid"], metadata["parent"])
            self.assertEqual(result.status, ResultStatus.FAILED if mode == "leader_exit" else ResultStatus.CANCELLED,
                             result.message)
            self.assertFalse(result.can_advance)
            self.assertIsNone(unrelated.poll(), "Runner signaled an unrelated owned-by-test process")
            self.assertEqual([e.sequence for e in events], list(range(len(events))))
            self.assertEqual({e.identity for e in events}, {self.identity})
            if mode != "cooperative":
                self.assertIn("killing", [event.kind for event in events])
            # Popen.wait in the runner already reaped its direct child.
            with self.assertRaises(ChildProcessError):
                os.waitpid(metadata["parent"], os.WNOHANG)
            self.reap_grandchild(metadata["child"], expect_kill=mode != "cooperative")
            group_cleaned = True
        finally:
            stop_monitor.set()
            watchdog.cancel()
            watchdog.join()
            monitoring.join(timeout=2)
            unrelated.terminate()
            try:
                unrelated.wait(timeout=3)
            except subprocess.TimeoutExpired:
                unrelated.kill()
                unrelated.wait(timeout=3)
            # On assertion failure, still clean only the exact fixture group
            # whose leader ID came from the runner, never global daemon names.
            started = next((event for event in events if event.kind == "started"), None)
            if not group_cleaned and started is not None:
                try:
                    os.killpg(started.payload["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if metadata is None and marker.exists():
                metadata = json.loads(marker.read_text())
            if metadata is not None:
                try:
                    self.reap_grandchild(metadata["child"], expect_kill=False)
                except (AssertionError, ChildProcessError):
                    pass

    def test_cooperative_parent_reaps_grandchild_on_term(self):
        self.group_case("cooperative")

    def test_term_resistant_grandchild_killed_after_parent_exits(self):
        self.group_case("parent_exit")

    def test_term_resistant_parent_and_grandchild_killed(self):
        self.group_case("both_resistant")

    def test_successful_parent_exit_cannot_leave_live_descendants(self):
        self.group_case("leader_exit")

    def test_select_pipe_drain_and_bounded_tail(self):
        program = ("import os, threading\n"
                   "def write(fd, char):\n"
                   " for _ in range(2048): os.write(fd, char * 120 + b'\\n')\n"
                   "a=threading.Thread(target=write,args=(1,b'A')); b=threading.Thread(target=write,args=(2,b'B'))\n"
                   "a.start(); b.start(); a.join(); b.join()\n")
        command = CommandSpec((sys.executable, "-u", "-c", program), self.root, self.identity, dict(os.environ))
        counts = {"A": 0, "B": 0}
        def logger(line):
            for char in counts:
                counts[char] += line.count(char)
        cancel = Event()
        watchdog = Timer(12, cancel.set)
        watchdog.start()
        try:
            result = CommandRunner(policy=replace(self.policy, poll_interval_s=.001)).run(command,
                cancellation=cancel, log_sink=logger, validate_outputs=lambda command: OutputValidation(True))
        finally:
            watchdog.cancel()
            watchdog.join()
        self.assertTrue(result.can_advance, result.message)
        self.assertEqual(counts, {"A": 2048 * 120, "B": 2048 * 120})
        self.assertEqual(len(result.log_tail), 20)


@pytest.mark.linux
@pytest.mark.fr("003-FR-5", "003-FR-6", "003-FR-7", "003-FR-16")  # spec 003 T6
class LinuxDaqdChecks(unittest.TestCase):
    """T6 on real Unix sockets/files with a dummy daemon; never PETsys tools or /tmp/d.sock."""

    @classmethod
    def setUpClass(cls):
        private = Path.home() / ".cache/process_petsys"
        private.mkdir(parents=True, exist_ok=True)
        cls.output = Path(tempfile.mkdtemp(prefix="pm-daqd-", dir=private))
        cls.count = 0

    def setUp(self):
        type(self).count += 1
        self.root = self.output / f"t{self.count}"
        (self.root / "shm").mkdir(parents=True)
        self.ini = self.root / "selected.ini"
        self.ini.write_text("[fixture]\n", encoding="utf-8")
        profile = SimpleNamespace(socket_path=str(self.root / "d.sock"),
                                  shared_memory_path=str(self.root / "shm/daqd_shm"))
        self.settings = SimpleNamespace(profile=profile, paths={"ini_file": self.ini})
        self.foreign = None

    def tearDown(self):
        if self.foreign is not None and self.foreign.poll() is None:
            self.foreign.terminate()  # test-owned foreign dummy
            self.foreign.wait(timeout=5)

    def daqd_argv(self, mode, delay):
        return (sys.executable, "-u", "-c", DUMMY_DAQD, "--socket-name", self.settings.profile.socket_path,
                "--shm", self.settings.profile.shared_memory_path, "--mode", mode, "--delay", str(delay))

    def service(self, mode="normal", *, delay=0.4, init="ok", startup=5.0):
        def build_daqd(settings, identity):
            return CommandSpec(self.daqd_argv(mode, delay), self.root, identity, dict(os.environ))
        def build_init(settings, identity):
            return CommandSpec((sys.executable, "-u", "-c", DUMMY_INIT, settings.profile.socket_path, init),
                               self.root, identity, dict(os.environ))
        policy = DaqdPolicy(startup_timeout_s=startup, probe_interval_s=0.02, probe_timeout_s=0.5,
                            terminate_grace_s=0.5)
        self.logs = []
        return DaqdService(policy=policy, build_command=build_daqd, build_init=build_init,
                           log_sink=self.logs.append)

    def wait(self, service, state, timeout=10):
        deadline = time.monotonic() + timeout
        while service.status().state != state and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(service.status().state, state, service.status().message)
        return service.status()

    def resources(self):
        profile = self.settings.profile
        return tuple(p for p in (profile.socket_path, profile.shared_memory_path) if os.path.lexists(p))

    def test_daqd_linux_ready_only_after_protocol_reply_then_clean_stop(self):
        service = self.service(delay=0.6)
        self.assertEqual(service.start(self.settings).state, DaqdState.STARTING)
        deadline = time.monotonic() + 5
        while not Path(self.settings.profile.socket_path).exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(service.status().state, DaqdState.STARTING)  # bound, cards "opening"
        status = self.wait(service, DaqdState.READY)
        self.assertTrue(Path(f"/proc/{status.pid}").exists())
        outcome = service.initialize(self.settings)
        self.assertTrue(outcome.initialized, outcome.message)
        self.assertEqual(service.acquisition_ready(self.settings), (True, "Initialized"))
        pid = status.pid
        final = service.close(10)
        self.assertEqual((final.state, final.stale_resources), (DaqdState.OFF, ()))
        self.assertFalse(Path(f"/proc/{pid}").exists())
        self.assertEqual(self.resources(), ())
        self.assertFalse(any("client error" in line for line in self.logs), self.logs)  # one waiting probe

    def test_daqd_linux_foreign_daemon_is_not_touched(self):
        self.foreign = subprocess.Popen(self.daqd_argv("normal", 0), stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 5
        while len(self.resources()) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        service = self.service()
        status = service.start(self.settings)
        self.assertEqual(status.state, DaqdState.OFF)
        self.assertEqual(len(status.stale_resources), 2)
        service.stop()
        service.close(1)
        self.assertIsNone(self.foreign.poll())
        self.assertEqual(len(self.resources()), 2)

    def test_daqd_linux_stale_shared_memory_is_not_removed(self):
        shm = Path(self.settings.profile.shared_memory_path)
        shm.write_bytes(b"foreign")
        status = self.service().start(self.settings)
        self.assertEqual((status.state, status.stale_resources), (DaqdState.OFF, (str(shm),)))
        self.assertEqual(shm.read_bytes(), b"foreign")
        self.assertFalse(Path(self.settings.profile.socket_path).exists())

    def test_daqd_linux_crash_invalidates_initialization_and_reports_leftovers(self):
        service = self.service()
        service.start(self.settings)
        pid = self.wait(service, DaqdState.READY).pid
        self.assertTrue(service.initialize(self.settings).initialized)
        os.kill(pid, signal.SIGKILL)  # simulated crash of the owned dummy
        status = self.wait(service, DaqdState.FAILED)
        self.assertFalse(status.initialized)
        self.assertFalse(service.acquisition_ready(self.settings)[0])
        self.assertEqual(len(status.stale_resources), 2)
        self.assertEqual(len(self.resources()), 2)
        self.assertEqual(service.start(self.settings).state, DaqdState.FAILED)  # leftovers block restart

    def test_daqd_linux_term_resistant_daemon_killed_and_reaped(self):
        service = self.service("resistant")
        service.start(self.settings)
        pid = self.wait(service, DaqdState.READY).pid
        final = service.close(10)
        self.assertEqual(final.state, DaqdState.OFF)
        self.assertFalse(Path(f"/proc/{pid}").exists())
        self.assertEqual(len(final.stale_resources), 2)  # KILL skipped the dummy's own cleanup

    def test_daqd_linux_early_exit_and_never_serving_fail(self):
        service = self.service("exit_early")
        service.start(self.settings)
        self.assertIn("exit code 3", self.wait(service, DaqdState.FAILED).message)
        service = self.service("never_serve", startup=1.0)
        service.start(self.settings)
        status = self.wait(service, DaqdState.FAILED)
        self.assertIn("not ready after", status.message)
        self.assertEqual(self.resources(), ())  # dummy cleaned up after TERM

    def test_daqd_linux_wrong_name_and_failed_init(self):
        service = self.service("wrong_name")
        service.start(self.settings)
        self.assertIn("reported shared memory", self.wait(service, DaqdState.FAILED).message)
        service = self.service(init="fail")
        service.start(self.settings)
        self.wait(service, DaqdState.READY)
        outcome = service.initialize(self.settings)
        self.assertFalse(outcome.initialized)
        self.assertFalse(service.acquisition_ready(self.settings)[0])
        service.close(10)


@pytest.mark.linux
@pytest.mark.fr("003-FR-5", "003-FR-7", "003-FR-8", "003-FR-9", "003-FR-16")  # spec 003 T7
class LinuxAcquisitionChecks(unittest.TestCase):
    """T7 with the real clock, real growing files and real TERM of a dummy acquisition group."""

    @classmethod
    def setUpClass(cls):
        private = Path.home() / ".cache/process_petsys"
        private.mkdir(parents=True, exist_ok=True)
        cls.output = Path(tempfile.mkdtemp(prefix="pm-acquisition-", dir=private))
        cls.count = 0

    def setUp(self):
        type(self).count += 1
        self.root = self.output / f"t{self.count}"
        self.root.mkdir()
        self.logs = []

    def run_modes(self, modes, *, stop_after_launch=False, max_attempts=3):
        safety = SimpleNamespace(startup_timeout_s=2.0, growth_window_s=0.5, poll_interval_s=0.1,
                                 min_growth_bytes=100_000, max_loss_percent=5.0, max_attempts=max_attempts,
                                 retry_delay_s=0.1, terminate_grace_s=1.0)
        settings = SimpleNamespace(profile=SimpleNamespace(safety=safety))
        def build(settings, identity, prefix):
            number = int(identity.attempt_id.split("-")[1]) if identity.attempt_id.startswith("attempt-") else 1
            mode, loss = modes[number - 1]
            return CommandSpec((sys.executable, "-u", "-c", DUMMY_ACQUIRE, "-o", str(prefix), "--mode", mode,
                                "--time", "1.2", "--loss", str(loss)), self.root, identity, dict(os.environ))
        def bias_off(settings, identity):
            marker = self.root / f"{identity.attempt_id}.bias-off"
            return CommandSpec((sys.executable, "-c", f"open({str(marker)!r}, \"x\").close()"), self.root, identity,
                               dict(os.environ))
        service = AcquisitionService(build_command=build, build_bias_off=bias_off, log_sink=self.logs.append)
        store = RunStore.reserve(self.root, {"fixture": "linux-acquisition"})
        handle = service.start(settings, store)
        if stop_after_launch:
            pid_file = store.root / "acquisition/attempt-1/acquisition.pid"
            deadline = time.monotonic() + 10
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            handle.stop()
        outcome = handle.wait(60)
        self.assertIsNotNone(outcome)
        return outcome, store

    def assertReaped(self, attempt):
        pid = int((attempt.directory / "acquisition.pid").read_text())
        self.assertFalse(Path(f"/proc/{pid}").exists(), f"dummy acquisition {pid} still present")

    def test_acquisition_linux_growth_passes_and_outputs_recorded(self):
        outcome, store = self.run_modes([("good", 0.0)])
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        attempt = outcome.attempts[0]
        self.assertEqual((attempt.growth.state, attempt.loss.state, attempt.bias.state), ("passed", "within", "not_needed"))
        self.assertFalse(list(self.root.glob("*.bias-off")))
        self.assertEqual(sorted(a.kind for a in attempt.artifacts), ["idxf", "modf", "rawf", "tmpf"])
        self.assertGreater((attempt.directory / "acquisition.rawf").stat().st_size, 1_000_000)
        self.assertEqual(read_manifest(store.root)["attempts"][0]["status"], "succeeded")
        self.assertReaped(attempt)

    def test_acquisition_linux_stall_times_out_then_retry_succeeds(self):
        outcome, store = self.run_modes([("stall", 0.0), ("good", 1.0)])
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        first, second = outcome.attempts
        self.assertEqual((first.retry_reason, second.number), ("startup_timeout", 2))
        self.assertReaped(first)
        self.assertTrue((first.directory / "acquisition.rawf").exists())  # failed attempt kept
        self.assertIn("[acquisition attempt-1] aborting: Early abort", "\n".join(self.logs))

    def test_acquisition_linux_slow_growth_and_loss_exhaust_attempts(self):
        outcome, _ = self.run_modes([("slow", 0.0), ("good", 12.5)], max_attempts=2)
        self.assertEqual(outcome.status, ResultStatus.FAILED)
        self.assertEqual([a.retry_reason for a in outcome.attempts], ["insufficient_growth", "frame_loss"])
        self.assertEqual(outcome.attempts[1].loss.all_lost_percent, 12.5)
        self.assertIn("after 2 attempts", outcome.message)
        for attempt in outcome.attempts:
            self.assertReaped(attempt)

    def test_acquisition_linux_stop_terminates_and_never_retries(self):
        outcome, store = self.run_modes([("stall", 0.0), ("good", 0.0)], stop_after_launch=True)
        self.assertEqual(outcome.status, ResultStatus.CANCELLED)
        self.assertEqual(outcome.attempts[0].bias.state, "requested")
        self.assertTrue((self.root / "attempt-1-bias-off.bias-off").exists())  # dummy bias tool really ran
        self.assertEqual(len(outcome.attempts), 1)
        self.assertReaped(outcome.attempts[0])
        self.assertFalse((store.root / "acquisition/attempt-2").exists())


PETSYS_TOOL = """#!/usr/bin/env python3
import json, os, subprocess, sys
nested = subprocess.run(["env", "python3", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True)
print(json.dumps({"executable": sys.executable, "conda": sorted(k for k in os.environ if k.startswith("CONDA_")),
                  "pythonpath": os.environ.get("PYTHONPATH"), "nested": nested.stdout.strip(),
                  "nested_code": nested.returncode}))
"""


@pytest.mark.linux
@pytest.mark.fr("003-FR-3", "003-FR-4", "003-FR-16", "003-FR-23")  # spec 003 T22
class LinuxPetsysPythonChecks(unittest.TestCase):
    """FR-23 with real processes: a conda-like env python3 first on PATH, the tool's own shebang and a
    nested ``env python3``; ``petsys_python`` /usr/bin/python3 must run the tool and its children."""

    @classmethod
    def setUpClass(cls):
        private = Path.home() / ".cache/process_petsys"
        private.mkdir(parents=True, exist_ok=True)
        cls.output = Path(tempfile.mkdtemp(prefix="pm-python-", dir=private))

    def test_petsys_python_linux_tool_and_nested_children_use_the_named_interpreter(self):
        from unittest.mock import patch
        from src.petsys_manager.commands import build_initialize
        from src.petsys_manager.contracts import Action
        from src.petsys_manager.settings import MachineProfile, SystemProbe, preflight

        class Probe(SystemProbe):
            def device(self, path):
                return path.is_file()

        root = self.output
        env_bin = root / "conda env/bin"
        env_bin.mkdir(parents=True)
        shim = env_bin / "python3"          # the manager's env python: lacks the PETsys packages
        shim.write_text("#!/bin/sh\necho 'ModuleNotFoundError: env python3' >&2\nexit 3\n")
        tools = root / "tools"
        tools.mkdir()
        for name in ("daqd", "init_system"):
            (tools / name).write_text(PETSYS_TOOL)
        for path in (shim, *tools.iterdir()):
            path.chmod(0o755)
        (root / "config.ini").write_text("fixture")
        cards = [root / "card0", root / "card1"]
        for card in cards:
            card.write_text("fixture")
        activated = {"PATH": f"{env_bin}:/usr/bin:/bin", "CONDA_PREFIX": str(env_bin.parent),
                     "CONDA_DEFAULT_ENV": "process_petsys", "PYTHONPATH": str(root), "HOME": str(Path.home())}
        results = {}
        for label, python in (("shebang", None), ("petsys_python", "/usr/bin/python3")):
            profile = MachineProfile(petsys_folder=str(tools), petsys_python=python, ini_file=str(root / "config.ini"),
                                     cards=tuple(str(c) for c in cards))
            with patch.dict(os.environ, activated, clear=True):
                report = preflight(profile, Action.INITIALIZE, repo_root=REPO, probe=Probe())
                self.assertTrue(report.ready, report.issues)
                command = build_initialize(report.settings, Identity("run", "initialize", "attempt-1"))
            done = subprocess.run(command.argv, cwd=command.cwd, env=dict(command.environment),
                                  capture_output=True, text=True, timeout=30)
            results[label] = (done.returncode, done.stdout, done.stderr)
        code, _, stderr = results["shebang"]          # reproduces the Cornell failure without the setting
        self.assertEqual(code, 3)
        self.assertIn("env python3", stderr)
        code, stdout, stderr = results["petsys_python"]
        self.assertEqual(code, 0, stderr)
        seen = json.loads(stdout)
        self.assertEqual(Path(seen["executable"]).parent, Path("/usr/bin"))
        self.assertEqual((seen["nested_code"], Path(seen["nested"]).parent), (0, Path("/usr/bin")))
        self.assertEqual((seen["conda"], seen["pythonpath"]), ([], None))
