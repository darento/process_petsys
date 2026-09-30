"""DAQD ownership/readiness and initialization validity (T6); acquisition monitoring is T7.

Readiness contract, from the inspected sibling daqd source (installed Cornell
version still pending, T19):

- daqd binds its socket before opening cards/shared memory, so a socket file
  is not readiness. It answers GetDataFrameSharedMemoryName (command 0x02, the
  query ``daqd.Connection()`` makes on connect) only from its client loop,
  which starts after cards and the frame server are up. READY therefore needs
  that reply from our own child (SO_PEERCRED where available) naming the
  expected shared memory.
- A daqd whose O_EXCL shared-memory create fails still shm_unlink()s that name
  on exit. Launching while the socket or shared memory exists could delete
  another daemon's resources, so either one blocks start. Nothing here removes
  them: leftovers are reported for the operator to resolve.

Service methods never touch Tk. ``initialize``/``close`` block; call them from
a worker. Status snapshots carry a revision so consumers can drop stale ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import os
from pathlib import Path
import socket
import stat
import struct
from threading import Event, Lock, Thread
import time

from .contracts import CommandResult, Identity, ResultStatus
from .runner import Clock, CommandRunner, RunnerPolicy

SHM_NAME_COMMAND = 0x02
_REQUEST = struct.Struct("@HH")      # CmdHeader_t {type, length}
_REPLY = struct.Struct("@HQQQ")      # {length, sizes[3]}, then the name bytes
_MAX_SHM_NAME = 255
_MAX_INI_BYTES = 4 * 1024 * 1024


class DaqdState(str, Enum):
    OFF = "off"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


def _positive(value, name):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class DaqdPolicy:
    startup_timeout_s: float = 30.0
    probe_interval_s: float = 0.25
    probe_timeout_s: float = 2.0     # post-initialization re-query only
    terminate_grace_s: float = 10.0

    def __post_init__(self):
        for name in ("startup_timeout_s", "probe_interval_s", "probe_timeout_s", "terminate_grace_s"):
            _positive(getattr(self, name), name)


@dataclass(frozen=True)
class DaqdConfig:
    """What a running daemon depends on: its exact argv/cwd and resources."""
    argv: tuple[str, ...]
    cwd: str
    socket_path: str
    shared_memory_path: str

    @property
    def expected_shm_name(self):
        return "/" + Path(self.shared_memory_path).name


@dataclass(frozen=True)
class InitConfig:
    """What an initialization depends on; INI content included (plan: INI changes invalidate)."""
    daqd: DaqdConfig
    argv: tuple[str, ...]
    cwd: str
    ini_path: str
    ini_sha256: str


@dataclass(frozen=True)
class DaqdStatus:
    revision: int
    state: DaqdState
    generation: int
    pid: int | None = None
    initialized: bool = False
    initializing: bool = False
    message: str = ""
    stale_resources: tuple[str, ...] = ()
    config: DaqdConfig | None = None


@dataclass(frozen=True)
class InitOutcome:
    initialized: bool
    message: str
    result: CommandResult | None = None
    status: DaqdStatus | None = None


class DaqdResources:
    """Read-only resource checks and the protocol query; never deletes anything."""

    def existing(self, config):
        return tuple(path for path in (config.socket_path, config.shared_memory_path) if os.path.lexists(path))

    def is_socket(self, path):
        try:
            return stat.S_ISSOCK(os.lstat(path).st_mode)
        except OSError:
            return False

    def query(self, socket_path, timeout, abort):
        """Return (shared-memory name, serving pid or None). Raises OSError/ValueError."""
        family = getattr(socket, "AF_UNIX", None)
        if family is None:
            raise OSError("Unix sockets are unavailable on this platform")
        deadline = time.monotonic() + timeout
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(socket_path)
            peer = None
            if hasattr(socket, "SO_PEERCRED"):
                pid, _, _ = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                                                 struct.calcsize("3i")))
                peer = pid
            sock.sendall(_REQUEST.pack(SHM_NAME_COMMAND, _REQUEST.size))
            length, _, _, _ = _REPLY.unpack(_receive(sock, _REPLY.size, deadline, abort))
            if not _REPLY.size < length <= _REPLY.size + _MAX_SHM_NAME:
                raise ValueError(f"Implausible shared-memory reply length {length}")
            name = _receive(sock, length - _REPLY.size, deadline, abort).decode("ascii")
        return name, peer


def _receive(sock, size, deadline, abort):
    data = b""
    while len(data) < size:
        if abort():
            raise OSError("Query abandoned")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OSError("No reply from DAQD yet")
        sock.settimeout(min(remaining, 0.1))
        try:
            chunk = sock.recv(size - len(data))
        except socket.timeout:
            continue
        if not chunk:
            raise OSError("DAQD closed the query connection")
        data += chunk
    return data


def _file_sha256(path):
    path = Path(path)
    if path.stat().st_size > _MAX_INI_BYTES:
        raise ValueError(f"INI exceeds {_MAX_INI_BYTES} bytes: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DaqdService:
    def __init__(self, *, runner=None, init_runner=None, resources=None, clock=None, policy=None,
                 status_sink=None, log_sink=None, build_command=None, build_init=None):
        self.policy = DaqdPolicy() if policy is None else policy
        if not isinstance(self.policy, DaqdPolicy):
            raise ValueError("DAQD service requires a typed policy")
        self._runner = runner or CommandRunner(policy=RunnerPolicy(terminate_grace_s=self.policy.terminate_grace_s))
        self._init_runner = init_runner or CommandRunner()
        self._resources = DaqdResources() if resources is None else resources
        self._clock = Clock() if clock is None else clock
        self._status_sink = status_sink
        self._log_sink = log_sink
        if build_command is None or build_init is None:
            from .commands import build_daqd, build_initialize  # needs PyYAML via settings
            build_command = build_command or build_daqd
            build_init = build_init or build_initialize
        # Builders validate the typed RunSettings snapshot; checks may inject dummies.
        self._build_command = build_command
        self._build_init = build_init
        self._lock = Lock()
        self._revision = 0
        self._generation = 0
        self._state = DaqdState.OFF
        self._pid = None
        self._config = None
        self._message = ""
        self._stale = ()
        self._stop = None
        self._stop_reason = None
        self._done = Event()
        self._done.set()
        self._init_record = None
        self._initializing = False
        self._init_count = 0

    # Status -------------------------------------------------------------

    def _snapshot(self, changed=True):
        if changed:
            self._revision += 1
        return DaqdStatus(self._revision, self._state, self._generation, self._pid,
                          self._init_record is not None, self._initializing, self._message,
                          self._stale, self._config)

    def _publish(self, status):
        if self._status_sink is not None:
            try:
                self._status_sink(status)
            except Exception:
                pass  # A broken sink must not break daemon ownership; status() stays authoritative.
        return status

    def status(self):
        with self._lock:
            return self._snapshot(changed=False)

    def daqd_config(self, settings):
        command = self._build_command(settings, Identity("daqd", "daqd", "config"))
        profile = settings.profile
        return DaqdConfig(command.argv, str(command.cwd), profile.socket_path, profile.shared_memory_path)

    def init_config(self, settings):
        command = self._build_init(settings, Identity("daqd", "initialize", "config"))
        ini = settings.paths.get("ini_file")
        if ini is None:
            raise ValueError("Initialization requires a selected INI")
        return InitConfig(self.daqd_config(settings), command.argv, str(command.cwd), str(ini), _file_sha256(ini))

    # Lifecycle ----------------------------------------------------------

    def start(self, settings):
        """Launch an owned daemon; returns the status (check state == STARTING)."""
        config = self.daqd_config(settings)
        with self._lock:
            if self._state in (DaqdState.STARTING, DaqdState.READY, DaqdState.STOPPING):
                self._message = f"DAQD is {self._state.value}; stop it before starting another"
                return self._publish(self._snapshot())
            existing = self._resources.existing(config)
            if existing:
                self._stale = existing
                self._message = ("Existing DAQD resources block start (another daemon or a stale "
                                 f"leftover; not removed): {', '.join(existing)}")
                return self._publish(self._snapshot())
            self._generation += 1
            generation = self._generation
            command = self._build_command(settings, Identity(f"daqd-{generation}", "daqd", f"launch-{generation}"))
            self._state, self._pid, self._config = DaqdState.STARTING, None, config
            self._message, self._stale, self._init_record = "Starting DAQD", (), None
            self._stop, self._stop_reason = Event(), None
            self._done = Event()
            stop, done = self._stop, self._done
            status = self._snapshot()
        Thread(target=self._run, args=(generation, command, stop, done), daemon=True,
               name=f"petsys-daqd-{generation}").start()
        Thread(target=self._watch_ready, args=(generation, config, stop), daemon=True,
               name=f"petsys-daqd-ready-{generation}").start()
        return self._publish(status)

    def _run(self, generation, command, stop, done):
        def events(event):
            if event.kind == "started":
                with self._lock:
                    if generation == self._generation:
                        self._pid = event.payload["pid"]
        try:
            result = self._runner.run(command, cancellation=stop, event_sink=events, log_sink=self._log_sink)
            outcome = f"exit code {result.exit_code}, {result.status.value}: {result.message}"
            stop_failed = result.status in (ResultStatus.FAILED, ResultStatus.LAUNCH_ERROR) and stop.is_set()
        except Exception as exc:
            outcome, stop_failed = f"runner error: {exc}", True
        with self._lock:
            if generation != self._generation:
                return
            config = self._config
            try:
                leftover = self._resources.existing(config)
            except Exception as exc:
                leftover = (f"unknown: {exc}",)
            if self._stop_reason == "operator" and not stop_failed:
                self._state, self._message = DaqdState.OFF, f"DAQD stopped ({outcome})"
            else:
                reason = self._stop_reason if self._stop_reason not in (None, "operator") else "DAQD exited"
                self._state, self._message = DaqdState.FAILED, f"{reason} ({outcome})"
            if leftover:
                self._message += f"; leftover resources not removed: {', '.join(leftover)}"
            self._pid, self._stale, self._init_record, self._initializing = None, leftover, None, False
            status = self._snapshot()
        self._publish(status)
        done.set()  # After publishing: close() returning implies the sink saw the final state.

    def _fail_startup(self, generation, stop, message):
        with self._lock:
            if generation != self._generation or self._state != DaqdState.STARTING:
                return
            self._state, self._stop_reason, self._message = DaqdState.STOPPING, message, message
            status = self._snapshot()
        stop.set()
        self._publish(status)

    def _watch_ready(self, generation, config, stop):
        deadline = self._clock.monotonic() + self.policy.startup_timeout_s
        last = "socket not yet created"
        while not stop.is_set():
            with self._lock:
                if generation != self._generation or self._state != DaqdState.STARTING:
                    return
                pid = self._pid
            now = self._clock.monotonic()
            if now >= deadline:
                self._fail_startup(generation, stop, f"DAQD not ready after {self.policy.startup_timeout_s:g} s ({last})")
                return
            if pid is not None and self._resources.is_socket(config.socket_path):
                try:
                    # One connection waits for the reply until the startup deadline:
                    # daqd queues it while opening cards and answers once polling.
                    name, peer = self._resources.query(config.socket_path, deadline - now, stop.is_set)
                except (OSError, ValueError) as exc:
                    last = f"query: {exc}"
                else:
                    if name != config.expected_shm_name:
                        self._fail_startup(generation, stop,
                                           f"DAQD reported shared memory {name!r}, expected {config.expected_shm_name!r}")
                        return
                    if peer is not None and peer != pid:
                        self._fail_startup(generation, stop, f"Socket is served by pid {peer}, not owned DAQD {pid}")
                        return
                    if config.shared_memory_path not in self._resources.existing(config):
                        last = "shared memory not present"
                    else:
                        with self._lock:
                            if generation != self._generation or self._state != DaqdState.STARTING:
                                return
                            self._state = DaqdState.READY
                            self._message = f"DAQD ready: pid {pid} answered the shared-memory query ({name})"
                            status = self._snapshot()
                        self._publish(status)
                        return
            self._clock.wait(stop, self.policy.probe_interval_s)

    def stop(self):
        """Request an owned daemon stop (TERM, KILL after grace); returns immediately."""
        with self._lock:
            if self._state not in (DaqdState.STARTING, DaqdState.READY):
                self._message = "No owned DAQD running" if self._state != DaqdState.STOPPING else "DAQD stopping"
                return self._publish(self._snapshot())
            self._state, self._stop_reason, self._init_record = DaqdState.STOPPING, "operator", None
            self._message = "Stopping owned DAQD"
            stop = self._stop
            status = self._snapshot()
        stop.set()
        return self._publish(status)

    def close(self, timeout):
        """Stop and wait for reaping; the returned status says if it is still stopping."""
        self.stop()
        with self._lock:
            done = self._done
        done.wait(timeout)
        return self.status()

    # Initialization -----------------------------------------------------

    def initialize(self, settings, cancellation=None):
        """Run init_system against the current owned READY daemon. Blocking."""
        try:
            config = self.init_config(settings)
        except (OSError, ValueError) as exc:
            return InitOutcome(False, f"Cannot initialize: {exc}", status=self.status())
        with self._lock:
            if self._state != DaqdState.READY:
                return InitOutcome(False, f"DAQD is {self._state.value}, not ready", status=self._snapshot())
            if config.daqd != self._config:
                return InitOutcome(False, "Settings differ from the running DAQD; restart it", status=self._snapshot())
            if self._initializing:
                return InitOutcome(False, "Initialization already running", status=self._snapshot())
            generation = self._generation
            self._initializing, self._init_record = True, None
            self._init_count += 1
            identity = Identity(f"daqd-{generation}", "initialize", f"init-{self._init_count}")
            status = self._snapshot()
        self._publish(status)
        result = None
        try:
            command = self._build_init(settings, identity)
            result = self._init_runner.run(command, cancellation=cancellation, log_sink=self._log_sink)
            message = result.message
            ok = result.status == ResultStatus.SUCCEEDED
            if ok:
                name, peer = self._resources.query(config.daqd.socket_path, self.policy.probe_timeout_s, lambda: False)
                with self._lock:
                    pid = self._pid
                ok = name == config.daqd.expected_shm_name and (peer is None or peer == pid)
                message = "Initialized" if ok else f"DAQD answered {name!r} from pid {peer} after init"
        except Exception as exc:
            ok, message = False, f"Initialization failed: {exc}"
        with self._lock:
            self._initializing = False
            current = generation == self._generation and self._state == DaqdState.READY
            if ok and current:
                self._init_record = (generation, config)
                self._message = "System initialized"
            else:
                if ok:
                    message = "DAQD changed or stopped during initialization"
                ok = False
                self._message = f"Initialization not valid: {message}"
            status = self._snapshot()
        self._publish(status)
        return InitOutcome(ok, message, result, status)

    def acquisition_ready(self, settings):
        """(ready, reason). A settings mismatch discards the initialization (sticky)."""
        try:
            config = self.init_config(settings)
        except (OSError, ValueError) as exc:
            return False, f"Cannot check initialization: {exc}"
        with self._lock:
            if self._state != DaqdState.READY:
                return False, f"DAQD is {self._state.value}"
            if self._init_record is None:
                return False, "System not initialized for the running DAQD"
            generation, recorded = self._init_record
            if generation != self._generation:
                self._init_record = None
                return False, "Initialization belongs to a previous DAQD"
            if config != recorded:
                self._init_record = None
                self._message = "Hardware/INI settings changed; initialize again"
                status = self._snapshot()
            else:
                return True, "Initialized"
        self._publish(status)
        return False, status.message
