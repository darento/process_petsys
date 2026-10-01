"""DAQD ownership/readiness and initialization validity (T6); monitored acquisition attempts (T7).

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

Acquisition (reference ``gui_cornell`` safety intent, inspected sibling tools):
``acquire_sipm_data -o PREFIX`` writes PREFIX.rawf/.idxf/.tmpf (write_raw) and
PREFIX.modf. write_raw reports per step on stderr ``some events were lost for
N (x%) frames; all events were lost for M (y%) frames``; like the reference,
the "all events" percentage is compared (> limit retries). Every attempt gets
its own run-store directory, so no retry deletes earlier data.

The tool switches SiPM bias off only at the end of a normal run; TERM skips
that. After any launched attempt that did not exit 0 by itself, the service
runs ``set_bias --power off`` (FR-19); if that fails the bias state is
reported unknown and no further attempt starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import os
from pathlib import Path
import re
import socket
import stat
import struct
from threading import Event, Lock, RLock, Thread, Timer
import time

from .contracts import (Artifact, CommandResult, Identity, OutputValidation, ResultStatus, RunEvent,
                        to_plain)
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


# Acquisition attempts, monitoring and retries (T7) ---------------------------

ACQUISITION_STAGE = "acquisition"
RAW_SUFFIXES = (".rawf", ".idxf", ".tmpf", ".modf")
REQUIRED_RAW_SUFFIXES = (".rawf", ".idxf")
# Stall/no-data/loss symptoms retry, as in the reference. A nonzero exit,
# launch error, lost prerequisite, STOP or storage failure never does.
RETRYABLE_REASONS = frozenset({"startup_timeout", "insufficient_growth", "no_data", "frame_loss"})
_LOSS_MARKER = "events were lost for"
_LOSS_LINE = re.compile(r"writeRaw:: some events were lost for (\d+) \(\s*(\d+(?:\.\d+)?)%\) frames; "
                        r"all events were lost for (\d+) \(\s*(\d+(?:\.\d+)?)%\) frames")
_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}")


@dataclass(frozen=True)
class FrameLoss:
    """Reported write_raw loss (maximum over steps). ``unknown`` never means zero."""
    state: str                              # within | exceeded | unknown
    all_lost_percent: float | None = None
    some_lost_percent: float | None = None
    reports: int = 0
    malformed: int = 0
    message: str = ""


@dataclass(frozen=True)
class GrowthCheck:
    state: str          # passed | insufficient | startup_timeout | not_exercised
    first_size: int | None = None
    last_size: int | None = None
    growth_bytes: int | None = None
    message: str = ""


@dataclass(frozen=True)
class BiasOff:
    """FR-19. ``requested``: set_bias --power off exited 0 (the tool's report, not a voltage reading)."""
    state: str          # not_needed | requested | unknown
    message: str = ""
    exit_code: int | None = None


@dataclass(frozen=True)
class AttemptSummary:
    identity: Identity
    number: int
    directory: Path
    prefix: Path
    status: ResultStatus
    exit_code: int | None
    message: str
    retry_reason: str | None
    growth: GrowthCheck
    loss: FrameLoss
    bias: BiasOff = BiasOff("not_needed")
    artifacts: tuple[Artifact, ...] = ()


@dataclass(frozen=True)
class AcquisitionOutcome:
    status: ResultStatus
    message: str
    attempts: tuple[AttemptSummary, ...] = ()

    @property
    def succeeded(self):
        return self.status == ResultStatus.SUCCEEDED

    @property
    def raw_prefix(self):
        """The successful attempt's PETsys prefix; None unless succeeded."""
        return self.attempts[-1].prefix if self.succeeded else None

    @property
    def bias_unknown(self):
        """True when some attempt could not get SiPM bias switched off: warn the operator."""
        return any(attempt.bias.state == "unknown" for attempt in self.attempts)


class _LossTracker:
    """Bounded: counts and maxima only, never the whole log."""

    def __init__(self):
        self.reports = self.malformed = 0
        self.all_lost = self.some_lost = None
        self.example = ""

    def feed(self, line):
        if _LOSS_MARKER not in line:
            return
        match = _LOSS_LINE.search(line)
        some, lost = (float(match.group(2)), float(match.group(4))) if match else (None, None)
        if match is None or some > 100 or lost > 100:
            self.malformed += 1  # e.g. "-nan%" from a step without frames
            self.example = self.example or line[:200]
            return
        self.reports += 1
        self.some_lost = some if self.some_lost is None else max(self.some_lost, some)
        self.all_lost = lost if self.all_lost is None else max(self.all_lost, lost)

    def result(self, limit):
        values = {"all_lost_percent": self.all_lost, "some_lost_percent": self.some_lost,
                  "reports": self.reports, "malformed": self.malformed}
        if self.all_lost is not None and self.all_lost > limit:
            return FrameLoss("exceeded", **values, message=f"Frame loss {self.all_lost:g}% exceeds {limit:g}%")
        if self.malformed:
            return FrameLoss("unknown", **values, message=f"Unparseable frame-loss report: {self.example!r}")
        if self.reports:
            return FrameLoss("within", **values, message=f"Frame loss {self.all_lost:g}% within {limit:g}%")
        return FrameLoss("unknown", **values, message="No frame-loss report in acquisition output")


def parse_frame_loss(lines, max_loss_percent):
    tracker = _LossTracker()
    for line in lines:
        tracker.feed(line)
    return tracker.result(max_loss_percent)


def _regular_size(path):
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    return info.st_size if stat.S_ISREG(info.st_mode) else None


class _AnyCancellation:
    """Runner cancellation: operator STOP or this attempt's monitor abort."""

    def __init__(self, operator, abort):
        self._operator, self._abort = operator, abort

    def is_set(self):
        return self._operator.is_set() or self._abort.is_set()

    def wait(self, timeout):
        self._abort.wait(timeout)  # Runner polls in short slices, so STOP is seen promptly too.
        return self.is_set()


@dataclass
class _Live:
    """Mutable per-attempt monitor state; guarded by the service lock."""
    identity: Identity
    rawf: Path
    abort: Event = field(default_factory=Event)
    wake: Event = field(default_factory=Event)     # attempt over: its monitor must not act
    exited: Event = field(default_factory=Event)   # child exited: stop judging growth
    launched: Event = field(default_factory=Event)  # child started: bias may be on
    started_at: float = 0.0
    state: str = "waiting"
    first_size: int | None = None
    last_size: int | None = None
    window_start: float | None = None
    growth_bytes: int | None = None
    previous: tuple[float, int] | None = None     # (time, size) of the last progress sample
    message: str = ""
    reason: str | None = None
    detail: str = ""        # why the monitor aborted the attempt
    warned: bool = False

    def growth(self, window):
        if self.state in ("passed", "insufficient", "startup_timeout"):
            return GrowthCheck(self.state, self.first_size, self.last_size, self.growth_bytes, self.message)
        text = ("No .rawf data observed before the attempt ended" if self.first_size is None
                else f"Attempt ended before the {window:g} s growth window completed")
        return GrowthCheck("not_exercised", self.first_size, self.last_size, None, text)


class AcquisitionHandle:
    """STOP is sticky: it cancels the running attempt and every pending retry."""

    def __init__(self):
        self._cancel = Event()
        self._done = Event()
        self._outcome = None

    def stop(self):
        self._cancel.set()

    @property
    def cancelled(self):
        return self._cancel.is_set()

    @property
    def done(self):
        return self._done.is_set()

    def wait(self, timeout=None):
        """Outcome, or None while still running."""
        self._done.wait(timeout)
        return self._outcome

    def _finish(self, outcome):
        self._outcome = outcome
        self._done.set()


class AttemptFilter:
    """Consumer-side guard for queued updates: newest sequence and current attempt only."""

    def __init__(self):
        self._sequence = -1
        self._current = None

    def accept(self, event):
        if event.sequence <= self._sequence:
            return False
        if event.kind == "attempt_started":
            self._current = event.identity
        elif event.kind != "finished" and self._current is not None and event.identity != self._current:
            return False
        self._sequence = event.sequence
        return True


class AcquisitionService:
    """Monitored acquisition with bounded retries; one active acquisition per service.

    ``start`` returns at once. Attempts, retry waits and monitoring run in worker
    threads. Update sinks are called in order under the service lock and log
    sinks from worker/reader threads: both must only enqueue, never touch Tk.
    Each attempt is a new run-store attempt directory; failed ones are kept.
    """

    def __init__(self, *, runner=None, clock=None, size_probe=None, update_sink=None, log_sink=None,
                 build_command=None, build_bias_off=None, bias_off_timeout_s=30.0):
        _positive(bias_off_timeout_s, "bias_off_timeout_s")
        self._bias_timeout = bias_off_timeout_s
        self._runner = runner
        self._clock = Clock() if clock is None else clock
        self._size = _regular_size if size_probe is None else size_probe
        self._update_sink = update_sink
        self._log_sink = log_sink
        if build_command is None or build_bias_off is None:
            from .commands import build_acquisition, build_bias_off as bias_builder  # needs PyYAML via settings
            build_command = build_command or build_acquisition
            build_bias_off = build_bias_off or bias_builder
        self._build_command = build_command
        self._build_bias_off = build_bias_off
        self._lock = RLock()
        self._sequence = 0
        self._active = None

    def _log(self, line):
        if self._log_sink is not None:
            try:
                self._log_sink(line)
            except Exception:
                pass  # Logging cannot change an acquisition verdict.

    def _emit(self, identity, kind, message="", payload=None, *, log=True):
        with self._lock:
            event = RunEvent(identity, self._sequence, kind, message, payload or {})
            self._sequence += 1
            if message and log:
                self._log(f"[acquisition {identity.attempt_id}] {kind}: {message}")
            if self._update_sink is not None:
                try:
                    self._update_sink(event)
                except Exception:
                    pass
        return event

    @staticmethod
    def _check(prerequisite):
        if prerequisite is None:
            return True, ""
        try:
            ok, reason = prerequisite()
        except Exception as exc:
            return False, f"check failed: {exc}"
        return bool(ok), str(reason)

    def start(self, settings, store, *, basename="acquisition", prerequisite=None):
        """Begin in a worker; ``prerequisite() -> (ok, reason)`` gates each attempt and is polled."""
        if not isinstance(basename, str) or not _BASENAME.fullmatch(basename):
            raise ValueError("Acquisition basename must be a portable file name")
        safety = settings.profile.safety
        # Validate the argv contract before reserving anything on disk.
        self._build_command(settings, Identity(store.run_id, ACQUISITION_STAGE, "preflight"),
                            store.root / "preflight" / basename)
        self._build_bias_off(settings, Identity(store.run_id, ACQUISITION_STAGE, "preflight-bias-off"))
        runner = self._runner or CommandRunner(policy=RunnerPolicy(terminate_grace_s=safety.terminate_grace_s),
                                               clock=self._clock)
        with self._lock:
            if self._active is not None and not self._active.done:
                raise RuntimeError("An acquisition is already active")
            handle = self._active = AcquisitionHandle()
        Thread(target=self._run_all, args=(handle, settings, store, basename, prerequisite, runner, safety),
               daemon=True, name=f"petsys-acquisition-{store.run_id}").start()
        return handle

    def _run_all(self, handle, settings, store, basename, prerequisite, runner, safety):
        attempts = []
        identity = Identity(store.run_id, ACQUISITION_STAGE, "run")
        status, message = ResultStatus.FAILED, "Acquisition did not run"
        try:
            for number in range(1, safety.max_attempts + 1):
                if handle.cancelled:
                    status, message = ResultStatus.CANCELLED, f"Stopped before attempt {number}"
                    break
                ok, reason = self._check(prerequisite)
                if not ok:
                    status, message = ResultStatus.FAILED, f"Acquisition prerequisite not met: {reason}"
                    break
                summary = self._attempt(handle, settings, store, basename, prerequisite, runner, safety, number)
                attempts.append(summary)
                identity = summary.identity
                if summary.status == ResultStatus.SUCCEEDED:
                    status = summary.status
                    message = f"Acquisition succeeded on attempt {number}/{safety.max_attempts}"
                    break
                if summary.retry_reason not in RETRYABLE_REASONS:
                    status, message = summary.status, summary.message
                    break
                if number == safety.max_attempts:
                    status = ResultStatus.FAILED
                    message = f"Acquisition failed after {number} attempts; last: {summary.message}"
                    break
                self._emit(identity, "retry_wait",
                           f"Retrying ({summary.retry_reason}) as attempt {number + 1}/{safety.max_attempts} "
                           f"in {safety.retry_delay_s:g} s",
                           {"reason": summary.retry_reason, "next_attempt": number + 1})
                if not handle.cancelled:
                    self._clock.wait(handle._cancel, safety.retry_delay_s)
                if handle.cancelled:
                    status, message = ResultStatus.CANCELLED, "Stopped during the retry delay; no further attempts"
                    break
        except Exception as exc:
            status, message = ResultStatus.FAILED, f"Acquisition worker failed: {exc}"
        outcome = AcquisitionOutcome(status, message, tuple(attempts))
        try:
            self._emit(identity, "finished", message, {"status": status.value, "attempts": len(attempts)})
        finally:
            handle._finish(outcome)

    def _inventory(self, paths):
        return tuple(Artifact(path, suffix[1:]) for suffix, path in paths.items() if _regular_size(path) is not None)

    def _attempt(self, handle, settings, store, basename, prerequisite, runner, safety, number):
        attempt = store.reserve_attempt(ACQUISITION_STAGE, attempt_id=f"attempt-{number}")
        identity = attempt.identity
        prefix = attempt.directory / basename
        paths = {suffix: Path(f"{prefix}{suffix}") for suffix in RAW_SUFFIXES}
        live = _Live(identity, paths[".rawf"])
        loss = _LossTracker()
        rejected = []
        tag = f"{number}/{safety.max_attempts}"

        def log(line):
            loss.feed(line)
            self._log(f"[{identity.attempt_id}] {line}")

        def events(event):
            if event.kind == "started":
                live.launched.set()
            elif event.kind == "exited":
                live.exited.set()

        def validate(command):
            problems = [f"{paths[suffix].name} missing or empty" for suffix in REQUIRED_RAW_SUFFIXES
                        if not _regular_size(paths[suffix])]
            if problems:
                rejected.append("No acquisition data: " + "; ".join(problems))
                return OutputValidation(False, rejected[-1])
            return OutputValidation(True, "RAW outputs present and nonempty", self._inventory(paths))

        try:
            command = self._build_command(settings, identity, prefix)
            self._emit(identity, "attempt_started", f"Starting acquisition attempt {tag}",
                       {"attempt": number, "max_attempts": safety.max_attempts, "directory": str(attempt.directory)})
            live.started_at = self._clock.monotonic()
            Thread(target=self._monitor, args=(live, safety, prerequisite), daemon=True,
                   name=f"petsys-acquisition-monitor-{identity.attempt_id}").start()
            result = runner.run(command, cancellation=_AnyCancellation(handle._cancel, live.abort),
                                validate_outputs=validate, event_sink=events, log_sink=log)
        except Exception as exc:
            result = CommandResult(identity, ResultStatus.FAILED, None, f"Acquisition attempt failed to run: {exc}")
        finally:
            with self._lock:
                live.wake.set()
                reason, detail = live.reason, live.detail
                growth = live.growth(safety.growth_window_s)

        natural = result.exit_code == 0 and reason is None and not handle.cancelled
        if not live.launched.is_set():
            bias = BiasOff("not_needed", "Acquisition tool never started")
        elif natural:
            bias = BiasOff("not_needed", "Tool ended normally; it switches bias off itself", 0)
        else:
            bias = self._bias_off(settings, identity, runner, tag)

        loss_result = FrameLoss("unknown", message="Not evaluated: attempt was stopped")
        if not handle.cancelled and reason is None:
            loss_result = loss.result(safety.max_loss_percent)
        artifacts = result.artifacts
        if handle.cancelled:
            status, message, retry = ResultStatus.CANCELLED, "Stopped by operator; no retry", None
        elif reason is not None:
            status, message, retry = ResultStatus.FAILED, detail, reason
        elif result.status == ResultStatus.SUCCEEDED and loss_result.state == "exceeded":
            status, message, retry = ResultStatus.FAILED, loss_result.message, "frame_loss"
        elif result.status == ResultStatus.SUCCEEDED:
            status, message, retry = result.status, f"RAW outputs complete; {loss_result.message}", None
        elif result.status == ResultStatus.FAILED and result.exit_code == 0 and rejected:
            status, message, retry = ResultStatus.FAILED, rejected[-1], "no_data"
        else:
            status, message, retry = result.status, result.message or result.status.value, None
        if bias.state == "unknown":
            retry = None  # Never re-enable bias for a retry while its state is unknown.
            message = f"{message}; {bias.message}"
        if status != ResultStatus.SUCCEEDED:
            artifacts = self._inventory(paths)  # Partial/failed output is recorded and kept.
        exit_code = None if result.status == ResultStatus.LAUNCH_ERROR else result.exit_code
        final = CommandResult(identity, status, exit_code, message, artifacts, result.log_tail,
                              status == ResultStatus.SUCCEEDED and result.outputs_validated)
        details = {"attempt": number, "retry_reason": retry, "growth": to_plain(growth),
                   "frame_loss": to_plain(loss_result), "bias": to_plain(bias)}
        try:
            store.finish_attempt(attempt, final, details=details)
        except Exception as exc:
            # The attempt stays partial, so no further attempt can be reserved.
            status, retry = ResultStatus.FAILED, None
            message = f"{message}; run record failed: {exc}"
        self._emit(identity, "attempt_finished", f"Attempt {tag} {status.value}: {message}",
                   {"status": status.value, "retry_reason": retry, "growth": growth.state,
                    "frame_loss": loss_result.state, "bias": bias.state})
        return AttemptSummary(identity, number, attempt.directory, prefix, status, exit_code, message, retry,
                              growth, loss_result, bias, artifacts)

    def _bias_off(self, settings, identity, runner, tag):
        """FR-19: bounded by its own timeout, deliberately not by STOP."""
        bias_identity = Identity(identity.run_id, identity.stage_id, f"{identity.attempt_id}-bias-off")
        expired = Event()
        timer = Timer(self._bias_timeout, expired.set)
        timer.daemon = True
        timer.start()
        unknown = f"SiPM bias state UNKNOWN after attempt {tag}; check the hardware and switch bias off manually"
        try:
            command = self._build_bias_off(settings, bias_identity)
            result = runner.run(command, cancellation=expired,
                                log_sink=lambda line: self._log(f"[{bias_identity.attempt_id}] {line}"))
        except Exception as exc:
            bias = BiasOff("unknown", f"{unknown} (bias-off could not run: {exc})")
        else:
            if result.status == ResultStatus.SUCCEEDED:
                bias = BiasOff("requested", "set_bias --power off exited 0", 0)
            elif expired.is_set():
                bias = BiasOff("unknown", f"{unknown} (set_bias timed out after {self._bias_timeout:g} s)",
                               result.exit_code)
            else:
                bias = BiasOff("unknown", f"{unknown} (set_bias {result.status.value}: {result.message})",
                               result.exit_code)
        finally:
            timer.cancel()
        self._emit(identity, "bias_off" if bias.state == "requested" else "bias_unknown", bias.message,
                   {"state": bias.state, "exit_code": bias.exit_code})
        return bias

    def _monitor(self, live, safety, prerequisite):
        """Reference growth monitor: startup timeout, then one growth window; STOP-free."""
        while True:
            now = self._clock.monotonic()
            try:
                size, problem = self._size(live.rawf), None
            except OSError as exc:
                size, problem = None, str(exc)
            with self._lock:
                if live.wake.is_set() or live.exited.is_set():
                    return
                if problem and not live.warned:
                    live.warned = True
                    self._log(f"[acquisition {live.identity.attempt_id}] warning: cannot check .rawf size: {problem}")
                if size is not None:
                    live.last_size = size
                if live.first_size is not None and size is not None and live.previous is not None:
                    before, previous_size = live.previous
                    rate = (size - previous_size) / (now - before) if now > before else 0.0
                    growing = size > previous_size
                    self._emit(live.identity, "rawf_progress",
                               f".rawf {size:,} bytes, {rate / 1e6:.2f} MB/s"
                               + ("" if growing else "; NOT growing since the previous check"),
                               {"size": size, "bytes_per_s": rate, "growing": growing}, log=not growing)
                verdict = None
                if live.first_size is None:
                    if size:
                        live.first_size, live.window_start, live.state = size, now, "started"
                        self._emit(live.identity, "growth_started", f".rawf started writing ({size:,} bytes)",
                                   {"size": size})
                    elif now - live.started_at >= safety.startup_timeout_s:
                        live.state = "startup_timeout"
                        verdict = ("startup_timeout",
                                   f"Early abort: .rawf did not start writing data within {safety.startup_timeout_s:g} s")
                elif live.state == "started" and now - live.window_start >= safety.growth_window_s:
                    live.growth_bytes = (size or 0) - live.first_size
                    text = (f"{live.growth_bytes:,} bytes in {safety.growth_window_s:g} s from file start "
                            f"(required >= {safety.min_growth_bytes:,})")
                    if live.growth_bytes < safety.min_growth_bytes:
                        live.state = "insufficient"
                        verdict = ("insufficient_growth", f"Early abort: .rawf growth below threshold: {text}")
                    else:
                        live.state, live.message = "passed", f"Growth check passed, RAW file growing as expected: {text}"
                        self._emit(live.identity, "growth_passed", live.message, {"growth_bytes": live.growth_bytes})
                if live.first_size is not None and size is not None:
                    live.previous = (now, size)
                if verdict is not None:
                    live.reason, live.detail = verdict
                    live.message = live.detail
                    self._emit(live.identity, "aborting", live.detail, {"reason": live.reason})
                    live.abort.set()
                    return
            ok, why = self._check(prerequisite)
            if not ok:
                with self._lock:
                    if live.wake.is_set() or live.exited.is_set():
                        return
                    live.reason, live.detail = "prerequisite_lost", f"Acquisition prerequisite lost: {why}"
                    self._emit(live.identity, "aborting", live.detail, {"reason": live.reason})
                    live.abort.set()
                return
            if not live.wake.is_set():
                self._clock.wait(live.wake, safety.poll_interval_s)
