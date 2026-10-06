"""Blocking, toolkit-free owned-command runner; call from a workflow worker.

Linux launches use a fresh session and signal only that owned process group.
Pipe readers feed a bounded queue; line fragments and the returned tail are
bounded too. Sinks must consume promptly (or provide bounded queueing), never
touch Tk. No callbacks launch later stages: callers must check can_advance.
"""

from __future__ import annotations

import codecs
from collections import deque
from dataclasses import dataclass
import math
import os
from queue import Empty, Full, Queue
import select
import signal
import subprocess
import sys
from threading import Event, Thread
import time

from .contracts import CommandResult, CommandSpec, OutputValidation, ResultStatus, RunEvent


@dataclass(frozen=True)
class RunnerPolicy:
    log_tail_lines: int = 1000
    chunk_bytes: int = 4096
    queue_chunks: int = 32
    line_chars: int = 4096
    poll_interval_s: float = 0.02
    terminate_grace_s: float = 3.0
    reap_timeout_s: float = 5.0
    drain_timeout_s: float = 1.0
    descendant_grace_s: float = 2.0   # helpers (e.g. multiprocessing's resource tracker) exit just after the child

    def __post_init__(self):
        for name in ("log_tail_lines", "chunk_bytes", "queue_chunks", "line_chars"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("poll_interval_s", "terminate_grace_s", "reap_timeout_s", "drain_timeout_s",
                     "descendant_grace_s"):
            value = getattr(self, name)
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class Clock:
    def monotonic(self):
        return time.monotonic()

    def wait(self, cancellation, seconds):
        if cancellation.is_set():
            time.sleep(seconds)  # Avoid busy-waiting through TERM/KILL grace.
        else:
            cancellation.wait(seconds)


class LinuxChild:
    """Only created from this backend's successful start_new_session launch."""

    selectable_pipes = True

    def __init__(self, process):
        self.process = process
        self.pid = process.pid
        self.stdout = process.stdout
        self.stderr = process.stderr

    def poll(self):
        return self.process.poll()

    def wait(self, timeout):
        return self.process.wait(timeout=timeout)

    def group_alive(self):
        try:
            os.killpg(self.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def _signal(self, sig):
        try:
            os.killpg(self.pid, sig)
        except ProcessLookupError:
            pass  # A concurrently exited owned group needs no signal.

    def terminate(self):
        self._signal(signal.SIGTERM)

    def kill(self):
        self._signal(signal.SIGKILL)


class LinuxProcessBackend:
    def launch(self, command):
        if not sys.platform.startswith("linux"):
            raise OSError("Production command execution requires Linux; use an explicit dummy backend for checks")
        process = subprocess.Popen(command.argv, shell=False, cwd=command.cwd,
                                   env=dict(command.environment), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        return LinuxChild(process)


class CommandRunner:
    def __init__(self, *, policy=None, backend=None, clock=None):
        self.policy = RunnerPolicy() if policy is None else policy
        if not isinstance(self.policy, RunnerPolicy):
            raise ValueError("Runner requires a typed policy")
        self.backend = LinuxProcessBackend() if backend is None else backend
        self.clock = Clock() if clock is None else clock

    def run(self, command, *, cancellation=None, validate_outputs=None, event_sink=None, log_sink=None):
        """Run/reap one command. Exit zero without validation cannot advance.

        Validator returns OutputValidation; exceptions or missing outputs fail
        closed. Sink/reader/cleanup errors are failures, never successful runs.
        Direct child is reaped here; killed descendants are reparented/reaped by
        Linux's init. No global subreaper policy is changed by this library.
        """
        if not isinstance(command, CommandSpec):
            raise ValueError("Runner requires a typed command")
        cancel = Event() if cancellation is None else cancellation
        policy = self.policy
        tail = deque(maxlen=policy.log_tail_lines)
        failures = []
        sequence = 0
        sink = event_sink
        logger = log_sink

        def fail(message):
            if message not in failures:
                failures.append(message)

        def emit(kind, message="", payload=None):
            nonlocal sequence, sink
            event = RunEvent(command.identity, sequence, kind, message, payload or {})
            sequence += 1
            if sink is not None:
                try:
                    sink(event)
                except Exception as exc:
                    fail(f"Event sink failed: {exc}")
                    sink = None

        def log(stream, text):
            nonlocal logger
            line = f"[{stream}] {text}"
            tail.append(line)
            if logger is not None:
                try:
                    logger(line)
                except Exception as exc:
                    fail(f"Log sink failed: {exc}")
                    logger = None
            emit("log", line, {"stream": stream})

        def result(status, code=None, message="", validation=None):
            # Emission itself can fail, so build the final status afterwards.
            emit("completed", message, {"status": status.value, "exit_code": code,
                                      "outputs_validated": bool(validation and validation.valid)})
            if cancel.is_set() and status == ResultStatus.SUCCEEDED:
                status = ResultStatus.CANCELLED
                message = "Cancelled before publishing the command result"
                validation = None
            if failures:
                status = ResultStatus.FAILED
                message = "; ".join(failures)
                validation = None
            return CommandResult(command.identity, status, code, message,
                                 validation.artifacts if validation else (), tuple(tail),
                                 bool(validation and validation.valid))

        if cancel.is_set():
            return result(ResultStatus.CANCELLED, message="Cancelled before launch")
        emit("starting")
        if failures:
            return result(ResultStatus.FAILED, message="; ".join(failures))
        try:
            child = self.backend.launch(command)
        except Exception as exc:
            return result(ResultStatus.LAUNCH_ERROR, message=f"Launch failed: {exc}")

        chunks = Queue(maxsize=policy.queue_chunks)
        stop_readers = Event()
        readers = []
        decoders = {name: codecs.getincrementaldecoder("utf-8")("replace") for name in ("stdout", "stderr")}
        pending = {name: "" for name in decoders}
        ended = set()
        code = None
        stopping_at = None
        exited_at = None
        killed_at = None

        def put(item):
            while not stop_readers.is_set():
                try:
                    chunks.put(item, timeout=policy.poll_interval_s)
                    return
                except Full:
                    pass

        def read_pipe(name, pipe):
            try:
                while not stop_readers.is_set():
                    if child.selectable_pipes:
                        if not select.select([pipe], [], [], policy.poll_interval_s)[0]:
                            continue
                        data = os.read(pipe.fileno(), policy.chunk_bytes)
                    else:  # Explicit test seam; production Linux uses select.
                        data = (getattr(pipe, "read1", pipe.read))(policy.chunk_bytes)
                    if not data:
                        break
                    put((name, data, None))
            except Exception as exc:
                put((name, b"", f"{name} reader failed: {exc}"))
            finally:
                put((name, None, None))

        def consume(name, data, error):
            # Like a terminal, a "\r" not followed by "\n" overwrites its line (T36): progress written
            # with "\r" only (acquire_sipm_data) logs its newest line once per chunk; one followed by a
            # "\n" line is overwritten without a log line. "\r" at the end of a chunk waits for the next.
            if error:
                fail(error)
            final = data is None
            pending[name] += decoders[name].decode(data or b"", final=final)
            overwritten = None
            while pending[name]:
                text = pending[name]
                newline = text.find("\n")
                limit = newline if newline >= 0 else len(text)
                carriage = text.find("\r", 0, limit)
                if 0 <= carriage <= policy.line_chars and carriage + 1 != newline and (
                        carriage + 1 < len(text) or final):
                    overwritten = text[:carriage]
                    pending[name] = text[carriage + 1:]
                elif 0 <= newline <= policy.line_chars:
                    overwritten = None
                    log(name, text[:newline].rstrip("\r"))
                    pending[name] = text[newline + 1:]
                elif len(text) >= policy.line_chars:
                    if overwritten is not None:
                        log(name, overwritten)
                        overwritten = None
                    log(name, text[:policy.line_chars])
                    pending[name] = text[policy.line_chars:]
                elif final:
                    log(name, text)
                    pending[name] = ""
                else:
                    break
            if overwritten:
                log(name, overwritten)
            if final:
                ended.add(name)

        try:
            emit("started", payload={"pid": child.pid})
            for name in decoders:
                reader = Thread(target=read_pipe, args=(name, getattr(child, name)),
                                name=f"petsys-{name}-{child.pid}", daemon=True)
                reader.start()
                readers.append(reader)
            while True:
                # Drain a bounded number per turn; STOP cannot starve in a flood.
                for _ in range(policy.queue_chunks):
                    try:
                        consume(*chunks.get_nowait())
                    except Empty:
                        break
                now = self.clock.monotonic()
                code = child.poll()
                if code is not None and exited_at is None:
                    exited_at = now
                    emit("exited", payload={"exit_code": code})  # Monitors stop judging a finished child.
                # A descendant still alive after the grace is a leftover (FR-7); within it, keep polling.
                lingering = code is not None and stopping_at is None and child.group_alive()
                if lingering and now - exited_at >= policy.descendant_grace_s:
                    fail("Child exited while owned descendants remained; terminating the group")
                if (stopping_at is None and not cancel.is_set() and exited_at is not None and
                        len(ended) < 2 and now - exited_at >= policy.drain_timeout_s):
                    fail("Output drain timed out after child exit")
                if stopping_at is None and (cancel.is_set() or failures):
                    stopping_at = now
                    emit("terminating", "Cancellation requested" if cancel.is_set() else "; ".join(failures))
                    child.terminate()
                if stopping_at is not None and killed_at is None and now - stopping_at >= policy.terminate_grace_s:
                    if child.group_alive():
                        emit("killing", "TERM grace expired; killing owned group")
                        child.kill()
                    killed_at = now
                # After KILL, zombies can keep a PGID present until init reaps
                # them; wait for our child and EOF, not forever for killpg(0).
                stopped = stopping_at is None or killed_at is not None or not child.group_alive()
                if code is not None and len(ended) == 2 and stopped and not lingering:
                    break
                if killed_at is not None and now - killed_at >= policy.reap_timeout_s:
                    fail("Owned child/output cleanup timed out after KILL")
                    break
                self.clock.wait(cancel, policy.poll_interval_s)
        except Exception as exc:
            fail(f"Runner failed: {exc}")
        finally:
            # Also covers partial reader startup, sink failures and exceptions.
            try:
                needs_kill = child.poll() is None or (failures and child.group_alive())
            except Exception as exc:
                fail(f"Failed to query owned child during cleanup: {exc}")
                needs_kill = True
            if needs_kill:
                try:
                    child.kill()
                except Exception as exc:
                    fail(f"Failed to kill owned child: {exc}")
            try:
                code = child.wait(policy.reap_timeout_s)
            except Exception as exc:
                fail(f"Failed to terminate/reap owned child: {exc}")
            stop_readers.set()
            for reader in readers:
                reader.join(timeout=policy.drain_timeout_s)
                if reader.is_alive():
                    fail(f"Pipe reader did not stop: {reader.name}")
            if not any(reader.is_alive() for reader in readers):
                for name in decoders:
                    try:
                        getattr(child, name).close()
                    except Exception as exc:
                        fail(f"Failed to close {name}: {exc}")

        if failures:
            return result(ResultStatus.FAILED, code, "; ".join(failures))
        if cancel.is_set():
            return result(ResultStatus.CANCELLED, code, "Cancelled; owned child reaped")
        if code != 0:
            return result(ResultStatus.FAILED, code, f"Command exited with code {code}")
        validation = None
        if validate_outputs is not None:
            try:
                validation = validate_outputs(command)
                if not isinstance(validation, OutputValidation):
                    raise ValueError("Validator must return OutputValidation")
            except Exception as exc:
                return result(ResultStatus.FAILED, code, f"Output validation failed: {exc}")
            if not validation.valid:
                return result(ResultStatus.FAILED, code, validation.message, validation)
        if cancel.is_set():
            return result(ResultStatus.CANCELLED, code, "Cancelled during output validation")
        return result(ResultStatus.SUCCEEDED, code,
                      validation.message if validation else "Exit zero; required outputs not yet validated", validation)
