"""Owned command runner: draining, cancellation, reaping, validation (spec 003 T3, T36; spec 007 T6).

Moved from scripts/petsys_manager_check.py --runner; real dummy children are Python subprocesses.
"""

from __future__ import annotations

from dataclasses import replace
import io
import json
import os
import subprocess
import sys
from threading import Event, Timer
import time
import unittest
from unittest.mock import patch

import pytest

from manager_helpers import DirectDummyChild, FakeBackend, FakeChild, PrivateOutput, SettingsFixtures
from src.petsys_manager.contracts import (Artifact, CommandResult, CommandSpec, Identity, OutputValidation,
    ResultStatus)
from src.petsys_manager.runner import CommandRunner, LinuxProcessBackend, RunnerPolicy


class FakeClock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def wait(self, cancellation, seconds):
        self.now += seconds
        time.sleep(0.001)  # Let fixture pipe-reader threads deliver EOF.



class DirectDummyBackend:
    def launch(self, command):
        return DirectDummyChild(subprocess.Popen(command.argv, shell=False, cwd=command.cwd,
            env=dict(command.environment), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE))



@pytest.mark.fr("003-FR-4", "003-FR-5", "003-FR-7", "003-FR-10", "003-FR-16")  # spec 003 T3
class RunnerChecks(SettingsFixtures, PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-runner-"

    def setUp(self):
        super().setUp()
        self.identity = Identity("run", "stage", "attempt")
        self.command = CommandSpec((sys.executable, "-u", "-c", "print('fixture')"), self.root,
                                   self.identity, dict(os.environ))
        self.policy = RunnerPolicy(log_tail_lines=10, poll_interval_s=.01, terminate_grace_s=.05,
                                   reap_timeout_s=.2, drain_timeout_s=.2, descendant_grace_s=.1)

    def fake_run(self, child=None, backend=None, **kwargs):
        backend = FakeBackend(child) if backend is None else backend
        clock = FakeClock()
        result = CommandRunner(policy=self.policy, backend=backend, clock=clock).run(self.command, **kwargs)
        return result, backend, clock

    def test_runner_zero_exit_without_validation_cannot_advance(self):
        result, backend, _ = self.fake_run()
        self.assertEqual(result.status, ResultStatus.SUCCEEDED)
        self.assertFalse(result.can_advance)
        self.assertTrue(backend.child.waited)

    def test_runner_validated_success_and_artifacts(self):
        artifact = Artifact(self.root / "owned/output.ldat", "fixture")
        result, _, _ = self.fake_run(validate_outputs=lambda command: OutputValidation(True, "validated", [artifact]))
        self.assertTrue(result.can_advance)
        self.assertEqual(result.artifacts, (artifact,))
        self.assertEqual(result.identity, self.identity)

    def test_runner_nonzero_exit_never_calls_validator(self):
        calls = []
        result, _, _ = self.fake_run(FakeChild(code=7), validate_outputs=lambda command: calls.append(command))
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertEqual(result.exit_code, 7)
        self.assertFalse(result.can_advance)
        self.assertEqual(calls, [])

    def test_runner_launch_error_distinct_no_child_reaping(self):
        result, backend, _ = self.fake_run(backend=FakeBackend(error=OSError("launch refused")))
        self.assertEqual(result.status, ResultStatus.LAUNCH_ERROR)
        self.assertIsNone(result.exit_code)
        self.assertFalse(backend.child.waited)

    def test_runner_precancel_does_not_launch(self):
        cancellation = Event()
        cancellation.set()
        result, backend, _ = self.fake_run(cancellation=cancellation)
        self.assertEqual(result.status, ResultStatus.CANCELLED)
        self.assertEqual(backend.launched, [])

    def test_runner_cancel_term_reaps_and_skips_validation(self):
        cancellation = Event()
        events = []
        def sink(event):
            events.append(event)
            if event.kind == "started":
                cancellation.set()
        result, backend, _ = self.fake_run(FakeChild(code=None), cancellation=cancellation, event_sink=sink,
            validate_outputs=lambda command: self.fail("validator after cancellation"))
        self.assertEqual(result.status, ResultStatus.CANCELLED)
        self.assertEqual(backend.child.signals, ["TERM"])
        self.assertTrue(backend.child.waited)
        self.assertFalse(result.can_advance)
        self.assertEqual([e.sequence for e in events], list(range(len(events))))
        self.assertEqual({e.identity for e in events}, {self.identity})
        self.assertEqual(events[-1].kind, "completed")

    def test_runner_cancel_escalates_term_resistant_group(self):
        cancellation = Event()
        events = []
        def sink(event):
            events.append(event)
            if event.kind == "started":
                cancellation.set()
        result, backend, clock = self.fake_run(FakeChild(code=None, term_exits=False),
            cancellation=cancellation, event_sink=sink)
        self.assertEqual(result.status, ResultStatus.CANCELLED)
        self.assertEqual(backend.child.signals, ["TERM", "KILL"])
        self.assertGreaterEqual(clock.now, self.policy.terminate_grace_s)
        self.assertIn("killing", [event.kind for event in events])
        self.assertTrue(backend.child.waited)

    def test_runner_cancel_kills_descendants_even_if_parent_exits_on_term(self):
        cancellation = Event()
        def sink(event):
            if event.kind == "started":
                cancellation.set()
        result, backend, _ = self.fake_run(FakeChild(code=None, descendants=True),
            cancellation=cancellation, event_sink=sink)
        self.assertEqual(result.status, ResultStatus.CANCELLED)
        self.assertEqual(backend.child.signals, ["TERM", "KILL"])

    def test_runner_normal_exit_with_leftover_descendants_fails_and_kills(self):
        result, backend, _ = self.fake_run(FakeChild(code=0, descendants=True),
            validate_outputs=lambda command: self.fail("leftover descendants must not validate"))
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("descendants", result.message)
        self.assertEqual(backend.child.signals, ["TERM", "KILL"])

    def test_runner_descendant_exiting_within_grace_is_not_a_leftover(self):
        """T25.5 Cornell finding: multiprocessing's resource tracker exits just after the CLI child; a group that
        empties within ``descendant_grace_s`` succeeds and is not signalled; one that stays still fails."""
        result, backend, clock = self.fake_run(FakeChild(code=0, linger_checks=5),
                                               validate_outputs=lambda command: OutputValidation(True))
        self.assertEqual(result.status, ResultStatus.SUCCEEDED, result.message)
        self.assertEqual(backend.child.signals, [])
        self.assertLess(clock.now, self.policy.descendant_grace_s)
        result, backend, clock = self.fake_run(FakeChild(code=0, descendants=True))
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertIn("descendants", result.message)
        self.assertEqual(backend.child.signals, ["TERM", "KILL"])
        self.assertGreaterEqual(clock.now, self.policy.descendant_grace_s)
        with self.assertRaises(ValueError):
            RunnerPolicy(descendant_grace_s=0)

    def test_runner_missing_required_output_and_validator_errors_fail(self):
        for validator in (lambda command: OutputValidation(False, "required output missing"),
                          lambda command: True):
            result, _, _ = self.fake_run(validate_outputs=validator)
            self.assertEqual(result.status, ResultStatus.FAILED)
            self.assertEqual(result.exit_code, 0)
            self.assertFalse(result.can_advance)
        def raises(command):
            raise ValueError("corrupted required output")
        result, _, _ = self.fake_run(validate_outputs=raises)
        self.assertIn("corrupted", result.message)

    def test_runner_cancel_during_validation_no_advance(self):
        cancellation = Event()
        def validator(command):
            cancellation.set()
            return OutputValidation(True)
        result, _, _ = self.fake_run(cancellation=cancellation, validate_outputs=validator)
        self.assertEqual(result.status, ResultStatus.CANCELLED)
        self.assertFalse(result.can_advance)

    def test_runner_cancel_at_result_publication_no_advance(self):
        cancellation = Event()
        def sink(event):
            if event.kind == "completed":
                cancellation.set()
        result, _, _ = self.fake_run(cancellation=cancellation, event_sink=sink,
            validate_outputs=lambda command: OutputValidation(True))
        self.assertEqual(result.status, ResultStatus.CANCELLED)
        self.assertFalse(result.can_advance)

    def test_runner_sink_error_cleans_up_no_success(self):
        def fails(event):
            if event.kind == "started":
                raise OSError("event storage unavailable")
        result, backend, _ = self.fake_run(FakeChild(code=None), event_sink=fails)
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertIn("sink failed", result.message)
        self.assertTrue(backend.child.waited)
        self.assertIn("TERM", backend.child.signals)
        def log_fails(line):
            raise OSError("full log destination")
        result, _, _ = self.fake_run(log_sink=log_fails)
        self.assertEqual(result.status, ResultStatus.FAILED)

    def test_runner_reader_and_cleanup_failures_visible(self):
        child = FakeChild()
        child.stdout = io.BytesIO()
        child.stdout.close()
        result, _, _ = self.fake_run(child)
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertIn("reader failed", result.message)
        child = FakeChild(code=None, term_exits=False)
        def ineffective_kill():
            child.signals.append("KILL")
        child.kill = ineffective_kill
        cancellation = Event()
        def sink(event):
            if event.kind == "started":
                cancellation.set()
        result, _, _ = self.fake_run(child, cancellation=cancellation, event_sink=sink)
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertIn("reap", result.message)
        self.assertIsNone(result.exit_code)

    def test_runner_partial_thread_startup_failure_kills_reaps(self):
        with patch("src.petsys_manager.runner.Thread.start", side_effect=RuntimeError("thread startup failed")):
            result, backend, _ = self.fake_run(FakeChild(code=None))
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertEqual(backend.child.signals, ["KILL"])
        self.assertTrue(backend.child.waited)

    def test_runner_term_signal_failure_still_kills_and_reaps(self):
        child = FakeChild(code=None)
        def denied():
            raise PermissionError("TERM denied")
        child.terminate = denied
        cancellation = Event()
        def sink(event):
            if event.kind == "started":
                cancellation.set()
        result, _, _ = self.fake_run(child, cancellation=cancellation, event_sink=sink)
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertIn("TERM denied", result.message)
        self.assertEqual(child.signals, ["KILL"])
        self.assertTrue(child.waited)

    def test_runner_poll_failure_cannot_bypass_kill_and_wait(self):
        child = FakeChild(code=None)
        def poll_fails():
            raise OSError("poll unavailable")
        child.poll = poll_fails
        result, _, _ = self.fake_run(child)
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertIn("poll unavailable", result.message)
        self.assertEqual(child.signals, ["KILL"])
        self.assertTrue(child.waited)

    def test_runner_terminal_event_sink_failure_prevents_advance(self):
        def sink(event):
            if event.kind == "completed":
                raise OSError("terminal event unavailable")
        result, _, _ = self.fake_run(event_sink=sink,
            validate_outputs=lambda command: OutputValidation(True))
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertFalse(result.can_advance)

    def test_runner_linux_launch_and_signals_target_only_owned_group(self):
        child = FakeChild()
        with patch("src.petsys_manager.runner.sys.platform", "linux"), \
                patch("src.petsys_manager.runner.subprocess.Popen", return_value=child) as launch:
            owned = LinuxProcessBackend().launch(self.command)
        kwargs = launch.call_args.kwargs
        self.assertEqual(launch.call_args.args, (self.command.argv,))
        self.assertFalse(kwargs["shell"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["cwd"], self.root)
        self.assertEqual(kwargs["env"], dict(self.command.environment))
        with patch("src.petsys_manager.runner.os.killpg", create=True) as killpg:
            owned.terminate()
            # Windows has no SIGKILL attribute; fixture pins the exact Linux value.
            with patch("src.petsys_manager.runner.signal.SIGKILL", 9, create=True):
                owned.kill()
            self.assertEqual([call.args[0] for call in killpg.call_args_list], [child.pid, child.pid])
            self.assertEqual([int(call.args[1]) for call in killpg.call_args_list], [15, 9])
        with patch("src.petsys_manager.runner.os.killpg", side_effect=ProcessLookupError, create=True):
            self.assertFalse(owned.group_alive())
            owned.terminate()

    def test_runner_production_backend_refuses_windows(self):
        with patch("src.petsys_manager.runner.sys.platform", "win32"), \
                patch("src.petsys_manager.runner.subprocess.Popen") as launch:
            result = CommandRunner().run(self.command)
        self.assertEqual(result.status, ResultStatus.LAUNCH_ERROR)
        launch.assert_not_called()

    def test_runner_real_dummy_literal_paths_cwd_env_and_validated_file(self):
        script = self.root / "private/dummy ; & [x].py"
        output = self.root / "private/output ; & [x].json"
        script.write_text("import json, os, pathlib, sys\n"
            "with pathlib.Path(sys.argv[1]).open('x') as f:\n"
            " json.dump({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'env': os.environ['LITERAL']}, f)\n"
            "print('dummy complete')\n", encoding="utf-8")
        env = dict(os.environ, LITERAL="value ; & [literal]")
        command = CommandSpec((sys.executable, "-u", str(script), str(output), "literal ; & [arg] $HOME"),
                              self.root, self.identity, env)
        result = CommandRunner(policy=self.policy, backend=DirectDummyBackend()).run(command,
            validate_outputs=lambda command: OutputValidation(output.is_file(),
                "dummy output check", (Artifact(output, "fixture"),)))
        self.assertTrue(result.can_advance, result.message)
        content = json.loads(output.read_text())
        self.assertEqual(content, {"argv": [str(output), "literal ; & [arg] $HOME"],
                                  "cwd": str(self.root), "env": "value ; & [literal]"})

    def test_runner_real_dummy_floods_both_pipes_without_deadlock_bounded_tail(self):
        program = ("import os, threading\n"
            "def write(fd, value):\n"
            " for _ in range(4096): os.write(fd, value * 120 + b'\\n')\n"
            "threads = [threading.Thread(target=write, args=(1,b'A')), threading.Thread(target=write, args=(2,b'B'))]\n"
            "[t.start() for t in threads]\n[t.join() for t in threads]\n")
        command = replace(self.command, argv=(sys.executable, "-u", "-c", program))
        counts = {"A": 0, "B": 0}
        def logger(line):
            for value in counts:
                counts[value] += line.count(value)
        cancellation = Event()
        watchdog = Timer(15, cancellation.set)
        watchdog.start()
        try:
            result = CommandRunner(policy=replace(self.policy, poll_interval_s=.001),
                backend=DirectDummyBackend()).run(command, cancellation=cancellation, log_sink=logger)
        finally:
            watchdog.cancel()
            watchdog.join()
        self.assertEqual(result.status, ResultStatus.SUCCEEDED, result.message)
        self.assertEqual(counts, {"A": 4096 * 120, "B": 4096 * 120})
        self.assertEqual(len(result.log_tail), 10)
        self.assertTrue(all(len(line) <= self.policy.line_chars + 9 for line in result.log_tail))

    def test_runner_real_dummy_cancellation_terminates_and_reaps_direct_child(self):
        command = replace(self.command, argv=(sys.executable, "-u", "-c",
            "import time; print('READY', flush=True); time.sleep(60)"))
        cancellation = Event()
        def sink(event):
            if event.kind == "log" and "READY" in event.message:
                cancellation.set()
        watchdog = Timer(10, cancellation.set)
        watchdog.start()
        try:
            result = CommandRunner(policy=self.policy, backend=DirectDummyBackend()).run(command,
                cancellation=cancellation, event_sink=sink,
                validate_outputs=lambda command: self.fail("validator after STOP"))
        finally:
            watchdog.cancel()
            watchdog.join()
        self.assertEqual(result.status, ResultStatus.CANCELLED, result.message)
        self.assertIsNotNone(result.exit_code)
        self.assertFalse(result.can_advance)

    def test_runner_real_dummy_nonzero_and_spawn_error_are_distinct(self):
        runner = CommandRunner(policy=self.policy, backend=DirectDummyBackend())
        nonzero = runner.run(replace(self.command, argv=(sys.executable, "-c", "import sys; sys.exit(7)")),
            validate_outputs=lambda command: self.fail("validator after nonzero exit"))
        missing = runner.run(replace(self.command, argv=(str(self.root / "missing-executable"),)))
        self.assertEqual(nonzero.status, ResultStatus.FAILED)
        self.assertEqual(nonzero.exit_code, 7)
        self.assertEqual(missing.status, ResultStatus.LAUNCH_ERROR)
        self.assertIsNone(missing.exit_code)
        self.assertFalse(nonzero.can_advance or missing.can_advance)

    def test_runner_huge_unterminated_utf8_lines_are_fragmented_not_retained(self):
        policy = replace(self.policy, chunk_bytes=127, line_chars=256, queue_chunks=8)
        stdout = ("á" * 3000).encode() + b'\xff'
        counts = {"characters": 0}
        def logger(line):
            counts["characters"] += len(line[9:])
        result = CommandRunner(policy=policy, backend=FakeBackend(FakeChild(stdout=stdout)),
            clock=FakeClock()).run(self.command, log_sink=logger)
        self.assertEqual(result.status, ResultStatus.SUCCEEDED)
        self.assertEqual(counts["characters"], 3001)
        self.assertTrue(all(len(line) <= 265 for line in result.log_tail))
        self.assertIn("�", result.log_tail[-1])

    @pytest.mark.fr("003-FR-1")
    def test_runner_carriage_return_progress_is_logged_at_most_every_interval(self):
        """T36: like a terminal, a '\r' not followed by '\n' overwrites the line. acquire_sipm_data
        writes one '\r' progress line per read, ten a second (2026-10-06 Cornell: 398 log lines for
        30 s); the newest is logged at most every progress_interval_s; '\n' lines are unchanged."""
        def run(stdout, chunk_bytes, clock=None):
            lines = []
            result = CommandRunner(policy=replace(self.policy, chunk_bytes=chunk_bytes),
                backend=FakeBackend(FakeChild(stdout=stdout)), clock=clock or FakeClock()).run(
                self.command, log_sink=lines.append)
            self.assertEqual(result.status, ResultStatus.SUCCEEDED, result.message)
            self.assertTrue(all("\r" not in line for line in lines), lines)
            return lines

        clock = FakeClock()

        class Paced:
            """One write per read, 0.1 s of clock apart, as the real tool flushes."""
            def __init__(self, parts):
                self.parts = list(parts)

            def read1(self, size):
                if not self.parts:
                    child.code = 0    # exits after its last write, like the tool
                    return b""
                clock.now += 0.1
                return self.parts.pop(0)

            read = read1

            def close(self):
                pass

        parts = ([b"INFO: Setting BIAS power  ON\n"]
                 + [f"Python:: Acquired {19535 * i} frames in {i / 10:4.1f} seconds\r".encode() for i in range(1, 301)]
                 + [b"\nINFO: Setting BIAS power OFF\n"])
        child = FakeChild(None, stdout=b"")
        child.stdout = Paced(parts)
        lines = []
        result = CommandRunner(policy=self.policy, backend=FakeBackend(child), clock=clock).run(
            self.command, log_sink=lines.append)
        self.assertEqual(result.status, ResultStatus.SUCCEEDED, result.message)
        acquired = [line for line in lines if "Acquired" in line]
        self.assertTrue(4 <= len(acquired) <= 12, acquired)                     # ~30 s / 5 s, not 300
        self.assertTrue(all(line.count("Acquired") == 1 and line.startswith("[stdout] Python:: Acquired ")
                            and "\r" not in line for line in acquired), acquired)
        self.assertEqual(acquired[0], "[stdout] Python:: Acquired 19535 frames in  0.1 seconds")
        frames = [int(line.split()[3]) for line in acquired]
        self.assertEqual(frames, sorted(set(frames)))
        self.assertEqual(lines[0], "[stdout] INFO: Setting BIAS power  ON")
        self.assertEqual(lines[-2:], ["[stdout] Python:: Acquired 5860500 frames in 30.0 seconds",
                                      "[stdout] INFO: Setting BIAS power OFF"])
        self.assertEqual(run(b"plain\rover\nx\ry\rz\n", 64), ["[stdout] over", "[stdout] z"])
        self.assertEqual(run(b"abc\r\ndef\r\n", 4), ["[stdout] abc", "[stdout] def"])   # CRLF split across reads
        self.assertEqual(run(b"a\rb\rc\r", 2), ["[stdout] a", "[stdout] c"])  # first, then the last at EOF
        with self.assertRaises(ValueError):
            RunnerPolicy(progress_interval_s=0)

    def test_runner_policy_and_validation_contract_bounds(self):
        for kwargs in ({"queue_chunks": 0}, {"line_chars": 0}, {"log_tail_lines": 0},
                       {"terminate_grace_s": float("nan")}, {"reap_timeout_s": -1}, {"poll_interval_s": True},
                       {"reap_timeout_s": 10**1000}):
            with self.assertRaises(ValueError):
                RunnerPolicy(**kwargs)
        with self.assertRaises(ValueError):
            OutputValidation(False)
        with self.assertRaises(ValueError):
            CommandResult(self.identity, "failed", 2, outputs_validated=True)
