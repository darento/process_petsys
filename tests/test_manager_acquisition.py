"""Acquisition attempts, monitoring, retries and bias-off (spec 003 T7, FR-19, FR-20; spec 007 T5).

Moved from scripts/petsys_manager_acquisition_check.py. No PETsys tools or hardware: fake
children write small real files into their run-store attempt directory, a fake size probe
scripts .rawf growth, and a scaled fake clock runs the reference safety defaults 100x faster.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from pathlib import Path
import threading
from threading import Event, Lock
import time
from typing import Callable
import unittest

import pytest

from helpers import REPO
from manager_helpers import FakeChild, FixtureProbe, PrivateOutput, profile_fixture
from src.petsys_manager.acquisition import (AcquisitionService, AttemptFilter, parse_frame_loss)
from src.petsys_manager.commands import build_bias_off
from src.petsys_manager.artifacts import RunStore, read_manifest
from src.petsys_manager.contracts import Action, CommandSpec, Identity, ResultStatus, RunEvent
from src.petsys_manager.runner import CommandRunner, RunnerPolicy
from src.petsys_manager.settings import AcquisitionSafety, preflight, profile_from_mapping

SPEED = 100.0
RUNNER = RunnerPolicy(poll_interval_s=0.2, terminate_grace_s=3.0, reap_timeout_s=10.0, drain_timeout_s=10.0)
GOOD_FILES = {".rawf": b"R" * 64, ".idxf": b"0\t64\t0\t10\t0.0\t0.0\n", ".tmpf": b"0.0\t0.0\t0\t0\t",
              ".modf": b"#portID\tslaveID\tchipID\tchannelID\tmode\n"}


def loss_line(percent):
    return (f"writeRaw:: some events were lost for 3 (  0.1%) frames; "
            f"all events were lost for 7 ({percent:5.1f}%) frames")


def growing(t):
    return None if t < 2 else int(64 + (t - 2) * 2e6)  # 40 MB per 20 s window


def slow(t):
    return None if t < 2 else int(64 + (t - 2) * 1e4)


def no_data(t):
    return 0


def stalls_after_30(t):
    return growing(min(t, 30.0))


class ScaledClock:
    """Fake monotonic time at SPEED x real time; waits block on their event, so wakes are prompt."""

    def __init__(self, speed=SPEED):
        self.speed = speed
        self.t0 = time.monotonic()
        self.threads = set()
        self.lock = Lock()

    def monotonic(self):
        return (time.monotonic() - self.t0) * self.speed

    def wait(self, cancellation, seconds):
        with self.lock:
            self.threads.add(threading.current_thread().name)
        if cancellation.is_set():
            time.sleep(seconds / self.speed)
        else:
            cancellation.wait(seconds / self.speed)


@dataclass
class Scenario:
    end: float | None = 10.0            # fake seconds after launch; None runs until signalled
    code: int = 0
    files: dict = field(default_factory=lambda: dict(GOOD_FILES))
    stderr: bytes = (loss_line(0.0) + "\n").encode()
    size: Callable = growing
    term_exits: bool = True
    error: Exception | None = None


class ScenarioChild(FakeChild):
    def __init__(self, clock, scenario):
        super().__init__(None, term_exits=scenario.term_exits, stdout=b"Python:: Acquired frames\n",
                         stderr=scenario.stderr)
        self.clock, self.scenario, self.t0 = clock, scenario, clock.monotonic()

    def poll(self):
        end = self.scenario.end
        if self.code is None and end is not None and self.clock.monotonic() - self.t0 >= end:
            self.code = self.scenario.code
        return self.code


class ScenarioBackend:
    def __init__(self, clock, scenarios):
        self.clock = clock
        self.scenarios = list(scenarios)
        self.launched, self.children, self.sizes = [], [], {}
        self.bias, self.order = [], []
        self.bias_code, self.bias_hang, self.bias_error = 0, False, None
        self.lock = Lock()

    def launch(self, command):
        if Path(command.argv[0]).name == "set_bias":
            with self.lock:
                self.bias.append(command)
                self.order.append("set_bias")
            if self.bias_error:
                raise self.bias_error
            return FakeChild(None if self.bias_hang else self.bias_code, stdout=b"", stderr=b"")
        with self.lock:
            self.order.append("acquire")
            scenario = self.scenarios[len(self.launched)]
            self.launched.append(command)
        if scenario.error:
            raise scenario.error
        prefix = command.argv[command.argv.index("-o") + 1]
        for suffix, data in scenario.files.items():
            Path(prefix + suffix).write_bytes(data)
        child = ScenarioChild(self.clock, scenario)
        with self.lock:
            self.children.append(child)
            self.sizes[prefix + ".rawf"] = (child.t0, scenario.size)
        return child

    def size(self, path):
        with self.lock:
            entry = self.sizes.get(str(path))
        if entry is None:
            return None
        t0, curve = entry
        return curve(self.clock.monotonic() - t0)


@pytest.mark.fr("003-FR-5", "003-FR-7", "003-FR-8", "003-FR-9", "003-FR-16")  # spec 003 T7
class AcquisitionChecks(PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-acquisition-"

    def setUp(self):
        self.root = self.output / self._testMethodName
        self.profile = profile_fixture(self.root)
        report = preflight(self.profile, Action.ACQUIRE, repo_root=self.root, probe=FixtureProbe())
        self.assertTrue(report.ready, report.issues)
        self.settings = report.settings
        self.updates, self.logs = [], []
        self.update_threads = set()

    def with_safety(self, **changes):
        return replace(self.settings, profile=replace(self.settings.profile,
                       safety=replace(self.settings.profile.safety, **changes)))

    def service(self, scenarios, *, size_probe=None, on_update=None, **options):
        self.clock = ScaledClock()
        self.backend = ScenarioBackend(self.clock, scenarios)
        def sink(event):
            self.update_threads.add(threading.current_thread().name)
            self.updates.append(event)
            if on_update is not None:
                on_update(event)
        return AcquisitionService(runner=CommandRunner(policy=RUNNER, backend=self.backend, clock=self.clock),
                                  clock=self.clock, size_probe=size_probe or self.backend.size,
                                  update_sink=sink, log_sink=self.logs.append, **options)

    def store(self, settings=None):
        return RunStore.reserve(self.root / "data", settings or self.settings)

    def run_to_end(self, scenarios, *, settings=None, prerequisite=None, backend_setup=None, **service_options):
        service = self.service(scenarios, **service_options)
        if backend_setup is not None:
            backend_setup(self.backend)
        store = self.store(settings)
        handle = service.start(settings or self.settings, store, prerequisite=prerequisite)
        outcome = handle.wait(30)
        self.assertIsNotNone(outcome, "acquisition did not finish")
        return outcome, store, handle

    def manifest_attempts(self, store):
        return read_manifest(store.root)["attempts"]

    def kinds(self):
        return [event.kind for event in self.updates]

    # Success paths ----------------------------------------------------------

    def test_acquisition_adequate_growth_succeeds(self):
        outcome, store, _ = self.run_to_end([Scenario(end=60)])
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        attempt = outcome.attempts[0]
        self.assertEqual((attempt.number, attempt.retry_reason, attempt.growth.state), (1, None, "passed"))
        self.assertGreaterEqual(attempt.growth.growth_bytes, 20_000_000)
        self.assertEqual((attempt.loss.state, attempt.loss.all_lost_percent), ("within", 0.0))
        self.assertEqual(outcome.raw_prefix, store.root / "acquisition/attempt-1/acquisition")
        self.assertEqual(sorted(a.kind for a in attempt.artifacts), ["idxf", "modf", "rawf", "tmpf"])
        argv = self.backend.launched[0].argv
        self.assertEqual(argv[argv.index("-o") + 1], str(outcome.raw_prefix))
        self.assertEqual(argv[argv.index("--time") + 1], str(self.settings.options.duration_s))
        record = self.manifest_attempts(store)[0]
        self.assertEqual((record["status"], record["attempt_id"]), ("succeeded", "attempt-1"))
        self.assertEqual(record["details"]["growth"]["state"], "passed")
        self.assertEqual(record["details"]["frame_loss"]["state"], "within")
        self.assertTrue(all(item["validated"] for item in record["artifacts"]))
        self.assertIn("growth_passed", self.kinds())

    def test_acquisition_short_success_records_unexercised_growth(self):
        outcome, store, _ = self.run_to_end([Scenario(end=10)])
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        growth = outcome.attempts[0].growth
        self.assertEqual(growth.state, "not_exercised")
        self.assertIsNone(growth.growth_bytes)
        self.assertIn("before the 20 s growth window", growth.message)
        self.assertEqual(self.manifest_attempts(store)[0]["details"]["growth"]["state"], "not_exercised")
        self.assertNotIn("growth_passed", self.kinds())

    # Monitor aborts and retries -------------------------------------------

    def test_acquisition_startup_timeout_retries_until_three_attempts_exhausted(self):
        never = dict(end=None, size=no_data, files={".rawf": b"", ".idxf": b""})
        outcome, store, _ = self.run_to_end([Scenario(**never) for _ in range(3)])
        self.assertEqual(outcome.status, ResultStatus.FAILED)
        self.assertIn("after 3 attempts", outcome.message)
        self.assertEqual([a.retry_reason for a in outcome.attempts], ["startup_timeout"] * 3)
        self.assertEqual([a.growth.state for a in outcome.attempts], ["startup_timeout"] * 3)
        self.assertTrue(all(child.signals[:1] == ["TERM"] for child in self.backend.children))
        self.assertEqual(len(self.backend.launched), 3)
        records = self.manifest_attempts(store)
        self.assertEqual([r["attempt_id"] for r in records], ["attempt-1", "attempt-2", "attempt-3"])
        self.assertEqual({r["status"] for r in records}, {"failed"})
        for number in (1, 2, 3):  # every failed attempt's (empty) files are kept
            self.assertTrue((store.root / f"acquisition/attempt-{number}/acquisition.rawf").exists())
        self.assertEqual(self.kinds().count("retry_wait"), 2)
        self.assertEqual(self.kinds()[-1], "finished")

    def test_acquisition_insufficient_growth_retries_and_keeps_prior_attempt(self):
        outcome, store, _ = self.run_to_end([Scenario(end=None, size=slow), Scenario(end=10)])
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        first, second = outcome.attempts
        self.assertEqual((first.status, first.retry_reason, first.growth.state),
                         (ResultStatus.FAILED, "insufficient_growth", "insufficient"))
        self.assertLess(first.growth.growth_bytes, 20_000_000)
        self.assertEqual(self.backend.children[0].signals[:1], ["TERM"])
        self.assertEqual(second.number, 2)
        self.assertNotEqual(first.directory, second.directory)
        self.assertEqual((first.prefix.parent / "acquisition.rawf").read_bytes(), GOOD_FILES[".rawf"])
        self.assertEqual([r["status"] for r in self.manifest_attempts(store)], ["failed", "succeeded"])
        text = "\n".join(self.logs)
        self.assertIn("[acquisition attempt-1] aborting: Early abort: .rawf growth below threshold", text)
        self.assertIn("[acquisition attempt-1] retry_wait: Retrying (insufficient_growth) as attempt 2/3", text)
        self.assertIn("[attempt-2] [stderr] writeRaw::", text)

    def test_acquisition_frame_loss_parsing_limits(self):
        for percent, state in ((4.9, "within"), (5.0, "within"), (5.1, "exceeded")):
            loss = parse_frame_loss(["[stderr] " + loss_line(percent)], 5.0)
            self.assertEqual((loss.state, loss.all_lost_percent), (state, percent))
        steps = parse_frame_loss([loss_line(1.0), "Python:: Acquired", loss_line(6.0)], 5.0)
        self.assertEqual((steps.state, steps.reports, steps.all_lost_percent), ("exceeded", 2, 6.0))
        absent = parse_frame_loss(["Python:: Acquired 10 frames"], 5.0)
        self.assertEqual((absent.state, absent.all_lost_percent, absent.reports), ("unknown", None, 0))
        nan = parse_frame_loss(["writeRaw:: some events were lost for 0 ( -nan%) frames; "
                                "all events were lost for 0 ( -nan%) frames"], 5.0)
        self.assertEqual((nan.state, nan.malformed, nan.all_lost_percent), ("unknown", 1, None))
        self.assertEqual(parse_frame_loss([loss_line(100.5)], 5.0).state, "unknown")
        mixed = parse_frame_loss([loss_line(7.0), "all events were lost for garbage"], 5.0)
        self.assertEqual(mixed.state, "exceeded")

    def test_acquisition_frame_loss_above_limit_retries_equal_passes(self):
        outcome, store, _ = self.run_to_end([Scenario(stderr=(loss_line(7.5) + "\n").encode()),
                                             Scenario(stderr=(loss_line(5.0) + "\n").encode())])
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        first, second = outcome.attempts
        self.assertEqual((first.retry_reason, first.loss.state, first.exit_code), ("frame_loss", "exceeded", 0))
        self.assertEqual((second.loss.state, second.loss.all_lost_percent), ("within", 5.0))
        records = self.manifest_attempts(store)
        self.assertEqual(records[0]["status"], "failed")
        self.assertFalse(any(item["validated"] for item in records[0]["artifacts"]))
        self.assertEqual(records[0]["details"]["retry_reason"], "frame_loss")

    def test_acquisition_missing_or_malformed_loss_is_unknown_not_zero(self):
        outcome, store, _ = self.run_to_end([Scenario(stderr=b"")])
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        loss = outcome.attempts[0].loss
        self.assertEqual((loss.state, loss.all_lost_percent), ("unknown", None))
        self.assertIn("No frame-loss report", outcome.attempts[0].message)
        details = self.manifest_attempts(store)[0]["details"]["frame_loss"]
        self.assertEqual((details["state"], details["all_lost_percent"]), ("unknown", None))
        self.updates.clear()
        bad = b"writeRaw:: some events were lost for 0 ( -nan%) frames; all events were lost for 0 ( -nan%) frames\n"
        outcome, _, _ = self.run_to_end([Scenario(stderr=bad)])
        self.assertEqual(outcome.attempts[0].loss.state, "unknown")
        self.assertIn("Unparseable", outcome.attempts[0].loss.message)

    # Terminal failures -------------------------------------------------------

    def test_acquisition_nonzero_exit_fails_without_retry(self):
        outcome, store, _ = self.run_to_end([Scenario(code=1), Scenario()])
        self.assertEqual(outcome.status, ResultStatus.FAILED)
        self.assertEqual(len(outcome.attempts), 1)
        self.assertEqual((outcome.attempts[0].exit_code, outcome.attempts[0].retry_reason), (1, None))
        self.assertIn("exited with code 1", outcome.message)
        self.assertEqual(len(self.backend.launched), 1)
        record = self.manifest_attempts(store)[0]
        self.assertEqual((record["status"], record["exit_code"]), ("failed", 1))
        self.assertEqual(len(record["artifacts"]), 4)  # partial output recorded, never deleted

    def test_acquisition_missing_or_empty_output_is_not_success(self):
        scenarios = [Scenario(files={}), Scenario(files={".rawf": b"", ".idxf": b"x"}),
                     Scenario(files={".rawf": b"data"})]
        outcome, store, _ = self.run_to_end(scenarios)
        self.assertEqual(outcome.status, ResultStatus.FAILED)
        self.assertEqual([a.retry_reason for a in outcome.attempts], ["no_data"] * 3)
        self.assertIn("acquisition.rawf missing or empty", outcome.attempts[0].message)
        self.assertIn("acquisition.rawf missing or empty", outcome.attempts[1].message)
        self.assertIn("acquisition.idxf missing or empty", outcome.attempts[2].message)
        self.assertTrue(all(a.exit_code == 0 for a in outcome.attempts))
        self.assertEqual({r["status"] for r in self.manifest_attempts(store)}, {"failed"})

    def test_acquisition_launch_error_is_terminal(self):
        outcome, store, _ = self.run_to_end([Scenario(error=OSError("no tool")), Scenario()])
        self.assertEqual(outcome.status, ResultStatus.LAUNCH_ERROR)
        self.assertEqual((len(outcome.attempts), outcome.attempts[0].exit_code), (1, None))
        self.assertEqual(self.manifest_attempts(store)[0]["status"], "launch_error")

    def test_acquisition_prerequisite_gates_start_and_loss_aborts(self):
        outcome, store, _ = self.run_to_end([Scenario()], prerequisite=lambda: (False, "DAQD is off"))
        self.assertEqual(outcome.status, ResultStatus.FAILED)
        self.assertIn("prerequisite not met: DAQD is off", outcome.message)
        self.assertEqual((self.backend.launched, outcome.attempts, self.manifest_attempts(store)), ([], (), ()))
        state = {"ok": True}
        def lose_on_growth(event):
            if event.kind == "growth_started":
                state["ok"] = False
        outcome, store, _ = self.run_to_end([Scenario(end=None), Scenario()], on_update=lose_on_growth,
                                            prerequisite=lambda: (state["ok"], "DAQD failed"))
        self.assertEqual(outcome.status, ResultStatus.FAILED)
        self.assertEqual(outcome.attempts[0].retry_reason, "prerequisite_lost")
        self.assertIn("prerequisite lost: DAQD failed", outcome.message)
        self.assertEqual((len(self.backend.launched), self.backend.children[0].signals[:1]), (1, ["TERM"]))

    def test_acquisition_invalid_command_or_name_reserves_nothing(self):
        service = self.service([Scenario()])
        store = self.store()
        custom = replace(self.settings, profile=replace(self.settings.profile, socket_path="/tmp/other.sock"))
        with self.assertRaises(ValueError):
            service.start(custom, store)
        for name in ("../escape", "a b", "", ".hidden"):
            with self.assertRaises(ValueError):
                service.start(self.settings, store, basename=name)
        self.assertEqual(self.manifest_attempts(store), ())
        self.assertFalse((store.root / "acquisition").exists())
        self.assertEqual(self.backend.launched, [])

    def test_acquisition_second_start_refused_while_active(self):
        service = self.service([Scenario(end=None), Scenario()])
        store = self.store()
        handle = service.start(self.settings, store)
        with self.assertRaises(RuntimeError):
            service.start(self.settings, self.store())
        deadline = time.monotonic() + 10
        while not self.backend.children and time.monotonic() < deadline:
            time.sleep(0.005)
        handle.stop()
        self.assertEqual(handle.wait(30).status, ResultStatus.CANCELLED)
        self.assertEqual(len(self.backend.launched), 1)

    # STOP, stale events, threads -----------------------------------------------

    def test_acquisition_stop_during_attempt_never_retries(self):
        service = self.service([Scenario(end=None), Scenario()])
        store = self.store()
        handle = service.start(self.settings, store)
        deadline = time.monotonic() + 10
        while not self.backend.children and time.monotonic() < deadline:
            time.sleep(0.005)
        handle.stop()
        outcome = handle.wait(30)
        self.assertEqual(outcome.status, ResultStatus.CANCELLED)
        self.assertEqual((len(outcome.attempts), outcome.attempts[0].retry_reason), (1, None))
        self.assertEqual(self.backend.children[0].signals[:1], ["TERM"])
        self.assertEqual(len(self.backend.launched), 1)
        self.assertEqual(outcome.attempts[0].loss.state, "unknown")
        self.assertEqual(self.manifest_attempts(store)[0]["status"], "cancelled")
        self.assertNotIn("retry_wait", self.kinds())

    def test_acquisition_stop_during_retry_delay_cancels_pending_retry(self):
        holder = {}
        def stop_on_retry(event):
            if event.kind == "retry_wait":
                holder["handle"].stop()
        service = self.service([Scenario(files={}), Scenario()], on_update=stop_on_retry)
        settings = self.with_safety(retry_delay_s=100_000.0)  # 1000 s real if STOP were ignored
        started = time.monotonic()
        holder["handle"] = service.start(settings, self.store(settings))
        outcome = holder["handle"].wait(30)
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(outcome.status, ResultStatus.CANCELLED)
        self.assertIn("retry delay", outcome.message)
        self.assertEqual((len(outcome.attempts), len(self.backend.launched)), (1, 1))

    def test_acquisition_stale_monitor_cannot_touch_newer_attempt(self):
        release, blocked = Event(), Event()
        backend_holder = {}
        def probe(path):
            if "attempt-1" in str(path) and not release.is_set():
                blocked.set()
                release.wait(30)
                return 10 ** 9  # would announce growth / pass for attempt-1 if not guarded
            return backend_holder["backend"].size(path)
        def release_on_second(event):
            if event.kind == "attempt_started" and event.identity.attempt_id == "attempt-2":
                release.set()
        service = self.service([Scenario(files={}, end=5), Scenario(end=10)], size_probe=probe,
                               on_update=release_on_second)
        backend_holder["backend"] = self.backend
        store = self.store()
        handle = service.start(self.settings, store)
        outcome = handle.wait(30)
        self.assertTrue(blocked.is_set())
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        self.assertEqual([a.retry_reason for a in outcome.attempts], ["no_data", None])
        start2 = next(i for i, e in enumerate(self.updates)
                      if e.kind == "attempt_started" and e.identity.attempt_id == "attempt-2")
        late = [e for e in self.updates[start2:] if e.identity.attempt_id == "attempt-1"]
        self.assertEqual(late, [])
        sequences = [e.sequence for e in self.updates]
        self.assertEqual(sequences, sorted(set(sequences)))

    def test_acquisition_attempt_filter_drops_stale_updates(self):
        one, two = (Identity("run-x", "acquisition", f"attempt-{n}") for n in (1, 2))
        events = [RunEvent(one, 0, "attempt_started"), RunEvent(one, 1, "growth_started"),
                  RunEvent(two, 3, "attempt_started"), RunEvent(one, 2, "attempt_finished"),
                  RunEvent(one, 4, "aborting"), RunEvent(two, 5, "attempt_finished"),
                  RunEvent(two, 5, "attempt_finished"), RunEvent(two, 6, "finished")]
        gate = AttemptFilter()
        accepted = [(e.identity.attempt_id, e.sequence) for e in events if gate.accept(e)]
        self.assertEqual(accepted, [("attempt-1", 0), ("attempt-1", 1), ("attempt-2", 3),
                                    ("attempt-2", 5), ("attempt-2", 6)])

    def test_acquisition_work_and_waits_stay_off_main_thread(self):
        service = self.service([Scenario(end=None, size=no_data, files={".rawf": b""}), Scenario(end=10)])
        before = time.monotonic()
        handle = service.start(self.settings, self.store())
        self.assertLess(time.monotonic() - before, 0.5)  # start returns; attempts run in workers
        self.assertFalse(handle.done)
        outcome = handle.wait(30)
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        main = threading.main_thread().name
        self.assertTrue(self.clock.threads)
        self.assertNotIn(main, self.clock.threads)
        self.assertNotIn(main, self.update_threads)
        self.assertEqual(outcome.attempts[0].retry_reason, "startup_timeout")

    def test_acquisition_runner_reports_child_exit_once(self):
        events = []
        child = FakeChild(3)
        backend = type("B", (), {"launch": lambda self, command: child})()
        command = CommandSpec(("tool",), REPO, Identity("r", "s", "a"))
        CommandRunner(policy=RUNNER, backend=backend).run(command, event_sink=events.append)
        exited = [e for e in events if e.kind == "exited"]
        self.assertEqual([e.payload["exit_code"] for e in exited], [3])
        self.assertLess(events.index(exited[0]), [e.kind for e in events].index("completed"))

    def test_acquisition_module_has_no_sleep_tk_or_deletion(self):
        tree = ast.parse((REPO / "src/petsys_manager/acquisition.py").read_text(encoding="utf-8"))
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                 for node in ast.walk(tree) if isinstance(node, ast.Call)}
        self.assertFalse(calls & {"sleep", "unlink", "remove", "rmtree", "rmdir", "kill", "killpg", "after"})
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                   for alias in node.names}
        self.assertFalse({"tkinter", "customtkinter", "subprocess"} & imports)

    # FR-19 SiPM bias-off, FR-20 operator feedback/editable limits -------------------

    @pytest.mark.fr("003-FR-19")
    def test_acquisition_bias_off_after_abort_before_retry(self):
        never = dict(end=None, size=no_data, files={".rawf": b"", ".idxf": b""})
        outcome, store, _ = self.run_to_end([Scenario(**never), Scenario(end=10)])
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        first, second = outcome.attempts
        self.assertEqual((first.bias.state, second.bias.state), ("requested", "not_needed"))
        self.assertEqual(self.backend.order, ["acquire", "set_bias", "acquire"])  # bias off before the retry
        self.assertEqual(self.backend.bias[0].argv, (str(self.root / "tools/set_bias"), "--power", "off"))
        self.assertEqual(self.backend.bias[0].identity.attempt_id, "attempt-1-bias-off")
        records = self.manifest_attempts(store)
        self.assertEqual([r["details"]["bias"]["state"] for r in records], ["requested", "not_needed"])
        self.assertFalse(outcome.bias_unknown)
        self.assertIn("bias_off", self.kinds())

    @pytest.mark.fr("003-FR-19")
    def test_acquisition_bias_off_after_stop_and_nonzero_exit_not_after_normal_end(self):
        service = self.service([Scenario(end=None)])
        handle = service.start(self.settings, self.store())
        deadline = time.monotonic() + 10
        while not self.backend.children and time.monotonic() < deadline:
            time.sleep(0.005)
        handle.stop()
        outcome = handle.wait(30)
        self.assertEqual((outcome.status, outcome.attempts[0].bias.state), (ResultStatus.CANCELLED, "requested"))
        self.assertEqual(len(self.backend.bias), 1)  # STOP does not skip bias-off
        outcome, _, _ = self.run_to_end([Scenario(code=2)])
        self.assertEqual(outcome.attempts[0].bias.state, "requested")
        for scenarios in ([Scenario()], [Scenario(files={})] * 3, [Scenario(stderr=(loss_line(9.0) + "\n").encode())] * 3):
            outcome, _, _ = self.run_to_end(scenarios)
            self.assertEqual({a.bias.state for a in outcome.attempts}, {"not_needed"})
            self.assertEqual(self.backend.bias, [])
        outcome, _, _ = self.run_to_end([Scenario(error=OSError("no tool"))])
        self.assertEqual((outcome.attempts[0].bias.state, self.backend.bias), ("not_needed", []))

    @pytest.mark.fr("003-FR-19")
    def test_acquisition_bias_off_failure_warns_and_stops_retrying(self):
        never = dict(end=None, size=no_data, files={".rawf": b""})
        def failing(backend):
            backend.bias_code = 1
        outcome, store, _ = self.run_to_end([Scenario(**never), Scenario()], backend_setup=failing)
        self.assertEqual(outcome.status, ResultStatus.FAILED)
        self.assertEqual(len(outcome.attempts), 1)  # startup timeout would retry; unknown bias must not
        attempt = outcome.attempts[0]
        self.assertEqual((attempt.bias.state, attempt.retry_reason), ("unknown", None))
        self.assertIn("SiPM bias state UNKNOWN", outcome.message)
        self.assertTrue(outcome.bias_unknown)
        self.assertIn("bias_unknown", self.kinds())
        self.assertEqual(self.manifest_attempts(store)[0]["details"]["bias"]["state"], "unknown")
        def hanging(backend):
            backend.bias_hang = True
        outcome, _, _ = self.run_to_end([Scenario(code=3)], backend_setup=hanging, bias_off_timeout_s=0.3)
        self.assertEqual(outcome.attempts[0].bias.state, "unknown")
        self.assertIn("timed out after 0.3 s", outcome.attempts[0].bias.message)
        def missing(backend):
            backend.bias_error = OSError("set_bias not found")
        outcome, _, _ = self.run_to_end([Scenario(code=3)], backend_setup=missing)
        self.assertEqual(outcome.attempts[0].bias.state, "unknown")

    @pytest.mark.fr("003-FR-19")
    def test_acquisition_bias_off_command_contract(self):
        identity = Identity("run-x", "acquisition", "attempt-1-bias-off")
        command = build_bias_off(self.settings, identity)
        self.assertEqual(command.argv, (str(self.root / "tools/set_bias"), "--power", "off"))
        self.assertEqual(command.cwd, self.root / "tools")
        custom = replace(self.settings, profile=replace(self.settings.profile, socket_path="/tmp/other.sock"))
        with self.assertRaises(ValueError):
            build_bias_off(custom, identity)
        missing = replace(self.profile, petsys_folder="missing-tools")
        report = preflight(missing, Action.ACQUIRE, repo_root=self.root, probe=FixtureProbe())
        self.assertIn("tool:set_bias", {issue.field for issue in report.issues})

    @pytest.mark.fr("003-FR-20")
    def test_acquisition_progress_shows_growing_file_and_later_stall(self):
        outcome, _, _ = self.run_to_end([Scenario(end=60)])
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        progress = [e for e in self.updates if e.kind == "rawf_progress"]
        self.assertGreaterEqual(len(progress), 3)
        self.assertTrue(all(e.payload["growing"] and e.payload["bytes_per_s"] > 0 for e in progress))
        sizes = [e.payload["size"] for e in progress]
        self.assertEqual(sizes, sorted(sizes))
        passed = next(e for e in self.updates if e.kind == "growth_passed")
        self.assertIn("RAW file growing as expected", passed.message)
        self.assertFalse(any("rawf_progress" in line for line in self.logs))  # growing samples stay out of the log
        self.updates.clear()
        self.logs.clear()
        outcome, _, _ = self.run_to_end([Scenario(end=70, size=stalls_after_30)])
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)  # a warning, not an abort
        self.assertEqual(outcome.attempts[0].growth.state, "passed")
        stalled = [e for e in self.updates if e.kind == "rawf_progress" and not e.payload["growing"]]
        self.assertTrue(stalled)
        self.assertIn("NOT growing since the previous check", stalled[0].message)
        self.assertTrue(any("NOT growing" in line for line in self.logs))

    @pytest.mark.fr("003-FR-20")
    def test_acquisition_loss_limit_is_an_editable_setting(self):
        relaxed = self.with_safety(max_loss_percent=10.0)
        outcome, store, _ = self.run_to_end([Scenario(stderr=(loss_line(7.5) + "\n").encode())], settings=relaxed)
        self.assertEqual(outcome.status, ResultStatus.SUCCEEDED, outcome.message)
        self.assertEqual(outcome.attempts[0].loss.message, "Frame loss 7.5% within 10%")
        self.assertEqual(read_manifest(store.root)["settings"]["profile"]["safety"]["max_loss_percent"], 10.0)
        profile = profile_from_mapping({"schema_version": 1, "safety": {"max_loss_percent": 12.5}})
        self.assertEqual(profile.safety.max_loss_percent, 12.5)

    @pytest.mark.fr("003-FR-20")
    def test_acquisition_reference_safety_defaults(self):
        self.assertEqual(AcquisitionSafety(), AcquisitionSafety(startup_timeout_s=45.0, growth_window_s=20.0,
                         poll_interval_s=5.0, min_growth_bytes=20_000_000, max_loss_percent=5.0,
                         max_attempts=3, retry_delay_s=2.0))
        self.assertEqual(self.settings.profile.safety, AcquisitionSafety())

