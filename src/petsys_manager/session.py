"""Toolkit-free manager session: profile file, prerequisites, DAQD/workflow control and UI event queue.

Workers only put `ShellEvent`s on `events`; the GUI drains that queue on its Tk
thread. Every backend call (preflight probes, DAQD start/stop/initialization,
workflow preparation/start/wait, shutdown) runs in a worker thread, so a
blocking command never stalls the window. Prerequisite checks carry a
generation and workflow results a token, so a slower, older answer cannot
replace a newer one; DAQD statuses carry the service's own revision.

The DAQD service and workflow coordinator are created on first use: opening
the window launches nothing. Nothing here deletes DAQ resources, creates
destinations before a workflow starts, or writes processing configuration/maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import queue
import threading
import time

from .contracts import Action
from .settings import (MachineProfile, PrerequisiteIssue, SystemProbe, default_profile_path,
                       load_profile, preflight, save_profile)


CHECKOUT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ShellEvent:
    # "log", "readiness", "daqd" (DaqdStatus), "init_done" (InitOutcome), "refused" ((key, message)),
    # "workflow" (RunEvent), "workflow_done" (WorkflowResult), "shutdown" (ShutdownResult),
    # "inputs_probed" (InputProbe)
    kind: str
    payload: object


@dataclass(frozen=True)
class Readiness:
    generation: int
    issues: dict  # check key -> tuple[PrerequisiteIssue, ...]; empty means prerequisites met
    limit_plans: dict = field(default_factory=dict)  # calibrate/pipeline key -> event-limit plan (FR-21)


@dataclass(frozen=True)
class WorkflowResult:
    token: int
    outcome: object | None  # WorkflowOutcome when the workflow ran; None when it never started
    message: str


@dataclass(frozen=True)
class ShutdownResult:
    ok: bool
    message: str


@dataclass(frozen=True)
class InputProbe:
    key: str
    request: int                # only the newest request per key is shown
    results: tuple              # per file, in order: (path, ProbeSummary | None, error text | None)
    message: str = ""           # a reason no file was checked (configuration, cancellation)


class Refused(RuntimeError):
    pass


class ManagerSession:
    def __init__(self, profile_path=None, *, probe=None, repo_root=None, daqd_factory=None,
                 coordinator_factory=None):
        self.repo_root = Path(repo_root or CHECKOUT).resolve()
        self.explicit_profile = profile_path is not None
        self.profile_path = Path(profile_path) if profile_path is not None else default_profile_path()
        self.probe = SystemProbe() if probe is None else probe
        self.profile = MachineProfile()
        self.events = queue.SimpleQueue()
        self._wake = threading.Condition()
        self._pending = None
        self._generation = 0
        self._worker = None
        self._closed = False
        self._daqd_factory = daqd_factory
        self._coordinator_factory = coordinator_factory
        self._lock = threading.RLock()
        self._daqd = None
        self._coordinator = None
        self._token = 0
        self._active = None      # token of the requested/running foreground workflow
        self._handle = None      # its WorkflowHandle once started
        self._stop_requested = False
        self._shutdown = threading.Event()
        self._threads = []
        self._probes = {}        # selection key -> (request, cancellation Event)
        self._map_cache = None   # (file stamps, mapping) for the event-limit plan
        self._probe_request = 0

    @property
    def generation(self):
        return self._generation

    def log(self, message):
        self.events.put(ShellEvent("log", str(message)))

    # Profile ------------------------------------------------------------------------------------

    def open(self):
        """Load the startup profile, or keep defaults; never creates or rewrites a file."""
        if not self.profile_path.exists():
            self.log(f"No profile at {self.profile_path}; defaults in use, nothing written")
            return "defaults"
        try:
            self.load()
        except (OSError, ValueError) as exc:
            self.log(f"Profile not loaded, defaults in use and file left unchanged: {exc}")
            return "error"
        self.log(f"Loaded profile {self.profile_path}")
        return "loaded"

    def load(self, path=None):
        """Read a profile; on failure the current profile and path stay unchanged."""
        target = self.profile_path if path is None else Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Profile not found: {target}")
        profile = load_profile(target)
        self.profile_path, self.profile = target, profile
        return profile

    def save(self, profile, path=None):
        """Write only a manager profile; an existing file must already be a valid profile."""
        target = self.profile_path if path is None else Path(path)
        written = save_profile(profile, target, overwrite=target.exists())
        self.profile_path, self.profile = written, profile
        return written

    # Prerequisite checks ------------------------------------------------------------------------

    def check(self, profile, requests):
        """Queue read-only prerequisite checks: key -> (Action, RunOptions[, ordered InputDescriptors])."""
        requests = {key: (Action(value[0]), value[1], tuple(value[2]) if len(value) > 2 else ())
                    for key, value in requests.items()}
        with self._wake:
            if self._closed:
                raise RuntimeError("Session is closed")
            self._generation += 1
            self._pending = (self._generation, profile, requests)
            if self._worker is None:
                self._worker = threading.Thread(target=self._check_loop, name="petsys-preflight", daemon=True)
                self._worker.start()
            self._wake.notify()
            return self._generation

    def _check_loop(self):
        while True:
            with self._wake:
                while self._pending is None and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                generation, profile, requests = self._pending
                self._pending = None
            issues, plans = {}, {}
            for key, (action, options, inputs) in requests.items():
                try:
                    report = preflight(profile, action, options, inputs, repo_root=self.repo_root,
                                       probe=self.probe)
                    issues[key] = report.issues
                    if report.settings is not None and action in (Action.CALIBRATE, Action.PIPELINE):
                        try:
                            plans[key] = self._limit_plan(report.settings)
                        except (OSError, ValueError) as exc:   # display only: never a readiness reason
                            plans[key] = {"error": str(exc)}
                except Exception as exc:  # A probe/profile fault is a reason, never readiness.
                    issues[key] = (PrerequisiteIssue("settings", f"{type(exc).__name__}: {exc}"),)
            self.events.put(ShellEvent("readiness", Readiness(generation, issues, plans)))

    def _limit_plan(self, settings):
        """Calibration event-limit plan shown before a run (FR-21); the map's key count is cached per file.

        The pipeline's file count is its split count (the converter's actual outputs may differ)."""
        from src.cornell.calibration import event_limit_plan
        from src.cornell.inputs import load_processing_config
        yaml_path = Path(settings.paths["yaml_file"])
        stamp = []
        for path in (yaml_path, settings.paths.get("map_file")):
            info = os.stat(path) if path is not None else None
            stamp.append(None if info is None else (str(path), info.st_size, info.st_mtime_ns))
        stamp = tuple(stamp)
        if self._map_cache is None or self._map_cache[0] != stamp:
            mapping = load_processing_config(yaml_path, processing_root=settings.processing_root,
                                             action=Action.CALIBRATE).mapping
            self._map_cache = (stamp, mapping)
        options, limits = settings.options, settings.profile.limits
        files = len(settings.inputs) if settings.action == Action.CALIBRATE else options.splits
        plan = event_limit_plan(self._map_cache[1], options.regions, files,
                                limit_mode=options.calibration_limit_mode,
                                event_limit=limits.calibration_event_limit,
                                target_per_key=limits.calibration_target_per_key)
        return {**plan, "files_from_splits": settings.action == Action.PIPELINE}

    def probe_inputs(self, profile, key, descriptors, *, max_records=10000):
        """Check the first records of each selected file against the selected map, off the UI thread.

        Read-only operator feedback; a newer request for the same key cancels the older
        one. Processing still runs the full validation.
        """
        with self._lock:
            self._probe_request += 1
            request, cancel = self._probe_request, threading.Event()
            previous = self._probes.get(key)
            self._probes[key] = (request, cancel)
        if previous is not None:
            previous[1].set()

        def run():
            from src.cornell.inputs import InputError, ValidationCancelled, load_processing_config, probe_ldat
            results, message = [], ""
            try:
                report = preflight(profile, Action.CALIBRATE, repo_root=self.repo_root, probe=self.probe)
                yaml_file = report.paths.get("yaml_file")
                if yaml_file is None:
                    raise InputError("Select the processing YAML; its map defines the valid channels")
                mapping = load_processing_config(yaml_file, processing_root=report.paths["processing_root"],
                                                 action=Action.CALIBRATE).mapping
                for descriptor in descriptors:
                    try:
                        summary = probe_ldat(descriptor, mapping.modules, max_records=max_records,
                                             cancelled=cancel.is_set)
                        results.append((descriptor.path, summary, None))
                    except ValidationCancelled:
                        raise
                    except (InputError, OSError) as exc:
                        results.append((descriptor.path, None, str(exc)))
            except ValidationCancelled:
                message = "Superseded by a newer selection"
            except Exception as exc:
                known = isinstance(exc, (InputError, OSError, ValueError))
                message = f"Not checked: {exc}" if known else f"Not checked: {type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    if self._probes.get(key, (None,))[0] == request:
                        del self._probes[key]
            self.events.put(ShellEvent("inputs_probed", InputProbe(key, request, tuple(results), message)))
        thread = threading.Thread(target=run, name=f"petsys-inputs-{key}", daemon=True)
        thread.start()
        return request

    # Backend services (created on first use, never at startup) ----------------------------------

    def _services(self):
        with self._lock:
            if self._daqd is None:
                if self._daqd_factory is None:
                    from .acquisition import DaqdService
                    self._daqd_factory = DaqdService
                self._daqd = self._daqd_factory(status_sink=lambda status: self.events.put(ShellEvent("daqd", status)),
                                                log_sink=self.log)
            if self._coordinator is None:
                if self._coordinator_factory is None:
                    from .workflow import WorkflowCoordinator
                    self._coordinator_factory = lambda **kw: WorkflowCoordinator(checkout_root=self.repo_root, **kw)
                self._coordinator = self._coordinator_factory(
                    update_sink=lambda event: self.events.put(ShellEvent("workflow", event)), log_sink=self.log)
            return self._daqd, self._coordinator

    def _settings(self, profile, action, options=None, inputs=()):
        report = preflight(profile, action, options, inputs, repo_root=self.repo_root, probe=self.probe)
        if not report.ready:
            raise Refused("; ".join(f"{issue.field}: {issue.message}" for issue in report.issues))
        return report.settings

    def _spawn(self, key, target):
        def run():
            try:
                target()
            except Exception as exc:
                text = str(exc) if isinstance(exc, Refused) else f"{type(exc).__name__}: {exc}"
                self.events.put(ShellEvent("refused", (key, text)))
        thread = threading.Thread(target=run, name=f"petsys-{key}", daemon=True)
        with self._lock:
            self._threads = [item for item in self._threads if item.is_alive()] + [thread]
        thread.start()
        return thread

    def idle(self):
        """True when no workflow is requested/running and no owned DAQD process is alive."""
        from .acquisition import DaqdState
        with self._lock:
            if self._active is not None:
                return False
            daqd = self._daqd
        return daqd is None or daqd.status().state in (DaqdState.OFF, DaqdState.FAILED)

    def start_daqd(self, profile):
        def run():
            if self._shutdown.is_set():
                raise Refused("The manager is closing")
            settings = self._settings(profile, Action.DAQD)
            self._services()[0].start(settings)  # the outcome arrives as a "daqd" status
        return self._spawn("daqd", run)

    def stop_daqd(self):
        return self._spawn("daqd", lambda: self._services()[0].stop())

    def initialize(self, profile):
        def run():
            if self._shutdown.is_set():
                raise Refused("The manager is closing")
            settings = self._settings(profile, Action.INITIALIZE)
            outcome = self._services()[0].initialize(settings, cancellation=self._shutdown)
            self.events.put(ShellEvent("init_done", outcome))
        return self._spawn("initialize", run)

    def start_workflow(self, profile, action, options=None, inputs=()):
        """Request one foreground workflow; returns its token. The result arrives as "workflow_done".

        ``inputs`` are the exact ordered descriptors of a manual calibration, LM or offline QC.
        """
        from .workflow import WorkflowBusy, WorkflowError, prepare

        inputs = tuple(inputs)
        with self._lock:
            if self._active is not None:
                raise RuntimeError("A foreground workflow is already requested")
            self._token += 1
            token = self._active = self._token
            self._handle, self._stop_requested = None, False

        def run():
            outcome, message = None, "Not started"
            try:
                if self._shutdown.is_set():
                    raise Refused("The manager is closing")
                settings = self._settings(profile, action, options, inputs)
                plan = prepare(settings)
                daqd, coordinator = self._services()
                prerequisite = None
                if "acquisition" in plan.stages:
                    prerequisite = lambda: daqd.acquisition_ready(settings)
                with self._lock:
                    if self._stop_requested:
                        raise Refused("Stopped before the workflow started")
                    self._handle = handle = coordinator.start(plan, prerequisite=prerequisite)
                outcome = handle.wait()
                message = outcome.message
            except (Refused, WorkflowError, WorkflowBusy) as exc:
                message = f"Not started: {exc}"
            except Exception as exc:
                message = f"Not started: {type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    if self._active == token:
                        self._active, self._handle = None, None
                self.events.put(ShellEvent("workflow_done", WorkflowResult(token, outcome, message)))

        thread = threading.Thread(target=run, name=f"petsys-workflow-request-{token}", daemon=True)
        thread.start()
        return token

    def stop_workflow(self):
        """STOP: cancel the requested/running workflow, its retries and every later stage."""
        with self._lock:
            if self._active is None:
                return False
            self._stop_requested = True
            handle = self._handle
        if handle is not None:
            handle.stop()
        return True

    def shutdown(self, timeout):
        """Close asynchronously: STOP the workflow and wait for it (bias-off included), then stop DAQD.

        Reports ShutdownResult; DAQD is left running if the workflow has not finished,
        so an acquisition's bias-off is never cut short.
        """
        def run():
            from .acquisition import DaqdState
            self._shutdown.set()
            with self._lock:
                self._stop_requested = True
                handle, active, daqd, threads = self._handle, self._active, self._daqd, list(self._threads)
            if handle is not None:
                handle.stop()
                if handle.wait(timeout) is None:
                    return self._shut(False, f"The workflow is still finishing after {timeout:g} s (bias-off "
                                             "included); DAQD left running. Close again to retry.")
            if active is not None:  # its request thread clears _active/_handle just after the outcome
                end = time.monotonic() + timeout
                while time.monotonic() < end:
                    with self._lock:
                        if self._active != active:
                            break
                    time.sleep(0.05)
            for thread in threads:  # initialization/DAQD requests see the shutdown flag
                thread.join(timeout)
            with self._lock:
                current, daqd = self._handle, self._daqd
            if current is not None and current is not handle:
                return self._shut(False, "A workflow started during shutdown; close again to retry.")
            if daqd is None:
                return self._shut(True, "Nothing to stop")
            status = daqd.close(timeout)
            if status.state not in (DaqdState.OFF, DaqdState.FAILED):
                return self._shut(False, f"DAQD still {status.state.value} after {timeout:g} s: {status.message}. "
                                         "Close again to retry.")
            return self._shut(True, f"DAQD {status.state.value}: {status.message}")
        thread = threading.Thread(target=self._guard_shutdown(run), name="petsys-shutdown", daemon=True)
        thread.start()
        return thread

    def _guard_shutdown(self, run):
        def guarded():
            try:
                run()
            except Exception as exc:
                self._shut(False, f"Shutdown failed: {type(exc).__name__}: {exc}. Close again to retry.")
        return guarded

    def _shut(self, ok, message):
        if not ok:
            self._shutdown.clear()  # the operator may retry, or keep working
        self.events.put(ShellEvent("shutdown", ShutdownResult(ok, message)))

    def close(self, timeout=2.0):
        """Stop the preflight worker; call after a successful shutdown or when nothing ran."""
        with self._lock:
            for _, cancel in self._probes.values():
                cancel.set()
        with self._wake:
            self._closed = True
            self._pending = None
            self._wake.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout)
        return worker is None or not worker.is_alive()
