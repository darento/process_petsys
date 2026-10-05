"""Single foreground workflow coordinator: manual actions and fail-closed stage graphs (T12).

``prepare`` checks the whole chosen graph before anything is created or
launched: argv contracts, the destination, the processing YAML/map, every
external selected file (limits, maps, manual calibration), LM metadata and the
exact manual inputs. ``WorkflowCoordinator.start`` then runs the graph in one
worker thread under a single run store:

=================  ==============================================================
Action             Stages (each consumes only its predecessor's recorded outputs)
=================  ==============================================================
``acquire``        acquisition
``convert``        conversion to compact coincidence (the only manager route, FR-10)
``calibrate``      calibration of the selected compact coincidence files
``listmode``       listmode of the selected compact coincidence files
``qc_analyze``     QC of the selected compact coincidence files
``qc``             acquisition (60/180 s preset) -> compact coincidence conversion -> QC
``pipeline``       acquisition -> compact coincidence conversion -> calibration -> listmode
=================  ==============================================================

Each run gets one new readable folder (T29, FR-9/FR-13): ``run_name`` gives
``<data>_<action>[-<options>]_<YYYY-MM-DD>_<HHMM>``; a single-stage run writes in
it, a multi-stage run uses ``1_acquisition/``, ``2_conversion/`` ... (see
``RunStore``), and every finished run adds one line to ``<destination>/runs.tsv``.

A stage succeeds only with a validated exact output set: conversion outputs
are discovered by their exact new prefix in a fresh stage directory and
fully validated against the selected map; processing outputs come from the
``src.cornell.cli`` result manifest, checked against this stage's request and
attempt directory. Any failure, launch error or STOP ends the workflow: no
later stage starts and the run is finalized as failed/cancelled with every
attempt and partial output recorded. Settings are the immutable preflight
snapshot; nothing here writes a profile, processing YAML or map.

Worker code never touches Tk: sinks must only enqueue. QC stage success means
the processing completed; its findings are observations, never a PASS verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from threading import Event, RLock, Thread

from .acquisition import AcquisitionService
from .artifacts import RunStore
from .commands import build_acquisition, build_bias_off, build_conversion, build_internal
from .contracts import (MANAGER_ROUTE, Action, Artifact, CommandResult, DataFormat, FrozenMapping, Identity,
                        InputDescriptor, OutputValidation, Population, ResultStatus, RunEvent, SourceMode, freeze,
                        to_plain)
from .runner import Clock, CommandRunner, RunnerPolicy
from .settings import RunSettings, lm_header_metadata

STAGES = {
    Action.ACQUIRE: ("acquisition",),
    Action.CONVERT: ("conversion",),
    Action.CALIBRATE: ("calibration",),
    Action.LISTMODE: ("listmode",),
    Action.QC_ANALYZE: ("qc",),
    Action.QC: ("acquisition", "conversion", "qc"),
    Action.PIPELINE: ("acquisition", "conversion", "calibration", "listmode"),
}
# The run directory is created under this destination; every stage writes inside it.
DESTINATIONS = {Action.ACQUIRE: "data_dir", Action.CONVERT: "data_dir", Action.QC: "data_dir",
                Action.PIPELINE: "data_dir", Action.CALIBRATE: "calibration_dir", Action.LISTMODE: "lm_dir",
                Action.QC_ANALYZE: "report_dir"}
CONVERSION_CHECK_RECORDS = 10_000   # FR-24: converter outputs are structure-checked, not read whole
CLI_ACTIONS = {"calibration": "calibrate", "listmode": "listmode", "qc": "qc"}
STAGE_ACTIONS = {"calibration": Action.CALIBRATE, "listmode": Action.LISTMODE, "qc": Action.QC_ANALYZE}
REQUIRED_KINDS = {"calibrate": {"encal", "calibration_sidecar", "calibration_status", "calibration_plot"},
                  "listmode": {"listmode", "listmode_provenance", "listmode_job"},
                  "qc": {"qc_report", "qc_summary"}}
LM_BATCH_RECORDS = 1000      # reference LM reader batch (src.cornell.listmode.DEFAULT_BATCH_RECORDS)
REQUEST = "request.json"
RESULT = "result.json"
RUNS = "runs.tsv"                       # per-destination overview, one appended line per finished run (T29)
RUNS_HEADER = ("finished", "run_folder", "action", "inputs", "status", "main_output")
ACTION_CODES = {Action.ACQUIRE: "acq", Action.CONVERT: "conv", Action.CALIBRATE: "cal", Action.LISTMODE: "lm",
                Action.QC: "qc", Action.QC_ANALYZE: "qc", Action.PIPELINE: "pipeline"}
CONVERTER_SUFFIXES = ("_coincCompact", "_coincFixed", "_groupCompact", "_groupFixed")
DATA_NAME_LIMIT = 56                    # the whole name, with a _99 suffix, stays a 96-character component
MAIN_OUTPUT = {"acquisition": "rawf", "conversion": "ldat", "calibration": "encal", "listmode": "listmode",
               "qc": "qc_report"}


def format_elapsed(seconds):
    """Operator text for a stage's elapsed time (FR-1): 42.3 s, 12 min 04 s, 1 h 02 min."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {rest:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


class WorkflowError(ValueError):
    def __init__(self, *issues):
        self.issues = tuple(str(issue) for issue in issues)
        super().__init__("; ".join(self.issues))


class WorkflowBusy(RuntimeError):
    """Only one foreground workflow may run; DAQD is managed separately."""


@dataclass(frozen=True)
class WorkflowPlan:
    settings: RunSettings
    stages: tuple
    destination: Path
    basename: str
    conversion_format: DataFormat | None
    conversion_population: Population | None


@dataclass(frozen=True)
class StageOutcome:
    stage_id: str
    status: ResultStatus
    message: str
    attempt_id: str | None = None
    directory: Path | None = None
    artifacts: tuple = ()
    details: FrozenMapping = FrozenMapping()


@dataclass(frozen=True)
class WorkflowOutcome:
    action: Action
    status: ResultStatus
    message: str
    run_root: Path | None
    stages: tuple = ()
    qc_findings: FrozenMapping | None = None   # observations of the coincidence sample, not a verdict

    @property
    def succeeded(self):
        return self.status == ResultStatus.SUCCEEDED

    def outputs(self, stage_id):
        """Exact validated outputs recorded for a successful stage."""
        for stage in self.stages:
            if stage.stage_id == stage_id and stage.status == ResultStatus.SUCCEEDED:
                return stage.artifacts
        return ()


class WorkflowHandle:
    """STOP is sticky: cancels the running stage and prevents every later one."""

    def __init__(self, plan):
        self.plan = plan
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


# Preflight -------------------------------------------------------------------

def _raw_name(settings):
    name = Path(settings.paths["raw_input"]).name
    return name[:-len(".rawf")] if name.endswith(".rawf") else name


def _basename(settings):
    """File basename of the stage outputs: the RAW name (conversion) or the acquisition name."""
    name = settings.options.acquisition_name
    if settings.action == Action.QC:
        return f"{name}_qc_{settings.options.source_mode.value}_source"
    if settings.action == Action.CONVERT and settings.paths.get("raw_input") is not None:
        return _raw_name(settings)
    return name


def portable_name(text):
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text).lstrip("_-")[:DATA_NAME_LIMIT].rstrip("_-")
    return text or "data"


def common_base(paths):
    """Readable data name of input files: their common stem prefix, cut back to a ``_``/``-`` boundary when the
    stems differ, without a converter suffix (``run_coincCompact_0..33`` -> ``run``)."""
    stems = [Path(path).stem for path in paths]
    base = os.path.commonprefix(stems)
    if len(set(stems)) > 1:
        cut = max(base.rfind("_"), base.rfind("-"))
        base = base[:cut] if cut > 0 else ""
    base = base.rstrip("_-")
    for suffix in CONVERTER_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            base = base[:-len(suffix)]
            break
    return portable_name(base)


def run_name(settings, now=None):
    """``<data>_<action>[-<options>]_<YYYY-MM-DD>_<HHMM>`` (FR-9, T29); ``RunStore`` adds ``_2`` ... on a clash."""
    action, options = settings.action, settings.options
    if action == Action.CONVERT:
        data = _raw_name(settings)
    elif action in (Action.ACQUIRE, Action.PIPELINE, Action.QC):
        data = options.acquisition_name
    else:
        data = common_base(item.path for item in settings.inputs)
    extra = None
    if action == Action.CALIBRATE:
        extra = f"P{options.regions}-{options.calibration_limit_mode}"
    elif action == Action.PIPELINE:
        extra = f"P{options.regions}"
    elif action == Action.LISTMODE:
        from src.cornell.inputs import calibration_layout
        extra = f"P{calibration_layout(settings.paths['calibration_file'])[1]}"
    elif action == Action.QC:
        extra = f"{options.source_mode.value}-source"
    code = ACTION_CODES[action] + (f"-{extra}" if extra else "")
    return f"{portable_name(data)}_{code}_{(now or datetime.now()).strftime('%Y-%m-%d_%H%M')}"


def _cell(value):
    return " ".join(str(value).split()) or "-"


def overview_row(outcome, settings, finished):
    """One ``runs.tsv`` line: finished, run folder, action, inputs, status, main output (relative)."""
    root = outcome.run_root
    if settings.action == Action.CONVERT:
        inputs = Path(settings.paths["raw_input"]).name
    elif settings.inputs:
        inputs = f"{len(settings.inputs)} file(s): {Path(settings.inputs[0].path).name}"
    else:
        inputs = "-"
    main = "-"
    if outcome.succeeded and outcome.stages:
        stage = outcome.stages[-1]
        kind = MAIN_OUTPUT.get(stage.stage_id)
        found = [a.path for a in stage.artifacts if a.kind == kind]
        if found:
            main = Path(found[0]).relative_to(root).as_posix()
            if len(found) > 1:
                main += f" (+{len(found) - 1} more)"
    return tuple(_cell(value) for value in (finished.strftime("%Y-%m-%d %H:%M:%S"), root.name,
                                            settings.action.value, inputs, outcome.status.value, main))


def append_overview(destination, row):
    """Append one line to ``<destination>/runs.tsv`` (header when created); never rewrites or follows a link."""
    path = Path(destination) / RUNS
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    content = ("\t".join(row) + "\n").encode("utf-8")
    if os.path.islink(path):
        raise OSError(f"Not a plain file: {path}")
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o644)
        content = ("\t".join(RUNS_HEADER) + "\n").encode("utf-8") + content
    except FileExistsError:
        fd = os.open(path, flags)
    try:
        os.write(fd, content)
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def _conversion_prefix(plan):
    return f"{plan.basename}_coincCompact"


def _processing_config(settings, action):
    from src.cornell.inputs import load_processing_config
    path = settings.paths.get("yaml_file")
    if path is None:
        raise WorkflowError("Select the processing YAML (map and cuts) for this workflow")
    return load_processing_config(path, processing_root=settings.processing_root, action=action)


def prepare(settings):
    """Check every stage of the chosen graph; nothing is created or launched. Raises WorkflowError."""
    from src.cornell.inputs import InputError, load_calibration, load_limits, select_inputs

    if not isinstance(settings, RunSettings):
        raise WorkflowError("Workflows need a validated run-settings snapshot (preflight)")
    action = settings.action
    if action not in STAGES:
        raise WorkflowError(f"{action.value} is not a workflow; DAQD/initialization are managed separately")
    stages = STAGES[action]
    options, paths, profile = settings.options, settings.paths, settings.profile
    issues = []
    destination = paths.get(DESTINATIONS[action])
    if destination is None or not Path(destination).is_dir():
        issues.append(f"{DESTINATIONS[action]}: select an existing destination directory ({destination})")
    fmt = population = None
    if "conversion" in stages:
        fmt, population = options.output_format, options.population
        if (fmt, population) != MANAGER_ROUTE:
            issues.append("The manager converts to compact coincidence only (FR-10)")
    if any((item.format, item.population) != MANAGER_ROUTE for item in settings.inputs):
        issues.append("The manager processes compact coincidence inputs only (FR-10)")
    if action == Action.QC and options.duration_s != (60.0 if options.source_mode == SourceMode.WITH else 180.0):
        issues.append("QC acquisition must use the 60 s with-source / 180 s without-source preset")
    plan = WorkflowPlan(settings, stages, Path(destination) if destination else Path("/"), _basename(settings),
                        fmt, population)
    probe = Identity("preflight", "preflight", "preflight")
    scratch = Path(destination or "/") / "preflight"
    try:
        if "acquisition" in stages:
            build_acquisition(settings, probe, scratch / plan.basename)
            build_bias_off(settings, probe)
        if "conversion" in stages:
            raw = None if action == Action.CONVERT else scratch / f"{plan.basename}.rawf"
            build_conversion(settings, probe, scratch / _conversion_prefix(plan), raw_input=raw)
        for stage in stages:
            if stage in STAGE_ACTIONS:
                build_internal(settings, probe, STAGE_ACTIONS[stage], scratch / REQUEST, scratch / RESULT)
    except (ValueError, KeyError) as exc:
        issues.append(f"Command contract: {exc}")
    if stages[-1] in STAGE_ACTIONS:
        config_action = STAGE_ACTIONS[stages[-1]]
    else:   # compact conversion output is validated against the selected map
        config_action = Action.QC_ANALYZE
    try:
        config = _processing_config(settings, config_action) if stages != ("acquisition",) else None
        if "listmode" in stages or ("calibration" in stages and options.regions > 1):
            load_limits(paths.get("cog_limits_file"), config.mapping, kind="cog")
        if "listmode" in stages:
            from src.cornell.listmode import load_pair_map, load_region_map
            load_limits(paths.get("doi_limits_file"), config.mapping, kind="doi")
            load_pair_map(paths.get("pair_map_file"))
            load_region_map(paths.get("region_map_file"), config.mapping)
            missing = lm_header_metadata(profile, action, options).missing()
            if missing:
                issues.append("LM metadata unavailable: " + ", ".join(missing))
        if action == Action.LISTMODE:   # per-slab or position; its own region count is used
            load_calibration(paths.get("calibration_file"), config.mapping)
        if action in (Action.CALIBRATE, Action.LISTMODE, Action.QC_ANALYZE):
            select_inputs(settings.inputs, action, processing_root=settings.processing_root)
    except WorkflowError as exc:
        issues.extend(exc.issues)
    except (InputError, OSError, TypeError, ValueError) as exc:
        issues.extend(getattr(exc, "issues", None) or [str(exc)])
    if issues:
        raise WorkflowError(*dict.fromkeys(issues))
    return plan


# Requests --------------------------------------------------------------------

def _entry(descriptor):
    return {"path": str(descriptor.path), "format": descriptor.format.value,
            "population": descriptor.population.value}


def processing_request(settings, stage, inputs, directory, context):
    """Exact ``src.cornell.cli`` request for one stage; outputs inside ``directory``."""
    from src.cornell.parallel import resolve_workers
    options, paths, limits = settings.options, settings.paths, settings.profile.limits
    action = CLI_ACTIONS[stage]
    request = {"schema_version": 1, "action": action, "processing_root": str(settings.processing_root),
               "processing_config": str(paths["yaml_file"]), "inputs": [_entry(d) for d in inputs]}
    stem = Path(inputs[0].path).stem
    if action == "calibrate":
        positions = options.regions
        name = f"{stem}_resolved" if positions == 1 else f"{stem}_position_{positions}regions"
        cog = paths.get("cog_limits_file") if positions > 1 else None
        request.update(files={"cog_limits": None if cog is None else str(cog)},
                       options={"positions": positions, "event_limit": limits.calibration_event_limit,
                                "limit_mode": options.calibration_limit_mode,
                                "target_per_key": limits.calibration_target_per_key,
                                "memory_budget_mb": limits.calibration_memory_mb,
                                "workers": resolve_workers(limits.workers),
                                "batch_records": limits.batch_records},
                       outputs={"encal": str(directory / f"{name}.encal"),
                                "sidecar": str(directory / f"{name}.encal.json"),
                                "status": str(directory / f"{name}_status.txt"),
                                "plot": str(directory / f"{name}.png")})
    elif action == "listmode":
        from src.cornell.inputs import calibration_layout
        calibration = context.get("encal") or paths["calibration_file"]
        sidecar = context.get("sidecar")
        regions = options.regions if context.get("encal") else calibration_layout(calibration)[1]
        request.update(files={"calibration": str(calibration),
                              "calibration_sidecar": None if sidecar is None else str(sidecar),
                              "cog_limits": str(paths["cog_limits_file"]),
                              "doi_limits": str(paths["doi_limits_file"]),
                              "pair_map": str(paths["pair_map_file"]), "region_map": str(paths["region_map_file"])},
                       options={"num_regions": regions, "region_boundaries": None,
                                "metadata": to_plain(lm_header_metadata(settings.profile, settings.action,
                                                                        settings.options)),
                                "batch_records": LM_BATCH_RECORDS, "debug": options.debug, "resume": False,
                                "hit_limit": options.hit_limit,       # compact LM decodes at this width (FR-22)
                                "lm_seed": limits.lm_seed, "workers": resolve_workers(limits.workers),
                                "in_place": True},                    # the stage folder itself (T29)
                       outputs={"directory": str(directory)})
    else:
        live = settings.action == Action.QC      # offline files: source mode/duration not recorded
        request.update(files={},
                       options={"plots": options.plots, "slabs": options.slabs,
                                "source_mode": options.source_mode.value if live else None,
                                "acquisition_time_s": options.duration_s if live else None,
                                "pair_limit": limits.qc_pair_limit, "in_place": True,
                                "report_title": context.get("run_name")},
                       outputs={"directory": str(directory)})
    return request


def _write_exclusive(path, value):
    content = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with open(path, "xb") as out:
        out.write(content)
        out.flush()
        os.fsync(out.fileno())
    return hashlib.sha256(content).hexdigest()


# Coordinator -----------------------------------------------------------------

class WorkflowCoordinator:
    """One active foreground workflow. ``start`` returns at once; the graph runs in a worker."""

    def __init__(self, *, backend=None, clock=None, policy=None, acquisition_factory=None, update_sink=None,
                 log_sink=None, checkout_root=None):
        self._backend = backend
        self._clock = Clock() if clock is None else clock
        self._stage_started = self._clock.monotonic()
        self._policy = policy
        self._acquisition_factory = acquisition_factory
        self._update_sink = update_sink
        self._log_sink = log_sink
        self._checkout = checkout_root
        self._lock = RLock()
        self._sequence = 0
        self._active = None

    @property
    def active(self):
        with self._lock:
            return self._active is not None and not self._active.done

    def _log(self, line):
        if self._log_sink is not None:
            try:
                self._log_sink(line)
            except Exception:
                pass  # Logging cannot change a stage verdict.

    def _emit(self, identity, kind, message="", payload=None, *, log=True):
        with self._lock:
            event = RunEvent(identity, self._sequence, kind, message, payload or {})
            self._sequence += 1
            if message and log:
                self._log(f"[{identity.stage_id}] {kind}: {message}")
            if self._update_sink is not None:
                try:
                    self._update_sink(event)
                except Exception:
                    pass
        return event

    def start(self, plan, *, prerequisite=None):
        """``prerequisite() -> (ok, reason)`` gates live graphs (DAQD initialized) before anything is created."""
        if not isinstance(plan, WorkflowPlan):
            raise WorkflowError("Start a workflow from its prepared plan")
        if "acquisition" in plan.stages:
            ok, reason = AcquisitionService._check(prerequisite)
            if not ok:
                raise WorkflowError(f"Acquisition prerequisite not met: {reason}")
        with self._lock:
            if self._active is not None and not self._active.done:
                raise WorkflowBusy("A foreground workflow is already active")
            handle = self._active = WorkflowHandle(plan)
        Thread(target=self._run, args=(handle, prerequisite), daemon=True,
               name=f"petsys-workflow-{plan.settings.action.value}").start()
        return handle

    def close(self, timeout):
        """STOP the active workflow and wait for its children to be reaped. False if still running."""
        with self._lock:
            handle = self._active
        if handle is None:
            return True
        handle.stop()
        return handle.wait(timeout) is not None

    def _runner(self, settings, *, grace=None):
        policy = self._policy or RunnerPolicy(log_tail_lines=settings.profile.limits.log_tail_lines)
        if grace is not None:
            policy = replace(policy, terminate_grace_s=grace)
        return CommandRunner(policy=policy, backend=self._backend, clock=self._clock)

    def _run(self, handle, prerequisite):
        plan = handle.plan
        settings = plan.settings
        action = settings.action
        run_identity = Identity("unreserved", "workflow", "run")
        stages, store, findings = [], None, None
        status, message = ResultStatus.FAILED, "Workflow did not start"
        try:
            manual = action in (Action.CALIBRATE, Action.LISTMODE, Action.QC_ANALYZE)
            store = RunStore.reserve(plan.destination, settings, settings.inputs if manual else (),
                                     name=run_name(settings), stages=plan.stages)
            run_identity = Identity(store.run_id, "workflow", "run")
            self._emit(run_identity, "workflow_started", f"{action.value}: {' -> '.join(plan.stages)}",
                       {"run_root": str(store.root), "stages": list(plan.stages)})
            context = {"inputs": settings.inputs, "run_name": store.run_id}
            for stage in plan.stages:
                if handle.cancelled:
                    status, message = ResultStatus.CANCELLED, f"Stopped before the {stage} stage"
                    break
                outcome = self._stage(stage, handle, plan, store, context, prerequisite)
                stages.append(outcome)
                if outcome.status != ResultStatus.SUCCEEDED:
                    status = outcome.status
                    message = f"{stage} {outcome.status.value}: {outcome.message}; later stages not started"
                    break
            else:
                status = ResultStatus.SUCCEEDED
                message = "All stages completed with validated outputs"
                if action == Action.CONVERT:     # FR-24: what was checked, not more
                    message = (f"Outputs structure-checked (first {CONVERSION_CHECK_RECORDS:,} records of each "
                               "file); each processing stage validates the records it reads")
                elif action == Action.PIPELINE:
                    message = ("All stages completed; conversion outputs structure-checked, calibration and LM "
                               "validated the records they read")
                if action in (Action.QC, Action.QC_ANALYZE):
                    findings = stages[-1].details.get("findings")
                    message = ("QC processing completed; findings are observations of the coincidence "
                               "sample, not a detector verdict")
            if status == ResultStatus.SUCCEEDED and handle.cancelled:
                status, message = ResultStatus.CANCELLED, "Stopped before the run was finalized"
            if status != ResultStatus.SUCCEEDED:
                findings = None
            store.finish(status, message)
        except Exception as exc:
            # An unrecorded verdict is not a verdict: the manifest stays partial and this reports failure.
            status = ResultStatus.FAILED
            message = f"{message}; run record failed: {exc}" if store is not None else f"Run not created: {exc}"
            findings = None
        outcome = WorkflowOutcome(action, status, message, store.root if store else None, tuple(stages),
                                  None if findings is None else freeze(findings))
        if store is not None:
            try:
                append_overview(plan.destination, overview_row(outcome, settings, datetime.now()))
            except Exception as exc:     # an index line never changes the run's verdict
                self._log(f"[workflow] {RUNS} not updated in {plan.destination}: {exc}")
        try:
            self._emit(run_identity, "workflow_finished", message,
                       {"status": status.value, "run_root": str(store.root) if store else None,
                        "stages": [s.stage_id for s in stages]})
        finally:
            handle._finish(outcome)

    def _stage(self, stage, handle, plan, store, context, prerequisite):
        """Run one stage and record its elapsed time in its details (FR-1)."""
        self._stage_started = self._clock.monotonic()      # one foreground workflow, one stage at a time
        outcome = self._run_stage(stage, handle, plan, store, context, prerequisite)
        elapsed = round(self._clock.monotonic() - self._stage_started, 1)
        return replace(outcome, details=freeze({**to_plain(outcome.details), "elapsed_s": elapsed}))

    def _run_stage(self, stage, handle, plan, store, context, prerequisite):
        context.pop("attempt", None)
        try:
            if stage == "acquisition":
                return self._acquisition(handle, plan, store, context, prerequisite)
            if stage == "conversion":
                return self._conversion(handle, plan, store, context)
            return self._processing(stage, handle, plan, store, context)
        except Exception as exc:
            attempt = context.pop("attempt", None)
            message = f"Stage failed: {exc}"
            if attempt is not None and any(r["attempt_id"] == attempt.identity.attempt_id and
                                           r["stage_id"] == stage and r["status"] == "partial"
                                           for r in store.snapshot["attempts"]):
                try:
                    self._finish(store, attempt, CommandResult(attempt.identity, ResultStatus.FAILED, None, message),
                                 {"error": message})
                except Exception as record:
                    message = f"{message}; attempt record failed: {record}"
            return StageOutcome(stage, ResultStatus.FAILED, message,
                                attempt.identity.attempt_id if attempt else None,
                                attempt.directory if attempt else None)

    # Acquisition ---------------------------------------------------------------

    def _acquisition(self, handle, plan, store, context, prerequisite):
        settings = plan.settings
        # The acquisition service already logged these lines; forward the events only.
        forward = lambda event: self._emit(event.identity, f"acquisition_{event.kind}", event.message,
                                           dict(event.payload), log=False)
        if self._acquisition_factory is not None:
            service = self._acquisition_factory(update_sink=forward, log_sink=self._log)
        else:
            runner = self._runner(settings, grace=settings.profile.safety.terminate_grace_s)
            service = AcquisitionService(runner=runner, clock=self._clock, update_sink=forward, log_sink=self._log)
        acquisition = service.start(settings, store, basename=plan.basename, prerequisite=prerequisite)
        while acquisition.wait(0.05) is None:
            if handle.cancelled:
                acquisition.stop()
        outcome = acquisition.wait()
        last = outcome.attempts[-1] if outcome.attempts else None
        details = {"attempts": len(outcome.attempts), "bias_unknown": outcome.bias_unknown,
                   "growth": last.growth.state if last else None, "frame_loss": last.loss.state if last else None}
        if outcome.bias_unknown:
            self._emit(last.identity, "bias_unknown", "SiPM bias state unknown after an abnormal end")
        if outcome.succeeded:
            context["raw"] = Path(f"{outcome.raw_prefix}.rawf")
        return StageOutcome("acquisition", outcome.status, outcome.message,
                            last.identity.attempt_id if last else None, last.directory if last else None,
                            last.artifacts if last and outcome.succeeded else (), freeze(details))

    # Conversion ----------------------------------------------------------------

    def _conversion(self, handle, plan, store, context):
        from src.cornell.inputs import InputError, ValidationCancelled, probe_ldat

        settings = plan.settings
        options = settings.options
        mapping = _processing_config(settings, Action.CALIBRATE if settings.action == Action.PIPELINE
                                     else Action.QC_ANALYZE).mapping
        attempt = context["attempt"] = store.reserve_attempt("conversion", attempt_id="attempt-1")
        identity = attempt.identity
        prefix = attempt.directory / _conversion_prefix(plan)
        raw = context.get("raw") if "acquisition" in plan.stages else None
        self._emit(identity, "stage_started", f"Converting to {plan.conversion_format.value} "
                   f"{plan.conversion_population.value}",
                   {"directory": str(attempt.directory), "raw": str(raw or settings.paths.get("raw_input"))})
        before = store.inventory(attempt)
        validated, empty, counts = [], [], []

        def validate(command):
            found = store.discover_ldat(attempt, prefix, plan.conversion_format, plan.conversion_population,
                                        before=before)
            for artifact in found:
                if artifact.path.stat().st_size == 0:
                    empty.append(replace(artifact, disposable=True))
                    continue
                try:   # FR-24: structure check only; the consuming stage validates what it reads
                    summary = probe_ldat(artifact.input_descriptor, mapping.modules,
                                         max_records=CONVERSION_CHECK_RECORDS, expected_hit_limit=options.hit_limit,
                                         cancelled=handle._cancel.is_set)
                except ValidationCancelled:
                    return OutputValidation(False, "Cancelled while checking converter output")
                except InputError as exc:
                    return OutputValidation(False, f"Converter output invalid: {exc}")
                checked = replace(summary.descriptor, validated=summary.complete)   # whole small file: validated
                validated.append(replace(artifact, input_descriptor=checked))
                counts.append({"path": str(artifact.path), "records": summary.records_total,
                               "records_checked": summary.records_checked, "whole_file_checked": summary.complete,
                               "channel_hits_checked": summary.channel_hits_checked, "file_bytes": summary.file_size})
            if empty:
                store.record_artifacts(attempt, empty)      # recorded, kept (no cleanup by default)
            if not validated:
                return OutputValidation(False, f"No nonempty converter output for prefix {prefix.name}")
            return OutputValidation(True, f"{len(validated)} structure-checked LDAT file(s)", validated)

        command = build_conversion(settings, identity, prefix, raw_input=raw)
        result = self._runner(settings).run(command, cancellation=handle._cancel, validate_outputs=validate,
                                            log_sink=self._log)
        details = {"ldat": counts, "empty_ldat": [str(a.path) for a in empty],
                   "format": plan.conversion_format.value, "population": plan.conversion_population.value,
                   "output_check": f"structure: first {CONVERSION_CHECK_RECORDS:,} records of each file; "
                                   "each later stage validates the records it reads (FR-24)"}
        result = self._finish(store, attempt, result, details, exclude={str(a.path) for a in empty})
        if result.status == ResultStatus.SUCCEEDED:
            context["inputs"] = tuple(a.input_descriptor for a in result.artifacts)
        return StageOutcome("conversion", result.status, result.message, identity.attempt_id, attempt.directory,
                            result.artifacts if result.status == ResultStatus.SUCCEEDED else (), freeze(details))

    def _finish(self, store, attempt, result, details, *, exclude=(), extra=()):
        """Record the attempt; failed/cancelled ones keep every regular file as an unvalidated partial."""
        if result.status != ResultStatus.SUCCEEDED or not result.can_advance:
            status = result.status if result.status != ResultStatus.SUCCEEDED else ResultStatus.FAILED
            message = result.message or "Outputs were not validated"
            recorded = {item["path"] for record in store.snapshot["attempts"]
                        if record["attempt_id"] == attempt.identity.attempt_id
                        and record["stage_id"] == attempt.identity.stage_id for item in record["artifacts"]}
            partial = []
            for path in store.inventory(attempt):
                if path.is_file() and not path.is_symlink() and str(path) not in recorded | set(exclude):
                    partial.append(Artifact(path, "partial_output"))
            exit_code = None if status == ResultStatus.LAUNCH_ERROR else result.exit_code
            if status == ResultStatus.FAILED and exit_code in (None, 0) and not message:
                message = "Outputs were not validated"
            result = CommandResult(attempt.identity, status, exit_code, message, tuple(partial), result.log_tail)
        else:
            result = replace(result, artifacts=tuple(result.artifacts) + tuple(extra))
        elapsed = round(self._clock.monotonic() - self._stage_started, 1)
        details = {**(details or {}), "elapsed_s": elapsed}
        store.finish_attempt(attempt, result, details=details)
        self._emit(attempt.identity, "stage_finished", f"{attempt.identity.stage_id} {result.status.value}: "
                   f"{result.message} (in {format_elapsed(elapsed)})",
                   {"status": result.status.value, "directory": str(attempt.directory), "elapsed_s": elapsed})
        return result

    # Processing (src.cornell.cli) -------------------------------------------------

    def _processing(self, stage, handle, plan, store, context):
        from src.cornell import cli

        settings = plan.settings
        attempt = context["attempt"] = store.reserve_attempt(stage, attempt_id="attempt-1")
        identity = attempt.identity
        inputs = tuple(context["inputs"])
        request = processing_request(settings, stage, inputs, attempt.directory, context)
        request_path, result_path = attempt.directory / REQUEST, attempt.directory / RESULT
        digest = _write_exclusive(request_path, request)
        self._emit(identity, "stage_started", f"{stage}: {len(inputs)} input file(s)",
                   {"directory": str(attempt.directory), "inputs": [str(d.path) for d in inputs]})
        found = {}

        def log(line):
            event = cli.parse_event(line)
            if event is None:
                self._log(f"[{stage}] {line}")
            elif event.get("kind") == "progress":
                self._emit(identity, "stage_progress", "", {k: event.get(k) for k in
                                                            ("file_index", "files", "path", "records_read",
                                                             "records_written", "phase", "keys_done",
                                                             "keys_total")})

        def validate(command):
            value = cli.read_result(result_path, verify_hashes=True)
            found["result"] = value
            if value.get("request") != {"path": str(request_path), "sha256": digest} \
                    or value.get("action") != CLI_ACTIONS[stage]:
                return OutputValidation(False, "Processing result does not belong to this stage request")
            if value["status"] != "succeeded":
                return OutputValidation(False, "; ".join(value.get("errors") or ["processing failed"]))
            artifacts, kinds = [], set()
            for item in value["outputs"]:
                path = Path(item["path"])
                if not path.resolve().is_relative_to(attempt.directory.resolve()):
                    return OutputValidation(False, f"Processing output outside its attempt directory: {path}")
                artifacts.append(Artifact(path, item["kind"]))
                kinds.add(item["kind"])
            missing = REQUIRED_KINDS[CLI_ACTIONS[stage]] - kinds
            if missing:
                return OutputValidation(False, "Processing result lacks outputs: " + ", ".join(sorted(missing)))
            return OutputValidation(True, f"{len(artifacts)} verified output(s)", artifacts)

        command = build_internal(settings, identity, STAGE_ACTIONS[stage], request_path, result_path,
                                 checkout_root=self._checkout)
        result = self._runner(settings).run(command, cancellation=handle._cancel, validate_outputs=validate,
                                            log_sink=log)
        value = found.get("result")
        if value is None and result_path.is_file():
            try:
                value = cli.read_result(result_path, verify_hashes=False)
            except Exception:
                value = None
        details = {"request_sha256": digest, "cli_status": value and value.get("status"),
                   "cli_exit_code": value and value.get("exit_code"), "error_kind": value and value.get("error_kind"),
                   "errors": (value or {}).get("errors"),
                   "summary": (value or {}).get("summary") if result.can_advance else None}
        if result.status != ResultStatus.SUCCEEDED and details["errors"]:
            result = replace(result, message=f"{result.message}: {'; '.join(details['errors'])}")
        if stage == "listmode":
            metadata = request["options"]["metadata"]
            details["lm_header_times"] = {
                "acquisition_time_s": metadata["acquisition_time_s"],
                "measurement_time_s": metadata["measurement_time_s"],
                "source": "pipeline Acq. Time" if settings.action == Action.PIPELINE else "profile"}
        if stage == "qc" and result.can_advance:
            summary = value["summary"]
            details.update(process="completed", findings=summary.get("findings"),
                           note="Process completion is not a detector verdict")
        extra = [Artifact(path, kind) for path, kind in ((request_path, "processing_request"),
                                                         (result_path, "processing_result"))]
        result = self._finish(store, attempt, result, details, extra=extra)
        if result.status == ResultStatus.SUCCEEDED:
            by_kind = {a.kind: a.path for a in result.artifacts}
            if stage == "calibration":
                context.update(encal=by_kind["encal"], sidecar=by_kind["calibration_sidecar"])
        return StageOutcome(stage, result.status, result.message, identity.attempt_id, attempt.directory,
                            result.artifacts if result.status == ResultStatus.SUCCEEDED else (), freeze(details))
