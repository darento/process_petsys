"""Headless Cornell processing dispatch (spec 003 T11).

Usage::

    python -u -m src.cornell.cli {calibrate,listmode,qc} --request REQUEST.json --result RESULT.json

``commands.build_internal`` builds this argv with ``sys.executable``; no shell,
conda activation or ignored script is involved. The request is a bounded JSON
document listing every input, selected file, option and output path
explicitly (no defaults, nothing parsed from basenames). Paths are used
literally, so spaces and shell metacharacters need no quoting.

Outputs are created exclusively by the processing modules. The result manifest
is published once, without replacement, after the outputs have been verified.
Only ``status == "succeeded"`` (exit 0) lists outputs; a failure or
cancellation lists none, and the files written so far are kept as evidence.
Progress goes to stdout as ``@petsys-event {json}`` lines; ``parse_event``
reads them back, including from the runner's ``[stdout] `` log lines.

Exit codes: 0 succeeded, 1 failed (inputs, processing or verification),
2 invalid request or result path, 3 cancelled (SIGTERM/SIGINT).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
from threading import Event
import time
import traceback
from uuid import uuid4

from src.petsys_manager.contracts import Action, InputDescriptor, SourceMode
from src.petsys_manager.settings import LMMetadata, ProfileError
from .inputs import (InputError, MAX_METADATA_BYTES, ValidationCancelled, _finite_tree, _unique_pairs,
                     load_calibration, load_limits, load_processing_config, select_inputs)

SCHEMA_VERSION = 1
GENERATOR = "src.cornell.cli (spec 003 T11)"
EVENT_PREFIX = "@petsys-event "
MAX_REQUEST_BYTES = MAX_METADATA_BYTES
PROGRESS_INTERVAL_S = 0.5
EXIT_OK, EXIT_FAILED, EXIT_INVALID, EXIT_CANCELLED = 0, 1, 2, 3
STATUSES = ("succeeded", "failed", "cancelled")
ACTIONS = {"calibrate": Action.CALIBRATE, "listmode": Action.LISTMODE, "qc": Action.QC_ANALYZE}

# Exact request keys per action; every key is required. Kinds: "file" (existing),
# "file?" (or null), "new_file"/"new_file?" (absent, parent exists), "new_dir"
# (absent, parent exists), "job_dir" (new, or existing when resuming).
_COMMON = ("schema_version", "action", "processing_root", "processing_config", "inputs", "files", "options",
           "outputs")
_SPEC = {
    "calibrate": {
        "files": {"cog_limits": "file?"},
        "options": {"positions": "int", "event_limit": "int?", "limit_mode": "limit_mode",
                    "target_per_key": "int?", "memory_budget_mb": "int?", "workers": "int",
                    "batch_records": "int"},
        "outputs": {"encal": "new_file", "sidecar": "new_file", "status": "new_file", "plot": "new_file?"},
    },
    "listmode": {
        "files": {"calibration": "file", "calibration_sidecar": "file?", "cog_limits": "file",
                  "doi_limits": "file", "pair_map": "file", "region_map": "file"},
        "options": {"num_regions": "int", "region_boundaries": "numbers?", "metadata": "metadata",
                    "batch_records": "int", "debug": "bool", "resume": "bool", "hit_limit": "int?",
                    "lm_seed": "seed?", "workers": "int", "in_place": "bool"},
        "outputs": {"directory": "job_dir"},
    },
    "qc": {
        "files": {},
        "options": {"plots": "bool", "slabs": "bool", "source_mode": "source?", "acquisition_time_s": "number?",
                    "pair_limit": "int", "in_place": "bool", "report_title": "text?"},
        "outputs": {"directory": "new_dir"},
    },
}


class RequestError(InputError):
    """The request document itself is unusable (exit 2)."""


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class Request:
    path: Path
    sha256: str
    action: str
    processing_root: Path
    processing_config: Path
    inputs: tuple
    files: dict
    options: dict
    outputs: dict


# Request ---------------------------------------------------------------------

def _read_json(path, label, limit=MAX_REQUEST_BYTES, error=RequestError):
    try:
        with open(path, "rb") as stream:
            content = stream.read(limit + 1)
    except OSError as exc:
        raise error(f"{label} unavailable: {path}: {exc}") from exc
    if len(content) > limit:
        raise error(f"{label} exceeds {limit} bytes")

    def constant(value):
        raise error(f"{label}: nonfinite JSON number {value}")
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=constant)
        _finite_tree(value)
    except error:
        raise
    except (InputError, ValueError, UnicodeError, RecursionError) as exc:
        raise error(f"{label} is not valid bounded JSON: {exc}") from exc
    return value, hashlib.sha256(content).hexdigest()


def _absolute(value, label):
    if not isinstance(value, str) or not value or "\0" in value:
        raise RequestError(f"{label} must be an absolute path string")
    path = Path(value)
    if not path.is_absolute():
        raise RequestError(f"{label} must be an absolute path: {value!r}")
    return path


def _keys(value, expected, label):
    if not isinstance(value, dict):
        raise RequestError(f"{label} must be an object")
    missing = [key for key in expected if key not in value]
    unknown = sorted(set(value) - set(expected))
    if missing or unknown:
        raise RequestError(*([f"{label}: missing {key}" for key in missing]
                             + [f"{label}: unknown {key}" for key in unknown]))


def _value(kind, value, label):
    optional = kind.endswith("?")
    if optional and value is None:
        return None
    kind = kind.rstrip("?")
    if kind == "int":
        if type(value) is not int or value < 1:
            raise RequestError(f"{label} must be a positive integer")
    elif kind == "number":
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise RequestError(f"{label} must be a positive finite number")
        value = float(value)
    elif kind == "bool":
        if type(value) is not bool:
            raise RequestError(f"{label} must be true or false")
    elif kind == "numbers":
        if not isinstance(value, list) or any(type(v) not in (int, float) for v in value):
            raise RequestError(f"{label} must be a list of numbers or null")
        value = tuple(float(v) for v in value)
    elif kind == "source":
        try:
            value = SourceMode(value)
        except ValueError:
            raise RequestError(f"{label} must be 'with', 'without' or null (not recorded)") from None
    elif kind == "seed":
        if type(value) is not int or not 0 <= value < 2 ** 63:
            raise RequestError(f"{label} must be a non-negative integer or null")
    elif kind == "text":
        if not isinstance(value, str) or not value.strip() or len(value) > 200 or not value.isprintable():
            raise RequestError(f"{label} must be a printable text of at most 200 characters, or null")
    elif kind == "limit_mode":
        if value not in ("reference", "target"):
            raise RequestError(f"{label} must be 'reference' or 'target'")
    elif kind == "metadata":
        names = [item.name for item in fields(LMMetadata)]
        _keys(value, names, label)
        try:
            value = LMMetadata(**value)
        except (ProfileError, TypeError, ValueError) as exc:
            raise RequestError(f"{label}: {exc}") from exc
        if value.missing():
            raise RequestError(*(f"{label}: required LM metadata unavailable: {name}" for name in value.missing()))
    elif kind in ("file", "new_file", "new_dir", "job_dir"):
        value = _absolute(value, label)
    else:  # pragma: no cover - table error
        raise AssertionError(kind)
    return value


def _check_outputs(outputs, options, result_path, request_path):
    seen = {os.path.normcase(os.path.abspath(result_path)), os.path.normcase(os.path.abspath(request_path))}
    for name, path in outputs.items():
        if path is None:
            continue
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            raise RequestError(f"outputs.{name} repeats another output, the request or the result path")
        seen.add(key)
        resume = options.get("resume") is True
        if name == "directory" and options.get("in_place") is True:   # T29: the caller's existing folder
            if path.is_symlink() or not path.is_dir():
                raise RequestError(f"outputs.directory must be an existing plain directory (in_place): {path}")
            continue
        if not (resume and name == "directory") and os.path.lexists(path):
            raise RequestError(f"outputs.{name} already exists (outputs are never replaced): {path}")
        if not path.parent.is_dir():
            raise RequestError(f"outputs.{name} parent directory does not exist: {path.parent}")


def load_request(path, action, *, result_path=None):
    """Bounded, exact-key request for ``action``. Raises :class:`RequestError`."""
    if action not in _SPEC:
        raise RequestError(f"Unsupported processing action: {action!r}")
    path = _absolute(str(path), "request path")
    value, digest = _read_json(path, "Request")
    _keys(value, _COMMON, "request")
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise RequestError(f"Unsupported request schema_version (expected {SCHEMA_VERSION})")
    if value["action"] != action:
        raise RequestError(f"Request action {value['action']!r} differs from the command {action!r}")
    spec = _SPEC[action]
    root = _absolute(value["processing_root"], "processing_root")
    config = _absolute(value["processing_config"], "processing_config")
    entries = value["inputs"]
    if not isinstance(entries, list) or not entries:
        raise RequestError("inputs must be a nonempty ordered list")
    inputs = []
    for index, entry in enumerate(entries):
        label = f"inputs[{index}]"
        _keys(entry, ("path", "format", "population"), label)
        try:
            inputs.append(InputDescriptor(_absolute(entry["path"], f"{label}.path"), entry["format"],
                                          entry["population"]))
        except RequestError:
            raise
        except (TypeError, ValueError) as exc:
            raise RequestError(f"{label}: {exc}") from exc
    sections = {}
    for section in ("files", "options", "outputs"):
        _keys(value[section], spec[section], section)
        sections[section] = {name: _value(kind, value[section][name], f"{section}.{name}")
                             for name, kind in spec[section].items()}
    if action == "qc" and sections["options"]["slabs"] and not sections["options"]["plots"]:
        raise RequestError("options.slabs requires options.plots")
    _check_outputs(sections["outputs"], sections["options"], result_path or path, path)
    return Request(path, digest, action, root, config, tuple(inputs), sections["files"], sections["options"],
                   sections["outputs"])


# Events and result ------------------------------------------------------------

class Events:
    """``@petsys-event`` JSON lines with a monotonically increasing sequence."""

    def __init__(self, stream, action):
        self.stream, self.action, self.sequence = stream, action, 0
        self._last = {}

    def emit(self, kind, **payload):
        self.sequence += 1
        line = json.dumps({"sequence": self.sequence, "action": self.action, "kind": kind, **payload},
                          allow_nan=False, default=str)
        self.stream.write(EVENT_PREFIX + line + "\n")
        self.stream.flush()

    def progress(self, index, files, path, records, **extra):
        now = time.monotonic()
        key = str(path)
        if key in self._last and now - self._last[key] < PROGRESS_INTERVAL_S:
            return
        self._last[key] = now
        self.emit("progress", file_index=index, files=files, path=key, records_read=records, **extra)


def parse_event(line):
    """Event dict from a CLI stdout line (or the runner's ``[stdout] `` log line); else None."""
    if not isinstance(line, str):
        return None
    if line.startswith("[stdout] "):
        line = line[len("[stdout] "):]
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        value = json.loads(line[len(EVENT_PREFIX):])
    except ValueError:
        return None
    return value if isinstance(value, dict) and type(value.get("sequence")) is int else None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _output(kind, path, sha256=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise InputError(f"Expected output missing or not a regular file: {path}")
    return {"kind": kind, "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256 or _sha256(path)}


def _publish(path, content):
    """Write ``content`` beside ``path`` and link it into place; never replaces."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    with open(temporary, "xb") as out:
        out.write(content)
        out.flush()
        os.fsync(out.fileno())
    try:
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _result_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise RequestError(f"Result path must be absolute: {value!r}")
    if os.path.lexists(path):
        raise RequestError(f"Result already exists (never replaced): {path}")
    if not path.parent.is_dir():
        raise RequestError(f"Result parent directory does not exist: {path.parent}")
    return path


def read_result(path, *, verify_hashes=True):
    """Read a result manifest. A ``succeeded`` result is returned only when every listed
    output still exists with its recorded size (and SHA-256 when ``verify_hashes``)."""
    value, _ = _read_json(path, "Result", error=InputError)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION \
            or value.get("status") not in STATUSES or not isinstance(value.get("outputs"), list):
        raise InputError("Unsupported or malformed processing result")
    if value["status"] != "succeeded":
        if value["outputs"]:
            raise InputError("Unsuccessful result lists outputs")
        return value
    if value.get("exit_code") != EXIT_OK or not value["outputs"]:
        raise InputError("Succeeded result without exit 0 or outputs")
    for item in value["outputs"]:
        output = Path(item.get("path", ""))
        if not output.is_absolute() or not output.is_file() or output.stat().st_size != item.get("size_bytes"):
            raise InputError(f"Result output missing or changed: {output}")
        if verify_hashes and _sha256(output) != item.get("sha256"):
            raise InputError(f"Result output content changed: {output}")
    return value


# Actions ---------------------------------------------------------------------

def _config(request, action):
    return load_processing_config(request.processing_config, processing_root=request.processing_root,
                                  action=ACTIONS[action])


def _descriptors(request):
    return select_inputs(request.inputs, ACTIONS[request.action], processing_root=request.processing_root)


def run_calibrate(request, events, cancelled):
    from . import calibration as cal

    config = _config(request, "calibrate")
    descriptors = _descriptors(request)
    options, outputs = request.options, request.outputs
    if options["positions"] > 1 and request.files["cog_limits"] is None:
        raise RequestError("files.cog_limits is required for a position calibration (positions >= 2)")
    limits = (None if options["positions"] == 1 else
              load_limits(request.files["cog_limits"], config.mapping, kind="cog"))
    paths = [str(d.path) for d in descriptors]

    def progress(path, records, phase="read", **extra):     # FR-1: read / pass 2 per file, fits per keys
        index = None if path is None else paths.index(str(path))
        events.progress(index, len(paths), f"{phase}:{path}" if path is None else path, records, phase=phase,
                        **extra)
    result = cal.calibrate(descriptors, config, limits, positions=options["positions"],
                           event_limit=options["event_limit"], limit_mode=options["limit_mode"],
                           target_per_key=options["target_per_key"],
                           memory_budget=(None if options["memory_budget_mb"] is None
                                          else options["memory_budget_mb"] * 2 ** 20),
                           batch_records=options["batch_records"], workers=options["workers"],
                           cancelled=cancelled, progress=progress)
    if cancelled():
        raise Cancelled("Calibration cancelled")
    digest = cal.write_calibration(result, outputs["encal"], outputs["sidecar"], outputs["status"])
    events.emit("output", output_kind="encal", path=str(outputs["encal"]))
    written = [_output("encal", outputs["encal"], digest), _output("calibration_sidecar", outputs["sidecar"]),
               _output("calibration_status", outputs["status"])]
    if outputs["plot"] is not None:
        cal.plot_summary(result, outputs["plot"])
        written.append(_output("calibration_plot", outputs["plot"]))
    # The written pair must be what the listmode stage will accept.
    loaded = load_calibration(outputs["encal"], config.mapping, expected_regions=result.positions,
                              metadata_path=outputs["sidecar"])
    if len(loaded.values) != len(result.factors) or loaded.sha256 != digest or loaded.layout != result.layout:
        raise InputError("Written calibration does not read back as produced")
    summary = {
        "layout": result.layout, "positions": result.positions, "format": result.data_format.value,
        "population": result.population.value, "region_boundaries": list(result.boundaries),
        "min_ch": result.min_ch, "en_min_ch": result.en_min_ch,
        "passing_event_limit_per_file": result.event_limit, "batch_records": result.batch_records,
        "limit_plan": result.limit_plan, "coverage": result.coverage, "decoding": result.decoding,
        "workers": result.workers,
        "inputs": [{"path": str(f.path), "records_validated": f.records_validated, "records_read": f.records_read,
                    "events_passed": f.events_passed, "accepted_sides": f.accepted_sides,
                    "stopped_at_limit": f.stopped_at_limit, "rejected": f.rejected} for f in result.files],
        "status_counts": result.status_counts(), "keys": len(result.keys), "factors": len(result.factors),
        "zero_width_limit_keys": {"cog": len(limits.zero_width) if limits is not None else 0},
        "energy_units": "a.u.",
    }
    return written, summary


def run_listmode(request, events, cancelled):
    from . import listmode as lm

    config = _config(request, "listmode")
    descriptors = _descriptors(request)
    files, options = request.files, request.options
    mapping = config.mapping
    calibration = load_calibration(files["calibration"], mapping, expected_regions=options["num_regions"],
                                   region_boundaries=options["region_boundaries"],
                                   metadata_path=files["calibration_sidecar"])
    cog = load_limits(files["cog_limits"], mapping, kind="cog")
    doi = load_limits(files["doi_limits"], mapping, kind="doi")
    paths = [str(d.path) for d in descriptors]
    result = lm.generate_listmode(
        descriptors, config, calibration, cog, doi, lm.load_pair_map(files["pair_map"]),
        lm.load_region_map(files["region_map"], mapping), options["metadata"], request.outputs["directory"],
        resume=options["resume"], debug=options["debug"], batch_records=options["batch_records"],
        hit_limit=options["hit_limit"], lm_seed=options["lm_seed"], workers=options["workers"],
        in_place=options["in_place"], cancelled=cancelled,
        progress=lambda index, path, records, written: events.progress(index, len(paths), path, records,
                                                                       records_written=written))
    if cancelled():
        raise Cancelled("Listmode cancelled")
    expected = lm.HEADER_BYTES + result.records * lm.RECORD_DTYPE.itemsize
    if result.output.stat().st_size != expected:
        raise InputError(f"Listmode size {result.output.stat().st_size} differs from header + records {expected}")
    events.emit("output", output_kind="listmode", path=str(result.output))
    written = [_output("listmode", result.output, result.sha256), _output("listmode_provenance", result.sidecar),
               _output("listmode_job", Path(request.outputs["directory"]) / lm.JOB_FILE)]
    written += [_output("listmode_debug_plot", path) for path in result.debug_outputs]
    summary = {
        "records_written": result.records, "job_sha256": result.job_sha256, "resumed": result.resumed,
        "input_format": descriptors[0].format.value, "compact_hit_limit": options["hit_limit"],
        "ignored_partial_files": [str(path) for path in result.ignored],
        "inputs": [{"path": str(f.path), "validated_records": f.validated_records, "records_read": f.records_read,
                    "records_written": f.records_written, "rejected": f.rejected, "observations": f.observations,
                    "reused_segment": f.reused} for f in result.files],
        "rejected": result.totals("rejected"), "observations": result.totals("observations"),
        "en_min_ch_applied": False,
        "calibration_non_positive_mu_as_no_factor": len(calibration.non_positive),
        "zero_width_limit_keys": {"cog": len(cog.zero_width), "doi": len(doi.zero_width)},
    }
    return written, summary


def run_qc(request, events, cancelled):
    from . import qc, qc_report

    config = _config(request, "qc")
    descriptors = _descriptors(request)
    options = request.options
    paths = [str(d.path) for d in descriptors]
    result = qc.run_qc(descriptors, config, plots=options["plots"], slabs=options["slabs"],
                       source_mode=options["source_mode"], acquisition_time_s=options["acquisition_time_s"],
                       pair_limit=options["pair_limit"], cancelled=cancelled,
                       progress=lambda path, records: events.progress(paths.index(str(path)), len(paths), path,
                                                                      records))
    if cancelled():
        raise Cancelled("QC cancelled")
    report = qc_report.write_report(result, request.outputs["directory"], in_place=options["in_place"],
                                    title=options["report_title"])
    events.emit("output", output_kind="qc_report", path=str(request.outputs["directory"]))
    content, _ = _read_json(report[-1], "QC summary", error=InputError)
    recorded = {item["path"]: item["sha256"] for item in content["outputs"]}
    written = []
    for path in report[:-1]:
        item = _output("qc_report", path)
        if recorded.get(Path(path).name) != item["sha256"]:
            raise InputError(f"QC output differs from its summary record: {path}")
        written.append(item)
    written.append(_output("qc_summary", report[-1]))
    findings = content["findings"]
    summary = {
        "process": content["process"], "source_mode": content["source_mode"],
        "acquisition_time_s": content["acquisition_time_s"], "options": content["options"],
        "totals": content["totals"],
        "inputs": [{key: item[key] for key in ("path", "validated_records", "records_read", "occupancy_pairs",
                                               "accepted_pairs", "stopped_at_limit")} for item in content["inputs"]],
        "minimodule_fit_status": (content["minimodule_fits"] or {}).get("status_counts"),
        "slab_fit_status": (content["slab_fits"] or {}).get("status_counts"),
        "findings": {"expected_minimodules": findings["expected_minimodules"],
                     "minimodules_without_hits": len(findings["minimodules_without_hits"]),
                     "declared_unpopulated_with_hits": len(findings["declared_unpopulated_with_hits"]),
                     "missing_time_channels": findings["missing_time_channels"],
                     "missing_energy_channels": findings["missing_energy_channels"]},
        "energy_units": "a.u.",
        "note": content["note"],
    }
    return written, summary


RUNNERS = {"calibrate": run_calibrate, "listmode": run_listmode, "qc": run_qc}
_CANCELLED = (Cancelled, ValidationCancelled)


def _cancel_types():
    from .calibration import CalibrationCancelled
    from .listmode import ListmodeCancelled
    from .qc import QCCancelled
    return _CANCELLED + (CalibrationCancelled, ListmodeCancelled, QCCancelled)


def _issues(exc):
    issues = getattr(exc, "issues", None)
    return [str(i) for i in issues] if issues else [f"{type(exc).__name__}: {exc}"]


# Entry point -----------------------------------------------------------------

def main(argv=None, *, cancel_event=None, stdout=None, stderr=None):
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    for stream in (stdout, stderr):     # an unencodable path in a message must not block the result
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="python -m src.cornell.cli", description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=sorted(RUNNERS))
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    cancel_event = cancel_event or Event()
    cancelled = cancel_event.is_set
    started = datetime.now(timezone.utc)
    events = Events(stdout, args.action)
    try:
        result_path = _result_path(args.result)
    except RequestError as exc:
        print(f"Invalid result path; no result written: {exc}", file=stderr)
        return EXIT_INVALID
    record = {"schema_version": SCHEMA_VERSION, "generator": GENERATOR, "action": args.action,
              "request": {"path": args.request, "sha256": None}, "started_utc": started.isoformat(),
              "python": sys.version.split()[0], "executable": sys.executable, "pid": os.getpid(),
              "outputs": [], "summary": None, "errors": []}
    try:
        request = load_request(args.request, args.action, result_path=result_path)
        record["request"]["sha256"] = request.sha256
        # The inputs are in the request; a long path list would exceed the runner's line bound and be split.
        events.emit("started", request=str(request.path), inputs=len(request.inputs))
        if cancelled():
            raise Cancelled("Cancelled before processing")
        outputs, summary = RUNNERS[args.action](request, events, cancelled)
        if cancelled():
            raise Cancelled("Cancelled before publishing the result")
        status, code = "succeeded", EXIT_OK
        record.update(outputs=outputs, summary=summary)
    except RequestError as exc:
        status, code = "failed", EXIT_INVALID
        record.update(error_kind="invalid_request", errors=_issues(exc))
    except _cancel_types() as exc:
        status, code = "cancelled", EXIT_CANCELLED
        record.update(error_kind="cancelled", errors=_issues(exc))
    except (InputError, ProfileError) as exc:
        status, code = "failed", EXIT_FAILED
        record.update(error_kind="input_or_processing", errors=_issues(exc))
    except Exception as exc:  # numerical/unexpected failure: never a success result
        status, code = "failed", EXIT_FAILED
        record.update(error_kind="unexpected", errors=_issues(exc))
        traceback.print_exc(file=stderr)
    record.update(status=status, exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat())
    for message in record["errors"]:
        print(f"{status}: {message}", file=stderr)
    try:
        _publish(result_path, (json.dumps(record, indent=2, allow_nan=False, default=str) + "\n").encode("utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Could not publish result {result_path}: {exc}", file=stderr)
        return code if code != EXIT_OK else EXIT_FAILED
    events.emit("finished", status=status, exit_code=code, result=str(result_path))
    return code


def _signals(event):
    def handler(signum, frame):
        event.set()
    for name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)


if __name__ == "__main__":
    _event = Event()
    _signals(_event)
    sys.exit(main(cancel_event=_event))
