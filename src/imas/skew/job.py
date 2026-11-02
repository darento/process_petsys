"""Pooled IMAS skew job: ``python -m src.imas.skew run JOB.json --work DIR`` (spec 011 T13, FR-4, FR-6, FR-8).

The job file declares every run and split with explicit roles; nothing else
is read::

    {"schema": "imas-skew-job/1", "config": "configs/imas_2DAQ.yaml",
     "runs": [{"name": "r1", "splits": [{"path": ".../x_00000008.ldat", "roles": ["encal", "train"]},
                                         {"path": ".../x_00000009.ldat", "roles": ["train"]},
                                         {"path": ".../x_00000010.ldat", "roles": ["heldout"]}]}],
     "estimators": ["em", "pairs"], "per_run_solves": false, "workers": 4,
     "legacy": ".../skew_IMAS.txt", "transverse_sign": 1, "axial_sign": -1, "walk_lut": null}

Roles:

* ``encal``: the run's dedicated 511 keV calibration splits (FR-4).
* ``train``: enters the skew fit.
* ``heldout``: never enters any fit, and cannot be ``encal`` or ``train``.

Each run needs at least one ``encal`` and one ``train`` split. Unknown keys,
unknown roles, repeated paths and missing files are refused before any work.

Stages, each resumable in ``DIR`` (whose ``job_fingerprint.json`` must match
the job):

1. Per run, energy calibration from its ``encal`` splits (``encal.run``),
   reused when its inputs match.
2. Per run, a streamed observation set of its train and held-out splits (T8,
   held-out files in the ``test`` role), resumed per file.
3. Pooled solve over all runs' train rows: global EM and/or spec 008 pair
   fits and global solve. Canonical LUTs ``lut_<estimator>.csv``.
4. Checks:
   * the pooled held-out events with each LUT (``solve.held_out``) and with
     the legacy LUT;
   * per run, 511 keV peaks against the first run and EM residual widths
     with the pooled delays on its train and held-out events;
   * optionally, a solve per run compared with the pooled delays, so a
     run-dependent offset stays visible (FR-6).

Walk (spec 012 FR-4): the observation sets carry the raw walk ingredients.
``walk_lut`` names an IMAS walk LUT; stage 3 and the checks then read every
partition through ``walk.WalkCorrected`` (rows without every walk input
dropped and counted). ``walk_lut`` is not part of the job fingerprint, which
identifies the streamed events, so a walk solve reuses the work directory;
its outputs are ``lut_<estimator>_walk-<sha12>.csv`` and
``job_report_walk-<sha12>.json``, never overwritten. ``run --observe-only``
stops after stage 2 (``job_report_observe.json``).

Timestamp rule (spec 014 FR-4): ``candidates`` streams the other TIME hits of
each side into the observation sets (a new work directory; not part of the
job fingerprint, which identifies the events). ``timestamp_rule``
``top_earliest_mean`` (em, with a walk LUT fitted under the same rule; other
declared estimators stay in the fingerprint but are skipped and recorded)
solves the skew as a fixed point: from ``rule_start_skew`` (a frozen IMAS LUT
with its SHA-256), each pass runs the EM on the walk-corrected rows with the
rule's offsets of the previous delays (``timestamp.RuleAdjusted``) until the
applied delays change by less than ``rule_tolerance_ps`` (gauge-free RMS) or
after ``rule_passes``. Outputs gain ``_rule-mean``; the checks read the rows
under the rule with the final delays.

A disk preflight checks the free space in ``DIR`` against the expected
observation-set size (6.3 GB per TB of LDAT, T8) with a 2x margin.
``job_report.json`` holds stage timings, throughput and peak memory.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from src.imas.skew import cache as ic, encal, outputs, solve, timestamp
from src.imas.skew import walk as iw
from src.imas.skew.geometry import SELECTED_AXIAL_SIGN, SELECTED_TRANSVERSE_SIGN, ImasGeometry
from src.imas.skew.setup import ImasSetup
from src.timing.cache import OBSERVATION_DTYPE, OBSERVATION_WALK_DTYPE, observation_dtype
from src.timing.lut import DelayTable, LutError
from src.timing.report import peak_memory

SCHEMA = "imas-skew-job/1"
ROLES = ("encal", "train", "heldout")
# T8: 313 MB from 49.6 GB without walk inputs; rows carry them since spec 012 (FR-4)
OBSERVATION_BYTES_PER_LDAT_BYTE = 6.3e-3 * OBSERVATION_WALK_DTYPE.itemsize / OBSERVATION_DTYPE.itemsize
DISK_MARGIN = 2.0
JOB_KEYS = {"schema", "config", "runs", "estimators", "per_run_solves", "workers", "max_records", "legacy",
            "transverse_sign", "axial_sign", "walk_lut", "candidates", "timestamp_rule", "rule_start_skew",
            "rule_start_skew_sha256", "rule_passes", "rule_tolerance_ps"}
RULE_SUFFIX = {"top_earliest_mean": "_rule-mean"}


class JobError(ValueError):
    """The job file or its work directory is inconsistent; nothing was changed."""


@dataclass(frozen=True)
class Split:
    path: Path
    roles: frozenset


@dataclass(frozen=True)
class Run:
    name: str
    splits: tuple

    def having(self, role):
        return [s for s in self.splits if role in s.roles]


@dataclass(frozen=True)
class Job:
    config: Path
    runs: tuple
    estimators: tuple = solve.ESTIMATORS
    per_run_solves: bool = False
    workers: int = 1
    max_records: int | None = None
    legacy: Path | None = None
    transverse_sign: int = SELECTED_TRANSVERSE_SIGN
    axial_sign: int = SELECTED_AXIAL_SIGN
    source: dict | None = None
    walk_lut: Path | None = None        # spec 012 FR-4: applied before the solves; not in the fingerprint
    # spec 014 FR-2/FR-4, none in the fingerprint: candidates select the row dtype, the rule the solve
    candidates: bool = False
    timestamp_rule: str = "top"
    rule_start_skew: Path | None = None
    rule_start_skew_sha256: str | None = None
    rule_passes: int = 3
    rule_tolerance_ps: float = 0.5


def load_job(path):
    """Validated job; every refusal names the offending entry."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise JobError(f"{path.name}: schema must be {SCHEMA!r}")
    unknown = set(data) - JOB_KEYS
    if unknown:
        raise JobError(f"{path.name}: unknown keys {sorted(unknown)}")
    runs, seen = [], set()
    for run in data.get("runs") or []:
        if set(run) - {"name", "splits"} or not run.get("name") or not run.get("splits"):
            raise JobError(f"run {run.get('name')!r}: needs exactly a name and its splits")
        splits = []
        for split in run["splits"]:
            if set(split) - {"path", "roles"} or not split.get("path"):
                raise JobError(f"run {run['name']}: each split needs exactly a path and its roles")
            roles = frozenset(split.get("roles") or ())
            if not roles or roles - set(ROLES):
                raise JobError(f"{split['path']}: roles must be a non-empty subset of {ROLES}")
            if "heldout" in roles and len(roles) > 1:
                raise JobError(f"{split['path']}: a held-out split has no other role")
            resolved = Path(split["path"]).resolve()
            if resolved in seen:
                raise JobError(f"{split['path']}: declared twice")
            seen.add(resolved)
            splits.append(Split(Path(split["path"]), roles))
        result = Run(str(run["name"]), tuple(splits))
        if not result.having("encal") or not result.having("train"):
            raise JobError(f"run {result.name}: needs at least one encal and one train split")
        runs.append(result)
    if not runs:
        raise JobError(f"{path.name}: no runs")
    if len({r.name for r in runs}) != len(runs):
        raise JobError(f"{path.name}: run names must be unique")
    estimators = tuple(data.get("estimators") or solve.ESTIMATORS)
    if not estimators or set(estimators) - set(solve.ESTIMATORS):
        raise JobError(f"{path.name}: estimators must be among {solve.ESTIMATORS}")
    rule = data.get("timestamp_rule", "top")
    passes, tolerance = data.get("rule_passes", 3), data.get("rule_tolerance_ps", 0.5)
    start = data.get("rule_start_skew")
    if rule not in ("top", "top_earliest_mean"):
        raise JobError(f"{path.name}: timestamp_rule must be 'top' or 'top_earliest_mean' ('earliest' is for "
                       "skew-only comparisons)")
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1 or not float(tolerance) > 0:
        raise JobError(f"{path.name}: rule_passes must be a positive integer and rule_tolerance_ps positive")
    if rule == "top" and (start or data.get("rule_start_skew_sha256")):
        raise JobError(f"{path.name}: rule_start_skew is the start of a solve under a timestamp rule other than 'top'")
    if rule != "top":
        missing = [k for k in ("walk_lut", "rule_start_skew", "rule_start_skew_sha256") if not data.get(k)]
        if missing or not data.get("candidates"):
            raise JobError(f"{path.name}: timestamp_rule {rule!r} needs candidates true, walk_lut, rule_start_skew "
                           f"and rule_start_skew_sha256 (missing {missing or ['candidates']})")
        if "em" not in estimators:
            raise JobError(f"{path.name}: under timestamp_rule {rule!r} the estimators must include 'em' (the "
                           "others are skipped)")
    return Job(Path(data.get("config", "configs/imas_2DAQ.yaml")), tuple(runs), estimators,
               bool(data.get("per_run_solves", False)), int(data.get("workers", 1)), data.get("max_records"),
               Path(data["legacy"]) if data.get("legacy") else None,
               int(data.get("transverse_sign", SELECTED_TRANSVERSE_SIGN)),
               int(data.get("axial_sign", SELECTED_AXIAL_SIGN)), data,
               Path(data["walk_lut"]) if data.get("walk_lut") else None, bool(data.get("candidates", False)), rule,
               Path(start) if start else None, data.get("rule_start_skew_sha256"), passes, float(tolerance))


def _identity(path):
    stat = Path(path).stat()
    return {"name": Path(path).name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def preflight(job, work):
    """Every declared file present; free disk for the observation sets (2x margin)."""
    missing = [str(s.path) for r in job.runs for s in r.splits if not s.path.is_file()]
    if missing:
        raise JobError(f"declared files not found: {missing[:5]}")
    if job.legacy is not None and not job.legacy.is_file():
        raise JobError(f"legacy LUT not found: {job.legacy}")
    if job.walk_lut is not None and not job.walk_lut.is_file():
        raise JobError(f"walk LUT not found: {job.walk_lut}")
    if job.rule_start_skew is not None and not job.rule_start_skew.is_file():
        raise JobError(f"rule start skew LUT not found: {job.rule_start_skew}")
    ldat = sum(s.path.stat().st_size for r in job.runs for s in r.splits if s.roles & {"train", "heldout"})
    per_byte = OBSERVATION_BYTES_PER_LDAT_BYTE
    if job.candidates:
        per_byte *= observation_dtype(True, timestamp.SLOTS).itemsize / OBSERVATION_WALK_DTYPE.itemsize
    need = DISK_MARGIN * per_byte * ldat
    probe = Path(work)
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < need:
        raise JobError(f"{work}: {free / 1e9:.1f} GB free, {need / 1e9:.1f} GB needed for the observation sets")
    return {"ldat_bytes_observed": ldat, "disk_needed_bytes": need, "disk_free_bytes": free}


def fingerprint(job):
    declared = {"config": str(job.config), "estimators": list(job.estimators), "max_records": job.max_records,
                "signs": [job.transverse_sign, job.axial_sign],
                "runs": [{"name": r.name, "splits": [{**_identity(s.path), "path": str(s.path.resolve()),
                                                      "roles": sorted(s.roles)} for s in r.splits]}
                         for r in job.runs]}
    return {"schema": SCHEMA, "declared": declared,
            "sha256": hashlib.sha256(json.dumps(declared, sort_keys=True).encode()).hexdigest()}


def _claim(work, job):
    work = Path(work)
    marker = work / "job_fingerprint.json"
    stamp = fingerprint(job)
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8"))["sha256"] != stamp["sha256"]:
            raise JobError(f"{work}: work directory of another job (different runs, files, roles or settings)")
    elif work.exists() and any(work.iterdir()):
        raise JobError(f"{work}: not empty and not this job's work directory")
    else:
        work.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(stamp, indent=1) + "\n", encoding="utf-8")
    return stamp


def _energy_calibration(run, directory, job):
    """``encal.run`` on the run's encal splits, or its earlier output when the inputs match."""
    manifest = directory / "encal.json"
    expected = [_identity(s.path) for s in run.having("encal")]
    if manifest.exists():
        inputs = json.loads(manifest.read_text(encoding="utf-8"))["inputs"]
        if [{k: i[k] for k in ("name", "size_bytes", "mtime_ns")} for i in inputs] != expected:
            raise JobError(f"{directory}: energy calibration from other splits")
        return directory / "encal.tsv", json.loads(manifest.read_text(encoding="utf-8")), True
    if directory.exists():
        shutil.rmtree(directory)       # incomplete output of this job (no manifest)
    path, written = encal.run([s.path for s in run.having("encal")], directory, config_path=job.config,
                              max_records=job.max_records, workers=job.workers)
    return path, written, False


def _observations(run, directory, job, encal_path, geometry):
    splits = [s for s in run.splits if s.roles & {"train", "heldout"}]
    inputs = [{"index": k, "path": s.path} for k, s in enumerate(splits)]
    holdout = {"qualified": True, "roles_frozen_before_fitting": True,
               "partitions": [{"index": k, "role": "test" if "heldout" in s.roles else "train",
                               "anchor_start_ps": -2 ** 62, "anchor_end_ps": 2 ** 62} for k, s in enumerate(splits)]}
    if (directory / "fingerprint.json").is_file():
        existing = json.loads((directory / "fingerprint.json").read_text(encoding="utf-8"))
        if "walk_inputs" not in existing:
            raise JobError(f"{directory}: observation set streamed without walk inputs (before spec 012); this job "
                           "streams them: use a new work directory")
        if ("candidates" in existing) != job.candidates:
            raise JobError(f"{directory}: observation set streamed {'with' if 'candidates' in existing else 'without'}"
                           f" the candidates (spec 014); this job {'needs' if job.candidates else 'does not stream'} "
                           "them: use another work directory")
    return ic.stream_observations(directory, inputs, geometry, config_path=job.config, encal_path=encal_path,
                                  holdout=holdout, energy_calibration_prefix={k: 0 for k in range(len(splits))},
                                  max_records=job.max_records, workers=job.workers, carry_walk=True,
                                  carry_candidates=job.candidates)


class PooledObservations:
    """Several runs' observation sets as one, partition by partition (same pair hashing)."""

    def __init__(self, sets):
        self.sets = list(sets)
        counts = {s.partitions for s in self.sets}
        if len(counts) != 1:
            raise JobError("observation sets disagree on pair partitions")
        self.partitions = counts.pop()

    @property
    def carries_candidates(self):
        return all(getattr(s, "carries_candidates", False) for s in self.sets)

    def partition(self, p):
        return np.concatenate([s.partition(p) for s in self.sets])


def _peaks(calibrations):
    """511 keV peak per TIME channel, each run against the first: median ratio and robust spread."""
    names = list(calibrations)
    reference = calibrations[names[0]]
    result = {}
    for name in names[1:]:
        both = np.isfinite(reference) & np.isfinite(calibrations[name])
        ratio = calibrations[name][both] / reference[both]
        result[f"{name} / {names[0]}"] = {
            "channels": int(both.sum()), "median_ratio": float(np.median(ratio)) if both.any() else None,
            "robust_sd_ratio": float(1.4826 * np.median(np.abs(ratio - np.median(ratio)))) if both.any() else None}
    return result


def _mu(path, setup):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        channel, mu = line.split("\t")
        values[int(channel)] = float(mu)
    return np.array([values.get(int(c), np.nan) for c in setup.time_channels])


def _walk(job, work, setup, stamp):
    """The job's walk LUT (checked against the map and fitted on this job's events) and its output suffix; outputs
    that exist are refused."""
    if job.walk_lut is None:
        return None, ""
    try:
        table = iw.WalkLut.read(job.walk_lut)
    except (iw.core.WalkError, OSError, ValueError, KeyError) as error:
        raise JobError(f"walk LUT {job.walk_lut}: {error}") from None
    if table.manifest.get("map_sha256") != setup.provenance["map_sha256"]:
        raise JobError(f"walk LUT {job.walk_lut}: fitted for another map (map SHA-256 differs)")
    if table.manifest.get("job_sha256") != stamp["sha256"]:
        raise JobError(f"walk LUT {job.walk_lut}: fitted on another job's events (job SHA-256 differs)")
    try:
        timestamp.check_rule(timestamp.table_rule(table.manifest), job.timestamp_rule)
    except timestamp.RuleError as error:
        raise JobError(f"walk LUT {job.walk_lut}: {error}") from None
    suffix = f"_walk-{table.sha256[:12]}{RULE_SUFFIX.get(job.timestamp_rule, '')}"
    existing = [p for p in [work / f"job_report{suffix}.json", *(work / f"lut_{e}{suffix}.csv" for e in solved(job))]
                if p.exists()]
    if existing:
        raise JobError(f"outputs of this walk solve exist: {[p.name for p in existing]}; never overwritten")
    return table, suffix


def solved(job):
    """The estimators a job solves: under a timestamp rule only ``em`` (the others are declared, so they stay in
    the fingerprint, and skipped)."""
    return job.estimators if job.timestamp_rule == "top" else ("em",)


def _start_skew(job, setup):
    """The frozen start delays of a solve under a timestamp rule (SHA-256 and map checked)."""
    if hashlib.sha256(job.rule_start_skew.read_bytes()).hexdigest() != job.rule_start_skew_sha256:
        raise JobError(f"rule start skew LUT {job.rule_start_skew} is not the frozen {job.rule_start_skew_sha256}")
    try:
        return DelayTable.from_lut(outputs.read_imas_lut(job.rule_start_skew, setup))
    except LutError as error:
        raise JobError(f"rule start skew LUT {job.rule_start_skew}: {error}") from None


def delay_change(new, old):
    """Gauge-free change of applied delays over the common channels: RMS and maximum after removing the median."""
    common = sorted(set(new) & set(old))
    if len(common) < 2:
        return {"channels": len(common), "rms_ps": None, "max_abs_ps": None}
    difference = np.array([new[c][0] - old[c][0] for c in common])
    difference -= np.median(difference)
    return {"channels": len(common), "rms_ps": float(np.sqrt(np.mean(difference ** 2))),
            "max_abs_ps": float(np.abs(difference).max())}


def solve_under_rule(observations, setup, start, walk, rule, *, passes, tolerance_ps, progress=print):
    """FR-4 fixed point: EM passes on ``observations`` (walk-corrected rows with the candidates) adjusted by the
    rule's offsets of the previous pass's delays (``start`` first), until the applied delays change by less than
    ``tolerance_ps`` gauge-free RMS or after ``passes``. Returns the last solution, the per-pass record and its
    delays as a ``DelayTable``."""
    previous = start
    old = {c: (b, np.nan) for c, b in start.delays.items()}
    history = []
    for k in range(1, passes + 1):
        view = timestamp.RuleAdjusted(observations, previous, walk, rule)
        solution, _ = solve.em_estimator(view, setup)
        table = solve.applied_table(solution)
        change = delay_change(table, old)
        counts = view.counts()
        history.append({"pass": k, "change": change, "status_counts": solution.diagnostics["status_counts"],
                        "rule_counts": counts,
                        "rule_acts_share_train": counts.get("train", {}).get("rule acts", 0)
                        / max(counts.get("train", {}).get("rows", 0), 1)})
        progress(f"rule pass {k}: delays changed by {change['rms_ps']} ps RMS")
        previous, old = DelayTable({c: d for c, (d, _) in table.items()}), table
        if change["rms_ps"] is not None and change["rms_ps"] < tolerance_ps:
            break
    return solution, history, previous


def run_job(job, work, *, progress=print, observe_only=False):
    work = Path(work)
    started = time.perf_counter()
    disk = preflight(job, work)
    stamp = _claim(work, job)
    setup = ImasSetup.from_files(job.config)
    walk_lut, suffix = (None, "") if observe_only else _walk(job, work, setup, stamp)
    geometry = ImasGeometry.from_setup(setup, transverse_sign=job.transverse_sign, axial_sign=job.axial_sign)
    report = {"schema_version": 1, "scanner": "imas", "requirement": "011-FR-6/FR-12",
              "created_utc": datetime.now(timezone.utc).isoformat(), "job": stamp, "preflight": disk,
              "geometry": geometry.describe(), "runs": {}, "stages": {}}
    sets, calibrations = {}, {}
    for run in job.runs:
        clock = time.perf_counter()
        encal_path, manifest, reused = _energy_calibration(run, work / run.name / "encal", job)
        calibrations[run.name] = _mu(encal_path, setup)
        encal_seconds = time.perf_counter() - clock
        progress(f"{run.name}: energy calibration {'reused' if reused else 'done'} {manifest['status_counts']}")
        clock = time.perf_counter()
        observations = _observations(run, work / run.name / "observations", job, encal_path, geometry)
        stream_seconds = time.perf_counter() - clock
        sets[run.name] = observations
        ldat = sum(s.path.stat().st_size for s in run.splits if s.roles & {"train", "heldout"})
        report["runs"][run.name] = {
            "encal": {"path": str(encal_path), "sha256": hashlib.sha256(encal_path.read_bytes()).hexdigest(),
                      "status_counts": manifest["status_counts"], "reused": reused, "seconds": encal_seconds},
            "observations": {"path": str(observations.directory), **{k: observations.manifest[k] for k in (
                "observations", "first_failure", "geometry", "files")}, "seconds": stream_seconds,
                "ldat_bytes": ldat, "bytes_per_second": ldat / stream_seconds if stream_seconds > 0 else None}}
        progress(f"{run.name}: {observations.manifest['observations']} observations")
    if observe_only:
        report["stages"]["total_seconds"] = time.perf_counter() - started
        report["peak_memory"] = peak_memory()
        (work / "job_report_observe.json").write_text(json.dumps(report, indent=1, default=float) + "\n",
                                                      encoding="utf-8")
        return report
    pooled = PooledObservations(sets.values())
    provenance = {"job_sha256": stamp["sha256"], "setup": setup.provenance, "geometry": geometry.describe(),
                  "runs": {name: {"encal_sha256": r["encal"]["sha256"], "observations": r["observations"]["path"]}
                           for name, r in report["runs"].items()}}
    if walk_lut is not None:
        plain = [name for name, s in sets.items() if not s.carries_walk]
        if plain:
            raise JobError(f"observation sets without walk inputs: {plain}; stream them again in a new work directory")
        pooled = iw.WalkCorrected(pooled, walk_lut)
        sets = {name: iw.WalkCorrected(s, walk_lut) for name, s in sets.items()}
        provenance |= {"walk_lut": str(job.walk_lut), "walk_lut_sha256": walk_lut.sha256}
        report["walk"] = {"walk_lut": str(job.walk_lut), "walk_lut_sha256": walk_lut.sha256,
                          "skew_lut_sha256_at_fit": walk_lut.manifest.get("skew_lut_sha256"),
                          "note": "every solve and check (legacy included) reads walk-corrected dt"}
    ruled = {}
    if job.timestamp_rule != "top":
        start = _start_skew(job, setup)
        try:
            timestamp.require_candidates(pooled, job.timestamp_rule)
        except timestamp.RuleError as error:
            raise JobError(str(error)) from None
        clock = time.perf_counter()
        ruled["em"], history, final = solve_under_rule(pooled, setup, start, walk_lut, job.timestamp_rule,
                                                        passes=job.rule_passes, tolerance_ps=job.rule_tolerance_ps,
                                                        progress=progress)
        report["stages"]["rule_passes_seconds"] = time.perf_counter() - clock
        converged = history[-1]["change"]["rms_ps"] is not None and \
            history[-1]["change"]["rms_ps"] < job.rule_tolerance_ps
        rule = {"timestamp_rule": job.timestamp_rule, "rule_start_skew": str(job.rule_start_skew),
                "rule_start_skew_sha256": job.rule_start_skew_sha256, "rule_passes": history,
                "rule_converged": converged, "rule_tolerance_ps": job.rule_tolerance_ps}
        provenance |= rule
        report["rule"] = rule | {"note": "the checks (legacy included) read the rows under the rule with the final "
                                         "pass's delays",
                                 "skipped_estimators": [e for e in job.estimators if e not in solved(job)]}
        pooled = timestamp.RuleAdjusted(pooled, final, walk_lut, job.timestamp_rule)
        sets = {name: timestamp.RuleAdjusted(s, final, walk_lut, job.timestamp_rule) for name, s in sets.items()}
    solutions, tables = {}, {}
    for name in solved(job):
        clock = time.perf_counter()
        if name in ruled:
            solution = ruled[name]
        elif name == "em":
            solution, _ = solve.em_estimator(pooled, setup, directory=None)
        else:
            solution, _ = solve.pair_estimator(pooled, setup)
        report["stages"][f"solve_{name}_seconds"] = time.perf_counter() - clock
        outputs.write_imas_lut(work / f"lut_{name}{suffix}.csv", solution, {**provenance, "estimator": name}, setup)
        solutions[name], tables[name] = solution, solve.applied_table(solution)
        report.setdefault("estimators", {})[name] = solve.summary(solution)
        progress(f"pooled {name}: {solution.diagnostics['status_counts']}")
    if job.legacy is not None:
        tables["legacy"] = solve.legacy_table(job.legacy)
    names = list(tables)
    report["comparisons"] = {f"{a} vs {b}": solve.compare(tables[a], tables[b])
                             for k, a in enumerate(names) for b in names[k + 1:]}
    clock = time.perf_counter()
    populated = {name: table for name, table in tables.items() if table}
    report["held_out"] = {name: solve.held_out({name: table}, pooled, setup, roles=(2,))
                          for name, table in populated.items()}
    if len(populated) > 1:
        report["held_out"]["common"] = solve.held_out(populated, pooled, setup, roles=(2,))
    report["stages"]["held_out_seconds"] = time.perf_counter() - clock
    report["per_run"] = {"peaks_511_kev": _peaks(calibrations)}
    if "em" in solutions:
        pooled_table = tables["em"]
        for run in job.runs:
            entry = {role: solve.held_out({"pooled em": pooled_table}, sets[run.name], setup, roles=(code,))
                     for role, code in (("train", 0), ("heldout", 2))}
            if job.per_run_solves:
                alone, _ = solve.em_estimator(sets[run.name], setup)
                entry["alone_versus_pooled"] = solve.compare(solve.applied_table(alone), pooled_table)
            report["per_run"][run.name] = entry
    if walk_lut is not None:
        report["walk"]["pooled_counts"] = (pooled.observations if job.timestamp_rule != "top" else pooled).counts()
    report["stages"]["total_seconds"] = time.perf_counter() - started
    report["peak_memory"] = peak_memory()
    (work / f"job_report{suffix}.json").write_text(json.dumps(report, indent=1, default=float) + "\n",
                                                   encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", type=Path, help="job file (schema imas-skew-job/1)")
    parser.add_argument("--work", type=Path, required=True, help="work directory (resumed when it is this job's)")
    parser.add_argument("--observe-only", action="store_true",
                        help="stop after the energy calibrations and observation sets (spec 012 round-trip step 0)")
    args = parser.parse_args(argv)
    try:
        report = run_job(load_job(args.job), args.work, observe_only=args.observe_only)
    except JobError as error:
        print(f"job refused: {error}")
        return 2
    keys = ("estimators", "comparisons", "held_out", "walk", "rule", "stages")
    print(json.dumps({k: report[k] for k in keys if k in report},
                     indent=1, default=float))
    suffix = '' if 'walk' not in report else '_walk-' + report['walk']['walk_lut_sha256'][:12]
    suffix += RULE_SUFFIX.get((report.get("rule") or {}).get("timestamp_rule"), "")
    name = "job_report_observe.json" if args.observe_only else f"job_report{suffix}.json"
    print(f"-> {args.work / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
