"""Pooled multi-run Cornell TIME-channel calibration job (spec 008 T7a, FR-18).

Usage::

    python -u -m src.cornell.timing_job JOB.json

The job file lists every input explicitly (nothing is globbed or parsed from
basenames): runs with their conversion directory, ``run.json`` ledger and
split prefix; config; the reused energy calibration; acquisition-specific
down regions; file-index roles; work directory and LUT path. Relative paths
resolve against the job file's directory, except ``config``, which (like its
``map_file``) is relative to the repository root.

Per run: inventory (``timing_dataset``), frozen file roles, feature cache and
selected-geometry observation set, each reused when complete and matching.
Pooled (owner decision 2026-10-13): train events of all runs are fitted
together per channel pair with one global kernel, then one global solve and
canonical LUT. A run may name its own ``.encal`` (owner decision 2026-10-13
for a run whose 511 keV peak moved); its held-out prefix can be read from the
Manager calibrate run that produced it. Per run, the report adds 511 keV peak
positions per SM against the reference run (a reused ``.encal`` is checked,
never silently trusted) and train/validation pair residuals against the
pooled delays; with ``per_run_solves`` each run is also solved alone and its
delays compared with the pooled and reference-run solutions, so a
rate-dependent offset stays visible. Test files are never evaluated here. A failing run stops the job; the pool is never silently shrunk.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from src.cornell import timing_cache as tc
from src.cornell import timing_dataset as td
from src.cornell import timing_pairs as tp
from src.cornell import timing_solver as ts
from src.cornell.timing_ingest import GeometryHypothesis, LegacySetup, time_channel_classes
from src.cornell.timing_petsys import write_lut
from src.mapping_generator import map_factory

ROOT = Path(__file__).resolve().parents[2]
JOB_SCHEMA = "cornell-timing-job/1"
# T5 run7: 13.18 GB of feature blocks from 222.8 GB of compact LDAT.
FEATURE_BYTES_PER_LDAT_BYTE = 0.0592
ENERGY_WINDOW_KEV = (357.0, 665.0)
_REQUIRED = {"schema", "config", "encal", "reference_run", "down_regions", "runs", "roles", "geometry",
             "work_directory", "lut"}
_OPTIONAL = {"workers": 1, "seed": 0, "guard_ps": 1_000_000, "partitions": 16, "energy_peak_tolerance": 0.02,
             "disk_margin": 1.25, "min_peak_entries": 200, "per_run_solves": False, "reproducibility_floor": False,
             "outlier_threshold": 5.0}
_RUN_KEYS = {"name", "directory", "ledger", "prefix", "energy_calibration_prefix"}
_RUN_OPTIONAL = {"encal"}


class JobError(ValueError):
    """The job file is invalid, an input fails qualification or a stage cannot run."""


@dataclass(frozen=True)
class RunSpec:
    name: str
    directory: Path
    ledger: Path
    prefix: str
    energy_calibration_prefix: dict | None
    calibration_run: Path | None = None
    encal: Path | None = None


@dataclass(frozen=True)
class Job:
    path: Path
    sha256: str
    config: Path
    encal: Path
    reference_run: str
    down_regions: tuple
    runs: tuple
    validation: tuple
    test: tuple
    work: Path
    lut: Path
    workers: int
    seed: int
    guard_ps: int
    partitions: int
    energy_peak_tolerance: float
    disk_margin: float
    min_peak_entries: int
    per_run_solves: bool
    reproducibility_floor: bool
    outlier_threshold: float


def load_job(path):
    """Validated job; unknown or missing keys are errors, never defaults for inputs."""
    path = Path(path)
    raw = path.read_bytes()
    spec = json.loads(raw)
    if not isinstance(spec, dict) or spec.get("schema") != JOB_SCHEMA:
        raise JobError(f"job schema must be {JOB_SCHEMA!r}")
    missing, unknown = _REQUIRED - set(spec), set(spec) - _REQUIRED - set(_OPTIONAL)
    if missing or unknown:
        raise JobError(f"job keys missing {sorted(missing)} / unknown {sorted(unknown)}")
    if spec["geometry"] != "selected":
        raise JobError("geometry must be 'selected' (faceted, 7 mm, owner/data-selected signs)")
    base = path.resolve().parent

    def local(value):
        value = Path(value)
        return value if value.is_absolute() else base / value

    runs = []
    for item in spec["runs"]:
        if not isinstance(item, dict) or not _RUN_KEYS <= set(item) <= _RUN_KEYS | _RUN_OPTIONAL:
            raise JobError(f"run entries need {sorted(_RUN_KEYS)} (optional {sorted(_RUN_OPTIONAL)}): {item!r}")
        # "none": no record of this run fed its .encal; {split: records} excluded prefixes; or
        # {"calibration_run": run.json} of the Manager calibrate run that produced this run's .encal
        value = item["energy_calibration_prefix"]
        prefix, calibration_run = None, None
        if value == "none":
            pass
        elif isinstance(value, dict) and set(value) == {"calibration_run"}:
            calibration_run = local(value["calibration_run"])
        elif isinstance(value, dict) and all(str(k).isdigit() and int(v) >= 0 for k, v in value.items()):
            prefix = {int(k): int(v) for k, v in value.items()}
        else:
            raise JobError(f"{item['name']}: energy_calibration_prefix must be 'none', {{split: records}} "
                           "or {\"calibration_run\": path}")
        runs.append(RunSpec(str(item["name"]), local(item["directory"]), local(item["ledger"]), str(item["prefix"]),
                            prefix, calibration_run, local(item["encal"]) if "encal" in item else None))
    names = [run.name for run in runs]
    if not runs or len(set(names)) != len(names) or any(not n or os.sep in n or "/" in n for n in names):
        raise JobError("runs need unique simple names")
    if spec["reference_run"] not in names:
        raise JobError("reference_run must be one of the runs")
    options = _OPTIONAL | {k: spec[k] for k in _OPTIONAL if k in spec}
    if options["reproducibility_floor"] and (not options["per_run_solves"] or len(runs) < 2):
        raise JobError("reproducibility_floor needs per_run_solves and at least two runs")
    roles = spec["roles"]
    if set(roles) != {"validation", "test"}:
        raise JobError("roles must list 'validation' and 'test' file indices")
    if not (int(options["workers"]) >= 1 and int(options["partitions"]) >= 1 and 0 < options["energy_peak_tolerance"] < 1
            and options["disk_margin"] >= 1 and int(options["guard_ps"]) >= 0):
        raise JobError("invalid workers/partitions/tolerance/disk margin/guard")
    regions = tuple(tuple(int(v) for v in region) for region in spec["down_regions"])
    if any(len(region) != 2 for region in regions):
        raise JobError("down regions are [SM, MM] pairs")
    config = Path(spec["config"])
    return Job(path, hashlib.sha256(raw).hexdigest(), config if config.is_absolute() else ROOT / config,
               local(spec["encal"]), spec["reference_run"], regions, tuple(runs),
               tuple(int(v) for v in roles["validation"]), tuple(int(v) for v in roles["test"]),
               local(spec["work_directory"]), local(spec["lut"]), int(options["workers"]), int(options["seed"]),
               int(options["guard_ps"]), int(options["partitions"]), float(options["energy_peak_tolerance"]),
               float(options["disk_margin"]), int(options["min_peak_entries"]), bool(options["per_run_solves"]),
               bool(options["reproducibility_floor"]), float(options["outlier_threshold"]))


def peak_memory():
    """Peak resident memory in bytes for this process (and finished children where the OS reports it)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return {"parent_bytes": int(counters.PeakWorkingSetSize), "largest_child_bytes": None}
    import resource
    scale = 1 if sys.platform == "darwin" else 1024
    return {"parent_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale,
            "largest_child_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale}


def _tree_bytes(directory):
    directory = Path(directory)
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) if directory.exists() else 0


def _print(message):
    print(message, flush=True)


def _write_json(path, value):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _scan(path, lookup):
    return td.scan_file(path, lookup)


def inventory(run, lookup, directory, workers, log=_print):
    """Full raw inventory of a run's ledger outputs, reused while identities match."""
    inputs = td.ledger_inputs(run.ledger, run.directory, run.prefix)
    target = Path(directory) / "inventory.json"
    identities = [{"name": i["path"].name, "size_bytes": i["path"].stat().st_size,
                   "mtime_ns": i["path"].stat().st_mtime_ns} for i in inputs]
    if target.exists():
        stored = json.loads(target.read_text(encoding="utf-8"))
        if stored["identities"] == identities:
            return inputs, stored["scans"]
        raise JobError(f"{run.name}: inputs changed since {target}; use a new work directory")
    paths = [item["path"] for item in inputs]
    if workers > 1 and len(paths) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            scans = list(pool.map(_scan, paths, [lookup] * len(paths)))
    else:
        scans = [_scan(p, lookup) for p in paths]
    for item, scan in zip(inputs, scans):
        scan["index"] = item["index"]
        log(f"{run.name} split {item['index']:02d}: {scan['physical_coincidences']:,} records")
    Path(directory).mkdir(parents=True, exist_ok=True)
    _write_json(target, {"run": run.name, "identities": identities, "scans": scans})
    return inputs, scans


class PooledObservations:
    """Observation sets of several runs seen as one, partition by partition (same pair hashing)."""

    def __init__(self, sets):
        sets = list(sets)
        counts = {s.partitions for s in sets}
        if len(counts) != 1:
            raise JobError("observation sets disagree on pair partitions")
        self.sets, self.partitions = list(sets), counts.pop()

    def partition(self, p):
        return np.concatenate([s.partition(p) for s in self.sets])


def kernel_rows(observations, *, role=0, min_events=30, max_pairs=2000):
    """Rows of up to ``max_pairs`` pairs with >= ``min_events`` events, read one partition at a time."""
    chosen, total = [], 0
    for p in range(observations.partitions):
        rows = observations.partition(p)
        rows = rows[rows["role"] == role]
        key = tp._pair_keys(rows)
        values, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
        keep = np.flatnonzero(counts >= min_events)[:max_pairs - total]
        chosen.append(rows[np.isin(inverse, keep)])
        total += len(keep)
        if total >= max_pairs:
            break
    return np.concatenate(chosen) if chosen else np.zeros(0, tc.OBSERVATION_DTYPE)


def energy_peaks(observations, *, bins_kev=2.0, smooth=5, min_entries=200):
    """511 keV peak per SM from accepted endpoints: mode of a smoothed histogram in the energy window."""
    edges = np.arange(ENERGY_WINDOW_KEV[0], ENERGY_WINDOW_KEV[1] + bins_kev, bins_kev)
    histograms = {}
    for p in range(observations.partitions):
        rows = observations.partition(p)
        for sm_name, energy_name in (("sm_i", "energy_kev_i"), ("sm_j", "energy_kev_j")):
            for sm in np.unique(rows[sm_name]):
                values = rows[energy_name][rows[sm_name] == sm]
                histograms.setdefault(int(sm), np.zeros(len(edges) - 1))
                histograms[int(sm)] += np.histogram(values, edges)[0]
    peaks = {}
    for sm, counts in sorted(histograms.items()):
        if counts.sum() < min_entries:
            peaks[sm] = {"peak_kev": None, "entries": int(counts.sum())}
            continue
        smoothed = np.convolve(counts, np.ones(smooth) / smooth, mode="same")
        k = int(np.argmax(smoothed))
        peaks[sm] = {"peak_kev": float(edges[k] + bins_kev / 2), "entries": int(counts.sum())}
    return peaks


def peak_flags(peaks, reference, tolerance):
    """SMs whose peak moved by more than ``tolerance`` (fraction) against the reference run."""
    flags = []
    for sm, value in peaks.items():
        ref = reference.get(sm, {}).get("peak_kev")
        if value["peak_kev"] is None or ref is None:
            continue
        shift = value["peak_kev"] / ref - 1
        if abs(shift) > tolerance:
            flags.append({"sm": sm, "peak_kev": value["peak_kev"], "reference_kev": ref, "relative_shift": shift})
    return flags


def pair_residuals(fits, solution):
    """Usable pair shifts against pooled delays (both channels applied)."""
    delays = {int(r["time_channel"]): (float(r["delay_ps"]), float(r["sigma_ps"]))
              for r in solution.channels if r["apply"]}
    ok = (fits["status"] == tp.USABLE) & np.array([int(a) in delays and int(b) in delays
                                                   for a, b in zip(fits["channel_i"], fits["channel_j"])], bool)
    if not ok.any():
        return {"compared": 0}
    f = fits[ok]
    predicted = np.array([delays[int(a)][0] - delays[int(b)][0] for a, b in zip(f["channel_i"], f["channel_j"])])
    sigma = np.array([math.hypot(delays[int(a)][1], delays[int(b)][1]) for a, b in zip(f["channel_i"], f["channel_j"])])
    residual = f["shift_ps"] - predicted
    z = residual / np.hypot(f["shift_sigma_ps"], sigma)
    return {"compared": int(ok.sum()), "residual_ps_mean": float(residual.mean()),
            "residual_ps_median": float(np.median(residual)), "z_rms": float(np.sqrt(np.mean(z ** 2))),
            "abs_z_gt_5": int((np.abs(z) > 5).sum())}


def calibration_prefix(calibration_run, inputs, encal):
    """{split index: records_validated} from the Manager calibrate run that produced ``encal``.

    Every input split must appear once in the run's summary; the run's ``encal``
    artifact must have the same name and size as the ``.encal`` in use.
    """
    ledger = json.loads(Path(calibration_run).read_text(encoding="utf-8"))
    attempts = [a for a in ledger.get("attempts", []) if (a.get("details") or {}).get("cli_status") == "succeeded"]
    if not attempts:
        raise JobError(f"{calibration_run}: no succeeded calibrate attempt")
    attempt = attempts[-1]
    artifacts = [a for a in attempt.get("artifacts", []) if a.get("kind") == "encal"]
    if (len(artifacts) != 1 or artifacts[0]["path"].replace("\\", "/").rsplit("/", 1)[-1] != Path(encal).name
            or artifacts[0].get("size_bytes") != Path(encal).stat().st_size):
        raise JobError(f"{calibration_run}: its .encal artifact does not match {encal}")
    used = {}
    for entry in attempt["details"]["summary"]["inputs"]:
        name = entry["path"].replace("\\", "/").rsplit("/", 1)[-1]
        if name in used:
            raise JobError(f"{calibration_run}: {name} listed twice")
        used[name] = int(entry["records_validated"])
    missing = [item["path"].name for item in inputs if item["path"].name not in used]
    if missing:
        raise JobError(f"{calibration_run}: no records_validated for {missing[:3]}")
    return {item["index"]: used[item["path"].name] for item in inputs}


def compare_delays(a, b):
    """Applied delays of two solutions after removing the free relative gauge (weighted mean difference)."""
    da = {int(r["time_channel"]): (r["delay_ps"], r["sigma_ps"]) for r in a.channels if r["apply"]}
    db = {int(r["time_channel"]): (r["delay_ps"], r["sigma_ps"]) for r in b.channels if r["apply"]}
    common = sorted(set(da) & set(db))
    if not common:
        return {"channels": 0}
    diff = np.array([da[c][0] - db[c][0] for c in common])
    sigma = np.array([math.hypot(da[c][1], db[c][1]) for c in common])
    weight = 1 / sigma ** 2
    offset = float((weight * diff).sum() / weight.sum())
    diff -= offset
    z = diff / sigma
    return {"channels": len(common), "gauge_offset_ps": offset, "diff_rms_ps": float(np.sqrt(np.mean(diff ** 2))),
            "diff_ps_quantiles_1_50_99": np.percentile(diff, [1, 50, 99]).tolist(),
            "z_rms": float(np.sqrt(np.mean(z ** 2))), "abs_z_gt_5": int((np.abs(z) > 5).sum())}


def _disk_preflight(job, run, inputs, cache_dir):
    done = tc.FeatureCache(cache_dir).files if (Path(cache_dir) / "manifest.json").exists() else {}
    pending = sum(item["path"].stat().st_size for item in inputs if item["index"] not in done)
    need = pending * FEATURE_BYTES_PER_LDAT_BYTE * job.disk_margin
    free = shutil.disk_usage(job.work).free
    if need > free:
        raise JobError(f"{run.name}: feature cache needs ~{need / 1e9:.1f} GB, {free / 1e9:.1f} GB free in {job.work}")
    return {"estimated_bytes": int(need), "free_bytes": int(free)}


def _prepare_run(job, run, info, lookup, hypothesis, log):
    """Inventory, frozen roles, feature cache and observation set of one run."""
    run_dir = job.work / run.name
    clock = time.perf_counter()
    inputs, scans = inventory(run, lookup, run_dir, job.workers, log)
    info["stages"]["inventory_seconds"] = time.perf_counter() - clock
    holdout = td.freeze_holdout(scans, validation=job.validation, test=job.test, guard_ps=job.guard_ps)
    if not holdout["qualified"]:
        raise JobError(f"{run.name}: holdout not qualified: {holdout['reasons']}")
    info["files"] = len(inputs)
    info["ldat_bytes"] = sum(item["path"].stat().st_size for item in inputs)
    info["physical_coincidences"] = sum(s["physical_coincidences"] for s in scans)
    encal = run.encal or job.encal
    digest = hashlib.sha256(encal.read_bytes()).hexdigest()
    # a run-specific .encal gets its own cache directories; the shared one keeps the plain names
    tag = "" if run.encal is None else f"-{digest[:12]}"
    info["encal"], info["encal_sha256"] = str(encal), digest
    info["disk"] = _disk_preflight(job, run, inputs, run_dir / f"features{tag}")
    if run.calibration_run is not None:
        prefix = calibration_prefix(run.calibration_run, inputs, encal)
        prefix = {index: prefix[index] for index in job.validation + job.test}
    elif run.energy_calibration_prefix is not None:
        prefix = run.energy_calibration_prefix
    else:
        prefix = {index: 0 for index in job.validation + job.test}
    info["energy_calibration_prefix"] = {str(k): v for k, v in sorted(prefix.items())}
    clock = time.perf_counter()
    features = tc.build_feature_cache(run_dir / f"features{tag}", inputs, config_path=job.config,
                                      encal_path=encal, root=ROOT, holdout=holdout,
                                      energy_calibration_prefix=prefix,
                                      seed=job.seed, workers=job.workers)
    info["stages"]["feature_cache_seconds"] = time.perf_counter() - clock
    features.require_complete()
    clock = time.perf_counter()
    observations = tc.build_observations(features, hypothesis, run_dir / f"observations{tag}",
                                         partitions=job.partitions)
    info["stages"]["observations_seconds"] = time.perf_counter() - clock
    info["first_failure"] = observations.manifest["first_failure"]
    info["geometry"] = observations.manifest["geometry"]
    info["cache_bytes"] = _tree_bytes(run_dir)
    log(f"{run.name}: cache ready ({info['cache_bytes'] / 1e9:.2f} GB)")
    return observations


def run_job(job, *, log=_print):
    """Run every stage; returns the report (also written to ``work/job-report.json``)."""
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    job.work.mkdir(parents=True, exist_ok=True)
    report = {"schema": "cornell-timing-job-report/1", "job": str(job.path), "job_sha256": job.sha256,
              "started_utc": started_utc, "status": "running", "runs": {}, "stages": {}}
    target = job.work / "job-report.json"
    try:
        if job.lut.exists() or job.lut.with_suffix(".json").exists():
            raise JobError(f"output LUT exists: {job.lut}; choose a new name")
        for path in (job.config, job.encal):
            if not path.is_file():
                raise JobError(f"missing input: {path}")
        setup = LegacySetup.from_files(job.config, job.encal, root=ROOT)
        _, operational, down = time_channel_classes(setup, job.down_regions)
        _, modules, _, _ = map_factory(str(ROOT / setup.provenance["map"]))
        lookup = td.channel_lookup(modules, job.down_regions)
        hypothesis = GeometryHypothesis.selected(job.config, setup)
        sets = {}
        for run in job.runs:
            if not run.ledger.is_file() or not run.directory.is_dir():
                raise JobError(f"{run.name}: missing ledger or directory")
            for path in (run.encal, run.calibration_run):
                if path is not None and not path.is_file():
                    raise JobError(f"{run.name}: missing input: {path}")
            report["runs"][run.name] = {"directory": str(run.directory), "stages": {}}
            try:
                sets[run.name] = _prepare_run(job, run, report["runs"][run.name], lookup, hypothesis, log)
            except (ValueError, OSError) as error:
                if isinstance(error, JobError):
                    raise
                raise JobError(f"{run.name}: {type(error).__name__}: {error}") from error

        pooled = PooledObservations(sets[run.name] for run in job.runs)
        clock = time.perf_counter()
        try:
            model, history = tp.estimate_model(kernel_rows(pooled))
        except ValueError as error:
            raise JobError(f"kernel estimation: {error}") from error
        report["stages"]["kernel_seconds"] = time.perf_counter() - clock
        report["kernel"] = {"sigma_ps": model.sigma_ps, "background_fraction": model.background_fraction,
                            "window_ps": model.window_ps, "history": history}
        clock = time.perf_counter()
        fits = tp.fit_observation_set(pooled, model, roles=(0,))
        report["stages"]["pooled_pair_fit_seconds"] = time.perf_counter() - clock
        clock = time.perf_counter()
        solution = ts.solve(fits, operational, down)
        report["stages"]["solve_seconds"] = time.perf_counter() - clock
        report["pooled"] = {"pairs": int(len(fits)), "usable_pairs": int((fits["status"] == tp.USABLE).sum()),
                            "pair_status": {n: int((fits["status"] == k).sum()) for k, n in enumerate(tp.STATUSES)},
                            "solve": solution.diagnostics}

        peaks = {run.name: energy_peaks(sets[run.name], min_entries=job.min_peak_entries) for run in job.runs}
        for run in job.runs:
            info = report["runs"][run.name]
            info["energy_peaks"] = peaks[run.name]
            info["energy_peak_flags"] = peak_flags(peaks[run.name], peaks[job.reference_run],
                                                   job.energy_peak_tolerance)
            # test files stay closed here: they are evaluated once, explicitly, in T9
            info["residuals_vs_pooled"] = {
                role: pair_residuals(tp.fit_observation_set(sets[run.name], model, roles=(tc.ROLES.index(role),)),
                                     solution)
                for role in ("train", "validation")}
        if job.per_run_solves:
            # rate-dependence diagnostic: each run solved alone with the pooled kernel, never applied
            clock = time.perf_counter()
            alone = {}
            for run in job.runs:
                alone[run.name] = ts.solve(tp.fit_observation_set(sets[run.name], model, roles=(0,)), operational, down)
                target_lut = job.work / run.name / "per_run_delays.csv"
                for old in (target_lut, target_lut.with_suffix(".json")):
                    old.unlink(missing_ok=True)
                write_lut(target_lut, alone[run.name], {"diagnostic": "per-run solve; not for correction",
                                                        "run": run.name, "kernel": report["kernel"]})
            report["per_run_solves"] = {
                name: {"status_counts": sol.diagnostics["status_counts"],
                       "birge_ratio": (sol.diagnostics["components"] or [{}])[0].get("birge_ratio"),
                       "vs_pooled": compare_delays(sol, solution),
                       "vs_reference_run": None if name == job.reference_run else compare_delays(sol, alone[job.reference_run])}
                for name, sol in alone.items()}
            floors = {}
            if job.reproducibility_floor:
                # owner decision 2026-10-13: run-to-run spread enters the LUT uncertainty in quadrature
                try:
                    floors, pairs = ts.reproducibility_floor(alone)
                except ValueError as error:
                    raise JobError(f"reproducibility floor: {error}") from error
                solution = ts.apply_reproducibility_floor(solution, alone, floors,
                                                          precision_gate_ps=ts.SolveOptions().precision_gate_ps)
                report["reproducibility_floor"] = {"per_run_ps": floors, "pairs": pairs,
                                                   "status_counts": solution.diagnostics["status_counts"],
                                                   "pooled_floor_ps_quantiles_10_50_90":
                                                       solution.diagnostics["reproducibility_floor"][
                                                           "pooled_floor_ps_quantiles_10_50_90"]}
            # owner decision 2026-10-13: listed for inspection, still applied
            report["cross_run_outliers"] = ts.cross_run_outliers(solution, alone, floors,
                                                                 threshold=job.outlier_threshold)
            report["stages"]["per_run_solve_seconds"] = time.perf_counter() - clock
        provenance = {"job": str(job.path), "job_sha256": job.sha256, "runs": [r.name for r in job.runs],
                      "reference_run": job.reference_run, "kernel": report["kernel"],
                      "geometry": hypothesis.name,
                      "encal_sha256": {name: info["encal_sha256"] for name, info in report["runs"].items()},
                      "observation_fingerprints_sha256": {
                          name: hashlib.sha256(json.dumps(s.manifest["fingerprint"], sort_keys=True).encode()).hexdigest()
                          for name, s in sets.items()},
                      "cross_run_outliers": report.get("cross_run_outliers"),
                      "status": "pooled train solve; native qualification (T9a) pending"}
        job.lut.parent.mkdir(parents=True, exist_ok=True)
        manifest = write_lut(job.lut, solution, provenance)
        report["lut"] = {"path": str(job.lut), "sha256": manifest["table"]["sha256"], "rows": manifest["table"]["rows"]}
        report["status"] = "succeeded"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        report["peak_memory"] = peak_memory()
        report["work_bytes"] = _tree_bytes(job.work)
        _write_json(target, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pooled multi-run Cornell TIME-channel calibration (spec 008).")
    parser.add_argument("job", type=Path, help="job JSON file")
    args = parser.parse_args(argv)
    try:
        report = run_job(load_job(args.job))
    except JobError as error:
        print(f"job failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({k: report[k] for k in ("status", "lut", "stages")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
