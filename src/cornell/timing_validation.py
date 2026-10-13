"""Frozen-LUT validation: training size, channel sufficiency and test files (spec 008 T9, FR-16).

Usage::

    python -u -m src.cornell.timing_validation VALIDATION.json

The validation file names a timing job file and the canonical LUT that job
wrote. The LUT is frozen: its manifest must carry the job's SHA-256 and the
fingerprints of the job's observation sets, which are reused as they are
(no LDAT reads); the pair kernel is the one in the manifest.

Owner decisions (2026-10-13):

* Training size: pooled train events are thinned per event by a seeded hash
  of (run, file, record), so a smaller fraction is a subset of every larger
  one; each fraction is fitted and solved like the job and compared with the
  full-fraction solve after removing the weighted gauge offset. For nested
  subsets ``var(d_f - d_1) = sigma_f^2 - sigma_1^2``. The full fraction must
  reproduce the LUT delays.
* No block bootstrap: the job's run-to-run reproducibility floor stands for
  resampling uncertainty.
* A channel is sufficient when its LUT status is calibrated (uncertainty with
  floor within the precision gate, applied component), its half-data delay
  agrees with the full one within ``stability_z`` nested standard deviations
  and it is on neither the cross-run nor the per-ASIC pattern outlier list.
* The test files are evaluated with pair residuals (the job's train/validation
  metric), per run, at most once per LUT: a marker naming the LUT is written
  in the job's work directory before the first test fit, and a later run
  with ``evaluate_test`` refuses. Independent CTR stays pending (T9b).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from src.cornell import timing_cache as tc
from src.cornell import timing_job as tj
from src.cornell import timing_pairs as tp
from src.cornell import timing_plots as tpl
from src.cornell import timing_solver as ts
from src.cornell.timing_ingest import LegacySetup, time_channel_classes
from src.cornell.timing_petsys import CanonicalLut

SCHEMA = "cornell-timing-validation/1"
DEFAULT_FRACTIONS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
_REQUIRED = {"schema", "job", "lut", "output", "evaluate_test"}
_OPTIONAL = {"fractions": list(DEFAULT_FRACTIONS), "seed": 0, "stability_z": 3.0,
             "pattern_outlier_ps": tpl.DEFAULT_THRESHOLD_PS, "prior_test_looks": []}
TEST = tc.ROLES.index("test")
TRAIN = tc.ROLES.index("train")


class ValidationError(ValueError):
    """The validation file, the LUT binding or an output is invalid."""


def _mix(x):
    """splitmix64 finaliser on uint64 arrays (wrapping arithmetic)."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def event_uniform(rows, run_index, seed=0):
    """Deterministic U[0, 1) per event from (run, file, record, seed); same event, same value."""
    with np.errstate(over="ignore"):
        key = (_mix(np.full(len(rows), np.uint64(run_index * 1_000_003 + seed)))
               ^ (rows["file"].astype(np.uint64) << np.uint64(40)) ^ rows["record"].astype(np.uint64))
        return (_mix(key) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


class Thinned:
    """Train rows of several observation sets kept when their event value is below ``fraction``."""

    def __init__(self, sets, fraction, seed=0):
        self.sets, self.fraction, self.seed = list(sets), float(fraction), seed
        counts = {s.partitions for s in self.sets}
        if len(counts) != 1:
            raise ValidationError("observation sets disagree on pair partitions")
        self.partitions = counts.pop()

    def partition(self, p):
        parts = []
        for k, observations in enumerate(self.sets):
            rows = observations.partition(p)
            rows = rows[rows["role"] == TRAIN]
            parts.append(rows[event_uniform(rows, k, self.seed) < self.fraction])
        return np.concatenate(parts) if parts else np.zeros(0, tc.OBSERVATION_DTYPE)


def _applied(solution):
    rows = solution.channels[solution.channels["apply"]]
    return {int(c): (float(d), float(s)) for c, d, s in zip(rows["time_channel"], rows["delay_ps"], rows["sigma_ps"])}


def nested_comparison(subset, full, *, stability_z=3.0):
    """Gauge-aligned delay differences of a nested subset solve against the full solve.

    Returns a summary and ``{channel: z}``; the nested standard deviation is
    sqrt(max(sigma_f^2 - sigma_1^2, (0.1 sigma_f)^2)).
    """
    a, b = _applied(subset), _applied(full)
    common = sorted(set(a) & set(b))
    if not common:
        return {"channels": 0}, {}
    diff = np.array([a[c][0] - b[c][0] for c in common])
    sd = np.array([math.sqrt(max(a[c][1] ** 2 - b[c][1] ** 2, (0.1 * a[c][1]) ** 2)) for c in common])
    weight = 1 / sd ** 2
    offset = float((weight * diff).sum() / weight.sum())
    diff -= offset
    z = diff / sd
    summary = {"channels": len(common), "gauge_offset_ps": offset, "diff_rms_ps": float(np.sqrt(np.mean(diff ** 2))),
               "diff_ps_quantiles_1_50_99": np.percentile(diff, [1, 50, 99]).tolist(),
               "predicted_rms_ps": float(np.sqrt(np.mean(sd ** 2))), "z_rms": float(np.sqrt(np.mean(z ** 2))),
               f"abs_z_gt_{stability_z:g}": int((np.abs(z) > stability_z).sum())}
    return summary, dict(zip(common, z))


SUFFICIENCY_REASONS = ("not calibrated", "not solved at half data", "unstable at half data", "cross-run outlier",
                       "ASIC pattern outlier")


def classify(lut_rows, half_z, cross_run, pattern, *, stability_z=3.0):
    """Per operational channel: (sufficient, reasons); hardware-unavailable rows are excluded.

    ``half_z``: {channel: z} of the half-data comparison; ``cross_run``/``pattern``: listed channel ids.
    """
    out = {}
    unavailable = ts.CHANNEL_STATUSES.index("hardware unavailable")
    for row in lut_rows:
        if row["status"] == unavailable:
            continue
        channel, reasons = int(row["time_channel"]), []
        if not (row["status"] == ts.CALIBRATED and row["apply"]):
            reasons.append("not calibrated")
        if channel not in half_z:
            reasons.append("not solved at half data")
        elif abs(half_z[channel]) > stability_z:
            reasons.append("unstable at half data")
        if channel in cross_run:
            reasons.append("cross-run outlier")
        if channel in pattern:
            reasons.append("ASIC pattern outlier")
        out[channel] = (not reasons, reasons)
    return out


def dataset_class(classes):
    sufficient = sum(ok for ok, _ in classes.values())
    if classes and sufficient == len(classes):
        return "sufficient"
    return "sufficient for a named subset" if sufficient else "insufficient"


def load_validation(path):
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise ValidationError(f"validation schema must be {SCHEMA!r}")
    missing, unknown = _REQUIRED - set(spec), set(spec) - _REQUIRED - set(_OPTIONAL)
    if missing or unknown:
        raise ValidationError(f"validation keys missing {sorted(missing)} / unknown {sorted(unknown)}")
    options = _OPTIONAL | {k: spec[k] for k in _OPTIONAL if k in spec}
    fractions = sorted(float(f) for f in options["fractions"])
    if not fractions or not all(0 < f <= 1 for f in fractions) or 0.5 not in fractions or fractions[-1] != 1.0 \
            or len(set(fractions)) != len(fractions):
        raise ValidationError("fractions must be distinct values in (0, 1] including 0.5 and 1")
    if not (options["stability_z"] > 0 and options["pattern_outlier_ps"] > 0) or not isinstance(spec["evaluate_test"], bool):
        raise ValidationError("invalid stability_z, pattern_outlier_ps or evaluate_test")
    base = path.resolve().parent

    def local(value):
        value = Path(value)
        return value if value.is_absolute() else base / value

    return {"path": path, "job": local(spec["job"]), "lut": local(spec["lut"]), "output": local(spec["output"]),
            "evaluate_test": spec["evaluate_test"], "fractions": fractions, "seed": int(options["seed"]),
            "stability_z": float(options["stability_z"]), "pattern_outlier_ps": float(options["pattern_outlier_ps"]),
            "prior_test_looks": list(options["prior_test_looks"])}


def observation_sets(job, lut):
    """The job's observation sets, checked against the fingerprints frozen in the LUT manifest."""
    provenance = lut.manifest["provenance"]
    if provenance.get("job_sha256") != job.sha256:
        raise ValidationError("LUT was not produced by this job file (job SHA-256 differs)")
    if provenance.get("runs") != [run.name for run in job.runs]:
        raise ValidationError("LUT runs differ from the job's runs")
    sets = {}
    for run in job.runs:
        encal = run.encal or job.encal
        tag = "" if run.encal is None else f"-{hashlib.sha256(encal.read_bytes()).hexdigest()[:12]}"
        observations = tc.ObservationSet(job.work / run.name / f"observations{tag}")
        digest = hashlib.sha256(json.dumps(observations.manifest["fingerprint"], sort_keys=True).encode()).hexdigest()
        if provenance["observation_fingerprints_sha256"].get(run.name) != digest:
            raise ValidationError(f"{run.name}: observation set differs from the one that produced the LUT")
        sets[run.name] = observations
    return sets


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=1, default=float) + "\n", encoding="utf-8")


def run_validation(spec, *, log=tj._print):
    output = spec["output"]
    channels_csv = output.with_name(output.stem + "_channels.csv")
    for path in (output, channels_csv):
        if path.exists():
            raise ValidationError(f"output exists: {path}; choose a new name")
    job = tj.load_job(spec["job"])
    lut = CanonicalLut.read(spec["lut"])
    frozen = ts.Solution(lut.rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {})
    lut_sha = lut.manifest["table"]["sha256"]
    marker = job.work / f"test-evaluated-{lut_sha[:12]}.json"
    if spec["evaluate_test"] and marker.exists():
        raise ValidationError(f"test files of this LUT were already evaluated ({marker}); they are opened once")
    sets = observation_sets(job, lut)
    kernel = lut.manifest["provenance"]["kernel"]
    model = tp.PairModel(kernel["sigma_ps"], kernel["background_fraction"], kernel["window_ps"])
    setup = LegacySetup.from_files(job.config, job.encal, root=tj.ROOT)
    _, operational, down = time_channel_classes(setup, job.down_regions)
    report = {"schema": "cornell-timing-validation-report/1", "validation": str(spec["path"]), "job": str(job.path),
              "job_sha256": job.sha256, "lut": str(spec["lut"]), "lut_sha256": lut_sha,
              "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "kernel": {k: kernel[k] for k in ("sigma_ps", "background_fraction", "window_ps")},
              "bootstrap": "not run (owner decision 2026-10-13: reproducibility floor stands for resampling)",
              "prior_test_looks": spec["prior_test_looks"], "stages": {}, "training_size": []}

    # training size: nested event fractions of pooled train, full fraction last
    solutions = {}
    for fraction in sorted(spec["fractions"], reverse=True):
        clock = time.perf_counter()
        view = Thinned(sets.values(), fraction, spec["seed"])
        fits = tp.fit_observation_set(view, model, roles=(TRAIN,))
        solution = ts.solve(fits, operational, down)
        solutions[fraction] = solution
        report["training_size"].append({
            "fraction": fraction, "train_events": int(fits["events"].sum()), "pairs": int(len(fits)),
            "usable_pairs": int((fits["status"] == tp.USABLE).sum()), "status_counts": solution.diagnostics["status_counts"],
            "seconds": time.perf_counter() - clock})
        log(f"fraction {fraction:g}: {report['training_size'][-1]['usable_pairs']} usable pairs")
    full = solutions[1.0]
    reference = lut.rows
    same = {int(c): d for c, d in zip(reference["time_channel"], reference["delay_ps"]) if np.isfinite(d)}
    resolved = {int(r["time_channel"]): float(r["delay_ps"]) for r in full.channels if np.isfinite(r["delay_ps"])}
    if set(same) != set(resolved) or max((abs(same[c] - resolved[c]) for c in same), default=0.0) > 1e-6:
        raise ValidationError("the full-fraction solve does not reproduce the LUT delays (different inputs or code)")
    # per-channel floor share: LUT uncertainty (with floor) beyond the full-fraction statistical one
    statistical = _applied(full)
    floor = {c: math.sqrt(max(s ** 2 - statistical[c][1] ** 2, 0.0)) for c, (_, s) in _applied(frozen).items()
             if c in statistical}
    gate = ts.SolveOptions().precision_gate_ps
    half_z = {}
    for entry in report["training_size"]:
        solution = solutions[entry["fraction"]]
        applied = solution.channels[solution.channels["apply"]]
        sigma = applied["sigma_ps"]
        with_floor = np.hypot(sigma, [floor.get(int(c), 0.0) for c in applied["time_channel"]])
        entry["applied"] = int(len(applied))
        entry["sigma_ps_quantiles_10_50_90"] = np.percentile(sigma, [10, 50, 90]).tolist() if len(sigma) else None
        entry["within_gate_with_floor"] = int((with_floor <= gate).sum())
        entry["vs_full"], z = nested_comparison(solution, full, stability_z=spec["stability_z"])
        if entry["fraction"] == 0.5:
            half_z = z

    # sufficiency
    cross_run = {item["time_channel"] for item in lut.manifest["provenance"].get("cross_run_outliers") or []}
    listed = lut.manifest["provenance"].get("asic_pattern_outliers")
    applied = lut.rows[lut.rows["apply"]]
    if listed is None or listed["threshold_ps"] != spec["pattern_outlier_ps"]:
        pattern_list = tpl.pattern_outliers(tpl.asic_pattern(applied["time_channel"], applied["delay_ps"]),
                                            spec["pattern_outlier_ps"])
        source = "computed here from the LUT (manifest has none at this threshold)"
    else:
        pattern_list, source = listed["channels"], "LUT manifest"
    pattern = {item["time_channel"] for item in pattern_list}
    classes = classify(lut.rows, half_z, cross_run, pattern, stability_z=spec["stability_z"])
    reasons = {name: sum(name in r for _, r in classes.values()) for name in SUFFICIENCY_REASONS}
    report["sufficiency"] = {
        "rule": f"calibrated (sigma with floor <= {gate:g} ps, applied) and |half-data z| <= {spec['stability_z']:g} "
                "and on no cross-run or per-ASIC pattern outlier list",
        "operational_channels": len(classes), "hardware_unavailable": int(len(lut.rows) - len(classes)),
        "sufficient": int(sum(ok for ok, _ in classes.values())), "reasons": reasons,
        "dataset": dataset_class(classes), "pattern_outliers_source": source,
        "pattern_outlier_threshold_ps": spec["pattern_outlier_ps"], "channels_csv": str(channels_csv)}

    # test files: once per LUT, marker first
    if spec["evaluate_test"]:
        _write_json(marker, {"lut": str(spec["lut"]), "lut_sha256": lut_sha, "validation": str(spec["path"]),
                             "opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        clock = time.perf_counter()
        delays = _applied(frozen)
        report["test"] = {"marker": str(marker), "runs": {}}
        every = []
        for name, observations in sets.items():
            fits = tp.fit_observation_set(observations, model, roles=(TEST,))
            every.append(fits)
            result = tj.pair_residuals(fits, frozen)
            usable = fits[fits["status"] == tp.USABLE]
            touched = {int(c) for pair in zip(usable["channel_i"], usable["channel_j"])
                       if int(pair[0]) in delays and int(pair[1]) in delays for c in pair}
            counts = observations.pair_counts()
            result.update({"test_events": int(counts["count"][counts["role"] == TEST].sum()), "pairs": int(len(fits)),
                           "usable_pairs": int(len(usable)), "applied_channels_covered": len(touched),
                           "applied_channels": len(delays)})
            report["test"]["runs"][name] = result
        report["test"]["all_runs"] = tj.pair_residuals(np.concatenate(every), frozen)
        report["stages"]["test_seconds"] = time.perf_counter() - clock
    else:
        report["test"] = {"status": "not evaluated in this validation run"}

    with open(channels_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("time_channel", "lut_status", "sufficient", "half_data_z", "reasons"))
        status = {int(r["time_channel"]): ts.CHANNEL_STATUSES[r["status"]] for r in lut.rows}
        for channel, (ok, why) in sorted(classes.items()):
            z = half_z.get(channel)
            writer.writerow((channel, status[channel], int(ok), "" if z is None else f"{z:.3f}", ";".join(why)))
    report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["peak_memory"] = tj.peak_memory()
    _write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen-LUT validation of a Cornell timing job (spec 008 T9).")
    parser.add_argument("validation", type=Path, help="validation JSON file")
    args = parser.parse_args(argv)
    try:
        report = run_validation(load_validation(args.validation))
    except (ValidationError, tj.JobError, tc.CacheMismatch) as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"sufficiency": report["sufficiency"], "test": report["test"],
                      "training_size": [{k: e[k] for k in ("fraction", "train_events", "applied", "within_gate_with_floor",
                                                           "vs_full")} for e in report["training_size"]]},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
