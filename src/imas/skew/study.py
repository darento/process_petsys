"""Frame-sign and opening-angle studies with the global EM (spec 011 T11, FR-2, FR-3).

``python -m src.imas.skew study signs FEATURES --work DIR`` builds one
observation set per frame-sign hypothesis ``(s_t, s_a)`` from a feature cache
(no LDAT reads). It runs the EM with a free global branch-separation scale on
the events accepted under every hypothesis: same events, same ``dt``, only the
geometry differs. It reports the log-likelihood, scale, ``sigma`` and
background of each hypothesis. The highest likelihood is the selected pair.
The evidence is its margin per axis and how far each other hypothesis' delays
differ from the selected ones. A sign whose margin stays small changes the
TOF by only a few ps, mostly absorbed by the channel delays, so it is reported
as weakly identified and immaterial to the LUT rather than as selected.

``python -m src.imas.skew study angles FEATURES --work DIR`` builds the
nominal-frame sets with opening angle >= 165 (production), >= 150 and >= 135
degrees and runs the EM on the train files for each. Each delay set is
compared with the 165-degree delays and checked on the 165-degree held-out
events (``solve.held_out``). A change of the production cut is a spec change
(FR-3), not made here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from src.imas.skew import cache as ic, solve
from src.imas.skew.geometry import (NOMINAL_AXIAL_SIGN, NOMINAL_TRANSVERSE_SIGN, OPPOSITION_TOLERANCE_DEG,
                                    ImasGeometry)
from src.imas.skew.setup import ImasSetup
from src.timing.cache import FeatureCache
from src.timing.report import peak_memory

SIGNS = ((1, -1), (1, 1), (-1, -1), (-1, 1))
SIGN_OPTIONS = replace(solve.EM_OPTIONS, fit_separation_scale=True)
ANGLE_TOLERANCES_DEG = (OPPOSITION_TOLERANCE_DEG, 30.0, 45.0)


def event_key(rows):
    return (rows["file"].astype(np.int64) << 40) | rows["record"].astype(np.int64)


def event_keys(observations, *, files=None, roles=(0,)):
    keys = []
    for p in range(observations.partitions):
        rows = observations.partition(p)
        mask = np.isin(rows["role"], roles) & (True if files is None else np.isin(rows["file"], files))
        keys.append(event_key(rows[mask]))
    return np.unique(np.concatenate(keys)) if keys else np.empty(0, np.int64)


def sign_study(sets, setup, *, files=None, options=SIGN_OPTIONS, progress=None):
    """``sets``: {(s_t, s_a): ObservationSet}; EM on the events every hypothesis accepts."""
    common = None
    for observations in sets.values():
        keys = event_keys(observations, files=files)
        common = keys if common is None else np.intersect1d(common, keys, assume_unique=True)
    results, tables = {}, {}
    for signs, observations in sets.items():
        started = time.perf_counter()
        if progress is not None:
            progress(f"signs s_t {signs[0]:+d} s_a {signs[1]:+d}: EM on {len(common):,} common events")
        solution, _ = solve.em_estimator(observations, setup, files=files,
                                         where=lambda rows: np.isin(event_key(rows), common), options=options,
                                         progress=progress)
        diagnostics = solution.diagnostics
        if diagnostics["observations"] != len(common):
            raise ValueError("hypotheses must be compared on the same events")
        results[signs] = {"log_likelihood": diagnostics["log_likelihood"],
                          **{k: diagnostics["model"][k] for k in ("separation_scale", "sigma_ps",
                                                                   "background_fraction", "branch_fraction")},
                          "applied": int(solution.channels["apply"].sum()), "converged": diagnostics["converged"],
                          "seconds": time.perf_counter() - started}
        tables[signs] = solve.applied_table(solution)
    ranked = sorted(results, key=lambda s: -results[s]["log_likelihood"])
    best = ranked[0]
    for signs in ranked[1:]:
        results[signs]["delays_versus_selected"] = solve.compare(tables[signs], tables[best])

    def margin(axis):
        other = [s for s in ranked if s[axis] != best[axis]]
        return results[best]["log_likelihood"] - results[other[0]]["log_likelihood"] if other else None

    return {"common_events": int(len(common)),
            "selected": {"transverse_sign": best[0], "axial_sign": best[1]},
            "margin_log_likelihood": (results[best]["log_likelihood"] - results[ranked[1]]["log_likelihood"]
                                      if len(ranked) > 1 else None),
            "transverse_margin_log_likelihood": margin(0), "axial_margin_log_likelihood": margin(1),
            "hypotheses": [{"transverse_sign": s[0], "axial_sign": s[1], **results[s]} for s in ranked]}


def angle_study(sets, setup, *, train_files, heldout_files, reference=OPPOSITION_TOLERANCE_DEG,
                options=solve.EM_OPTIONS, progress=None):
    """``sets``: {opposition tolerance (deg): ObservationSet}; EM per cut, compared with ``reference``."""
    name = {tolerance: f">= {180 - tolerance:g} deg" for tolerance in sets}
    solutions, results = {}, {}
    for tolerance, observations in sets.items():
        started = time.perf_counter()
        if progress is not None:
            progress(f"opening {name[tolerance]}: EM on train files {list(train_files)}")
        solutions[tolerance], _ = solve.em_estimator(observations, setup, files=train_files, options=options,
                                                     progress=progress)
        diagnostics = solutions[tolerance].diagnostics
        results[name[tolerance]] = {"seconds": time.perf_counter() - started, **solve.summary(solutions[tolerance]),
                                    "observations": diagnostics["observations"], "converged": diagnostics["converged"]}
    reference_table = solve.applied_table(solutions[reference])
    for tolerance, solution in solutions.items():
        if tolerance != reference:
            results[name[tolerance]]["versus_reference"] = solve.compare(solve.applied_table(solution),
                                                                         reference_table)
    if progress is not None:
        progress(f"held-out check on the {name[reference]} events of files {list(heldout_files)}")
    held = solve.held_out({name[t]: solve.applied_table(s) for t, s in solutions.items()}, sets[reference], setup,
                          heldout_files)
    return {"reference": name[reference], "cuts": results, "held_out_on_reference_events": held}


def _sets(features, setup, work, hypotheses, progress=print):
    cache = FeatureCache(features)
    sets = {}
    for key, (geometry, folder) in hypotheses.items():
        started = time.perf_counter()
        sets[key] = ic.build_observations(cache, geometry, work / folder)
        progress(f"{geometry.name}: {sets[key].manifest['observations']} ({time.perf_counter() - started:.0f} s)")
    return sets


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew study", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=("signs", "angles"))
    parser.add_argument("features", type=Path, help="feature cache directory (observe's work/features)")
    parser.add_argument("--work", type=Path, required=True, help="directory for observation sets and the report")
    parser.add_argument("--config", type=Path, default=Path("configs/imas_2DAQ.yaml"))
    parser.add_argument("--files", type=int, nargs="+", help="signs: input files used (default all)")
    parser.add_argument("--train-files", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--heldout-files", type=int, nargs="+", default=[2])
    parser.add_argument("--transverse-sign", type=int, choices=(-1, 1), default=NOMINAL_TRANSVERSE_SIGN)
    parser.add_argument("--axial-sign", type=int, choices=(-1, 1), default=NOMINAL_AXIAL_SIGN)
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="EM iterations per stage (default 50; a stage stopped there is reported unconverged)")
    args = parser.parse_args(argv)
    report_path = args.work / f"{args.study}.json"
    if report_path.exists():
        parser.error(f"{report_path} exists; use a new work directory or remove the report")
    args.work.mkdir(parents=True, exist_ok=True)
    setup = ImasSetup.from_files(args.config)
    started = time.perf_counter()

    def progress(text):
        print(f"[{time.perf_counter() - started:7.0f} s] {text}", flush=True)
    if args.study == "signs":
        hypotheses = {(t, a): (ImasGeometry.from_setup(setup, transverse_sign=t, axial_sign=a),
                               f"obs_st{t:+d}_sa{a:+d}") for t, a in SIGNS}
        result = sign_study(_sets(args.features, setup, args.work, hypotheses, progress), setup, files=args.files,
                            options=replace(SIGN_OPTIONS, max_iterations=args.max_iterations), progress=progress)
    else:
        hypotheses = {tol: (ImasGeometry.from_setup(setup, transverse_sign=args.transverse_sign,
                                                    axial_sign=args.axial_sign, opposition_tolerance_deg=tol),
                            f"obs_tol{tol:g}") for tol in ANGLE_TOLERANCES_DEG}
        result = angle_study(_sets(args.features, setup, args.work, hypotheses, progress), setup,
                             train_files=args.train_files, heldout_files=args.heldout_files,
                             options=replace(solve.EM_OPTIONS, max_iterations=args.max_iterations), progress=progress)
    report = {"schema_version": 1, "scanner": "imas", "requirement": "011-FR-2" if args.study == "signs" else "011-FR-3",
              "created_utc": datetime.now(timezone.utc).isoformat(), "features": str(args.features),
              "feature_fingerprint": FeatureCache(args.features).manifest["fingerprint"],
              "em_max_iterations": args.max_iterations, "seconds": time.perf_counter() - started, **result,
              "peak_memory": peak_memory()}
    report_path.write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()}, indent=1, default=float))
    print(f"-> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
