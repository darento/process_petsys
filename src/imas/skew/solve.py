"""Both FR-7 estimators on an IMAS observation set, held-out check and comparison (spec 011 T10-T11).

* ``pair_estimator``: spec 008 unchanged in the core. The kernel width and
  background come from pairs with >= 30 events (``estimate_model``); when no
  such pair exists the declared model is used and named. Pair fits use the
  IMAS window (scan +-9 ns, flat background over +-10 ns). The global solve
  covers every IMAS TIME channel, so a channel without a usable pair gets an
  explicit status.
* ``em_estimator``: the global EM of ``src.timing.em`` with the IMAS
  data-only coarse start (SM, then SM x minimodule, then channel) and the
  same window.

Both return a ``src.timing.solver.Solution``; ``src.timing.lut.write_lut``
writes either one.

``python -m src.imas.skew solve OBS --estimator em|pairs|both --out DIR``
fits on the train files. It writes ``lut_<estimator>.csv/.json`` and
``solve_report.json``: model, statuses, a held-out check and the
comparisons. The held-out check refits ``sigma``, background and branch
fraction with each LUT's delays fixed (the legacy LUT included) on the
held-out files' events. It uses the events whose two channels every compared
LUT applies, so a smaller held-out ``sigma`` means better delays on the same
events. The comparisons cover
estimator against estimator and against the legacy LUT (``skew = -b``,
``9999`` excluded).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from src.imas.skew import outputs, select as sel
from src.imas.skew.pairstats import DECLARED_MODEL, POPULATED, kernel_rows
from src.imas.skew.setup import ImasSetup
from src.timing import em, pairs as tp, solver as ts
from src.timing.cache import ObservationSet
from src.timing.report import peak_memory

WINDOW_PS = 2.0 * sel.COINCIDENCE_WINDOW_PS
PAIR_OPTIONS = tp.FitOptions(scan_ps=sel.COINCIDENCE_WINDOW_PS - 1000.0)
EM_OPTIONS = em.EmOptions(window_ps=WINDOW_PS)
LEGACY_MISSING = 9999.0
ESTIMATORS = ("em", "pairs")


class FileView:
    """Rows of an ``ObservationSet`` from some input files only (same partitions)."""

    def __init__(self, observations, files):
        self.observations, self.files = observations, None if files is None else np.asarray(files)
        self.partitions = observations.partitions

    def partition(self, p):
        rows = self.observations.partition(p)
        return rows if self.files is None else rows[np.isin(rows["file"], self.files)]


def pair_model(observations, *, options=PAIR_OPTIONS, max_pairs=2000):
    """(model, provenance) from train pairs with >= 30 events, else the declared model."""
    rows = kernel_rows(observations, min_events=POPULATED, max_pairs=max_pairs)
    if rows is None or not len(rows):
        return DECLARED_MODEL, {"source": f"declared (no pair with >= {POPULATED} events)", "history": []}
    try:
        model, history = tp.estimate_model(rows, start=DECLARED_MODEL, options=options)
    except ValueError as error:
        return DECLARED_MODEL, {"source": f"declared ({error})", "history": []}
    return model, {"source": f"estimated from pairs with >= {POPULATED} events", "history": history}


def pair_estimator(observations, setup, *, roles=(0,), files=None, model=None, options=PAIR_OPTIONS,
                   solve_options=ts.SolveOptions()):
    """Spec 008 per-pair fits and global solve; returns (solution, pair fits)."""
    observations = FileView(observations, files)
    provenance = {"source": "given", "history": []}
    if model is None:
        model, provenance = pair_model(observations, options=options)
    fits = tp.fit_observation_set(observations, model, options, roles=roles)
    solution = ts.solve(fits, setup.time_channels, (), solve_options)
    solution.diagnostics.update({
        "estimator": "spec 008 pair fits + global solve",
        "model": {"sigma_ps": model.sigma_ps, "background_fraction": model.background_fraction,
                  "window_ps": model.window_ps, **provenance},
        "fit_options": {k: getattr(options, k) for k in tp.FitOptions.__dataclass_fields__},
        "pair_status_counts": {name: int((fits["status"] == k).sum()) for k, name in enumerate(tp.STATUSES)}})
    return solution, fits


def coarse_groups(setup, channels):
    """IMAS data-only coarse start: SM, then SM x minimodule."""
    lookup = setup.lookup()
    sm = lookup.sm[channels].astype(np.int64)
    return (("sm", sm), ("minimodule", sm * 16 + lookup.mm[channels]))


def em_estimator(observations, setup, *, roles=(0,), files=None, where=None, options=EM_OPTIONS, directory=None):
    """Global EM over every IMAS TIME channel; ``files`` and ``where(rows)`` restrict the rows;
    ``directory`` memory-maps the observation chunks."""
    def keep(rows):
        mask = np.ones(len(rows), bool) if files is None else np.isin(rows["file"], files)
        return mask if where is None else mask & where(rows)

    data = em.EmData.from_observation_set(observations, setup.time_channels, roles=roles, where=keep,
                                          directory=directory)
    solution = em.solve_em(data, coarse_groups(setup, data.channels), options)
    return solution, data


# --- held-out check and comparisons ----------------------------------------------------------------------------

def held_out(tables, observations, setup, files=None, *, roles=(0, 1, 2), options=EM_OPTIONS):
    """Each {channel: (delay, sigma)} table on the held-out events (``files`` and/or ``roles``) whose two
    channels every table has."""
    allowed = setup.time_channels
    for table in tables.values():
        allowed = allowed[np.isin(allowed, np.fromiter(table, np.int64, len(table)))]
    if not len(allowed):
        return {"events": 0, "note": "no channel applied by every compared LUT"}

    def keep(rows):
        inside = np.isin(rows["channel_i"], allowed) & np.isin(rows["channel_j"], allowed)
        return inside if files is None else inside & np.isin(rows["file"], files)

    data = em.EmData.from_observation_set(observations, allowed, roles=roles, where=keep)
    result = {"files": None if files is None else [int(f) for f in files], "roles": list(roles),
              "channels": int(len(allowed)), "events": data.observations}
    if data.observations:
        for name, table in tables.items():
            result[name] = em.evaluate(data, np.array([table[int(c)][0] for c in data.channels]), options)
    return result


def compare(a, b):
    """Gauge-free comparison of two {channel: (delay, sigma)} tables over their common channels."""
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return {"channels": len(common)}
    da, db = np.array([a[c][0] for c in common]), np.array([b[c][0] for c in common])
    sa, sb = np.array([a[c][1] for c in common]), np.array([b[c][1] for c in common])
    difference = da - db
    difference -= np.median(difference)
    combined = np.hypot(np.nan_to_num(sa), np.nan_to_num(sb))
    robust = 1.4826 * float(np.median(np.abs(difference)))
    result = {"channels": len(common), "correlation": float(np.corrcoef(da, db)[0, 1]),
              "robust_sd_ps": robust, "difference_ps_quantiles_5_50_95": np.percentile(difference, [5, 50, 95]).tolist(),
              "beyond_5_robust_sd": int((np.abs(difference) > 5 * robust).sum())}
    if (combined > 0).all():
        result["robust_sd_over_combined_sigma"] = 1.4826 * float(np.median(np.abs(difference / combined)))
    return result


def applied_table(solution):
    rows = solution.channels[solution.channels["apply"]]
    return {int(c): (float(d), float(s)) for c, d, s in zip(rows["time_channel"], rows["delay_ps"], rows["sigma_ps"])}


def legacy_table(path):
    """Legacy ``ch skew`` (additive) as {channel: (b = -skew, NaN)}, ``9999`` excluded."""
    table = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            channel, skew = line.split()
            if float(skew) != LEGACY_MISSING:
                table[int(channel)] = (-float(skew), np.nan)
    return table


def summary(solution):
    channels = solution.channels[solution.channels["apply"]]
    return {"status_counts": solution.diagnostics["status_counts"], "model": solution.diagnostics.get("model"),
            "applied": int(len(channels)),
            "sigma_ps_quantiles_5_50_95": np.percentile(channels["sigma_ps"], [5, 50, 95]).tolist()
            if len(channels) else None,
            "delay_ps_quantiles_1_50_99": np.percentile(channels["delay_ps"], [1, 50, 99]).tolist()
            if len(channels) else None}


def run(observations_dir, out, *, estimators=ESTIMATORS, train_files=(0, 1), heldout_files=(2,),
        config_path="configs/imas_2DAQ.yaml", encal_path=None, legacy=None, root="."):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    observations = ObservationSet(observations_dir)
    setup = ImasSetup.from_files(config_path, encal_path, root=root)
    upstream = observations.manifest["fingerprint"]
    provenance = {"observations": str(observations_dir), "observation_fingerprint": upstream,
                  "setup": setup.provenance, "train_files": list(train_files), "heldout_files": list(heldout_files)}
    report = {"schema_version": 1, "scanner": "imas", "requirement": "011-FR-7",
              "created_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance, "estimators": {}}
    solutions = {}
    for name in estimators:
        started = time.perf_counter()
        if name == "em":
            solution, _ = em_estimator(observations, setup, files=train_files, directory=out / "em_chunks")
        else:
            solution, _ = pair_estimator(observations, setup, files=train_files)
        seconds = time.perf_counter() - started
        outputs.write_imas_lut(out / f"lut_{name}.csv", solution, {**provenance, "estimator": name}, setup)
        solutions[name] = solution
        report["estimators"][name] = {"seconds": seconds, **summary(solution)}
    tables = {name: applied_table(solution) for name, solution in solutions.items()}
    if legacy is not None:
        tables["legacy"] = legacy_table(legacy)
        provenance["legacy"] = str(legacy)
    names = list(tables)
    report["comparisons"] = {f"{a} vs {b}": compare(tables[a], tables[b])
                             for k, a in enumerate(names) for b in names[k + 1:]}
    if heldout_files:
        report["held_out"] = {name: held_out({name: table}, observations, setup, heldout_files)
                              for name, table in tables.items()}
        populated = {name: table for name, table in tables.items() if table}
        if len(populated) > 1:
            report["held_out"]["common"] = held_out(populated, observations, setup, heldout_files)
    report["peak_memory"] = peak_memory()
    (out / "solve_report.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew solve", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("observations", type=Path, help="observation set directory")
    parser.add_argument("--estimator", choices=("em", "pairs", "both"), default="both")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--train-files", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--heldout-files", type=int, nargs="*", default=[2])
    parser.add_argument("--config", type=Path, default=Path("configs/imas_2DAQ.yaml"))
    parser.add_argument("--encal", type=Path, help="energy calibration of the set (provenance)")
    parser.add_argument("--legacy", type=Path, help="legacy skew LUT (ch skew, additive) to compare against")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output exists; use a new directory")
    if set(args.train_files) & set(args.heldout_files):
        parser.error("a file cannot be both train and held out")
    estimators = ESTIMATORS if args.estimator == "both" else (args.estimator,)
    report = run(args.observations, args.out, estimators=estimators, train_files=args.train_files,
                 heldout_files=args.heldout_files, config_path=args.config, encal_path=args.encal,
                 legacy=args.legacy)
    print(json.dumps({k: report[k] for k in ("estimators", "comparisons", "held_out") if k in report}, indent=1,
                     default=float))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
