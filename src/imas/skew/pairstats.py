"""Pair statistics of an IMAS observation set: the FR-7 measurement (spec 011 T7).

From one observation set (train rows) and the LDAT bytes it came from:

- accepted observations, distinct TIME-channel pairs, observations per pair (log bins);
- the fraction of pairs whose two branches spec 008's pair fit resolves (status ``usable`` with its
  default thresholds: >= 3 expected events per branch, alias margin 3, separation >= 2 kernel widths),
  with the global kernel estimated from pairs of >= 30 events when there are any, else a declared one;
- partners and observations per TIME channel, connected components (all pairs, and pairs of >= 30);
- grouped alternatives: observations per minimodule pair and per ASIC pair (64 channels per ASIC);
- coverage per SM and ring;
- a linear projection to ``project_bytes`` of LDAT. Per-channel and grouped counts scale linearly;
  per-pair counts are scaled observed pairs only (pairs unseen here are not projected).

Nothing here is a calibration result; it informs the estimator decision (FR-7).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.imas.skew import select as sel
from src.imas.skew.setup import ImasSetup
from src.timing import pairs as tp
from src.timing.cache import ObservationSet

PAIR_BINS = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 100_000, 10 ** 12)
CHANNELS_PER_ASIC = 64
POPULATED = 30                        # spec 008: events per pair for kernel estimation
DECLARED_MODEL = tp.PairModel(200.0, 0.1, 2.0 * sel.COINCIDENCE_WINDOW_PS)


def histogram(values, edges=PAIR_BINS):
    counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=np.float64))
    return [{"from": int(lo), "below": int(hi), "count": int(n)} for lo, hi, n in zip(edges[:-1], edges[1:], counts)]


def quantiles(values):
    if not len(values):
        return None
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {"min": float(np.min(values)), "p5": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]), "max": float(np.max(values)), "mean": float(np.mean(values))}


def components(channel_i, channel_j):
    """Connected components of the channel graph: (count, sizes sorted descending)."""
    if not len(channel_i):
        return 0, []
    nodes, inverse = np.unique(np.concatenate([channel_i, channel_j]), return_inverse=True)
    i, j = inverse[:len(channel_i)], inverse[len(channel_i):]
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(nodes), len(nodes)))
    count, labels = connected_components(graph, directed=False)
    return int(count), sorted(np.bincount(labels).tolist(), reverse=True)


def grouped(keys_i, keys_j, counts):
    """Observations per unordered group pair."""
    lo, hi = np.minimum(keys_i, keys_j), np.maximum(keys_i, keys_j)
    pair = lo.astype(np.int64) * 10_000_000 + hi
    _, inverse = np.unique(pair, return_inverse=True)
    return np.bincount(inverse, weights=counts.astype(np.float64))


def kernel_rows(observations, *, min_events=POPULATED, max_pairs=2000):
    """Train rows of up to ``max_pairs`` pairs with >= ``min_events`` events, one partition at a time."""
    chosen, total = [], 0
    for p in range(observations.partitions):
        rows = observations.partition(p)
        rows = rows[rows["role"] == 0]
        _, inverse, counts = np.unique(tp._pair_keys(rows), return_inverse=True, return_counts=True)
        keep = np.flatnonzero(counts >= min_events)[:max_pairs - total]
        chosen.append(rows[np.isin(inverse, keep)])
        total += len(keep)
        if total >= max_pairs:
            break
    return np.concatenate(chosen) if chosen else None


def pair_statistics(observations, setup, *, ldat_bytes, project_bytes=1e12, fit=True):
    started = time.perf_counter()
    table = observations.pair_counts()
    table = table[table["role"] == 0]
    n = table["count"].astype(np.int64)
    ci, cj = table["channel_i"].astype(np.int64), table["channel_j"].astype(np.int64)
    times = setup.time_channels
    lookup = setup.lookup()
    per_channel = np.zeros(lookup.kind.size, np.int64)
    partners = np.zeros(lookup.kind.size, np.int64)
    np.add.at(per_channel, ci, n)
    np.add.at(per_channel, cj, n)
    np.add.at(partners, ci, 1)
    np.add.at(partners, cj, 1)
    observed = per_channel[times] > 0
    scale = project_bytes / ldat_bytes
    populated = n >= POPULATED
    report = {
        "accepted_observations": int(n.sum()), "distinct_pairs": int(len(n)),
        "possible_pairs_among_observed_channels": int(observed.sum() * (observed.sum() - 1) // 2),
        "observations_per_pair": quantiles(n), "observations_per_pair_histogram": histogram(n),
        "pairs_with_at_least": {str(k): int((n >= k).sum()) for k in (3, 6, 10, 30, 100)},
        "time_channels_observed": int(observed.sum()), "time_channels": int(len(times)),
        "observations_per_channel": quantiles(per_channel[times][observed]),
        "partners_per_channel": quantiles(partners[times][observed]),
        "components_all_pairs": dict(zip(("count", "sizes_top5"), (lambda c: (c[0], c[1][:5]))(components(ci, cj)))),
        "components_pairs_ge_30": dict(zip(("count", "sizes_top5"),
                                           (lambda c: (c[0], c[1][:5]))(components(ci[populated], cj[populated])))),
    }
    module_key = lookup.sm.astype(np.int64) * 16 + lookup.mm
    mm_pairs = grouped(module_key[ci], module_key[cj], n)
    asic_pairs = grouped(ci // CHANNELS_PER_ASIC, cj // CHANNELS_PER_ASIC, n)
    report["grouped"] = {
        "minimodule_pairs": int(len(mm_pairs)), "observations_per_minimodule_pair": quantiles(mm_pairs),
        "asic_pairs": int(len(asic_pairs)), "observations_per_asic_pair": quantiles(asic_pairs)}
    sm = lookup.sm[times]
    n_sm = setup.n_azimuth * len(setup.ring_z)
    per_sm = np.bincount(sm, weights=per_channel[times], minlength=n_sm)
    channels_per_sm = np.bincount(sm[observed], minlength=n_sm)
    report["coverage"] = {
        "observations_per_sm": quantiles(per_sm), "sms_without_observations": [int(s) for s in np.flatnonzero(per_sm == 0)],
        "observed_channels_per_sm": quantiles(channels_per_sm),
        "observations_per_ring": [int(per_sm[r * setup.n_azimuth:(r + 1) * setup.n_azimuth].sum())
                                  for r in range(len(setup.ring_z))],
        "observations_per_azimuth": [int(per_sm[a::setup.n_azimuth].sum()) for a in range(setup.n_azimuth)]}
    report["projection"] = {
        "ldat_bytes": int(ldat_bytes), "project_bytes": project_bytes, "scale": scale,
        "accepted_observations": n.sum() * scale,
        "observations_per_channel": quantiles(per_channel[times][observed] * scale),
        "observations_per_observed_pair": quantiles(n * scale),
        "observed_pairs_reaching": {str(k): int((n * scale >= k).sum()) for k in (6, 30, 100)},
        "observations_per_minimodule_pair": quantiles(mm_pairs * scale),
        "observations_per_asic_pair": quantiles(asic_pairs * scale),
        "note": "linear in LDAT bytes; pairs not seen here are not projected"}
    if fit:
        report["branch_resolution"] = branch_resolution(observations)
    report["seconds"] = time.perf_counter() - started
    return report


def branch_resolution(observations):
    """Spec 008 pair fits with default thresholds over all train pairs."""
    rows = kernel_rows(observations)
    options = tp.FitOptions(scan_ps=sel.COINCIDENCE_WINDOW_PS - 1000.0)
    model, history, source = DECLARED_MODEL, [], "declared (no pair with >= 30 events)"
    if rows is not None and len(rows):
        try:
            model, history = tp.estimate_model(rows, start=DECLARED_MODEL, options=options)
            source = "estimated from pairs with >= 30 events"
        except ValueError as error:
            source = f"declared ({error})"
    fits = tp.fit_observation_set(observations, model, options, roles=(0,))
    status = {name: int((fits["status"] == k).sum()) for k, name in enumerate(tp.STATUSES)}
    usable = fits["status"] == tp.USABLE
    events = fits["events"]
    by_events = {}
    for lo, hi in ((1, 6), (6, 10), (10, 30), (30, 100), (100, 10 ** 12)):
        inside = (events >= lo) & (events < hi)
        by_events[f"{lo}-{hi if hi < 10 ** 12 else 'max'}"] = {
            "pairs": int(inside.sum()), "usable": int((inside & usable).sum())}
    return {"model": {"sigma_ps": model.sigma_ps, "background_fraction": model.background_fraction,
                      "window_ps": model.window_ps, "source": source, "history": history},
            "thresholds": {"min_branch_events": options.min_branch_events,
                           "min_alias_margin": options.min_alias_margin,
                           "min_separation_sigmas": options.min_separation_sigmas, "scan_ps": options.scan_ps},
            "status_counts": status, "usable_fraction": float(usable.mean()) if len(fits) else None,
            "observations_in_usable_pairs": int(events[usable].sum()), "usable_by_pair_events": by_events,
            "median_branch_separation_ps": float(np.nanmedian(2 * fits["half_separation_ps"][usable]))
            if usable.any() else None,
            "median_shift_sigma_ps": float(np.nanmedian(fits["shift_sigma_ps"][usable])) if usable.any() else None}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew pairstats", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("observations", type=Path, help="observation set directory (observe's work/observations)")
    parser.add_argument("--out", type=Path, required=True, help="report JSON (new file)")
    parser.add_argument("--config", type=Path, default=Path("configs/imas_2DAQ.yaml"))
    parser.add_argument("--ldat-bytes", type=int, help="LDAT bytes behind the set (default: observe_report.json)")
    parser.add_argument("--project-tb", type=float, default=1.0)
    parser.add_argument("--no-fit", action="store_true", help="skip the spec 008 pair fits")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output exists; use a new name")
    ldat_bytes = args.ldat_bytes
    source_report = args.observations.parent / "observe_report.json"
    upstream = json.loads(source_report.read_text(encoding="utf-8")) if source_report.exists() else None
    if ldat_bytes is None:
        if upstream is None or upstream.get("max_records") is not None:
            parser.error("--ldat-bytes required (no full-file observe_report.json next to the set)")
        ldat_bytes = upstream["bytes"]["ldat"]
    report = pair_statistics(ObservationSet(args.observations), ImasSetup.from_files(args.config),
                             ldat_bytes=ldat_bytes, project_bytes=args.project_tb * 1e12, fit=not args.no_fit)
    report = {"schema_version": 1, "scanner": "imas", "requirement": "011-FR-7",
              "created_utc": datetime.now(timezone.utc).isoformat(), "observations": str(args.observations),
              "upstream": None if upstream is None else {k: upstream.get(k) for k in (
                  "inputs", "selection", "geometry", "records", "first_failure", "geometry_counts", "provenance")},
              **report}
    args.out.write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    keys = ("accepted_observations", "distinct_pairs", "observations_per_pair", "pairs_with_at_least",
            "partners_per_channel", "components_all_pairs")
    print(json.dumps({k: report[k] for k in keys}, indent=1, default=float))
    if "branch_resolution" in report:
        print(json.dumps({k: report["branch_resolution"][k] for k in ("status_counts", "usable_fraction")}))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
