"""Per-TIME-channel 511 keV photopeak calibration from IMAS skew splits (spec 011 T5, FR-4).

Sides of records passing selection cuts 1-5 (``select`` calibration mode) are
histogrammed per top TIME channel: the side's minimodule ENERGY sum (a.u.) in
0.1 a.u. bins over 0-220 a.u. (one pass, bounded by channels x bins, never by
events). Per channel, as spec 003's Cornell slab calibration
(``src.cornell.calibration``):

- fewer than 200 entries: ``few entries``;
- anchor at the highest 5-bin-smoothed bin of a 150-bin 0-220 a.u. histogram;
- ``fit_peak_background`` over 0.55-1.5 x anchor (100 bins), Gaussian plus
  constant or linear background, linear only when it improves the deviance by >= 8;
- no fitted peak: ``no peak``.

Values enter the fit at their 0.1 a.u. bin centres. Not taken from Cornell:
slab keys, the two-TIME-channel side rule, outer-slab borrowing and
neighbour estimates. A channel without a fit is written as ``nan``, so
``KevConverter(file_type="mu")`` gives NaN and the selection rejects its events.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.imas.skew import select as sel
from src.imas.skew.setup import ImasSetup
from src.ldat_inspector.engine import fit_peak_background

FINE_BIN_AU, RANGE_AU = 0.1, (0.0, 220.0)
FINE_BINS = int(round((RANGE_AU[1] - RANGE_AU[0]) / FINE_BIN_AU))
ANCHOR_BINS, FIT_BINS = 150, 100
MIN_ENTRIES = 200
LINEAR_DEVIANCE_GAIN = 8
FIT, FEW, NO_PEAK = "fit", "few entries", "no peak"


def accumulate(path, setup, *, window_ps=sel.COINCIDENCE_WINDOW_PS, max_records=None, full_hash=False):
    """One split: (histogram [TIME channel row, fine bin] uint32, entries outside range, counts, identity)."""
    lookup = setup.lookup()
    times = setup.time_channels
    row = np.full(lookup.kind.size, -1, np.int64)
    row[times] = np.arange(len(times))
    histogram = np.zeros((len(times), FINE_BINS), np.uint32)
    outside = 0
    counts = {}
    digest = hashlib.sha256() if full_hash and max_records is None else None
    path = Path(path)
    before = path.stat()
    selection = sel.Selection.production(lookup, window_ps=window_ps)
    for features in sel.feature_batches(path, lookup, selection, max_records=max_records, calibration=True,
                                        on_block=None if digest is None else digest.update):
        codes, found = np.unique(features.codes, return_counts=True)
        for code, n in zip(codes.tolist(), found.tolist()):
            counts[sel.code_label(code)] = counts.get(sel.code_label(code), 0) + n
        channel = np.concatenate([features.column("time_channel_1"), features.column("time_channel_2")])
        energy = np.concatenate([features.column("e_raw_1"), features.column("e_raw_2")])
        index = np.floor((energy - RANGE_AU[0]) / FINE_BIN_AU).astype(np.int64)
        inside = (index >= 0) & (index < FINE_BINS)
        outside += int(np.count_nonzero(~inside))
        np.add.at(histogram, (row[channel[inside]], index[inside]), 1)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise ValueError(f"{path.name}: input changed while it was read")
    identity = {"name": path.name, "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                "max_records": max_records, "sha256": None if digest is None else digest.hexdigest()}
    return histogram, outside, counts, identity


def _values(fine):
    """Stand-in values: each fine-bin count at its bin centre."""
    centres = RANGE_AU[0] + FINE_BIN_AU * (np.arange(FINE_BINS) + 0.5)
    return np.repeat(centres, np.asarray(fine, np.int64))


def fit_channel(fine):
    """(mu, sigma, status, details) from one channel's fine histogram."""
    entries = int(fine.sum())
    if entries < MIN_ENTRIES:
        return np.nan, np.nan, FEW, {"entries": entries}
    values = _values(fine)
    anchor_counts, anchor_edges = np.histogram(values, ANCHOR_BINS, range=RANGE_AU)
    smooth = np.convolve(anchor_counts, np.ones(5) / 5, mode="same")
    centres = (anchor_edges[:-1] + anchor_edges[1:]) / 2
    anchor = float(centres[np.argmax(smooth)])
    settings = dict(interval=(0.55 * anchor, 1.5 * anchor), search=(0.8 * anchor, 1.2 * anchor),
                    sigma0=0.07 * anchor, mu_halfwidth=0.15 * anchor, bins=FIT_BINS)
    linear = fit_peak_background(values, background_model="linear", **settings)
    flat = fit_peak_background(values, background_model="constant", **settings)
    best = (linear if linear["status"] == "FIT" and (flat["status"] != "FIT" or
                                                      flat["deviance"] - linear["deviance"] >= LINEAR_DEVIANCE_GAIN)
            else flat)
    details = {"entries": entries, "anchor": anchor, "fit_status": best["status"]}
    if best["status"] != "FIT":
        return np.nan, np.nan, NO_PEAK, details
    details["background"] = "linear" if best is linear else "constant"
    return float(best["mu"]), float(best["sigma"]), FIT, details


def calibrate(histogram, setup):
    """Fit every TIME channel; returns (mu array, sigma array, statuses, details) in ``time_channels`` order."""
    times = setup.time_channels
    mu, sigma = np.full(len(times), np.nan), np.full(len(times), np.nan)
    statuses, details = [], []
    for i in range(len(times)):
        mu[i], sigma[i], status, info = fit_channel(histogram[i])
        statuses.append(status)
        details.append(info)
    return mu, sigma, statuses, details


def write_outputs(out, setup, mu, sigma, statuses, details, manifest):
    """``encal.tsv`` (KevConverter ``mu``; ``nan`` without a fit), ``encal_status.csv`` and ``encal.json``."""
    out = Path(out)
    out.mkdir(parents=True)
    times = setup.time_channels.tolist()
    lines = ["ID\tmu"] + [f"{ch}\t{value:.6f}" if np.isfinite(value) else f"{ch}\tnan" for ch, value in zip(times, mu)]
    (out / "encal.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = ["channel,sm,minimodule,slab,status,entries,mu,sigma,background"]
    for ch, m, s, status, info in zip(times, mu, sigma, statuses, details):
        sm, mm = setup.sm_mM_map[ch]
        rows.append(f"{ch},{sm},{mm},{setup.local_map[ch][2]},{status},{info['entries']},"
                    f"{'' if not np.isfinite(m) else f'{m:.6f}'},{'' if not np.isfinite(s) else f'{s:.6f}'},"
                    f"{info.get('background', '')}")
    (out / "encal_status.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = dict(manifest)
    manifest["encal_sha256"] = hashlib.sha256((out / "encal.tsv").read_bytes()).hexdigest()
    (out / "encal.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out / "encal.tsv"


def per_sm_spread(setup, mu):
    """Median, 5th and 95th percentile of fitted mu per SM (a.u.)."""
    sm = np.array([setup.sm_mM_map[ch][0] for ch in setup.time_channels.tolist()])
    spread = {}
    for s in np.unique(sm).tolist():
        values = mu[(sm == s) & np.isfinite(mu)]
        spread[str(s)] = None if not len(values) else [float(np.median(values)), float(np.percentile(values, 5)),
                                                       float(np.percentile(values, 95))]
    return spread


def run(splits, out, *, config_path, root=".", window_ps=sel.COINCIDENCE_WINDOW_PS, max_records=None, workers=1,
        full_hash=False):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out}: output directory already exists; use a new name")
    setup = ImasSetup.from_files(config_path, root=root)
    started = time.perf_counter()
    arguments = [(path, setup) for path in splits]
    options = dict(window_ps=window_ps, max_records=max_records, full_hash=full_hash)
    if workers > 1 and len(splits) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(splits))) as pool:
            results = list(pool.map(_accumulate_star, [(a, options) for a in arguments]))
    else:
        results = [accumulate(*a, **options) for a in arguments]
    histogram = np.zeros_like(results[0][0], dtype=np.uint64)
    counts, identities, outside = {}, [], 0
    for h, o, c, identity in results:
        histogram += h
        outside += o
        identities.append(identity)
        for label, n in c.items():
            counts[label] = counts.get(label, 0) + n
    read_seconds = time.perf_counter() - started
    mu, sigma, statuses, details = calibrate(histogram, setup)
    status_counts = {s: statuses.count(s) for s in (FIT, FEW, NO_PEAK)}
    manifest = {
        "schema_version": 1, "scanner": "imas", "requirement": "011-FR-4",
        "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": identities,
        "provenance": setup.provenance,
        "selection": {"cuts": "spec 011 FR-3 cuts 1-5 (production), both sides", **sel.Selection.production(
            setup.lookup(), window_ps=window_ps).describe()},
        "histogram": {"value": "minimodule ENERGY sum per side (a.u.), keyed by the side's top TIME channel",
                      "fine_bin_au": FINE_BIN_AU, "range_au": list(RANGE_AU), "outside_range": outside},
        "fit": {"min_entries": MIN_ENTRIES, "anchor": f"highest 5-bin-smoothed bin, {ANCHOR_BINS} bins",
                "interval": "0.55-1.5 x anchor", "bins": FIT_BINS, "model": "fit_peak_background Gaussian + "
                f"constant or linear background (linear if deviance gain >= {LINEAR_DEVIANCE_GAIN})",
                "values": f"fine-bin centres ({FINE_BIN_AU} a.u.)"},
        "record_counts": counts, "status_counts": status_counts,
        "per_sm_mu_median_p5_p95": per_sm_spread(setup, mu),
        "read_and_select_seconds": read_seconds, "fit_seconds": time.perf_counter() - started - read_seconds,
        "nominal": "per-TIME-channel 511 keV peak of the minimodule ENERGY sum; no legacy calibration to compare",
    }
    path = write_outputs(out, setup, mu, sigma, statuses, details, manifest)
    return path, manifest


def _accumulate_star(argument):
    (path, setup), options = argument
    return accumulate(path, setup, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew encal", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("splits", type=Path, nargs="+", help="dedicated energy-calibration splits")
    parser.add_argument("--config", type=Path, default=Path("configs/imas_2DAQ.yaml"))
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--window-ps", type=int, default=sel.COINCIDENCE_WINDOW_PS)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--full-hash", action="store_true", help="SHA-256 of each whole split while reading")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output directory already exists; use a new name")
    path, manifest = run(args.splits, args.out, config_path=args.config, window_ps=args.window_ps,
                         max_records=args.max_records, workers=args.workers, full_hash=args.full_hash)
    print(f"{manifest['status_counts']} -> {path}")
    return 0
