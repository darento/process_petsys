"""Native time-offset roundtrip check on matched LDAT events (spec 008 T9a, FR-3/FR-17).

Usage::

    python -m src.cornell.skew roundtrip UNCORRECTED.ldat CORRECTED.ldat TABLE.tsv [--records N] [--clock-ps 5000]

Both files come from the same RAWF converted without and with the native
table (``lut.write_native_table``). PETsys subtracts each hit's
offset before grouping, so the compact writer gives ``t_B = trunc((T - b)
Tps)`` against ``t_A = trunc(T Tps)``: a corrected hit sits within 1 ps of
``t_A - b``. The first ``records`` corrected records are matched hit by hit
(same channel, ``t_B + k b`` nearest an uncorrected hit within
``tolerance_ps``) to the uncorrected records of the same time span, read
until that span is passed. Reported, never assumed:

* hypothesis counts for k = 1 (applied once), 0 (not applied), 2 (twice) and
  -1 (sign flipped) on hits of listed channels with ``|b| >= min_offset_ps``;
* exact equality of time and energy for hits of unlisted channels;
* energies of listed hits against PETsys's QDC side effect: ``ProcessHit``
  subtracts the offset from the hit time but not from the integration end,
  and ``energy = t_eq - (timeEnd - time)``, so a listed hit's energy moves by
  ``-b / clock_ps`` (within the 0.05 Newton tolerance of ``t_eq``). Valid for
  QDC calibration without PETsys energy calibration (Cornell's INI);
* per-channel median ``t_A - t_B`` against ``b`` (identity);
* matched records whose detector order swapped, with the corrected ``dt``
  checked against ``dt_A - b_1 + b_2`` in the uncorrected orientation;
* records found only in one conversion (grouping/coincidence membership
  changes), reported separately and never a failure by themselves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from src.cornell.skew.lut import read_native_table
from src.ldat_stream import LdatStream

HYPOTHESES = {"applied once": 1, "not applied": 0, "applied twice": 2, "sign flipped": -1}


def read_hits(path, *, max_records=None, stop_after_ps=None, margin_ps=1_000_000_000):
    """Hits of the first records: dict of arrays (record, side, channel, timestamp, energy) and record count.

    With ``stop_after_ps``, records are read until a record starts more than ``margin_ps`` after it.
    """
    parts, total = [], 0
    for batch in LdatStream(path):
        n = batch.n1.astype(np.int64) + batch.n2
        records = batch.count if max_records is None else min(batch.count, max_records - total)
        last = int(batch.first_hit[records - 1] + n[records - 1]) if records else 0
        start = np.repeat(np.arange(records), n[:records]) + total
        offset = np.arange(last) - np.repeat(batch.first_hit[:records], n[:records])
        side = (offset >= np.repeat(batch.n1[:records], n[:records])).astype(np.int8)
        hits = {"record": start, "side": side, "channel": batch.channel[:last].astype(np.int64),
                "timestamp": batch.timestamp[:last].astype(np.int64), "energy": batch.energy[:last]}
        if stop_after_ps is not None and records:
            first = np.minimum.reduceat(hits["timestamp"], np.r_[0, np.cumsum(n[:records])[:-1]])
            beyond = np.flatnonzero(first > stop_after_ps + margin_ps)
            if len(beyond):
                keep = hits["record"] < total + int(beyond[0])
                parts.append({k: v[keep] for k, v in hits.items()})
                total += int(beyond[0])
                break
        parts.append(hits)
        total += records
        if max_records is not None and total >= max_records:
            break
    if not parts:
        raise ValueError(f"{path}: no records")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, total


_TIME_BITS = 43  # key = channel << 43 | (t - base): a read spans far less than 2**43 ps (8.8 s)


def _sorted_keys(a_channel, a_time):
    """Sorted ``channel << 43 | (t - base)`` of uncorrected hits, the sort order and ``base``."""
    base = int(a_time.min()) - (1 << 20)
    if int(a_time.max()) - base >= 1 << _TIME_BITS or int(a_channel.max()) >= 1 << 20:
        raise ValueError("read span or channel id too large for the matching key")
    key = (a_channel << _TIME_BITS) | (a_time - base)
    order = np.argsort(key, kind="stable")
    return key[order], order, base


def _query(channel, time, base):
    offset = np.clip(time - base, 0, (1 << _TIME_BITS) - 1).astype(np.int64)
    return (channel << _TIME_BITS) | offset


def _nearest(a_channel, a_time, channel, time):
    """Index of the uncorrected hit of the same channel nearest ``time`` (-1 if the channel is absent)."""
    base = int(min(a_time.min(), np.floor(time.min()))) - (1 << 20)
    if max(a_time.max(), time.max()) - base >= 1 << _TIME_BITS or max(a_channel.max(), channel.max()) >= 1 << 20:
        raise ValueError("read span or channel id too large for the matching key")
    key_a = (a_channel << _TIME_BITS) | (a_time - base)
    order = np.argsort(key_a, kind="stable")
    sorted_a = key_a[order]
    query = (channel << _TIME_BITS) | (np.rint(time).astype(np.int64) - base)
    j = np.searchsorted(sorted_a, query)
    best = np.full(len(query), -1, np.int64)
    distance = np.full(len(query), np.iinfo(np.int64).max)
    for candidate in (j - 1, j):
        valid = (candidate >= 0) & (candidate < len(sorted_a))
        c = np.where(valid, candidate, 0)
        same = valid & ((sorted_a[c] >> _TIME_BITS) == channel)
        d = np.where(same, np.abs(sorted_a[c] - query), np.iinfo(np.int64).max)
        better = d < distance
        best[better], distance[better] = order[c[better]], d[better]
    return best


def compare(uncorrected, corrected, offsets, *, tolerance_ps=2.0, min_offset_ps=10.0, edge_ps=1_000_000,
            clock_ps=5000.0, energy_tolerance=0.05):
    """Roundtrip report for hit dicts from ``read_hits`` and ``{channel: offset ps}``."""
    a, b = uncorrected, corrected
    offset_b = np.array([offsets.get(int(c), 0.0) for c in b["channel"]])
    listed = np.array([int(c) in offsets for c in b["channel"]], bool)
    # compared span on the uncorrected time scale, away from both reads' edges
    restored = b["timestamp"] + offset_b
    span = (max(a["timestamp"].min(), restored.min()) + edge_ps, min(a["timestamp"].max(), restored.max()) - edge_ps)

    def record_start(hits, times):
        starts = np.full(hits["record"].max() + 1, np.inf)
        np.minimum.at(starts, hits["record"], times)
        return starts

    b_start, a_start = record_start(b, restored), record_start(a, a["timestamp"])
    b_in = (b_start >= span[0]) & (b_start <= span[1])
    a_in = (a_start >= span[0]) & (a_start <= span[1])
    hit_in = b_in[b["record"]]
    report = {"span_ps": [int(span[0]), int(span[1])], "corrected_records": int(b_in.sum()),
              "uncorrected_records": int(a_in.sum()), "tolerance_ps": tolerance_ps}

    strong = hit_in & listed & (np.abs(offset_b) >= min_offset_ps)
    counts = {}
    for name, k in HYPOTHESES.items():
        target = b["timestamp"][strong] + k * offset_b[strong]
        index = _nearest(a["channel"], a["timestamp"], b["channel"][strong], target)
        ok = index >= 0
        ok[ok] = np.abs(a["timestamp"][index[ok]] - target[ok]) <= tolerance_ps
        counts[name] = int(ok.sum())
    report["hypotheses_on_listed_hits"] = dict(counts, hits=int(strong.sum()), min_offset_ps=min_offset_ps)

    index = _nearest(a["channel"], a["timestamp"], b["channel"], restored)
    matched = index >= 0
    matched[matched] = np.abs(a["timestamp"][index[matched]] - restored[matched]) <= tolerance_ps
    residual = (b["timestamp"] - (a["timestamp"][np.maximum(index, 0)] - offset_b))[matched & hit_in]
    report["hits"] = {"compared": int(hit_in.sum()), "matched": int((matched & hit_in).sum()),
                      "residual_ps_min_max": [float(residual.min()), float(residual.max())] if len(residual) else None}
    unlisted = matched & hit_in & ~listed
    report["unlisted_channel_hits"] = {
        "matched": int(unlisted.sum()),
        "changed": int((b["timestamp"][unlisted] != a["timestamp"][index[unlisted]]).sum())}
    both = matched & hit_in
    d_energy = b["energy"].astype(np.float64)[both] - a["energy"][index[both]]
    on_listed = listed[both]
    model = d_energy[on_listed] + offset_b[both][on_listed] / clock_ps
    within = int((np.abs(model) <= energy_tolerance).sum())
    x = offset_b[both][on_listed]
    slope = float(np.polyfit(x, d_energy[on_listed], 1)[0]) if len(x) > 1 and np.ptp(x) > 0 else 0.0
    report["energy"] = {
        "model": "listed TIME channels: dE = -b/clock_ps (PETsys QDC ProcessHit); unlisted: unchanged",
        "clock_ps": clock_ps, "tolerance": energy_tolerance,
        "unlisted_changed": int((d_energy[~on_listed] != 0).sum()),
        "listed_matched": int(on_listed.sum()), "listed_within_model": within,
        "listed_outside_model": int(on_listed.sum()) - within,
        "model_residual_pct_1_50_99": np.percentile(model, [1, 50, 99]).round(4).tolist() if len(model) else None,
        "fitted_clock_ps": round(-1 / slope, 1) if slope else None}

    # record level: every hit has a candidate (same channel, restored time within tolerance) in one
    # uncorrected record of the same hit count. PETsys may write one photon into several coincidence
    # records, so a hit can have identical copies in other records; those are resolved per record.
    a_key, order, base = _sorted_keys(a["channel"], a["timestamp"])
    low = _query(b["channel"], np.ceil(restored - tolerance_ps), base)
    high = _query(b["channel"], np.floor(restored + tolerance_ps), base)
    lo, hi = np.searchsorted(a_key, low, "left"), np.searchsorted(a_key, high, "right")
    candidates = hi - lo
    a_rec_sorted = a["record"][order]
    n_b = np.bincount(b["record"], minlength=len(b_start))
    n_a = np.bincount(a["record"], minlength=len(a_start))
    first = np.r_[0, np.flatnonzero(np.diff(b["record"])) + 1]
    single = np.where(candidates == 1, a_rec_sorted[np.minimum(lo, len(a_rec_sorted) - 1)], -1)
    rec_of_first = b["record"][first]
    all_single = np.logical_and.reduceat(candidates == 1, first)
    same = np.minimum.reduceat(single, first) == np.maximum.reduceat(single, first)
    any_none = np.logical_or.reduceat(candidates == 0, first)
    good = np.zeros(len(b_start), bool)
    target = np.full(len(b_start), -1, np.int64)
    mapping = np.full(len(b["record"]), -1, np.int64)
    fast = all_single & same & b_in[rec_of_first]
    fast &= n_a[np.maximum(single[first], 0)] == n_b[rec_of_first]
    good[rec_of_first[fast]], target[rec_of_first[fast]] = True, single[first][fast]
    in_fast = np.repeat(fast, n_b[rec_of_first])
    mapping[in_fast] = order[lo[in_fast]]
    ambiguous = 0
    for k in np.flatnonzero(~fast & ~any_none & b_in[rec_of_first]):
        r, s = rec_of_first[k], first[k]
        hits = range(s, s + n_b[r])
        ambiguous += 1
        for record in np.unique(a_rec_sorted[lo[s]:hi[s]]):
            if n_a[record] != n_b[r]:
                continue
            chosen = []
            for h in hits:
                where = np.flatnonzero(a_rec_sorted[lo[h]:hi[h]] == record)
                if not len(where):
                    break
                chosen.append(order[lo[h] + where[0]])
            if len(chosen) == n_b[r]:
                good[r], target[r] = True, record
                mapping[s:s + n_b[r]] = chosen
                break
    matched_a = np.zeros(len(a_start), bool)
    matched_a[target[good]] = True
    report["records"] = {"matched": int(good.sum()), "only_corrected": int((b_in & ~good).sum()),
                         "only_uncorrected": int((a_in & ~matched_a).sum()),
                         "resolved_with_duplicate_hits": int(ambiguous)}

    # endpoint order and dt on matched records: first hit of each side (PETsys photon time)
    swapped, dt_bad, checked = 0, 0, 0
    for r, s in zip(rec_of_first, first):
        if not good[r]:
            continue
        rows = slice(s, s + n_b[r])
        sides = b["side"][rows]
        a_sides = a["side"][mapping[rows]]
        swapped += int((sides != a_sides).all())
        h1, h2 = s + int(np.argmax(sides == 0)), s + int(np.argmax(sides == 1))
        dt_b = b["timestamp"][h1] - b["timestamp"][h2]
        dt_expected = (a["timestamp"][mapping[h1]] - offset_b[h1]) - (a["timestamp"][mapping[h2]] - offset_b[h2])
        checked += 1
        dt_bad += int(abs(dt_b - dt_expected) > 2 * tolerance_ps)
    report["endpoints"] = {"checked": checked, "detector_order_swapped": swapped, "dt_mismatch": dt_bad}

    # identity: median t_A - t_B per listed channel against its offset
    rows = np.flatnonzero(both & listed)
    rows = rows[np.argsort(b["channel"][rows], kind="stable")]
    shift = a["timestamp"][index[rows]] - b["timestamp"][rows]
    channels, starts = np.unique(b["channel"][rows], return_index=True)
    per = {int(c): float(np.median(part)) for c, part in zip(channels, np.split(shift, starts[1:]))} if len(rows) else {}
    deviation = [abs(per[c] - offsets[c]) for c in per]
    report["channels"] = {"listed_seen": len(per), "listed_in_table": len(offsets),
                          "max_abs_median_minus_offset_ps": float(max(deviation)) if deviation else None}
    once = counts["applied once"]
    other = sum(v for k, v in counts.items() if k != "applied once")
    report["passed"] = bool(
        once > 0 and once >= 0.99 * (once + other) and report["unlisted_channel_hits"]["changed"] == 0
        and report["energy"]["unlisted_changed"] == 0 and within >= 0.99 * report["energy"]["listed_matched"]
        and dt_bad == 0
        and (not deviation or max(deviation) <= tolerance_ps))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.cornell.skew roundtrip", description="Matched-event check of a native PETsys time-offset conversion.")
    parser.add_argument("uncorrected", type=Path, help="LDAT converted without the table")
    parser.add_argument("corrected", type=Path, help="LDAT of the same RAWF converted with the table")
    parser.add_argument("table", type=Path, help="native time-offset TSV used for the corrected conversion")
    parser.add_argument("--records", type=int, default=500_000, help="corrected records compared (default 500000)")
    parser.add_argument("--clock-ps", type=float, default=5000.0,
                        help="acquisition clock period in ps for the energy model (default 5000, 200 MHz)")
    args = parser.parse_args(argv)
    offsets, skipped, duplicates = read_native_table(args.table)
    if skipped or duplicates:
        print(f"table has skipped lines {skipped[:5]} / duplicate channels {duplicates[:5]}", file=sys.stderr)
        return 2
    corrected, n_b = read_hits(args.corrected, max_records=args.records)
    restored_end = int((corrected["timestamp"] + np.array([offsets.get(int(c), 0.0) for c in corrected["channel"]])).max())
    uncorrected, n_a = read_hits(args.uncorrected, stop_after_ps=restored_end)
    report = compare(uncorrected, corrected, offsets, clock_ps=args.clock_ps)
    report["files"] = {"uncorrected": str(args.uncorrected), "corrected": str(args.corrected),
                       "table": str(args.table), "records_read": {"uncorrected": n_a, "corrected": n_b}}
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1

