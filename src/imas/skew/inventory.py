"""Raw IMAS skew-split inventory: records and hits per TIME channel, SM and minimodule (spec 011 T3, FR-6).

No analysis cuts: a hit counts if it is in the file. TIME channels with no
hits are listed as unobserved, so down regions appear from the data. With
``--legacy-skew`` the unobserved set is compared with the legacy LUT's
``9999`` (no delay) channels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from src.imas.skew.setup import ImasSetup
from src.timing.dataset import MAX_CHUNK_BYTES, channel_lookup, scan_file

LEGACY_NO_DELAY = 9999.0
LOW_HIT_FRACTION = 0.1        # low statistics: below this fraction of the median TIME-channel hits


def read_legacy_skew(path):
    """``ch skew`` lines as {channel: skew}; ``9999`` kept as is."""
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            ch, skew = line.split()
            values[int(ch)] = float(skew)
    return values


def summarize(setup, scans, *, legacy=None):
    """Per-TIME-channel, SM and minimodule hit totals over ``scans`` (``scan_file`` results)."""
    times = setup.time_channels
    hits = np.zeros(len(times), np.int64)
    position = {int(ch): i for i, ch in enumerate(times)}
    for scan in scans:
        for ch, n in scan["raw_channel_hits"].items():
            if int(ch) in position:
                hits[position[int(ch)]] += n
    sm = np.array([setup.sm_mM_map[int(ch)][0] for ch in times])
    mm = np.array([setup.sm_mM_map[int(ch)][1] for ch in times])
    observed = hits > 0
    median = float(np.median(hits[observed])) if observed.any() else 0.0
    low = observed & (hits < LOW_HIT_FRACTION * median)
    n_sm = setup.n_azimuth * len(setup.ring_z)
    per_sm = np.bincount(sm, weights=hits, minlength=n_sm).astype(np.int64)
    module_hits = {}
    for s, m, n in zip(sm.tolist(), mm.tolist(), hits.tolist()):
        module_hits[s, m] = module_hits.get((s, m), 0) + n
    dark_modules = sorted([s, m] for (s, m), n in module_hits.items() if n == 0)
    summary = {
        "time_channels": int(len(times)), "observed": int(observed.sum()), "unobserved": int((~observed).sum()),
        "low_statistics": int(low.sum()),
        "low_statistics_rule": f"hits > 0 and < {LOW_HIT_FRACTION} x median observed TIME-channel hits ({median:.0f})",
        "unobserved_channels": times[~observed].tolist(), "low_statistics_channels": times[low].tolist(),
        "dark_minimodules": dark_modules,
        "hits_per_sm": {str(s): int(n) for s, n in enumerate(per_sm)},
        "dark_sms": [s for s in range(n_sm) if per_sm[s] == 0],
    }
    if legacy is not None:
        legacy_none = {ch for ch, value in legacy.items() if value == LEGACY_NO_DELAY}
        unobserved = set(times[~observed].tolist())
        summary["legacy_comparison"] = {
            "legacy_entries": len(legacy), "legacy_no_delay": len(legacy_none),
            "unobserved_and_legacy_no_delay": len(unobserved & legacy_none),
            "unobserved_only": sorted(unobserved - legacy_none),
            "legacy_no_delay_but_observed": len(legacy_none - unobserved),
            "legacy_missing_time_channels": len(set(times.tolist()) - set(legacy)),
        }
    return summary, hits


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew inventory", description=__doc__)
    parser.add_argument("splits", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=Path("configs/imas_2DAQ.yaml"))
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--chunk-bytes", type=int, default=MAX_CHUNK_BYTES)
    parser.add_argument("--legacy-skew", type=Path, help="legacy ch/skew file to compare unobserved channels")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output directory already exists; use a new name")
    setup = ImasSetup.from_files(args.config)
    lookup = channel_lookup(setup.sm_mM_map)
    scans = []
    started = time.perf_counter()
    for path in args.splits:
        last = [time.perf_counter()]

        def progress(done, total, records):
            now = time.perf_counter()
            if now - last[0] >= 10:
                print(f"{path.name}: {done / total:.0%}, {records:,} records", flush=True)
                last[0] = now
        scan = scan_file(path, lookup, chunk_bytes=args.chunk_bytes, max_records=args.max_records, progress=progress)
        scan["read_rate_mb_s"] = scan["bytes_read"] / 1e6 / scan["wall_seconds"] if scan["wall_seconds"] else None
        print(f"{path.name}: {scan['physical_coincidences']:,} records, {scan['wall_seconds']:.0f} s, "
              f"{scan['read_rate_mb_s']:.0f} MB/s", flush=True)
        scans.append(scan)
    legacy = read_legacy_skew(args.legacy_skew) if args.legacy_skew else None
    summary, hits = summarize(setup, scans, legacy=legacy)
    report = {"schema_version": 1, "scanner": "imas", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": ("full" if args.max_records is None else f"prefix of {args.max_records} records")
              + " raw-hit inventory; no analysis cuts",
              "provenance": setup.provenance | ({"legacy_skew": str(args.legacy_skew)} if legacy else {}),
              "splits": [{k: v for k, v in scan.items() if k != "raw_channel_hits"} for scan in scans],
              "total_records": sum(scan["physical_coincidences"] for scan in scans),
              "total_bytes": sum(scan["processed_bytes"] for scan in scans),
              "wall_seconds": time.perf_counter() - started, "summary": summary}
    args.out.mkdir(parents=True)
    (args.out / "inventory.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    times = setup.time_channels
    lines = ["channel,sm,minimodule,slab,hits"] + [
        f"{ch},{setup.sm_mM_map[ch][0]},{setup.sm_mM_map[ch][1]},{setup.local_map[ch][2]},{n}"
        for ch, n in zip(times.tolist(), hits.tolist())]
    (args.out / "time_channel_hits.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{summary['observed']}/{summary['time_channels']} TIME channels observed, "
          f"{summary['unobserved']} unobserved, {summary['low_statistics']} low statistics -> {args.out}")
    return 0
