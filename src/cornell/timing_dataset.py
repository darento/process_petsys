"""Bounded raw Cornell timing dataset inventory; no analysis cuts (spec 008 T1b1).

Existing fastread kernels are reused without modifying Inspector or Manager.
Raw hit presence is distinct from accepted calibration observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numba
import numpy as np
import yaml

from src.ldat_inspector.fastread import _hits, _scan_headers
from src.mapping_generator import map_factory

MAX_CHUNK_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 2 + 16 * 510
BATCH_RECORDS = 250_000


def ledger_inputs(ledger_path, directory, prefix):
    """Select the last successful compact conversion, never glob co-located runs."""
    ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    if ledger.get("status") != "succeeded":
        raise ValueError("conversion ledger did not succeed")
    attempts = [a for a in ledger.get("attempts", []) if a.get("stage_id") == "conversion"
                and a.get("status") == "succeeded" and a.get("exit_code") == 0]
    if not attempts:
        raise ValueError("no successful conversion attempt")
    names = {}
    pattern = re.compile(re.escape(prefix) + r"(\d{8})\.ldat")
    for artifact in attempts[-1].get("artifacts", []):
        if artifact.get("kind") != "ldat" or artifact.get("cleanup") != "retained":
            continue
        name = artifact["path"].replace("\\", "/").rsplit("/", 1)[-1]
        match = pattern.fullmatch(name)
        descriptor = artifact.get("input_descriptor") or {}
        if not match or descriptor.get("format") != "compact" or descriptor.get("population") != "coincidence":
            raise ValueError(f"unexpected conversion output: {name}")
        if name in names:
            raise ValueError(f"duplicate conversion output: {name}")
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size != artifact.get("size_bytes"):
            raise ValueError(f"missing or size-mismatched conversion output: {name}")
        names[name] = {"index": int(match[1]), "path": path, "size_bytes": artifact["size_bytes"]}
    ordered = sorted(names.values(), key=lambda item: item["index"])
    if not ordered or [item["index"] for item in ordered] != list(range(len(ordered))):
        raise ValueError("conversion split indices are empty or not consecutive from zero")
    return ordered


def channel_lookup(channel_modules, unavailable_regions=()):
    channels = np.asarray(sorted(channel_modules), dtype=np.int32)
    if not len(channels) or channels[0] < 0 or int(channels[-1]) >= 16_000_000:
        raise ValueError("channel IDs exceed bounded lookup capacity")
    dense = np.full(int(channels[-1]) + 1, -1, dtype=np.int32)
    dense[channels] = np.arange(len(channels), dtype=np.int32)
    unavailable = set(tuple(region) for region in unavailable_regions)
    down = np.asarray([tuple(channel_modules[int(ch)]) in unavailable for ch in channels], dtype=np.bool_)
    return channels, dense, down


@numba.njit(cache=True)
def _count_hits(timestamp, energy, channel, dense, counts, down):
    unknown = nonfinite = unavailable = 0
    for i in range(len(channel)):
        if not np.isfinite(energy[i]):
            nonfinite += 1
        ch = channel[i]
        if ch < 0 or ch >= len(dense) or dense[ch] < 0:
            unknown += 1
        else:
            index = dense[ch]
            counts[index] += np.uint64(1)
            unavailable += int(down[index])
    return unknown, nonfinite, unavailable


def scan_file(path, lookup, *, chunk_bytes=MAX_CHUNK_BYTES, max_records=None, full_hash=False, progress=None):
    """Scan bounded blocks; a trailing partial record is carried, then rejected at EOF."""
    if not 2 <= chunk_bytes <= MAX_CHUNK_BYTES or (max_records is not None and max_records < 1):
        raise ValueError("invalid chunk or prefix limit")
    path = Path(path)
    before = path.stat()
    identity = (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev)
    channels, dense, down = lookup
    counts = np.zeros(len(channels), dtype=np.uint64)
    records = hits = processed = bytes_read = zero_sides = unknown = nonfinite = unavailable = 0
    time_min = time_max = anchor_min = anchor_max = None
    read_seconds = decode_seconds = 0.0
    digest = hashlib.sha256() if full_hash and max_records is None else None
    started = time.perf_counter()
    carry = b""
    with path.open("rb") as stream:
        while max_records is None or records < max_records:
            read_started = time.perf_counter()
            raw = stream.read(chunk_bytes)
            read_seconds += time.perf_counter() - read_started
            bytes_read += len(raw)
            if digest is not None:
                digest.update(raw)
            if not raw:
                if carry:
                    raise ValueError(f"truncated LDAT record at byte {processed}")
                break
            data = carry + raw
            buf = np.frombuffer(data, dtype=np.uint8)
            position = 0
            stage_started = time.perf_counter()
            while position < len(buf) and (max_records is None or records < max_records):
                want = BATCH_RECORDS if max_records is None else min(BATCH_RECORDS, max_records - records)
                offsets, n1, n2, count, end, truncated = _scan_headers(buf, position, want)
                if count:
                    timestamp, energy, channel, first = _hits(buf, offsets, n1, n2, count)
                    new_unknown, new_nonfinite, new_down = _count_hits(timestamp, energy, channel, dense, counts, down)
                    unknown += new_unknown
                    nonfinite += new_nonfinite
                    unavailable += new_down
                    records += int(count)
                    hits += len(channel)
                    zero_sides += int(np.count_nonzero(n1[:count] == 0) + np.count_nonzero(n2[:count] == 0))
                    if len(timestamp):
                        lo, hi = int(timestamp.min()), int(timestamp.max())
                        time_min = lo if time_min is None else min(time_min, lo)
                        time_max = hi if time_max is None else max(time_max, hi)
                        anchors = timestamp[first[(n1[:count] + n2[:count]) > 0]]
                        lo, hi = int(anchors.min()), int(anchors.max())
                        anchor_min = lo if anchor_min is None else min(anchor_min, lo)
                        anchor_max = hi if anchor_max is None else max(anchor_max, hi)
                processed += int(end - position)
                position = int(end)
                if truncated:
                    break
            carry = data[position:]
            decode_seconds += time.perf_counter() - stage_started
            if max_records is not None and records >= max_records:
                break
            if len(carry) >= MAX_RECORD_BYTES:
                raise ValueError("partial record exceeds LDAT header capacity")
            if progress:
                progress(processed, before.st_size, records)
    after = path.stat()
    if identity != (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev):
        raise ValueError("input changed during full/prefix scan")
    complete = max_records is None and processed == before.st_size
    return {
        "name": path.name, "identity": {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
        "full_scan": complete, "requested_max_records": max_records,
        "processed_bytes": processed, "bytes_read": bytes_read, "physical_coincidences": records,
        "raw_hits": hits, "zero_hit_sides": zero_sides, "unknown_channel_hits": unknown,
        "nonfinite_hit_energy": nonfinite, "known_unavailable_raw_hits": unavailable,
        "raw_channel_hits": {str(int(ch)): int(number) for ch, number in zip(channels, counts) if number},
        "timestamp_min_ps": time_min, "timestamp_max_ps": time_max,
        "anchor_min_ps": anchor_min, "anchor_max_ps": anchor_max,
        "timestamp_anchor": "first hit of detector 1; detector 2 if detector 1 is empty",
        "local_full_sha256": digest.hexdigest() if digest is not None and complete else None,
        "read_seconds": read_seconds, "decode_and_raw_statistics_seconds": decode_seconds,
        "wall_seconds": time.perf_counter() - started,
        "analysis_cuts": "none", "accepted_calibration_coincidences": None,
    }


def freeze_holdout(scans, *, validation=(5, 11), test=(3, 8, 14), guard_ps=1_000_000):
    indices = [scan["index"] for scan in scans]
    if len(set(indices)) != len(indices) or len(set(validation)) != len(validation) or len(set(test)) != len(test):
        raise ValueError("duplicate split assignment")
    if set(validation) & set(test) or not (set(validation) | set(test)) <= set(indices) or guard_ps < 0:
        raise ValueError("overlapping or unknown split assignment / invalid guard")
    partitions, reasons = [], [] if scans else ["no scanned splits"]
    ordered = sorted(scans, key=lambda row: row["index"])
    for a, b in zip(ordered, ordered[1:]):
        if a["anchor_min_ps"] is not None and b["anchor_min_ps"] is not None and a["anchor_min_ps"] >= b["anchor_min_ps"]:
            reasons.append(f"splits {a['index']}/{b['index']}: non-chronological split indices")
    for scan in ordered:
        index = scan["index"]
        role = "test" if index in test else "validation" if index in validation else "train"
        lo, hi = scan["anchor_min_ps"], scan["anchor_max_ps"]
        valid_interval = lo is not None and hi is not None and lo + guard_ps <= hi - guard_ps
        if not scan["full_scan"]:
            reasons.append(f"split {index}: prefix/incomplete scan")
        if not valid_interval:
            reasons.append(f"split {index}: missing or too-short timestamp interval")
        if scan["unknown_channel_hits"] or scan["nonfinite_hit_energy"] or scan["zero_hit_sides"]:
            reasons.append(f"split {index}: raw input quality requires qualification")
        partitions.append({"index": index, "name": scan["name"], "identity": scan["identity"], "role": role,
                           "anchor_start_ps": lo + guard_ps if valid_interval else None,
                           "anchor_end_ps": hi - guard_ps if valid_interval else None})
    intervals = sorted((p for p in partitions if p["anchor_start_ps"] is not None), key=lambda p: p["anchor_start_ps"])
    for a, b in zip(intervals, intervals[1:]):
        if a["anchor_end_ps"] >= b["anchor_start_ps"]:
            reasons.append(f"splits {a['index']}/{b['index']}: overlapping guarded timestamp intervals")
    return {"qualified": not reasons, "reasons": reasons, "guard_ps_per_file_edge": guard_ps,
            "partitions": partitions, "roles_frozen_before_fitting": True,
            "interpretation": "record-anchor intervals; raw scan counts include the excluded edges"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--chunk-bytes", type=int, default=MAX_CHUNK_BYTES)
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument("--unavailable-region", action="append", default=[], help="acquisition-specific SM:MM")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new name")
    inputs = ledger_inputs(args.ledger, args.directory, args.prefix)
    if len(inputs) != 16:
        parser.error("run7 frozen 11/2/3 assignment requires all 16 ledger outputs")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # Config map paths use repository root, as in the selected Cornell configuration.
    map_path = Path(config["map_file"])
    _, modules, _, _ = map_factory(str(map_path))
    regions = [tuple(int(part) for part in value.split(":")) for value in args.unavailable_region]
    if any(len(region) != 2 or region not in set(modules.values()) for region in regions):
        parser.error("unavailable regions must be mapped SM:MM pairs")
    lookup = channel_lookup(modules, regions)
    kernel_path = Path(_hits.__code__.co_filename)
    provenance_files = (args.ledger, args.config, map_path, Path(__file__), kernel_path)
    report = {"schema_version": 1,
              "scope": ("full" if args.max_records is None else "prefix") + " raw-record inventory; no analysis cuts",
              "created_utc": datetime.now(timezone.utc).isoformat(), "known_unavailable_regions": regions,
              "provenance_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in provenance_files},
              "splits": [], "limitations": ["Local full hashes are optional; source byte identity is unverified",
                                            "Read timing has uncontrolled OS cache state",
                                            "Raw presence is not calibrated acceptance or skew sufficiency",
                                            "TIMESTAMP ps convention follows compact writer contract"]}
    partial = args.output.with_name(args.output.name + ".partial")
    # Exclusive checkpoint keeps incomplete scans visibly separate from completed reports.
    with partial.open("x", encoding="utf-8") as checkpoint:
        for item in inputs:
            last_progress = time.perf_counter()
            def progress(done, total, records):
                nonlocal last_progress
                now = time.perf_counter()
                if now - last_progress >= 5:
                    print(f"split {item['index']:02d}: {done / total:.0%}, {records:,} raw coincidences", flush=True)
                    last_progress = now
            result = scan_file(item["path"], lookup, chunk_bytes=args.chunk_bytes,
                               max_records=args.max_records, full_hash=args.full_hash, progress=progress)
            result["index"] = item["index"]
            report["splits"].append(result)
            checkpoint.seek(0)
            json.dump(report, checkpoint, indent=2)
            checkpoint.truncate()
            checkpoint.flush()
            print(f"split {item['index']:02d} done: {result['physical_coincidences']:,} records, "
                  f"{result['wall_seconds']:.1f}s", flush=True)
        for path in provenance_files:
            if hashlib.sha256(path.read_bytes()).hexdigest() != report["provenance_sha256"][str(path)]:
                raise ValueError(f"provenance input changed during inventory: {path}")
        report["holdout"] = freeze_holdout(report["splits"])
        report["all_full_scans"] = all(row["full_scan"] for row in report["splits"])
        report["total_physical_coincidences"] = sum(row["physical_coincidences"] for row in report["splits"])
        report["total_processed_bytes"] = sum(row["processed_bytes"] for row in report["splits"])
        checkpoint.seek(0)
        json.dump(report, checkpoint, indent=2)
        checkpoint.truncate()
    # Hard-link creation is exclusive and atomic; preserves a concurrently created output.
    import os
    os.link(partial, args.output)
    partial.unlink()


if __name__ == "__main__":
    main()
