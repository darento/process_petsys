"""Bounded raw LDAT inventory per channel; no analysis cuts (spec 011 T1, FR-1).

Moved unchanged from ``src.cornell.skew.dataset`` (spec 008 T1b1). Records come
from the shared bounded ``src.ldat_stream`` reader. Raw hit presence is
distinct from accepted calibration observations.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numba
import numpy as np

from src.ldat_stream import LdatStream

MAX_CHUNK_BYTES = 64 * 1024 * 1024
BATCH_RECORDS = 250_000


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
    zero_sides = unknown = nonfinite = unavailable = 0
    time_min = time_max = anchor_min = anchor_max = None
    statistics_seconds = 0.0
    digest = hashlib.sha256() if full_hash and max_records is None else None
    started = time.perf_counter()
    stream = LdatStream(path, block_bytes=chunk_bytes, batch_records=BATCH_RECORDS, max_records=max_records,
                        on_block=digest.update if digest is not None else None)
    for batch in stream:
        stage_started = time.perf_counter()
        timestamp, n1, n2, first = batch.timestamp, batch.n1, batch.n2, batch.first_hit
        new_unknown, new_nonfinite, new_down = _count_hits(timestamp, batch.energy, batch.channel,
                                                           dense, counts, down)
        unknown += new_unknown
        nonfinite += new_nonfinite
        unavailable += new_down
        zero_sides += int(np.count_nonzero(n1 == 0) + np.count_nonzero(n2 == 0))
        if len(timestamp):
            lo, hi = int(timestamp.min()), int(timestamp.max())
            time_min = lo if time_min is None else min(time_min, lo)
            time_max = hi if time_max is None else max(time_max, hi)
            anchors = timestamp[first[(n1 + n2) > 0]]
            lo, hi = int(anchors.min()), int(anchors.max())
            anchor_min = lo if anchor_min is None else min(anchor_min, lo)
            anchor_max = hi if anchor_max is None else max(anchor_max, hi)
        statistics_seconds += time.perf_counter() - stage_started
        if progress:
            progress(stream.processed_bytes, before.st_size, stream.records)
    records, hits, processed, bytes_read = stream.records, stream.hits, stream.processed_bytes, stream.bytes_read
    read_seconds, decode_seconds = stream.read_seconds, stream.decode_seconds + statistics_seconds
    after = path.stat()
    if identity != (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev):
        raise ValueError("input changed during full/prefix scan")
    complete = max_records is None and stream.complete and processed == before.st_size
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
