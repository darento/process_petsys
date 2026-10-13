"""Bounded compact-LDAT coincidence record batches (spec 008 T3, FR-4).

A compact record is ``uint8 n1, uint8 n2`` followed by ``n1 + n2`` hits of
``<int64 timestamp (ps), float32 energy, int32 channel>``: detector 1 hits,
then detector 2 hits. Records are physical coincidences, never singles.

``LdatStream`` reads fixed-size blocks with ordinary file reads (no memory
map), carries a trailing partial record into the next block and rejects it at
end of file. Memory depends on the block and batch sizes, not on file length;
consumers must reduce each batch before requesting the next. Timestamps stay
raw int64 so differences can be taken exactly before any float conversion.

The header/hit kernels are shared with ``src.ldat_inspector.fastread``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numba
import numpy as np

HIT_DTYPE = np.dtype([("timestamp", "<i8"), ("energy", "<f4"), ("channel", "<i4")])
HEADER_BYTES = 2
MAX_RECORD_BYTES = HEADER_BYTES + HIT_DTYPE.itemsize * 2 * 255
DEFAULT_BLOCK_BYTES = 64 * 1024 * 1024
DEFAULT_BATCH_RECORDS = 250_000
TRUNCATED_HEADER, TRUNCATED_HIT = 1, 2


class LdatFormatError(ValueError):
    """Malformed compact LDAT input; ``offset`` is the failing record's byte offset."""

    def __init__(self, message, offset):
        super().__init__(f"{message} at byte {offset}")
        self.offset = offset


@numba.njit(cache=True)
def scan_headers(buf, start, limit):
    """Record offsets and side sizes from byte ``start``; stops at ``limit`` pairs.

    Returns (offsets, n1, n2, count, end, truncation code).
    """
    offsets = np.empty(limit, np.int64)
    n1 = np.empty(limit, np.int64)
    n2 = np.empty(limit, np.int64)
    pos, count, size = start, 0, buf.size
    while count < limit:
        if pos >= size:
            return offsets, n1, n2, count, pos, 0
        if pos + 2 > size:
            return offsets, n1, n2, count, pos, 1
        a, b = np.int64(buf[pos]), np.int64(buf[pos + 1])
        end = pos + 2 + 16 * (a + b)
        if end > size:
            return offsets, n1, n2, count, pos, 2
        offsets[count], n1[count], n2[count] = pos, a, b
        count += 1
        pos = end
    return offsets, n1, n2, count, pos, 0


def decode_hits(buf, offsets, n1, n2, count):
    """Aligned hit arrays for ``count`` records, and each record's first hit index."""
    start, stop = offsets[0], offsets[count - 1] + 2 + 16 * (n1[count - 1] + n2[count - 1])
    keep = np.ones(stop - start, np.bool_)
    header = offsets[:count] - start
    keep[header] = False
    keep[header + 1] = False
    raw = np.ascontiguousarray(buf[start:stop][keep])
    hits = raw.view(HIT_DTYPE)
    first = np.zeros(count, np.int64)
    np.cumsum(n1[:count - 1] + n2[:count - 1], out=first[1:])
    return hits["timestamp"].copy(), hits["energy"].copy(), hits["channel"].copy(), first


@dataclass(frozen=True)
class LdatBatch:
    """Consecutive complete records; hit arrays are owned copies.

    Record ``k`` has detector 1 hits ``first_hit[k] : first_hit[k] + n1[k]``
    and detector 2 hits up to ``first_hit[k] + n1[k] + n2[k]``.
    """
    first_record: int
    byte_offset: int
    n1: np.ndarray
    n2: np.ndarray
    first_hit: np.ndarray
    timestamp: np.ndarray
    energy: np.ndarray
    channel: np.ndarray

    @property
    def count(self):
        return len(self.n1)


class LdatStream:
    """Iterate ``LdatBatch`` objects from one file; reusable counters describe progress.

    ``max_records`` reads an explicit prefix (``complete`` stays False unless
    the file ends exactly there). ``on_block(raw)`` sees every block read, e.g.
    to hash the input. Iteration closes the file on completion, error or early
    ``close()`` of the generator.
    """

    def __init__(self, path, *, block_bytes=DEFAULT_BLOCK_BYTES,
                 batch_records=DEFAULT_BATCH_RECORDS, max_records=None, on_block=None):
        if isinstance(block_bytes, bool) or not isinstance(block_bytes, int) or block_bytes < HEADER_BYTES:
            raise ValueError("block_bytes must be an integer >= 2")
        if isinstance(batch_records, bool) or not isinstance(batch_records, int) or batch_records < 1:
            raise ValueError("batch_records must be a positive integer")
        if max_records is not None and (isinstance(max_records, bool) or not isinstance(max_records, int)
                                        or max_records < 1):
            raise ValueError("max_records must be None or a positive integer")
        self.path = Path(path)
        self.block_bytes, self.batch_records, self.max_records = block_bytes, batch_records, max_records
        self.on_block = on_block
        self.size_bytes = None
        self.records = self.hits = self.processed_bytes = self.bytes_read = 0
        self.read_seconds = self.decode_seconds = 0.0
        self.complete = False

    def __iter__(self):
        if self.size_bytes is not None:
            raise RuntimeError("an LdatStream is iterated once")
        self.size_bytes = self.path.stat().st_size
        limit = self.max_records
        carry = b""
        with self.path.open("rb") as stream:
            while limit is None or self.records < limit:
                started = time.perf_counter()
                raw = stream.read(self.block_bytes)
                self.read_seconds += time.perf_counter() - started
                self.bytes_read += len(raw)
                if self.on_block is not None and raw:
                    self.on_block(raw)
                if not raw:
                    if carry:
                        kind = "truncated LDAT pair header" if len(carry) < HEADER_BYTES else "truncated LDAT hit record"
                        raise LdatFormatError(kind, self.processed_bytes)
                    self.complete = True
                    return
                data = carry + raw
                buf = np.frombuffer(data, dtype=np.uint8)
                position = 0
                while position < len(buf) and (limit is None or self.records < limit):
                    want = self.batch_records if limit is None else min(self.batch_records, limit - self.records)
                    started = time.perf_counter()
                    offsets, n1, n2, count, end, truncated = scan_headers(buf, position, want)
                    if not count:
                        self.decode_seconds += time.perf_counter() - started
                        break
                    timestamp, energy, channel, first = decode_hits(buf, offsets, n1, n2, count)
                    self.decode_seconds += time.perf_counter() - started
                    batch = LdatBatch(self.records, self.processed_bytes, n1[:count].copy(), n2[:count].copy(),
                                      first, timestamp, energy, channel)
                    self.records += int(count)
                    self.hits += len(channel)
                    self.processed_bytes += int(end - position)
                    position = int(end)
                    yield batch
                    if truncated:
                        break
                carry = data[position:]
                del buf, data
                if limit is not None and self.records >= limit:
                    break   # unread bytes are later records, not a partial one
                if len(carry) >= MAX_RECORD_BYTES:
                    raise LdatFormatError("partial record exceeds LDAT record capacity", self.processed_bytes)
        self.complete = self.processed_bytes == self.size_bytes
