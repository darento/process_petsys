"""Isolated, bounded raw-decoder benchmark (008-FR-1, 008-FR-14).

This measures decoding plus packing aligned numeric arrays. It applies no hit,
energy, geometry or coincidence cuts and does not benchmark a skew calibration.
Existing Inspector kernels are reused read-only; no production API changes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import struct
import time

import numpy as np

MAX_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 250_000


@dataclass(frozen=True)
class Prefix:
    data: bytes
    records: int
    source_size: int
    bytes_read: int
    read_seconds: float
    path: str


def _boundaries(data, limit):
    position = 0
    sides = []
    while position < len(data) and len(sides) < limit:
        if position + 2 > len(data):
            raise ValueError(f"truncated record header at byte {position}")
        a, b = struct.unpack_from("2B", data, position)
        end = position + 2 + 16 * (a + b)
        if end > len(data):
            raise ValueError(f"truncated hit record at byte {position}")
        sides.append((position, a, b))
        position = end
    return sides, position


def read_prefix(path, *, max_records=10_000, max_bytes=4 * 1024 * 1024):
    """Read within hard byte/record caps; never silently shorten a requested limit.

    A short complete file is valid. If a byte budget cannot contain the requested
    records, raise explicitly rather than mislabeling that boundary as file EOF.
    """
    if not 1 <= max_records <= MAX_RECORDS or not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError("record/byte limits must be positive and within benchmark caps")
    path = Path(path)
    before = path.stat()
    started = time.perf_counter()
    with path.open("rb") as stream:
        raw = stream.read(max_bytes)
    read_seconds = time.perf_counter() - started
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("input changed during prefix read")
    try:
        boundaries, end = _boundaries(raw, max_records)
    except ValueError as exc:
        if len(raw) < before.st_size:
            raise ValueError("byte budget exhausted before requested records; increase max_bytes") from exc
        raise
    if len(boundaries) < max_records and len(raw) < before.st_size:
        raise ValueError("byte budget exhausted before requested records; increase max_bytes")
    return Prefix(raw[:end], len(boundaries), before.st_size, len(raw),
                  read_seconds, str(path.resolve()))


def decode_reference(data, limit):
    """Current reader's Python struct-unpacking pattern, without analysis cuts."""
    boundaries, end = _boundaries(data, limit)
    timestamps, energies, channels, first, n1, n2 = [], [], [], [], [], []
    for position, a, b in boundaries:
        first.append(len(timestamps))
        n1.append(a)
        n2.append(b)
        for hit in range(a + b):
            timestamp, energy, channel = struct.unpack_from("<qfi", data, position + 2 + 16 * hit)
            timestamps.append(timestamp)
            energies.append(energy)
            channels.append(channel)
    return (np.asarray(timestamps, dtype=np.int64), np.asarray(energies, dtype=np.float32),
            np.asarray(channels, dtype=np.int32), np.asarray(first, dtype=np.int64),
            np.asarray(n1, dtype=np.int64), np.asarray(n2, dtype=np.int64), end)


def decode_compiled(data, limit):
    from src.ldat_inspector.fastread import _hits, _scan_headers

    buf = np.frombuffer(data, dtype=np.uint8)
    offsets, n1, n2, count, end, truncated = _scan_headers(buf, 0, limit)
    if truncated:
        raise ValueError(f"truncated record at byte {end}")
    if count:
        timestamp, energy, channel, first = _hits(buf, offsets, n1, n2, count)
    else:
        timestamp, energy, channel, first = (np.empty(0, dtype=dtype)
                                             for dtype in (np.int64, np.float32, np.int32, np.int64))
    return timestamp, energy, channel, first, n1[:count], n2[:count], int(end)


def _verify(reference, compiled):
    for expected, observed in zip(reference[:-1], compiled[:-1]):
        np.testing.assert_array_equal(observed, expected)
    if reference[-1] != compiled[-1]:
        raise ValueError("decoded byte ranges differ")


def benchmark_prefix(prefix, *, repeats=5):
    if not 1 <= repeats <= 20 or not 0 <= prefix.records <= MAX_RECORDS:
        raise ValueError("invalid benchmark repetitions or record count")
    if len(prefix.data) > MAX_BYTES:
        raise ValueError("prefix exceeds benchmark byte cap")
    # Include lazy fastread imports and JIT/cache load in the first compiled call.
    first_calls, results = {}, {}
    for name, decoder in (("python", decode_reference), ("compiled", decode_compiled)):
        started = time.perf_counter()
        results[name] = decoder(prefix.data, max(prefix.records, 1))
        first_calls[name] = time.perf_counter() - started
    _verify(results["python"], results["compiled"])
    reference = results["python"]
    if len(reference[3]) != prefix.records or reference[-1] != len(prefix.data):
        raise ValueError("prefix metadata does not match complete record bytes")
    samples = {"python": [], "compiled": []}
    for iteration in range(repeats):
        order = ("python", "compiled") if iteration % 2 == 0 else ("compiled", "python")
        for name in order:
            decoder = decode_reference if name == "python" else decode_compiled
            started = time.perf_counter()
            decoded = decoder(prefix.data, max(prefix.records, 1))
            decode_seconds = time.perf_counter() - started
            audit_started = time.perf_counter()
            _verify(reference, decoded)
            audit_seconds = time.perf_counter() - audit_started
            samples[name].append({"decode_and_pack_seconds": decode_seconds,
                                  "parity_audit_seconds": audit_seconds,
                                  "measured_seconds": decode_seconds + audit_seconds})
    medians = {name: float(np.median([row["decode_and_pack_seconds"] for row in rows]))
               for name, rows in samples.items()}
    from src.ldat_inspector import fastread
    source_files = (Path(__file__), Path(fastread.__file__))
    return {
        "schema_version": 1,
        "scope": "bounded raw decoding plus aligned-array packing; no analysis cuts",
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "numba": fastread.numba.__version__,
        "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        "input": {"path": prefix.path, "source_size_bytes": prefix.source_size,
                  "processed_bytes": len(prefix.data), "bytes_read": prefix.bytes_read,
                  "physical_coincidences": prefix.records, "raw_hits": len(reference[0]),
                  "prefix_limited": len(prefix.data) < prefix.source_size,
                  "prefix_sha256": hashlib.sha256(prefix.data).hexdigest(),
                  "analysis_cuts": "none", "accepted_analysis_coincidences": None},
        "parity": "all timestamps, energies, channels, side sizes and boundaries match",
        "input_read_seconds": prefix.read_seconds,
        "first_call_seconds": first_calls, "samples": samples,
        "median_decode_and_pack_seconds": medians,
        "warm_decode_and_pack_speedup": medians["python"] / medians["compiled"],
        "numeric_output_bytes": sum(array.nbytes for array in reference[:-1]),
        "limitations": ["OS cache state uncontrolled; read time is not raw disk throughput",
                        "first compiled call includes imports and JIT/cache load, not process startup",
                        "Python baseline packs numeric arrays in addition to legacy-style unpacking",
                        "peak process/concurrent memory not measured",
                        "features, calibration, geometry, cache writes and solver not measured",
                        "no full-file integrity, coverage, CTR or calibration sufficiency claim"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-records", type=int, default=10_000)
    parser.add_argument("--max-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("output must differ from input")
    # Exclusive creation preserves previous benchmark reports and all user files.
    with args.output.open("x", encoding="utf-8") as output:
        prefix = read_prefix(args.input, max_records=args.max_records, max_bytes=args.max_bytes)
        json.dump(benchmark_prefix(prefix, repeats=args.repeats), output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
