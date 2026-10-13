"""Bounded compact-LDAT record stream (spec 008 T3, FR-4).

A pure ``struct`` decoder is the oracle; the stream must reproduce every
record, side boundary and raw field for any block/batch size, keep int64
timestamps exact, diagnose empty/truncated input and hold memory independent
of file length.
"""
import hashlib
import os
import struct
import tracemalloc

import numpy as np
import pytest

from src.ldat_stream import LdatFormatError, LdatStream

pytestmark = pytest.mark.fr("008-FR-4")
HIT = struct.Struct("<qfi")


def records(count, seed=7):
    rng = np.random.default_rng(seed)
    big = 2 ** 62
    out = []
    for k in range(count):
        n1, n2 = (0, 0) if k == 5 else (int(rng.integers(0, 6)), int(rng.integers(0, 6)))
        hits = []
        for _ in range(n1 + n2):
            t = int(rng.choice([rng.integers(-big, big), big - 1 - k, -big + k, rng.integers(0, 10 ** 15)]))
            e = float(rng.choice([rng.normal(10, 5), np.nan, np.inf, -0.0]))
            hits.append((t, e, int(rng.integers(-2 ** 31, 2 ** 31))))
        out.append((n1, n2, hits))
    return out


def encode(recs):
    return b"".join(struct.pack("<BB", n1, n2) + b"".join(HIT.pack(*h) for h in hits)
                    for n1, n2, hits in recs)


def reference(data):
    """Independent decoder: (n1, n2, timestamps, energies, channels, byte offsets)."""
    pos, n1s, n2s, ts, es, cs, offsets = 0, [], [], [], [], [], []
    while pos < len(data):
        offsets.append(pos)
        a, b = data[pos], data[pos + 1]
        n1s.append(a)
        n2s.append(b)
        for t, e, c in HIT.iter_unpack(data[pos + 2:pos + 2 + 16 * (a + b)]):
            ts.append(t)
            es.append(e)
            cs.append(c)
        pos += 2 + 16 * (a + b)
    return n1s, n2s, ts, np.asarray(es, np.float32), cs, offsets


def collect(stream):
    batches = list(stream)
    joined = {name: np.concatenate([getattr(b, name) for b in batches]) if batches else np.empty(0)
              for name in ("n1", "n2", "timestamp", "energy", "channel")}
    return batches, joined


@pytest.mark.parametrize("block_bytes", [2, 7, 18, 35, 1000, 1 << 20])
@pytest.mark.parametrize("batch_records", [1, 3, 1000])
def test_reference_parity_variable_records(tmp_path, block_bytes, batch_records):
    data = encode(records(40))
    path = tmp_path / "mixed.ldat"
    path.write_bytes(data)
    n1, n2, ts, es, cs, offsets = reference(data)
    stream = LdatStream(path, block_bytes=block_bytes, batch_records=batch_records)
    batches, joined = collect(stream)
    assert joined["n1"].tolist() == n1 and joined["n2"].tolist() == n2
    assert joined["timestamp"].dtype == np.int64 and joined["timestamp"].tolist() == ts
    assert joined["energy"].dtype == np.float32
    np.testing.assert_array_equal(joined["energy"].view(np.uint32), es.view(np.uint32))
    assert joined["channel"].dtype == np.int32 and joined["channel"].tolist() == cs
    assert all(b.count <= batch_records for b in batches)
    # Batch identities and record side boundaries are absolute and exact.
    record = 0
    for batch in batches:
        assert batch.first_record == record and batch.byte_offset == offsets[record]
        for k in range(batch.count):
            start = batch.first_hit[k]
            det1 = batch.timestamp[start:start + batch.n1[k]]
            det2 = batch.timestamp[start + batch.n1[k]:start + batch.n1[k] + batch.n2[k]]
            expected = ts[sum(n1[:record + k]) + sum(n2[:record + k]):]
            assert det1.tolist() == expected[:n1[record + k]]
            assert det2.tolist() == expected[n1[record + k]:n1[record + k] + n2[record + k]]
        record += batch.count
    assert (stream.records, stream.hits, stream.processed_bytes) == (40, len(ts), len(data))
    assert stream.complete and stream.bytes_read == len(data)


def test_int64_timestamp_difference_before_float(tmp_path):
    big = 2 ** 62
    path = tmp_path / "large.ldat"
    path.write_bytes(encode([(1, 1, [(big + 1, 511.0, 3), (big - 1, 511.0, 4)])]))
    (batch,) = list(LdatStream(path))
    t1, t2 = batch.timestamp[0], batch.timestamp[1]
    # float64 spacing at 2**62 is 1024 ps: subtract the raw int64 values first.
    assert t1 - t2 == 2 and float(t1) - float(t2) == 0.0


def test_truncated_and_empty_input(tmp_path):
    empty = tmp_path / "empty.ldat"
    empty.write_bytes(b"")
    stream = LdatStream(empty)
    assert list(stream) == [] and stream.complete and stream.records == 0
    with pytest.raises(RuntimeError):
        list(stream)
    data = encode(records(6))
    for tail, label in ((b"\x01", "pair header"), (b"\x01\x00" + b"\x00" * 5, "hit record")):
        path = tmp_path / f"trunc_{len(tail)}.ldat"
        path.write_bytes(data + tail)
        for block in (2, 9, 4096):
            stream = LdatStream(path, block_bytes=block)
            seen = 0
            with pytest.raises(LdatFormatError, match=f"truncated LDAT {label} at byte {len(data)}") as info:
                for batch in stream:
                    seen += batch.count
            # Complete records before the defect are delivered and counted, never the partial one.
            assert seen == stream.records == 6 and info.value.offset == len(data)
            assert not stream.complete
    full = tmp_path / "full.ldat"
    full.write_bytes(data)
    prefix = LdatStream(full, max_records=3, block_bytes=7)
    assert sum(b.count for b in prefix) == 3 and not prefix.complete
    assert prefix.processed_bytes == len(encode(records(6)[:3]))
    whole = LdatStream(full, max_records=6)
    assert sum(b.count for b in whole) == 6 and whole.complete
    # A prefix ending early in a large block leaves unread complete records,
    # not a partial record (found on the run7 real-data prefix check).
    many = tmp_path / "many.ldat"
    many.write_bytes(encode([(2, 2, [(k, 1.0, k) for k in range(4)])] * 2000))
    early = LdatStream(many, max_records=10, block_bytes=1 << 20)
    assert sum(b.count for b in early) == 10 and early.processed_bytes == 10 * 66
    assert not early.complete
    for bad in ({"block_bytes": 1}, {"block_bytes": True}, {"batch_records": 0},
                {"max_records": 0}, {"max_records": 2.0}):
        with pytest.raises(ValueError):
            LdatStream(full, **bad)


def test_chunk_boundaries_and_map_release(tmp_path):
    data = encode(records(25, seed=3))
    path = tmp_path / "blocks.ldat"
    path.write_bytes(data)
    for block in (2, 3, 17, 66, 67, 4096):
        seen = []
        stream = LdatStream(path, block_bytes=block, batch_records=4, on_block=seen.append)
        _, joined = collect(stream)
        # Every byte is offered to on_block once, in order (e.g. for hashing).
        assert hashlib.sha256(b"".join(seen)).digest() == hashlib.sha256(data).digest()
        assert joined["n1"].size == 25 and stream.complete
    # No mapping/handle survives an error or an explicitly closed iterator.
    broken = tmp_path / "broken.ldat"
    broken.write_bytes(data + b"\x02")
    with pytest.raises(LdatFormatError):
        list(LdatStream(broken, block_bytes=5))
    os.replace(broken, tmp_path / "moved.ldat")
    os.remove(tmp_path / "moved.ldat")
    iterator = iter(LdatStream(path, block_bytes=5, batch_records=1))
    next(iterator)
    iterator.close()
    os.remove(path)


def test_memory_independent_of_file_length(tmp_path):
    def peak(count):
        path = tmp_path / f"n{count}.ldat"
        one = encode([(2, 2, [(10 * k, 1.0, k) for k in range(4)])])
        path.write_bytes(one * count)
        stream = LdatStream(path, block_bytes=64 * 1024, batch_records=512)
        tracemalloc.start()
        total = 0
        for batch in stream:
            total += int(batch.channel.sum())   # reduce, never retain batches
        _, high = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert stream.records == count and total == 6 * count and stream.complete
        return high

    small, large = peak(20_000), peak(80_000)
    assert large <= 1.25 * small + 64 * 1024
    assert large < 2 * 1024 * 1024   # far below the 5.3 MB file
