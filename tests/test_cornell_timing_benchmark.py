"""Bounded benchmark contracts, isolated from Manager (spec 008 T1a)."""

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from src.cornell.skew.benchmark import (Prefix, benchmark_prefix, decode_compiled,
                                        decode_reference, read_prefix)


def records():
    # Int64 timestamps exceed float64's exact range; negative energies remain raw.
    pairs = [([(2**60 + 1, -0.25, 81), (2**60 + 9, 1.5, 19)],
              [(2**60 + 3, 2.25, 99)]), ([], []),
             ([(2**60 + 10, 3.0, 1)], [(2**60 + 14, 4.0, 2)])]
    return b"".join(struct.pack("2B", len(a), len(b)) +
                    b"".join(struct.pack("<qfi", *hit) for hit in a + b)
                    for a, b in pairs)


@pytest.mark.fr("008-FR-1", "008-FR-14")
def test_matching_workloads_and_stage_accounting(tmp_path):
    data = records()
    source = tmp_path / "source.ldat"
    source.write_bytes(data)
    prefix = read_prefix(source, max_records=2, max_bytes=len(data))
    report = benchmark_prefix(prefix, repeats=2)
    assert report["input"]["physical_coincidences"] == 2
    assert report["input"]["raw_hits"] == 3
    assert report["input"]["processed_bytes"] == 52
    assert report["input"]["bytes_read"] == len(data)
    assert report["input"]["source_size_bytes"] == len(data)
    assert report["input"]["prefix_limited"] is True
    assert report["input"]["prefix_sha256"] == hashlib.sha256(data[:52]).hexdigest()
    assert report["input"]["accepted_analysis_coincidences"] is None
    for name in ("python", "compiled"):
        assert len(report["samples"][name]) == 2
        for sample in report["samples"][name]:
            assert sample["measured_seconds"] == (sample["decode_and_pack_seconds"] +
                                                   sample["parity_audit_seconds"])
            assert sample["decode_and_pack_seconds"] > 0
    assert report["warm_decode_and_pack_speedup"] > 0  # no machine-specific speed assertion


@pytest.mark.fr("008-FR-1", "008-FR-14")
def test_exact_hit_and_side_boundaries():
    for decode in (decode_reference, decode_compiled):
        timestamp, energy, channel, first, n1, n2, end = decode(records(), 3)
        np.testing.assert_array_equal(timestamp, [2**60 + k for k in (1, 9, 3, 10, 14)])
        np.testing.assert_array_equal(energy, [-0.25, 1.5, 2.25, 3.0, 4.0])
        np.testing.assert_array_equal(channel, [81, 19, 99, 1, 2])
        np.testing.assert_array_equal(first, [0, 3, 3])
        np.testing.assert_array_equal(n1, [2, 0, 1])
        np.testing.assert_array_equal(n2, [1, 0, 1])
        assert timestamp[0] - timestamp[2] == -2
        assert end == 86
        assert timestamp.dtype == np.int64
        assert energy.dtype == np.float32
        assert channel.dtype == np.int32


@pytest.mark.fr("008-FR-1", "008-FR-14")
def test_prefix_budget_and_truncation(tmp_path):
    source = tmp_path / "source.ldat"
    source.write_bytes(records())
    with pytest.raises(ValueError, match="byte budget exhausted"):
        read_prefix(source, max_records=3, max_bytes=52)
    with pytest.raises(ValueError, match="byte budget exhausted"):
        read_prefix(source, max_records=3, max_bytes=10)
    prefix = read_prefix(source, max_records=20, max_bytes=1024)
    assert prefix.records == 3 and prefix.data == records()
    for raw, message in ((b"\x01", "truncated record header"),
                         (b"\x01\x00\x01", "truncated hit record")):
        source.write_bytes(raw)
        with pytest.raises(ValueError, match=message):
            read_prefix(source)
        for decode in (decode_reference, decode_compiled):
            with pytest.raises(ValueError, match="truncated"):
                decode(raw, 2)
    source.write_bytes(b"")
    empty = read_prefix(source)
    assert empty.records == 0 and empty.data == b""
    assert benchmark_prefix(empty, repeats=1)["input"]["raw_hits"] == 0


@pytest.mark.fr("008-FR-1")
def test_rejects_false_prefix_metadata_and_unbounded_requests(tmp_path):
    source = tmp_path / "source.ldat"
    source.write_bytes(records())
    for kwargs in ({"max_records": 0}, {"max_bytes": 0}, {"max_bytes": 65 * 1024 * 1024},
                   {"max_records": 250_001}):
        with pytest.raises(ValueError, match="limits"):
            read_prefix(source, **kwargs)
    with pytest.raises(ValueError, match="metadata"):
        benchmark_prefix(Prefix(records(), 1, 86, 86, 0.0, "synthetic"), repeats=1)


@pytest.mark.real_data
@pytest.mark.fr("008-FR-1", "008-FR-14")
def test_run7_prefix_raw_decoder_parity(real_data_file):
    baseline = json.loads((Path(__file__).parent / "data/baselines/cornell_run7_prefix_008.json")
                          .read_text(encoding="utf-8"))
    source = real_data_file(baseline["input"])
    prefix = read_prefix(source, max_records=baseline["max_records"])
    report = benchmark_prefix(prefix, repeats=2)
    assert report["input"]["physical_coincidences"] == baseline["max_records"]
    for key in ("processed_bytes", "raw_hits", "prefix_sha256", "analysis_cuts"):
        assert report["input"][key] == baseline[key]
    assert report["input"]["prefix_limited"] is True
