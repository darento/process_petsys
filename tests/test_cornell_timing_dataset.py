"""Known raw records and disjoint time partitions (spec 008 T1b1)."""
import hashlib
import json
from pathlib import Path
import struct

import pytest

from src.cornell.timing_dataset import channel_lookup, freeze_holdout, ledger_inputs, scan_file

pytestmark = pytest.mark.fr("008-FR-1", "008-FR-6", "008-FR-16")


def pair(a, b):
    return struct.pack("2B", len(a), len(b)) + b"".join(struct.pack("<qfi", *hit) for hit in a + b)


def raw_records():
    return (pair([(2**60 + 1, 0.5, 11), (2**60 + 2, 1.5, 12)], [(2**60 + 3, 2.0, 12)]) +
            pair([(2**60 + 101, 0.25, 12)], [(2**60 + 106, 3.0, 11)]))


@pytest.fixture
def lookup():
    return channel_lookup({11: (14, 0), 12: (0, 1)}, [(14, 0)])


def ledger(tmp_path, contents):
    artifacts = []
    for i, data in enumerate(contents):
        name = f"run7_coincCompact_{i:08d}.ldat"
        (tmp_path / name).write_bytes(data)
        artifacts.append({"path": f"/remote/{name}", "kind": "ldat", "cleanup": "retained",
                          "size_bytes": len(data),
                          "input_descriptor": {"format": "compact", "population": "coincidence"}})
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"status": "succeeded", "attempts": [
        {"stage_id": "conversion", "status": "succeeded", "exit_code": 0, "artifacts": artifacts}]}))
    return path


def test_split_identity_time_coverage_and_cut_counts(tmp_path, lookup):
    data = raw_records()
    manifest = ledger(tmp_path, [data])
    (tmp_path / "run6_coincCompact_00000000.ldat").write_bytes(data)
    inputs = ledger_inputs(manifest, tmp_path, "run7_coincCompact_")
    assert len(inputs) == 1 and inputs[0]["index"] == 0
    report = scan_file(inputs[0]["path"], lookup, chunk_bytes=35, full_hash=True)
    assert report["physical_coincidences"] == 2 and report["raw_hits"] == 5
    assert report["processed_bytes"] == len(data) == report["bytes_read"]
    assert report["full_scan"] is True
    assert report["raw_channel_hits"] == {"11": 2, "12": 3}
    assert report["known_unavailable_raw_hits"] == 2
    assert report["timestamp_min_ps"] == 2**60 + 1
    assert report["timestamp_max_ps"] == 2**60 + 106
    assert report["anchor_max_ps"] == 2**60 + 101
    assert report["unknown_channel_hits"] == report["nonfinite_hit_energy"] == report["zero_hit_sides"] == 0
    assert report["analysis_cuts"] == "none" and report["accepted_calibration_coincidences"] is None
    assert report["local_full_sha256"] == hashlib.sha256(data).hexdigest()
    inputs[0]["path"].write_bytes(data + b"\0")
    with pytest.raises(ValueError, match="size-mismatched"):
        ledger_inputs(manifest, tmp_path, "run7_coincCompact_")


def test_holdout_time_blocks_do_not_overlap(tmp_path, lookup):
    scans = []
    for index in range(16):
        path = tmp_path / f"{index}.ldat"
        path.write_bytes(pair([(index * 1000, 1.0, 12)], [(index * 1000 + 1, 1.0, 11)]) +
                         pair([(index * 1000 + 100, 1.0, 12)], [(index * 1000 + 101, 1.0, 11)]))
        report = scan_file(path, lookup, chunk_bytes=37)
        report["index"] = index
        scans.append(report)
    holdout = freeze_holdout(scans, guard_ps=2)
    assert holdout["qualified"] is True
    roles = {role: {p["index"] for p in holdout["partitions"] if p["role"] == role}
             for role in ("train", "validation", "test")}
    assert len(roles["train"]) == 11
    assert roles["test"] == {3, 8, 14} and roles["validation"] == {5, 11}
    assert not roles["train"] & roles["test"] and not roles["train"] & roles["validation"]
    scans[1]["anchor_min_ps"] = 0
    assert "overlapping" in " ".join(freeze_holdout(scans, guard_ps=2)["reasons"])
    with pytest.raises(ValueError, match="overlapping"):
        freeze_holdout(scans, validation=(1,), test=(1,))
    assert freeze_holdout([], validation=(), test=())["qualified"] is False


@pytest.mark.parametrize("chunk_bytes", [2, 7, 17, 35, 64, 1024])
def test_chunk_boundaries_and_truncation(tmp_path, lookup, chunk_bytes):
    path = tmp_path / "fixture.ldat"
    path.write_bytes(raw_records())
    report = scan_file(path, lookup, chunk_bytes=chunk_bytes)
    assert report["physical_coincidences"] == 2 and report["raw_channel_hits"] == {"11": 2, "12": 3}
    for data in (b"\x01", raw_records()[:-1], raw_records() + b"\x01\x01"):
        path.write_bytes(data)
        with pytest.raises(ValueError, match="truncated"):
            scan_file(path, lookup, chunk_bytes=chunk_bytes)


def test_prefix_never_qualifies_as_full_coverage(tmp_path, lookup):
    path = tmp_path / "fixture.ldat"
    path.write_bytes(raw_records())
    report = scan_file(path, lookup, max_records=1, chunk_bytes=1024)
    assert report["physical_coincidences"] == 1 and report["full_scan"] is False
    assert report["processed_bytes"] == 50 and report["bytes_read"] == 84
    report["index"] = 0
    holdout = freeze_holdout([report], validation=(), test=(), guard_ps=0)
    assert holdout["qualified"] is False and "prefix" in " ".join(holdout["reasons"])
    assert scan_file(path, lookup, max_records=20)["full_scan"] is False


def test_empty_unknown_nonfinite_and_count_overflow(tmp_path, lookup):
    path = tmp_path / "fixture.ldat"
    path.write_bytes(b"")
    empty = scan_file(path, lookup)
    assert empty["physical_coincidences"] == 0 and empty["anchor_min_ps"] is None
    path.write_bytes(pair([], []) + pair([(1, float("nan"), -1)], [(2, 1.0, 999)]))
    invalid = scan_file(path, lookup)
    assert invalid["zero_hit_sides"] == 2 and invalid["unknown_channel_hits"] == 2
    assert invalid["nonfinite_hit_energy"] == 1
    path.write_bytes(pair([(1, 1.0, 12)], [(2, 1.0, 11)]) * 65_540)
    counts = scan_file(path, lookup, chunk_bytes=64 * 1024)
    assert counts["raw_channel_hits"] == {"11": 65_540, "12": 65_540}


def test_rejects_duplicate_ledger_and_changed_input(tmp_path, lookup):
    manifest = ledger(tmp_path, [raw_records()])
    content = json.loads(manifest.read_text())
    content["attempts"][0]["artifacts"] *= 2
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="duplicate"):
        ledger_inputs(manifest, tmp_path, "run7_coincCompact_")
    path = tmp_path / "fixture.ldat"
    path.write_bytes(raw_records())
    changed = False
    def progress(*_):
        nonlocal changed
        if not changed:
            with path.open("ab") as stream:
                stream.write(pair([(200, 1.0, 11)], [(201, 1.0, 12)]))
            changed = True
    with pytest.raises(ValueError, match="changed"):
        scan_file(path, lookup, chunk_bytes=35, progress=progress)


@pytest.mark.real_data
def test_run7_raw_prefix_against_frozen_identity(real_data_file):
    from src.mapping_generator import map_factory
    repo = Path(__file__).resolve().parents[1]
    baseline = json.loads((repo / "tests/data/baselines/cornell_run7_prefix_008.json").read_text())
    _, modules, _, _ = map_factory(str(repo / "maps/cornell_map_full_system.yaml"))
    mapped = channel_lookup(modules, [(14, mm) for mm in (0, 1, 4, 5)])
    report = scan_file(real_data_file(baseline["input"]), mapped, max_records=10_000, chunk_bytes=4 * 1024 * 1024)
    assert report["physical_coincidences"] == 10_000
    assert report["processed_bytes"] == baseline["processed_bytes"] and report["raw_hits"] == baseline["raw_hits"]
    assert report["known_unavailable_raw_hits"] == report["unknown_channel_hits"] == report["nonfinite_hit_energy"] == 0
    assert report["full_scan"] is False
