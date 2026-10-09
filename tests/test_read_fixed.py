"""src/read_fixed.py on synthetic fixed LDAT files (spec 007 T16, from petsys_manager_reference_check).

Fixed layout: an int32 hit limit H, then per record one count byte per side and H 16-byte
(timestamp, energy a.u., channel ID) slots per side, unused slots padded with channel -1.
Coincidence records hold two detectors, group records one; there is no singles population.
The encoder below is an independent wire oracle, not generated from the reader's dtype.
"""

import struct

import pytest

from src.read_fixed import read_fixed_file_numpy

HIT = struct.Struct("<qfi")
PADDING = HIT.pack(0, 0.0, -1)
HIT_LIMIT = 16
PAIRS = [
    ([(100, 12.0, 4), (101, 5.0, 3)], [(90, 25.0, 121), (91, 25.0, 122), (92, 0.125, 123)]),
    ([(2**40, 0.5, 0)], [(2**40 + 7, 30.0, 131071)]),
    ([(5_000 + i, 1.0 + i, 4096 + i) for i in range(HIT_LIMIT)], [(5_100, 255.0, 131072)]),
]
GROUPS = [PAIRS[0][0], PAIRS[1][1], PAIRS[2][0], [(7, 2.5, 64)]]


def write_fixed(path, records, *, group=False):
    with open(path, "wb") as out:
        out.write(struct.pack("<i", HIT_LIMIT))
        for record in records:
            sides = [record] if group else list(record)
            out.write(bytes(len(side) for side in sides))
            for side in sides:
                for hit in side:
                    out.write(HIT.pack(*hit))
                out.write(PADDING * (HIT_LIMIT - len(side)))
    return path


def decode(path, *, group=False, batch_size=2):
    records = []
    for chunk in read_fixed_file_numpy(str(path), batch_size=batch_size, group_events=group):
        for row in chunk:
            sides = ([(row["hits"], int(row["header"]))] if group else
                     [(row["side1"], int(row["header"][0])), (row["side2"], int(row["header"][1]))])
            decoded = [[(int(h["time"]), float(h["energy"]), int(h["channelID"])) for h in hits[:count]]
                       for hits, count in sides]
            records.append(decoded[0] if group else tuple(decoded))
    return records


@pytest.mark.fr("003-FR-16")
@pytest.mark.parametrize("batch_size", [1, 2, 100])
def test_coincidence_round_trip(tmp_path, batch_size):
    path = write_fixed(tmp_path / "pairs_fixed.ldat", PAIRS)
    assert path.stat().st_size == 4 + len(PAIRS) * (2 + 2 * HIT_LIMIT * 16)
    assert decode(path, batch_size=batch_size) == PAIRS


@pytest.mark.fr("003-FR-16")
def test_group_round_trip(tmp_path):
    path = write_fixed(tmp_path / "groups_fixed.ldat", GROUPS, group=True)
    assert path.stat().st_size == 4 + len(GROUPS) * (1 + HIT_LIMIT * 16)
    assert decode(path, group=True) == GROUPS


@pytest.mark.fr("003-FR-16")
def test_unused_slots_are_padding_not_hits(tmp_path):
    path = write_fixed(tmp_path / "pairs_fixed.ldat", PAIRS[:1])
    (chunk,) = read_fixed_file_numpy(str(path), group_events=False)
    assert list(chunk["header"][0]) == [2, 3]
    assert set(chunk["side1"]["channelID"][0][2:]) == {-1}
    assert set(chunk["side2"]["channelID"][0][3:]) == {-1}


@pytest.mark.fr("003-FR-16")
def test_header_only_file_yields_no_records(tmp_path):
    path = write_fixed(tmp_path / "empty.ldat", [])
    assert decode(path) == []
    (tmp_path / "short.ldat").write_bytes(b"\x10\x00")
    assert decode(tmp_path / "short.ldat") == []
