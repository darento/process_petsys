"""src/read_compact.py on synthetic compact LDAT files (spec 006 FR-7).

Records are coincidences of two detectors, each a list of (timestamp, energy a.u.,
channel ID) hits; there is no singles population. Energies are float32-exact.
"""

import pytest

from helpers import write_ldat
from src.read_compact import read_binary_file

PAIRS = [
    ([(1_000, 12.5, 3)], [(1_010, 20.25, 70)]),
    ([(2**40, 0.5, 0), (2**40 + 7, 30.0, 63), (2**40 + 9, 1.0, 64)],
     [(2**40 + 3, 8.0, 7 * 131072 + 31 * 4096 + 63 * 64 + 63)]),
    ([(5_000, 4.0, 4095), (5_001, 4.0, 4096)], [(5_002, 255.0, 131071), (5_003, 2.0, 131072)]),
]
HEADER_BYTES = 2
HIT_BYTES = 16


def _as_lists(records):
    return [([tuple(h) for h in det1], [tuple(h) for h in det2]) for det1, det2 in records]


@pytest.mark.fr("006-FR-7")
def test_round_trip_returns_written_pairs(tmp_path):
    path = write_ldat(tmp_path / "pairs.ldat", PAIRS)
    assert _as_lists(read_binary_file(str(path))) == _as_lists(PAIRS)


@pytest.mark.fr("006-FR-7")
def test_empty_file_yields_no_pairs(tmp_path):
    path = write_ldat(tmp_path / "empty.ldat", [])
    assert list(read_binary_file(str(path))) == []


@pytest.mark.fr("006-FR-7")
def test_energy_filter_drops_only_low_hits(tmp_path):
    path = write_ldat(tmp_path / "pairs.ldat", PAIRS)
    threshold = 4.0
    expected = [([h for h in det1 if h[1] >= threshold], [h for h in det2 if h[1] >= threshold])
                for det1, det2 in PAIRS]
    got = _as_lists(read_binary_file(str(path), en_filter=threshold))
    assert got == _as_lists(expected)
    assert len(got) == len(PAIRS)  # a record stays a record; only its hits are filtered


def _complete_bytes(pairs):
    return sum(HEADER_BYTES + HIT_BYTES * (len(d1) + len(d2)) for d1, d2 in pairs)


# Cut points inside the last record: in its header, in detector 1 hits, in detector 2 hits.
TRUNCATIONS = {
    "header": 1,
    "det1 hits": HEADER_BYTES + HIT_BYTES - 5,
    "det2 hits": HEADER_BYTES + 3 * HIT_BYTES + 4,
}


@pytest.mark.fr("006-FR-7", "bug-read-compact-truncation")
@pytest.mark.xfail(strict=True, reason="bug-read-compact-truncation: partial record yields ([], []) "
                                       "or raises struct.error; fix pending")
@pytest.mark.parametrize("cut", TRUNCATIONS.values(), ids=TRUNCATIONS.keys())
def test_truncated_last_record_yields_no_pair(tmp_path, cut):
    complete = PAIRS[:1]
    full = write_ldat(tmp_path / "full.ldat", complete + PAIRS[1:2]).read_bytes()
    path = tmp_path / "truncated.ldat"
    path.write_bytes(full[:_complete_bytes(complete) + cut])

    got = []
    with pytest.raises(ValueError):
        for record in read_binary_file(str(path)):
            got.append(record)
    assert _as_lists(got) == _as_lists(complete)
