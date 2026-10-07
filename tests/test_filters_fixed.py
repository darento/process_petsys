"""src.filters_fixed on synthetic fixed-size chunks (spec 007 FR-9).

Chunks use the read_fixed group layout: a hit count and ``hit_limit`` hit slots, empty
slots padded with channel -1, time 0, energy 0. Each vectorized filter must agree with
its src.filters counterpart on the same events.
"""

import numpy as np
import pytest

from src.filters import filter_min_ch, filter_single_mM, filter_total_energy
from src.filters_fixed import (filter_min_ch_vectorized, filter_single_mM_vectorized,
                               filter_total_energy_vectorized)
from src.utils_fixed import create_mm_map_array, get_energy_channel_mask
from helpers import SYNTH_CHTYPE as CHTYPE, SYNTH_SM_MM as SM_MM

HIT_LIMIT = 5
HIT = np.dtype([("time", "i8"), ("energy", "f4"), ("channelID", "i4")])
GROUP = np.dtype([("header", "u1"), ("hits", HIT, (HIT_LIMIT,))])

# (channel, energy) per hit; the empty event has no hits at all.
EVENTS = [
    [(0, 20.0), (1, 30.0), (4, 5.0)],
    [(0, 20.0), (2, 30.0)],
    [(8, 40.0), (9, 50.0)],
    [(0, 3.0), (4, 2.0)],
    [(0, 60.0), (1, 30.0), (2, 10.0), (3, 5.0)],
    [(5, 200.0)],
    [],
]


def det(event):
    return [(1000 + i, e, ch) for i, (ch, e) in enumerate(event)]


def chunk(events):
    out = np.zeros(len(events), dtype=GROUP)
    out["hits"]["channelID"] = -1
    for row, event in enumerate(events):
        out["header"][row] = len(event)
        for slot, (time, energy, ch) in enumerate(det(event)):
            out["hits"][row, slot] = (time, energy, ch)
    return out


EMPTY = len(EVENTS) - 1


@pytest.mark.fr("007-FR-9")
@pytest.mark.parametrize("sum_rows_cols", [False, True])
@pytest.mark.parametrize("min_ch", [1, 2, 3])
def test_min_ch_agrees_with_scalar(min_ch, sum_rows_cols):
    expected = [filter_min_ch(det(ev), min_ch, CHTYPE, sum_rows_cols) for ev in EVENTS]
    from_dict = filter_min_ch_vectorized(chunk(EVENTS), CHTYPE, min_ch, sum_rows_cols)
    from_mask = filter_min_ch_vectorized(chunk(EVENTS), get_energy_channel_mask(CHTYPE), min_ch,
                                         sum_rows_cols)
    assert from_dict.tolist() == expected
    assert from_mask.tolist() == expected


@pytest.mark.fr("007-FR-9")
@pytest.mark.parametrize("bounds", [(10.0, 100.0), (25.0, 90.0), (4.0, 6.0)])
def test_total_energy_agrees_with_scalar(bounds):
    expected = [filter_total_energy(sum(e for _, e in ev), *bounds) for ev in EVENTS]
    assert filter_total_energy_vectorized(chunk(EVENTS), *bounds).tolist() == expected


@pytest.mark.fr("007-FR-9")
def test_single_mM_agrees_with_scalar():
    expected = [filter_single_mM(det(ev), SM_MM) for ev in EVENTS]
    assert filter_single_mM_vectorized(chunk(EVENTS), SM_MM).tolist() == expected
    assert filter_single_mM_vectorized(chunk(EVENTS), create_mm_map_array(SM_MM)).tolist() == expected


@pytest.mark.fr("007-FR-9")
def test_padding_is_ignored():
    # Channel 0 is an energy channel in minimodule (0, 0): padding read as channel 0 would count.
    one_hit = chunk([[(4, 5.0)]])
    assert not filter_min_ch_vectorized(one_hit, CHTYPE, 1)[0]
    assert filter_single_mM_vectorized(chunk([[(2, 5.0)]]), SM_MM)[0]
    assert filter_total_energy_vectorized(chunk([[(0, 50.0)]]))[0]


@pytest.mark.fr("007-FR-9")
def test_empty_event_rejected():
    empty = chunk(EVENTS)[[EMPTY]]
    assert not filter_min_ch_vectorized(empty, CHTYPE, 1)[0]
    assert not filter_min_ch_vectorized(empty, CHTYPE, 1, sum_rows_cols=True)[0]
    assert not filter_single_mM_vectorized(empty, SM_MM)[0]
    assert not filter_total_energy_vectorized(empty)[0]
