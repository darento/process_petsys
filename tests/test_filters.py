"""src.filters on small synthetic events (spec 007 FR-9).

Hits follow the read_compact contract: ``(timestamp, energy, channel ID)``. The channel
maps mimic map_factory's output: channel -> list of ChannelType, channel -> (SM, mM).
"""

import numpy as np
import pytest

from src.filters import (filter_channel_list, filter_coincidence, filter_max_sm, filter_min_ch,
                         filter_ROI, filter_single_mM, filter_specific_mm, filter_total_energy)
from helpers import SYNTH_CHTYPE as CHTYPE, SYNTH_SM_MM as SM_MM


def hits(*channels, energy=10.0, t0=1000):
    """One hit per channel ID, timestamps t0, t0 + 1, ..."""
    return [(t0 + i, energy, ch) for i, ch in enumerate(channels)]


@pytest.mark.fr("007-FR-9")
@pytest.mark.parametrize("total, expected", [(10.0, False), (10.01, True), (50.0, True),
                                             (99.99, True), (100.0, False)])
def test_total_energy_open_default_bounds(total, expected):
    assert filter_total_energy(total) is expected


@pytest.mark.fr("007-FR-9")
def test_total_energy_custom_bounds():
    assert filter_total_energy(400.0, 350.0, 650.0)
    assert not filter_total_energy(350.0, 350.0, 650.0)
    assert not filter_total_energy(650.0, 350.0, 650.0)


@pytest.mark.fr("007-FR-9")
def test_min_ch_counts_energy_channels():
    det = hits(0, 8, 4, 5)  # energy-only 0 and dual 8 are energy channels; 4, 5 are not
    assert filter_min_ch(det, 2, CHTYPE, sum_rows_cols=False)
    assert not filter_min_ch(det, 3, CHTYPE, sum_rows_cols=False)


@pytest.mark.fr("007-FR-9")
def test_min_ch_summed_needs_fewer_energy_channels_than_hits():
    with_time = hits(0, 1, 4)
    energy_only = hits(0, 1)
    assert filter_min_ch(with_time, 2, CHTYPE, sum_rows_cols=True)
    assert not filter_min_ch(with_time, 3, CHTYPE, sum_rows_cols=True)
    assert not filter_min_ch(energy_only, 2, CHTYPE, sum_rows_cols=True)
    assert filter_min_ch(energy_only, 2, CHTYPE, sum_rows_cols=False)


@pytest.mark.fr("007-FR-9")
@pytest.mark.parametrize("channels, expected", [((0, 1, 4), True), ((0, 2), False), ((0, 8), False),
                                                ((9,), True)])
def test_single_mM(channels, expected):
    assert filter_single_mM(hits(*channels), SM_MM) is expected


@pytest.mark.fr("007-FR-9")
def test_max_sm_counts_both_detectors():
    det1, det2 = hits(0, 1), hits(8)  # SM 0 and SM 1
    assert not filter_max_sm(det1, det2, 1, SM_MM)
    assert filter_max_sm(det1, det2, 2, SM_MM)
    assert filter_max_sm(det1, hits(4), 1, SM_MM)


@pytest.mark.fr("007-FR-9", "bug-filter-max-sm-minimodules")
def test_max_sm_counts_supermodules_not_minimodules():
    det1, det2 = hits(0), hits(2)  # SM 0 minimodules 0 and 1: one supermodule
    assert filter_max_sm(det1, det2, 1, SM_MM)


@pytest.mark.fr("007-FR-9")
def test_specific_mm_in_either_detector():
    det1, det2 = hits(0), hits(8)
    assert filter_specific_mm(det1, det2, 0, 0, SM_MM)
    assert filter_specific_mm(det1, det2, 1, 0, SM_MM)
    assert not filter_specific_mm(det1, det2, 0, 1, SM_MM)


@pytest.mark.fr("007-FR-9", "bug-filter-channel-list")
@pytest.mark.parametrize("det1, det2, expected", [
    (hits(0, 1, t0=1000), hits(9, t0=1000), True),   # det1 channels valid, timestamps not
    (hits(5, t0=0), hits(6, t0=1), False),           # timestamps 0, 1 valid, channels not
    (hits(5, t0=1000), hits(2, t0=1000), True),      # only det2 valid
], ids=["det1-valid", "timestamps-not-channels", "det2-valid"])
def test_channel_list_checks_channel_ids(det1, det2, expected):
    assert filter_channel_list(det1, det2, np.array([0, 1, 2])) is expected


@pytest.mark.fr("007-FR-9")
@pytest.mark.parametrize("x, y, expected", [(0.0, 0.0, True), (-5.0, 0.0, False), (5.0, 0.0, False),
                                            (0.0, -2.0, False), (0.0, 2.0, False), (4.9, -1.9, True)])
def test_roi_open_bounds(x, y, expected):
    assert bool(filter_ROI(x, y, (-5.0, 5.0), (-2.0, 2.0))) is expected


@pytest.mark.fr("007-FR-9")
def test_coincidence_uses_highest_energy_time_channels():
    det1 = [(100, 5.0, 4), (130, 9.0, 5), (0, 50.0, 0)]  # energy channel 0 is ignored for timing
    det2 = [(200, 7.0, 6), (500, 3.0, 7)]
    inside, t1, t2 = filter_coincidence(det1, det2, CHTYPE, 70.5)
    assert inside
    assert (t1, t2) == ((130, 9.0, 5), (200, 7.0, 6))
    assert not filter_coincidence(det1, det2, CHTYPE, 70)[0]  # |130 - 200| = 70 is outside
