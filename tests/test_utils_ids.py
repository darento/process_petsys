"""src.utils absolute channel ID <-> (portID, slaveID, chipID, channelID) (spec 006 FR-7).

Layout: 131072 per port, 4096 per slave, 64 per chip, 64 channels per chip.
"""

import random

import pytest

from src.utils import get_absolute_id, get_electronics_nums

BOUNDARIES = [0, 63, 64, 4095, 4096, 131071, 131072,
              get_absolute_id(7, 31, 63, 63)]


@pytest.mark.fr("006-FR-7")
@pytest.mark.parametrize("channel", BOUNDARIES)
def test_boundary_ids_round_trip(channel):
    assert get_absolute_id(*get_electronics_nums(channel)) == channel


@pytest.mark.fr("006-FR-7")
def test_boundary_decomposition():
    assert get_electronics_nums(63) == (0, 0, 0, 63)
    assert get_electronics_nums(64) == (0, 0, 1, 0)
    assert get_electronics_nums(4096) == (0, 1, 0, 0)
    assert get_electronics_nums(131072) == (1, 0, 0, 0)
    assert get_electronics_nums(get_absolute_id(7, 31, 63, 63)) == (7, 31, 63, 63)


@pytest.mark.fr("006-FR-7")
def test_seeded_sample_round_trips():
    rng = random.Random(1)
    for channel in (rng.randrange(8 * 131072) for _ in range(1000)):
        port, slave, chip, ch = get_electronics_nums(channel)
        assert 0 <= slave < 32 and 0 <= chip < 64 and 0 <= ch < 64, channel
        assert get_absolute_id(port, slave, chip, ch) == channel


@pytest.mark.fr("006-FR-7")
def test_every_chip_channel_round_trips():
    for port in (0, 1):
        for slave in (0, 1):
            for chip in range(64):
                for ch in range(64):
                    nums = (port, slave, chip, ch)
                    assert get_electronics_nums(get_absolute_id(*nums)) == nums
