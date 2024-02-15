from .utils import (
    get_electronics_nums,
    get_absolute_id,
    get_maxEnergy_sm_mM,
    get_num_eng_channels,
    get_max_en_channel,
)
from .mapping_generator import ChannelType

det_list = [
    [1033333333333333333, 11.0, 1],
    [1555555555555555555, 33.0, 2],
    [4666666666666666666, 15.0, 3],
    [7444444444444444444, 28.0, 4],
    [1333333333333333333, 21.0, 5],
    [4757575577777777777, 5.0, 6],
    [5555555555555555557, 5.5, 7],
]
chtype_map = {
    1: [ChannelType.TIME, ChannelType.ENERGY],
    2: [ChannelType.TIME, ChannelType.ENERGY],
    3: [ChannelType.TIME, ChannelType.ENERGY],
    4: [ChannelType.TIME, ChannelType.ENERGY],
    5: [ChannelType.TIME, ChannelType.ENERGY],
    6: [ChannelType.TIME, ChannelType.ENERGY],
    7: [ChannelType.TIME, ChannelType.ENERGY],
}

sm_mM_map = {
    1: (1, 2),
    2: (3, 4),
    3: (5, 6),
    4: (7, 8),
    5: (9, 10),
    6: (11, 12),
    7: (13, 14),
}


def test_get_electronics_nums():
    assert get_electronics_nums(131072) == (1, 0, 0, 0)
    assert get_electronics_nums(4096) == (0, 1, 0, 0)
    assert get_electronics_nums(64) == (0, 0, 1, 0)
    assert get_electronics_nums(1) == (0, 0, 0, 1)
    assert get_electronics_nums(131072 + 4096 + 64 + 1) == (1, 1, 1, 1)


def test_absolute_id():
    assert get_absolute_id(0, 0, 0, 0) == 0
    assert get_absolute_id(1, 0, 0, 0) == 131072
    assert get_absolute_id(0, 1, 0, 0) == 4096
    assert get_absolute_id(0, 0, 1, 0) == 64
    assert get_absolute_id(0, 0, 0, 1) == 1
    assert get_absolute_id(1, 1, 1, 1) == 131072 + 4096 + 64 + 1


def test_get_maxEnergy_sm_mM():
    assert get_maxEnergy_sm_mM(det_list, sm_mM_map, chtype_map) == (
        [[1555555555555555555, 33.0, 2]],
        33.0,
    )
    assert not get_maxEnergy_sm_mM(det_list, sm_mM_map, chtype_map) == (
        [[7444444444444444444, 28.0, 4]],
        28.0,
    )


def test_get_num_eng_channels():
    for key, value_list in chtype_map.items():
        for channel_type in value_list:
            print(
                f"Key: {key}, ChannelType: {channel_type}, Value: {channel_type.value}"
            )

    assert get_num_eng_channels(det_list, chtype_map) == 7


def test_get_max_en_channel():
    assert get_max_en_channel(det_list, chtype_map) == ([1555555555555555555, 33.0, 2])
    assert not get_max_en_channel(det_list, chtype_map) == (
        [7444444444444444444, 28.0, 4]
    )
