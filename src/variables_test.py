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

local_dict = {
    1: [0.1, 0.2],
    2: [0.3, 0.4],
    3: [0.5, 0.6],
    4: [0.7, 0.8],
    5: [0.9, 1.0],
    6: [1.1, 1.2],
    7: [1.3, 1.4],
}

det_list_2 = [
    [1033333333333333333, 11.0, 1],
    [1555555555555555555, 33.0, 2],
    [4666666666666666666, 15.0, 3],
    [7444444444444444444, 28.0, 4],
    [1333333333333333333, 21.0, 5],
    [4757575577777777777, 5.0, 6],
    [5555555555555555557, 5.5, 7],
    [2343333333333333333, 23.0, 8],
]
local_coord_dict = {
    1: [1, 1],
    2: [2, 2],
    3: [3, 3],
    4: [4, 4],
    5: [5, 5],
    6: [6, 6],
    7: [7, 7],
    8: [8, 8],
}
