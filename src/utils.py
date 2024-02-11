from src.mapping_generator import ChannelType


def get_num_eng_channels(det_list: list[list], chtype_map: dict) -> int:
    """
    Return the number of channels for energy
    measurement in the module data list.
    """
    return sum(ChannelType.ENERGY in chtype_map[x[2]] for x in det_list)
