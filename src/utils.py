from typing import Callable
from src.mapping_generator import ChannelType


def get_num_eng_channels(det_list: list[list], chtype_map: dict) -> int:
    """
    Counts the number of energy channels in a list.

    Parameters:
        det_list (list): The list of detectors.
        chtype_map (dict): A mapping from detector names to channel types.

    Returns:
        int: The number of energy channels in det_list.
    """
    return sum(ChannelType.ENERGY in chtype_map[x[2]] for x in det_list)


def select_max_energy(
    det_list: list[list], chtype_map: dict, chtype: ChannelType = None
) -> list:
    """
    Returns a function that selects the channel with the highest deposit from a list of channels.

    Parameters:
        det_list : List of hits with [tspt, energy, chid]
        chtype_map (dict): A mapping from channel names to channel types.
        chtype (ChannelType, optional): The type of channel to be compared. If None, all channels are considered. Defaults to None.

    Returns:
        list: The channel with the highest deposit.
    """

    def _is_type(hit: list) -> bool:
        """
        Parameters:
            hit (list): The channel to check. [tspt, energy, chid]
        """
        return chtype in chtype_map[hit[2]]

    filt_func = _is_type if chtype else lambda x: True

    try:
        return max(filter(filt_func, det_list), key=lambda y: y[1])
    except ValueError:
        return None
