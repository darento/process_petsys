from typing import Callable
from src.detector_features import calculate_total_energy
from src.mapping_generator import ChannelType


def get_electronics_nums(channel_id: int) -> tuple[int, int, int, int]:
    """
    Calculates the electronics numbers: portID, slaveID, chipID, channelID based on the given channel id.

    Parameters:
    channel_id (int): The channel id to calculate the electronics numbers from.

    Returns:
    tuple[int, int, int, int]: A tuple containing the portID, slaveID, chipID, and channelID.
    """
    portID = channel_id // 131072
    slaveID = (channel_id % 131072) // 4096
    chipID = ((channel_id % 131072) % 4096) // 64
    channelID = channel_id % 64
    return portID, slaveID, chipID, channelID


def get_absolute_id(portID: int, slaveID: int, chipID: int, channelID: int) -> int:
    """
    Calculates the absolute channel id from the given electronics numbers.

    Parameters:
    portID (int): The port id.
    slaveID (int): The slave id.
    chipID (int): The chip id.
    channelID (int): The channel id.

    Returns:
    int: The absolute channel id calculated from the electronics numbers.
    """
    return 131072 * portID + 4096 * slaveID + 64 * chipID + channelID


def get_maxEnergy_sm_mM(det_list: list[list], sm_mM_map: dict, chtype_map: dict) -> int:
    """
    Get the mini module with highest energy

    Parameters:
    det_event (list): The event data.
    sm_mM_map (dict): The mapping of the channels to the mod and mM.

    Returns:
    list[list]: The event data for the maximum energy miniModule and sm.
    float: The maximum energy in the event.
    """
    # First we need to find if there is more than 1 mM in the event
    mM_list = list(set([sm_mM_map[ch[2]] for ch in det_list]))
    if len(mM_list) == 1:
        # If there is only 1 mM, we return it with the corresponding sm and the energy
        return (
            det_list,
            calculate_total_energy(det_list, chtype_map),
        )
    else:
        # If there is more than 1 mM, we need to find the one with the maximum energy
        max_energy = 0
        max_mm_list = 0

        for sm_mM in mM_list:
            eng_ch = list(
                filter(
                    lambda x: ChannelType.ENERGY in chtype_map[x[2]],
                    [ch for ch in det_list if sm_mM_map[ch[2]] == sm_mM],
                )
            )
            energy = sum([ch[1] for ch in eng_ch])
            if energy > max_energy:
                max_energy = energy
                max_mm_list = [ch for ch in det_list if sm_mM_map[ch[2]] == sm_mM]

        return max_mm_list, max_energy


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


def get_max_en_channel(
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
