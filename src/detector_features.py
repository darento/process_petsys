import numpy as np


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


def get_maxEnergy_sm_mM(det_event: list, sm_mM_map: dict) -> int:
    """
    Get the maximum energy miniModule and sm in the event.

    Parameters:
    det_event (list): The event data.
    sm_mM_map (dict): The mapping of the channels to the mod and mM.

    Returns:
    int: The maximum energy mM in the event.
    int: The maximum energy sm in the event.
    """
    # First we need to find if there is more than 1 mM in the event
    mM_list = [sm_mM_map[ch[2]][1] for ch in det_event]
    mM_list = list(set(mM_list))
    if len(mM_list) == 1:
        # If there is only 1 mM, we return it with the corresponding sm and the energy
        return (
            mM_list[0],
            sm_mM_map[det_event[0][2]][0],
            sum([ch[1] for ch in det_event]),
        )
    else:
        # If there is more than 1 mM, we need to find the one with the maximum energy
        max_energy = 0
        max_mM = 0
        max_sm = 0
        for mM in mM_list:
            energy = sum([ch[1] for ch in det_event if sm_mM_map[ch[2]][1] == mM])
            if energy > max_energy:
                max_energy = energy
                max_mM = mM
                max_sm = sm_mM_map[det_event[0][2]][0]
        return max_mM, max_sm, energy


def calculate_total_energy(det_event: list) -> float:
    """
    Calculate the total energy of the event.

    Parameters:
    event: The event data.

    Returns:
    float: The total energy of the event.
    """
    return sum([ch[1] for ch in det_event])


def calculate_centroid(
    local_dict: dict, det_event: list, x_rtp: int, y_rtp: int
) -> tuple:
    """
    Calculate the centroid of the event.

    Parameters:
    event: The event data.
    offset_x: The offset in the x direction.
    offset_y: The offset in the y direction.

    Returns:
    tuple: The centroid of the event.
    """
    powers = [x_rtp, y_rtp]
    offsets = [0.00001, 0.00001]

    sum_xy = [0.0, 0.0]
    weights_xy = [0.0, 0.0]

    # Calculate the centroid of the event based on the local coordinates of the channels
    # and the energy deposited in each channel
    for hit in det_event:
        ch = hit[-1]
        energy = hit[1]
        x, y = local_dict[ch]
        weight_x = (energy + offsets[0]) ** powers[0]
        weight_y = (energy + offsets[1]) ** powers[1]
        sum_xy[0] += weight_x * x
        sum_xy[1] += weight_y * y
        weights_xy[0] += weight_x
        weights_xy[1] += weight_y

    return sum_xy[0] / weights_xy[0], sum_xy[1] / weights_xy[1]
