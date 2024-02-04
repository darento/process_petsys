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
