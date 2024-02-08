from collections import defaultdict
import math
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


def get_maxEnergy_sm_mM(det_list: list[list], sm_mM_map: dict) -> int:
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
    mM_list = [sm_mM_map[ch[2]][1] for ch in det_list]
    mM_list = list(set(mM_list))
    if len(mM_list) == 1:
        # If there is only 1 mM, we return it with the corresponding sm and the energy
        return (
            mM_list[0],
            sm_mM_map[det_list[0][2]][0],
            sum([ch[1] for ch in det_list]),
        )
    else:
        # If there is more than 1 mM, we need to find the one with the maximum energy
        max_energy = 0
        max_mM = 0
        max_sm = 0
        for mM in mM_list:
            energy = sum([ch[1] for ch in det_list if sm_mM_map[ch[2]][1] == mM])
            if energy > max_energy:
                max_energy = energy
                max_mM = mM
                max_sm = sm_mM_map[det_list[0][2]][0]
        return max_mM, max_sm, max_energy


def calculate_total_energy(det_list: list[list]) -> float:
    """
    Calculate the total energy of the event.

    Parameters:
    event: The event data.

    Returns:
    float: The total energy of the event.
    """
    return sum([ch[1] for ch in det_list])


def calculate_centroid(
    det_list: list[list], local_dict: dict, x_rtp: int, y_rtp: int
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
    for hit in det_list:
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


def calculate_DOI(det_list: list[list], local_dict: dict) -> float:
    """
    Calculate the depth of interaction (DOI) of the event.

    Parameters:
    event: The event data.
    local_dict: The local coordinates of the channels.

    Returns:
    float: The depth of interaction (DOI) of the event.
    """
    # DOI is calculated as the sum of the energies divided by the maximum energy,
    # previosy summing the energies in the same x position
    x_pos = defaultdict(float)
    for ch in det_list:
        x, _ = local_dict[ch[-1]]
        x_pos[x] += ch[1]
    max_energy = max(x_pos.values())
    sum_energy = sum(x_pos.values())
    return sum_energy / max_energy


def calculate_impact_hits(
    det_list: list[list], local_coord_dict: dict, FEM_instance
) -> tuple:
    """
    Calculate the impact hits of the event.

    Parameters:
    det_list (list): The event data.
    local_coord_dict (dict): The local coordinates of the channels.
    FEM_instance (FEM): The FEM instance.

    Returns:
    tuple: The impact hits of the event.
    """
    num_ASIC_ch = FEM_instance.channels / FEM_instance.num_ASICS
    num_boxes_side = int(math.sqrt(num_ASIC_ch))

    # Create a matrix to store the energy of each box
    energy_matrix = np.zeros((num_boxes_side, num_boxes_side))

    # Fill the matrix with the energy of each box
    for hit in det_list:
        channel, energy = (
            hit[2],
            hit[1],
        )
        x, y = local_coord_dict[channel]

        # Convert the local coordinates to the box index
        x_index = int(x / FEM_instance.x_pitch)
        y_index = int(y / FEM_instance.x_pitch)

        energy_matrix[y_index, x_index] = energy
    return energy_matrix
