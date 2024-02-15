from collections import defaultdict
import math
import numpy as np
from src.fem_handler import FEMBase

from src.mapping_generator import ChannelType


def calculate_total_energy(det_list: list[list], chtype_map: dict) -> float:
    """
    Calculate the total energy of the event.

    Parameters:
    det_list: The event data.
    chtype_map: A dictionary mapping the channel type to the channel number.

    Returns:
    float: The total energy of the event.
    """
    eng_ch = list(filter(lambda x: ChannelType.ENERGY in chtype_map[x[2]], det_list))
    return sum([ch[1] for ch in eng_ch])


def calculate_centroid(
    det_list: list[list], local_dict: dict, x_rtp: int, y_rtp: int
) -> tuple:
    # TODO: Generalize for sum_rows_cols case. Implement 2 functions for each case.
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
    # TODO: Generalize for sum_rows_cols case. Implement 2 functions for each case and slab orientation.
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
    det_list: list[list], local_coord_dict: dict, FEM_instance: FEMBase
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
