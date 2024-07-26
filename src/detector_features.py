from collections import defaultdict
import math
import numpy as np
from src.fem_handler import FEMBase

from src.mapping_generator import ChannelType


def calculate_total_energy(det_list: list[list], chtype_map: dict) -> float:
    """
    Calculate the total energy of the event.

    Parameters:
        - det_list: The event data.
        - chtype_map: A dictionary mapping the channel type to the channel number.

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
        - det_list: The event data.
        - local_dict: The local coordinates of the channels.
        - x_rtp: The power of the x coordinate.
        - y_rtp: The power of the y coordinate.

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


def calculate_centroid_sum(
    det_list: list[list], local_map: dict, chtype_map: dict, x_rtp: int, y_rtp: int
) -> tuple:
    """
    Calculate the centroid of the event.

    Parameters:
        - det_list: The event data.
        - local_map: The local coordinates of the channels.
        - chtype_map: A dictionary mapping the channel type to the channel number.
        - x_rtp: The power of the x coordinate.
        - y_rtp: The power of the y coordinate.

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
        en_t = chtype_map[ch][0].value - 1
        pos = local_map[ch][en_t]
        weight = (hit[1] + offsets[en_t]) ** powers[en_t]
        sum_xy[en_t] += weight * pos
        weights_xy[en_t] += weight
    return sum_xy[0] / weights_xy[0], sum_xy[1] / weights_xy[1]


def calculate_DOI(
    det_list: list[list],
    local_dict: dict,
    sum_rows_cols: bool,
    chtype_map: dict,
    slab_orientation: str = "x",
) -> float:
    """
    Calculate the depth of interaction (DOI) of the event.

    Parameters:
        - det_list: The event data.
        - local_dict: The local coordinates of the channels.
        - slab_orientation: The orientation of the slab. (x or y)
        - sum_rows_cols: A boolean indicating whether to sum the rows and columns.
        - chtype_map: A dictionary mapping the channel type to the channel number.

    Returns:
    float: The depth of interaction (DOI) of the event.
    """

    def calculate_DOI_sum_rows_cols(det_list, chtype_map: dict):
        max_energy = max(
            (hit[1] for hit in det_list if ChannelType.ENERGY in chtype_map[hit[-1]]),
            default=0,
        )
        sum_energy = calculate_total_energy(det_list, chtype_map)
        return sum_energy / max_energy

    def calculate_DOI_sum_xy(det_list, local_dict, slab_orientation):
        xy_pos = defaultdict(float)
        if slab_orientation == "x":
            for ch in det_list:
                x, _ = local_dict[ch[-1]]
                xy_pos[x] += ch[1]
        elif slab_orientation == "y":
            for ch in det_list:
                _, y = local_dict[ch[-1]]
                xy_pos[y] += ch[1]
        max_energy = max(xy_pos.values())
        sum_energy = sum(xy_pos.values())
        return sum_energy / max_energy

    if sum_rows_cols:
        return calculate_DOI_sum_rows_cols(det_list, chtype_map)
    else:
        return calculate_DOI_sum_xy(det_list, local_dict, slab_orientation)


def calculate_impact_hits(
    det_list: list[list], local_coord_dict: dict, FEM_instance: FEMBase
) -> tuple:
    """
    Calculate the impact hits of the event.

    Parameters:
        - det_list (list): The event data.
        - local_coord_dict (dict): The local coordinates of the channels.
        - FEM_instance (FEM): The FEM instance.

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


def calculate_dt(
    det1_list: list[list],
    det2_list: list[list],
    chtype_map: dict,
    det1_avg: int = 1,
    det2_avg: int = 1,
) -> float:
    """
    Calculate the time difference between the two detectors.

    Parameters:
        - det1_list (list): The list of hits for detector 1.
        - det2_list (list): The list of hits for detector 2.
        - chtype_map (dict): A mapping from detector names to channel types.
        - skew_map (dict): A mapping from channel names to skew values.
        - det1_avg (float): The number of tstp channels for detector 1 to average.
        - det2_avg (float): The number of tstp channels for detector 2 to average.

    Returns:
    float: The time difference between the two detectors.
    """
    # Get the time channels for each detector
    time_det1 = list(filter(lambda x: ChannelType.TIME in chtype_map[x[2]], det1_list))
    time_det2 = list(filter(lambda x: ChannelType.TIME in chtype_map[x[2]], det2_list))

    # Calculate the time difference for no average
    if det1_avg == 1 and det2_avg == 1:
        time_ch1 = time_det1[0][2]
        time_ch2 = time_det2[0][2]
        # print(time_ch1, time_ch2)
        # print(time_det1[0][0], time_det2[0][0])
        t1 = time_det1[0][0]
        t2 = time_det2[0][0]
        # print(t1, t2)
        return t1 - t2
    else:
        # Check if the number of channels to average is greater than the number of channels
        # in the event
        if det1_avg > len(time_det1):
            det1_avg = len(time_det1)
        if det2_avg > len(time_det2):
            det2_avg = len(time_det2)
        # Calculate the weighted average time for each detector
        time_det1 = sum([x[0] * x[1] for x in time_det1[0:det1_avg]]) / sum(
            [x[1] for x in time_det1[0:det1_avg]]
        )
        time_det2 = sum([x[0] * x[1] for x in time_det2[0:det2_avg]]) / sum(
            [x[1] for x in time_det2[0:det2_avg]]
        )
        return time_det1 - time_det2


def calculate_dt_singlehit(
    det1_list: list[list],
    det2_list: list[list],
    chtype_map: dict,
    skew_map: dict,
    det1_Ntstp: int = 1,
    det2_Ntstp: int = 1,
) -> float:
    """
    Calculate the time difference between two detectors using one timestamp each.
    det1_Ntstp and det2_Ntstp give the timestamp number to consider for each detector

    ! this is needed to evaluate and correct for time walk before performing the average

    Parameters:
        - det1_list (list): The list of hits for detector 1.
        - det2_list (list): The list of hits for detector 2.
        - chtype_map (dict): A mapping from detector names to channel types.
        - skew_map (dict): A mapping from channel names to skew values.
        - det1_Ntstp (float): The tstp number to consider for detector 1.
        - det2_Ntstp (float): The tstp number to consider for detector 2.

    Returns:
    float: The time difference between the det1_Ntstp timestamp of detector 1 and det2_Ntstp timestamp of detector 2,
    and the energy collected by the det1_Ntstp and the det2_Ntstp channels.
    """
    # Get the time channels for each detector
    #

    event_det1 = list(filter(lambda x: ChannelType.TIME in chtype_map[x[2]], det1_list))
    event_det2 = list(filter(lambda x: ChannelType.TIME in chtype_map[x[2]], det2_list))

    det1_tstp_idx = det1_Ntstp - 1
    det2_tstp_idx = det2_Ntstp - 1

    # Check if the timestamp number exist or it's larger than the number of channel of the event

    if det1_tstp_idx >= len(event_det1):
        det1_tstp_idx = len(event_det1) - 1
    if det2_tstp_idx >= len(event_det2):
        det2_tstp_idx = len(event_det2) - 1

    time_ch1 = event_det1[det1_tstp_idx][2]
    time_ch2 = event_det2[det2_tstp_idx][2]
    t1 = event_det1[det1_tstp_idx][0] - skew_map.get(time_ch1, 0)
    t2 = event_det2[det2_tstp_idx][0] - skew_map.get(time_ch2, 0)

    en_det1_ch = event_det1[det1_tstp_idx][1]
    en_det2_ch = event_det2[det2_tstp_idx][1]

    return t1 - t2, en_det1_ch, en_det2_ch
