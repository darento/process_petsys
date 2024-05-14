from typing import Callable
import pandas as pd
from src.detector_features import calculate_total_energy
from src.fem_handler import FEMBase
from src.mapping_generator import ChannelType


def get_electronics_nums(channel_id: int) -> tuple[int, int, int, int]:
    """
    Calculates the electronics numbers: portID, slaveID, chipID, channelID based on the given channel id.

    Parameters:
        - channel_id (int): The channel id to calculate the electronics numbers from.

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
        - portID (int): The port id.
        - slaveID (int): The slave id.
        - chipID (int): The chip id.
        - channelID (int): The channel id.

    Returns:
    int: The absolute channel id calculated from the electronics numbers.
    """
    return 131072 * portID + 4096 * slaveID + 64 * chipID + channelID


def get_maxEnergy_sm_mM(det_list: list[list], sm_mM_map: dict, chtype_map: dict) -> int:
    """
    Returns the mini module with highest energy

    Parameters:
        - det_event (list): The event data.
        - sm_mM_map (dict): The mapping of the channels to the mod and mM.

    Returns:
        - list[list]: The event data for the maximum energy miniModule and sm.
        - float: The energy in the minimodule with highest energy.
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
        max_mm_evt = 0

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
                max_mm_evt = [ch for ch in det_list if sm_mM_map[ch[2]] == sm_mM]

        return max_mm_evt, max_energy


def get_num_eng_channels(det_list: list[list], chtype_map: dict) -> int:
    """
    Counts the number of energy channels in a list.

    Parameters:
        - det_list (list): The list of detectors.
        - chtype_map (dict): A mapping from detector names to channel types.

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
        - det_list : List of hits with [tstp, energy, chid]
        - chtype_map (dict): A mapping from channel names to channel types.
        - chtype (ChannelType, optional): The type of channel to be compared. If None, all channels are considered. Defaults to None.

    Returns:
        list: The channel with the highest deposit.
    """

    def _is_type(hit: list) -> bool:
        """
        Parameters:
            hit (list): The channel to check. [tstp, energy, chid]
        """
        return chtype in chtype_map[hit[2]]

    filt_func = _is_type if chtype else lambda x: True

    try:
        return max(filter(filt_func, det_list), key=lambda y: y[1])
    except ValueError:
        return None


def get_max_num_ch(det_list: list[list], chtype_map: dict, max_ch: int) -> list[list]:
    """
    Returns the event up to the maximum number of channel specified.

    Parameters:
        - det_list : List of hits with [tstp, energy, chid]
        - chtype_map (dict): A mapping from channel names to channel types.
        - max_ch (int): The maximum number of channels to return.

    Returns:
    list: The event up to the maximum number of channels specified.
    """
    return sorted(
        filter(lambda x: ChannelType.ENERGY in chtype_map[x[2]], det_list),
        key=lambda x: x[1],
        reverse=True,
    )[:max_ch]


def get_neighbour_channels(
    det_list: list[list],
    chtype_map: dict,
    local_coord_dict: dict,
    neighbour_ch: int,
    FEM_instance: FEMBase,
    zero_neighbour_xy: str = "x",
) -> list[list]:
    """
    Return the event with the neighbour channels of the highest energy channel.

    Parameters:
        - det_list : List of hits with [tstp, energy, chid]
        - chtype_map (dict): A mapping from channel names to channel types.
        - local_coord_dict (dict): A dictionary mapping detector channels to local coordinates.
        - neighbour_ch (int): The number of neighbour channels to return.

    Returns:
    list: The event with the neighbour channels of the highest energy channel.
    """
    max_en_ch = get_max_en_channel(det_list, chtype_map, ChannelType.ENERGY)
    pos_max_en_ch = local_coord_dict[max_en_ch[2]]
    neighbour_channels = []
    for hit in det_list:
        pos_hit = local_coord_dict[hit[2]]
        dx = abs(pos_max_en_ch[0] - pos_hit[0])
        dy = abs(pos_max_en_ch[1] - pos_hit[1])
        if neighbour_ch == 0:
            # Look only for the nearest neighbour, up, down, or left and right
            if zero_neighbour_xy == "x":
                if abs(dx) <= FEM_instance.x_pitch and (pos_hit[1] == pos_max_en_ch[1]):
                    neighbour_channels.append(hit)
            elif zero_neighbour_xy == "y":
                if abs(dy) <= FEM_instance.y_pitch and (pos_hit[0] == pos_max_en_ch[0]):
                    neighbour_channels.append(hit)
        elif neighbour_ch == 1:
            # Look only for the nearest neighbours, up, down, left and right
            if (
                abs(dx) <= FEM_instance.x_pitch and (pos_hit[1] == pos_max_en_ch[1])
            ) or (abs(dy) <= FEM_instance.y_pitch and (pos_hit[0] == pos_max_en_ch[0])):
                neighbour_channels.append(hit)

        elif neighbour_ch >= 2:
            if abs(dx) <= FEM_instance.x_pitch * (neighbour_ch - 1) and abs(
                dy
            ) <= FEM_instance.y_pitch * (neighbour_ch - 1):
                neighbour_channels.append(hit)
    return neighbour_channels


def get_timestamp_sorted(det_list: list[list], skew_dict: dict) -> list[list]:
    """
    Returns the event sorted by timestamp.

    Parameters:
        - det_list : List of hits with [tstp, energy, chid]
        - skew_dict (dict): A dictionary with the skew values.

    Returns:
    list: The event sorted by timestamp.
    """
    return sorted(det_list, key=lambda x: x[0] - skew_dict[x[2]])


def read_skew(skew_file: str):
    """
    This function reads the skew file and returns a dictionary with the skew values.
    """
    skew_dict = {}
    with open(skew_file, "r") as f:
        for line in f:
            ch, skew = line.split()
            skew_dict[int(ch)] = float(skew)
    return skew_dict


class KevConverter:
    def __init__(self, kev_file: str, file_type: str):
        if file_type == "mu":
            self.kev_factors = (
                pd.read_csv(kev_file, sep="\t").set_index("ID")["mu"].to_dict()
            )
            self.convert = self.convert_mu
        elif file_type == "poly":
            self.poly_coeffs = (
                pd.read_csv(kev_file, sep="\t")
                .set_index("ID")[["coef0", "coef1", "coef2"]]
                .to_dict("index")
            )
            self.convert = self.convert_poly
        else:
            raise ValueError(f"Unknown file type: {file_type}")

    def convert_mu(self, id: int, energy: float) -> float:
        mu = self.kev_factors[id]

        return 511.0 / mu * energy if mu != 0 else 0

    def convert_poly(self, id: int, energy: float) -> float:
        coeffs = self.poly_coeffs[id]

        return coeffs["coef0"] * energy**2 + coeffs["coef1"] * energy + coeffs["coef2"]
