import random
from typing import Callable
import pandas as pd
import ast
from src.detector_features import calculate_total_energy
from src.fem_handler import FEMBase
from src.mapping_generator import ChannelType


class EventCounter:
    """
    Class to count the total number of events and the number of events that pass the filter.

    Methods:
        - increment_total(): Increments the total event count.
        - increment_pf(): Increments the count of events that pass the filter.
        - reset(): Resets counters to zero.
        - get_counts(): Returns the current counts.
    """

    def __init__(self):
        self.evt_count_t = 0
        self.evt_count_f = 0

    def increment_total(self):
        """
        Increments the total event count.
        """
        self.evt_count_t += 1

    def increment_pf(self):
        """
        Increments the count of events that pass the filter.
        """
        self.evt_count_f += 1

    def reset(self):
        """
        Resets counters to zero.
        """
        self.evt_count_t = 0
        self.evt_count_f = 0

    def get_counts(self):
        """
        Returns the current counts.
        """
        return self.evt_count_t, self.evt_count_f


def get_electronics_nums(channel_id: int) -> list[int, int, int, int]:
    """
    Calculates the electronics numbers: portID, slaveID, chipID, channelID based on the given channel id.

    Parameters:
        - channel_id (int): The channel id to calculate the electronics numbers from.

    Returns:
    list[int, int, int, int]: A tuple containing the portID, slaveID, chipID, and channelID.
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
        - chtype_map (dict): A mapping from detector names to channel types.

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


def get_max_num_ch(
    det_list: list[list],
    chtype_map: dict,
    max_ch: int,
    chtype: ChannelType = ChannelType.ENERGY,
) -> list[list]:
    """
    Returns the event up to the maximum number of channel specified.

    Parameters:
        - det_list : List of hits with [tstp, energy, chid]
        - chtype_map (dict): A mapping from channel names to channel types.
        - max_ch (int): The maximum number of channels to return.
        - chtype (ChannelType, optional): The type of channel to be compared. Defaults to ChannelType.ENERGY.

    Returns:
    list: The event up to the maximum number of channels specified.
    """
    return sorted(
        filter(lambda x: chtype in chtype_map[x[2]], det_list),
        key=lambda x: x[1],
        reverse=True,
    )[:max_ch]


def get_slab_cornell(det_list: list[list], chtype_map: dict, local_map: dict) -> int:
    """
    Returns the slab with highest energy in the event for the cornell (16 slabs per
    minimodule) detector.

    Parameters:
        - det_list : List of hits with [tstp, energy, chid]
        - chtype_map (dict): A mapping from channel names to channel types.
        - local_map (dict): A mapping from detector names to local coordinates.

    Returns:
    int: The slab with the highest energy in the event. (0-15)
    int: The flag indicating the type of slab. (0-3)
    float: The x position of the slab in local coordinates.
    """
    cornell_slabs = [(i, i + 1) for i in range(0, 16, 2)]
    edges = [0, 7]
    half_slab_width = 0.8
    time_chs = get_max_num_ch(det_list, chtype_map, 2, ChannelType.TIME)
    max_time_ch = time_chs[0][2]
    max_time_ch_pos = local_map[max_time_ch][2]
    if max_time_ch_pos in edges:
        if max_time_ch_pos == edges[0]:
            slab = cornell_slabs[max_time_ch_pos][len(time_chs) - 1]
            ad_factor = half_slab_width if len(time_chs) == 2 else -half_slab_width
            x_pos = local_map[max_time_ch][0] + ad_factor
        else:
            slab = cornell_slabs[max_time_ch_pos][1 - (len(time_chs) - 1)]
            ad_factor = -half_slab_width if len(time_chs) == 2 else half_slab_width
            x_pos = local_map[max_time_ch][0] + ad_factor
        return slab, 0, x_pos
    if len(time_chs) == 1:
        random_num = random.randint(0, 1)
        slab = cornell_slabs[max_time_ch_pos][random_num]
        ad_factor = -half_slab_width if random_num == 0 else half_slab_width
        x_pos = local_map[max_time_ch][0] + ad_factor
        return slab, 1, x_pos
    second_max_time_ch = local_map[time_chs[1][2]][2]
    diff = max_time_ch_pos - second_max_time_ch
    if abs(diff) > 1:
        return None, 2, None
    if diff == 1:
        slab = cornell_slabs[max_time_ch_pos][1]
        x_pos = local_map[max_time_ch][0] + half_slab_width
    else:
        slab = cornell_slabs[max_time_ch_pos][0]
        x_pos = local_map[max_time_ch][0] - half_slab_width
    return slab, 3, x_pos


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
        - FEM_instance (FEMBase): The FEM instance.
        - zero_neighbour_xy (str): The axis to look for the nearest neighbour. Can be either "x" or "y".

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


def apply_skew(det_list: list[list], chtype_map: dict, skew_dict: dict) -> list[list]:
    """
    Applies the skew per channel to the event.

    Parameters:
        - det_list : List of hits with [tstp, energy, chid]
        - chtype_map (dict): A mapping from channel names to channel types.
        - skew_dict (dict): A dictionary with the skew values.

    Returns:
    list: The event with the skew applied.
    """
    return [
        (
            (hit[0] - skew_dict.get(hit[-1], 0), *hit[1:])
            if ChannelType.TIME in chtype_map[hit[-1]]
            else hit
        )
        for hit in det_list
    ]


def get_timestamp_sorted(det_list: list[list]) -> list[list]:
    """
    Returns the event sorted by timestamp.

    Parameters:
        - det_list : List of hits with [tstp, energy, chid]

    Returns:
    list: The event sorted by timestamp.
    """
    return sorted(det_list, key=lambda x: x[0])


def read_skew(skew_file: str):
    """
    This function reads the skew file and returns a dictionary with the skew values.

    Parameters:
        - skew_file (str): The path to the skew file.

    Returns:
    dict: A dictionary with the skew values.
    """

    skew_dict = {}
    with open(skew_file, "r") as f:
        for line in f:
            ch, skew = line.split()
            skew_dict[int(ch)] = float(skew)
    return skew_dict


def read_skew_LORs(skew_file: str):
    """
    This function reads the skew file and returns a dataframe with the skew values.

    Parameters:
        - skew_file (str): The path to the skew file per LOR.

    Returns:
    dict: A dictionary with the skew values.
    """
    print("Reading skew file")
    skew_dict = pd.read_csv(skew_file, sep=",").set_index("Pair")["Skew"].to_dict()
    print("Skew file read")
    return skew_dict


def read_pair_map(pair_path: str) -> dict:
    """
    Reads the pair map file and returns a dictionary with the pair mapping.

    Parameters:
        - pair_path (str): The path to the pair map file.

    Returns:
    dict: A dictionary with the pair mapping.
    """
    pair_cols = ["p", "s0", "s1"]
    pair_map = (
        pd.read_csv(pair_path, sep="\t", names=pair_cols)
        .set_index(pair_cols[1:])
        .to_dict()["p"]
    )
    return pair_map


class KevConverter:
    """
    Class to convert energy from PETsys a.u. to keV, using either a mu factor or a polynomial fit.

    Parameters:
        - kev_file (str): The file containing the conversion factors.
        - file_type (str): The type of file containing the conversion factors. Can be either "mu" or "poly".

    Methods:
        - convert_mu(id: int, energy: float) -> float: Converts the energy using the mu factor.
        - convert_poly(id: int, energy: float) -> float: Converts the energy using the polynomial
        fit.
    """

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
        elif file_type == "cornell":
            string_map = (
                pd.read_csv(kev_file, sep="\t")
                .set_index("ID(t_ch, slab)")["mu"]
                .to_dict()
            )
            self.kev_factors = {
                ast.literal_eval(key): value for key, value in string_map.items()
            }
            self.convert = self.convert_mu
        else:
            raise ValueError(f"Unknown file type: {file_type}")

    def convert_mu(self, id: int, energy: float) -> float:
        mu = self.kev_factors[id]

        return 511.0 / mu * energy if mu != 0 else 0

    def convert_poly(self, id: int, energy: float) -> float:
        coeffs = self.poly_coeffs[id]

        return coeffs["coef0"] * energy**2 + coeffs["coef1"] * energy + coeffs["coef2"]
