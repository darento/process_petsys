from typing import Tuple
from src.fem_handler import FEMBase, get_FEM_instance
from src.yaml_handler import YAMLMapReader, get_optional_group_keys

from enum import Enum, auto


class ChannelType(Enum):
    TIME = auto()
    ENERGY = auto()


def _get_local_mapping(
    mod_feb_map: dict,
    channels_1: list,
    channels_2: list,
    FEM_instance: FEMBase,
) -> Tuple[dict, dict, dict]:
    # TODO: Need to generalize this function to handle slabs and other types of detectors.
    """
    Generates the basic mapping from the mapping file.

    Parameters:
    - mod_feb_map (dict): Mapping from modules to FEBs.
    - channels_1 (list): The first list of channels or other entities.
    - channels_2 (list): The second list of channels or other entities.
    - FEM_instance (FEMBase): FEM instance for coordinate calculations.

    Returns:
    Tuple[dict, dict, dict]: The local map, the sm_mM_map, and the chtype_map.
    """
    local_map = {}
    sm_mM_map = {}
    chtype_map = {}

    for mod, value in mod_feb_map.items():
        portID, slaveID, febport = value
        mod_min_chan = (
            131072 * portID + 4096 * slaveID + FEM_instance.channels * febport
        )
        for i, (ch_1, ch_2) in enumerate(zip(channels_1, channels_2)):
            # if sum_rows_cols is True, the number of channels is divided by 2 as we have half the number of channels per ASIC
            # for ch_1 and ch_2
            loc_x, loc_y = FEM_instance.get_coordinates(i)
            local_map[ch_1 + mod_min_chan] = (loc_x, loc_y)
            local_map[ch_2 + mod_min_chan] = (loc_x, loc_y)
            mM = int(i // FEM_instance.mM_channels)
            sm_mM_map[ch_1 + mod_min_chan] = (mod, mM)
            sm_mM_map[ch_2 + mod_min_chan] = (
                (mod, mM + 1) if not FEM_instance.sum_rows_cols else (mod, mM)
            )
            if not FEM_instance.sum_rows_cols:
                chtype_map[ch_1 + mod_min_chan] = [ChannelType.TIME, ChannelType.ENERGY]
                chtype_map[ch_2 + mod_min_chan] = [ChannelType.TIME, ChannelType.ENERGY]
            else:
                chtype_map[ch_1 + mod_min_chan] = [ChannelType.TIME]
                chtype_map[ch_2 + mod_min_chan] = [ChannelType.ENERGY]

    return local_map, sm_mM_map, chtype_map


def map_factory(mapping_file: str) -> Tuple[dict, dict, dict, FEMBase]:
    """
    This function reads a YAML mapping file and returns a local map and the keys of the mod_feb_map.

    Parameters:
    mapping_file (str): The path to the YAML mapping file.

    Returns:
    Tuple[dict, dict, dict, FEMBase]: The local map, the sm_mM_map, the chtype_map, and the FEM instance.
    """
    yaml_schema = {
        "mandatory": {
            "FEM": str,
            "FEBD": str,
            "mod_feb_map": dict,
            "x_pitch": (int, float),
            "y_pitch": (int, float),
            "mM_channels": int,
            "sum_rows_cols": bool,
        },
        "optional_groups": [
            {"group": ["channels_j1", "channels_j2"], "type": list},
            {"group": ["Time", "Energy"], "type": list},
        ],
    }

    reader = YAMLMapReader(yaml_schema)
    yaml_map = reader.read_yaml_file(mapping_file)
    FEM_type = yaml_map["FEM"]
    mM_channels = yaml_map["mM_channels"]
    sum_rows_cols = yaml_map["sum_rows_cols"]
    x_pitch = yaml_map["x_pitch"]
    y_pitch = yaml_map["y_pitch"]
    mod_feb_map = yaml_map["mod_feb_map"]
    mM_disposition = yaml_map["mM_disposition"]

    # Assuming get_optional_group_keys logic is included here or adjusted as needed
    channel_group_keys = get_optional_group_keys(
        yaml_map
    )  # Implement this function based on your application logic
    channels_1 = yaml_map[channel_group_keys[0]]
    channels_2 = yaml_map[channel_group_keys[1]]

    FEM_instance = get_FEM_instance(
        FEM_type, x_pitch, y_pitch, mM_channels, sum_rows_cols
    )

    # Implement _get_local_mapping based on the refactored approach
    local_map, sm_mM_map, chtype_map = _get_local_mapping(
        mod_feb_map,
        channels_1,
        channels_2,
        FEM_instance,
    )
    return local_map, sm_mM_map, chtype_map, FEM_instance
