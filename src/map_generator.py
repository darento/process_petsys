import numpy as np
import yaml

from src.plots import plot_chan_position


def _read_yaml_file(mapping_file: str) -> dict:
    """
    This function reads a YAML file and returns its contents as a dictionary.

    Parameters:
    mapping_file (str): The path to the YAML file.

    Output:
    dict: The contents of the YAML file as a dictionary.

    Raises:
    RuntimeError: If the YAML file is not readable or not found, or if it does not have the correct format.
    """
    try:
        with open(mapping_file) as map_buffer:
            yaml_map = yaml.safe_load(map_buffer)
    except yaml.YAMLError:
        raise RuntimeError("Mapping file not readable.")
    except FileNotFoundError:
        raise RuntimeError("Mapping file not found, please check directory")
    if (
        type(yaml_map) is not dict
        or "channels_j1" not in yaml_map.keys()
        or "channels_j2" not in yaml_map.keys()
        or "FEM" not in yaml_map.keys()
        or "FEBD" not in yaml_map.keys()
        or "mod_feb_map" not in yaml_map.keys()
        or "x_pitch" not in yaml_map.keys()
        or "y_pitch" not in yaml_map.keys()
    ):
        raise RuntimeError("Mapping file not correct format.")
    return yaml_map


def _get_coordinates_FEM128(channel_pos: int, x_pitch: float, y_pitch: float) -> tuple:
    """
    This function calculates the x and y coordinates for a given channel position in the FEM128.

    Parameters:
    channel_pos (int): The position of the channel.
    x_pitch (float): The x pitch.
    y_pitch (float): The y pitch.

    Output:
    tuple: The x and y coordinates.
    """
    # Calculate row and column in the grid
    row = (channel_pos) // 8
    col = (channel_pos) % 8

    # Calculate x and y coordinates
    loc_x = (col + 0.5) * x_pitch
    loc_y = (row + 0.5) * y_pitch

    print(f"channel: {channel_pos}, row: {row}, col: {col}, x: {loc_x}, y: {loc_y}")
    return (loc_x, loc_y)


def _get_coordinates_FEM256(channel_pos: int, x_pitch: float, y_pitch: float) -> tuple:
    pass


def _get_local_mapping(
    FEM_chan: int,
    mod_feb_map: dict,
    channels_j1: list,
    channels_j2: list,
    x_pitch: float,
    y_pitch: float,
) -> dict:
    """
    This function generates the basic mapping from the mapping file.

    Parameters:
    FEM_chan (int): The number of channels in the FEM.
    mod_feb_map (dict): The mapping from modules to FEBs.
    channels_j1 (list): The list of channels in J1.
    channels_j2 (list): The list of channels in J2.
    x_pitch (float): The x pitch.
    y_pitch (float): The y pitch.

    Output:
    dict: The local mapping.
    """
    get_coord_from = (
        _get_coordinates_FEM128 if FEM_chan == 128 else _get_coordinates_FEM256
    )
    local_map = {}
    for mod, value in mod_feb_map.items():
        portID, slaveID, febport = value
        # Calculate the minimum channel number for the module
        mod_min_chan = 131072 * portID + 4096 * slaveID + FEM_chan * febport
        for i, (ch_j1, ch_j2) in enumerate(zip(channels_j1, channels_j2)):
            loc_x, loc_y = get_coord_from(i, x_pitch, y_pitch)
            local_map[ch_j1 + mod_min_chan] = (loc_x, loc_y)
            local_map[ch_j2 + mod_min_chan] = (loc_x, loc_y)
    print(local_map)
    return local_map


def map_factory(mapping_file: str) -> dict:
    """
    This function reads a mapping file and generates a mapping.

    Parameters:
    mapping_file (str): The path to the mapping file.

    Output:
    dict: The mapping.

    Raises:
    RuntimeError: If the FEM is not in the correct format.
    """
    yaml_map = _read_yaml_file(mapping_file)
    FEM = yaml_map["FEM"]
    FEBD = yaml_map["FEBD"]
    x_pitch = yaml_map["x_pitch"]
    y_pitch = yaml_map["y_pitch"]
    mod_feb_map = yaml_map["mod_feb_map"]

    if FEM == "FEM128":
        FEM_chan = 128
        channels_j1 = yaml_map["channels_j1"]
        channels_j2 = yaml_map["channels_j2"]
        local_map = _get_local_mapping(
            FEM_chan, mod_feb_map, channels_j1, channels_j2, x_pitch, y_pitch
        )
    elif FEM == "FEM256":
        # TODO: Add FEM256 mapping
        FEM_chan = 256
        pass
    else:
        raise RuntimeError("FEM not correct format.")

    return local_map
