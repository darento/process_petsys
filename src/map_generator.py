import numpy as np
import yaml


def validate_yaml_map(yaml_map: dict, schema: dict) -> bool:
    # Validate mandatory keys
    for key, value_type in schema["mandatory"].items():
        if key not in yaml_map:
            raise RuntimeError(f"Missing mandatory key in YAML map: {key}")
        if not isinstance(yaml_map[key], value_type):
            raise RuntimeError(
                f"Incorrect type for mandatory key {key}: expected {value_type.__name__}, got {type(yaml_map[key]).__name__}"
            )

    # Check for the presence and validity of one and only one optional group
    found_group = None
    for group_info in schema["optional_groups"]:
        group, expected_type = group_info["group"], group_info["type"]
        if all(
            key in yaml_map and isinstance(yaml_map[key], expected_type)
            for key in group
        ):
            if found_group is not None:
                # Found more than one valid group
                raise RuntimeError(
                    "More than one of the optional groups are present and correctly formatted in the YAML map."
                )
            found_group = group
    if found_group is None:
        raise RuntimeError(
            "Exactly one of the optional groups must be present and correctly formatted in the YAML map.\n"
            "Name[channels_j1, channels_j2] and [Time, Energy] are the optional groups."
        )

    return True


def _read_yaml_file(mapping_file: str) -> dict:
    try:
        with open(mapping_file) as map_buffer:
            yaml_map = yaml.safe_load(map_buffer)
    except yaml.YAMLError:
        raise RuntimeError("Mapping file not readable.")
    except FileNotFoundError:
        raise RuntimeError("Mapping file not found, please check directory")

    # Define YAML schema
    yaml_schema = {
        "mandatory": {
            "FEM": str,
            "FEBD": str,
            "mod_feb_map": dict,
            "x_pitch": (int, float),
            "y_pitch": (int, float),
        },
        "optional_groups": [
            {
                "group": ["channels_j1", "channels_j2"],
                "type": list,
            },
            {
                "group": ["Time", "Energy"],
                "type": list,
            },
        ],
    }

    validate_yaml_map(yaml_map, yaml_schema)
    return yaml_map


class FEMBase:
    def __init__(self, x_pitch: float, y_pitch: float, channels: int):
        self.x_pitch = x_pitch
        self.y_pitch = y_pitch
        self.channels = channels

    def get_coordinates(self, channel_pos: int) -> tuple:
        raise NotImplementedError("Subclass must implement abstract method")


class FEM128(FEMBase):
    def __init__(self, x_pitch: float, y_pitch: float):
        super().__init__(x_pitch, y_pitch, 128)

    def get_coordinates(self, channel_pos: int) -> tuple:
        row = channel_pos // 8
        col = channel_pos % 8
        loc_x = (col + 0.5) * self.x_pitch
        loc_y = (row + 0.5) * self.y_pitch
        return (loc_x, loc_y)


class FEM256(FEMBase):
    def __init__(self, x_pitch: float, y_pitch: float):
        super().__init__(x_pitch, y_pitch, 256)

    def get_coordinates(self, channel_pos: int) -> tuple:
        # Implement the FEM256 coordinate calculation logic here
        pass


def get_FEM_instance(FEM_type: str, x_pitch: float, y_pitch: float):
    if FEM_type == "FEM128":
        return FEM128(x_pitch, y_pitch)
    elif FEM_type == "FEM256":
        return FEM256(x_pitch, y_pitch)
    else:
        raise ValueError("Unsupported FEM type")


def get_optional_group_keys(yaml_map: dict) -> tuple:
    """
    Determines which optional group is present in the yaml_map and returns the keys.

    Parameters:
    - yaml_map (dict): The loaded YAML map.

    Returns:
    - tuple: A tuple containing the keys of the present optional group.
    """
    if "channels_j1" in yaml_map and "channels_j2" in yaml_map:
        return ("channels_j1", "channels_j2")
    elif "Time" in yaml_map and "Energy" in yaml_map:
        return ("Time", "Energy")
    else:
        raise RuntimeError("No valid optional group found.")


def _get_local_mapping(
    mod_feb_map: dict, channels_1: list, channels_2: list, FEM_instance: FEMBase
) -> dict:
    """
    Generates the basic mapping from the mapping file.

    Parameters:
    - mod_feb_map (dict): Mapping from modules to FEBs.
    - channels_1 (list): The first list of channels or other entities.
    - channels_2 (list): The second list of channels or other entities.
    - FEM_instance (FEMBase): FEM instance for coordinate calculations.

    Returns:
    - dict: The local mapping.
    """
    local_map = {}
    for mod, value in mod_feb_map.items():
        portID, slaveID, febport = value
        mod_min_chan = (
            131072 * portID + 4096 * slaveID + FEM_instance.channels * febport
        )
        for i, (ch_1, ch_2) in enumerate(zip(channels_1, channels_2)):
            loc_x, loc_y = FEM_instance.get_coordinates(i)
            local_map[ch_1 + mod_min_chan] = (loc_x, loc_y)
            local_map[ch_2 + mod_min_chan] = (loc_x, loc_y)

    return local_map


def map_factory(mapping_file: str) -> dict:
    """
    Reads a mapping file and generates a mapping.

    Parameters:
    - mapping_file (str): The path to the mapping file.

    Returns:
    - dict: The mapping.
    """
    yaml_map = _read_yaml_file(mapping_file)
    FEM_type = yaml_map["FEM"]
    x_pitch = yaml_map["x_pitch"]
    y_pitch = yaml_map["y_pitch"]
    mod_feb_map = yaml_map["mod_feb_map"]

    # Extracting the necessary lists based on the optional group present
    channel_group_keys = get_optional_group_keys(yaml_map)
    channels_1 = yaml_map[channel_group_keys[0]]
    channels_2 = yaml_map[channel_group_keys[1]]

    # Creating the FEM instance
    FEM_instance = get_FEM_instance(FEM_type, x_pitch, y_pitch)

    # Generating the local map without passing the whole yaml_map
    local_map = _get_local_mapping(mod_feb_map, channels_1, channels_2, FEM_instance)
    return local_map
