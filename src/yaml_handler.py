import yaml


class YAMLMapReader:
    def __init__(self, schema):
        self.schema = schema

    def validate_yaml_map(self, yaml_map):
        # Validate mandatory keys
        for key, value_type in self.schema["mandatory"].items():
            if key not in yaml_map:
                raise RuntimeError(f"Missing mandatory key in YAML map: {key}")
            if not isinstance(yaml_map[key], value_type):
                raise RuntimeError(
                    f"Incorrect type for mandatory key {key}: expected {value_type.__name__}, got {type(yaml_map[key]).__name__}"
                )

        # Check for the presence and validity of one and only one optional group
        found_group = None
        for group_info in self.schema["optional_groups"]:
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

    def read_yaml_file(self, mapping_file):
        try:
            with open(mapping_file) as map_buffer:
                yaml_map = yaml.safe_load(map_buffer)
            self.validate_yaml_map(yaml_map)
            return yaml_map
        except yaml.YAMLError:
            raise RuntimeError("Mapping file not readable.")
        except FileNotFoundError:
            raise RuntimeError("Mapping file not found, please check directory.")


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
