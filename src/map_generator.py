import numpy as np
import matplotlib.pyplot as plt
import yaml


def _read_yaml_file(mapping_file: str) -> dict:
    """
    Read and return yaml mapping file.
    """
    try:
        with open(mapping_file) as map_buffer:
            channel_map = yaml.safe_load(map_buffer)
    except yaml.YAMLError:
        raise RuntimeError("Mapping file not readable.")
    except FileNotFoundError:
        raise RuntimeError("Mapping file not found, please check directory")
    if (
        type(channel_map) is not dict
        or "channels_j1" not in channel_map.keys()
        or "channels_j2" not in channel_map.keys()
    ):
        raise RuntimeError("Mapping file not correct format.")
    return channel_map


# Function to convert channel numbers to x, y coordinates
def get_coordinates(mapping_file: str) -> dict:
    channel_map = _read_yaml_file(mapping_file)

    print(channel_map)

    exit(0)
    """# Calculate row and column in the grid
    row = (channel_num) // 8
    col = (channel_num) % 8

    # Calculate x and y coordinates
    x = (col + 0.5) * distance_between_channels
    y = (row + 0.5) * distance_between_channels

    print(f"channel: {channel_num}, row: {row}, col: {col}, x: {x}, y: {y}")
    return (x, y)"""


def plot_chan_position(channels: list) -> None:
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Distance between channels
    distance = 3  # in millimeters

    # Plot each channel for J1
    for i, channel in enumerate(channels):
        x, y = get_coordinates(i, distance)
        ax.scatter(x, y, label=f"Channel {channel}")
        ax.text(x, y, str(channel), va="center", ha="center")

    # Similarly plot for J2 and any other ports if needed

    # Set plot properties
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.set_title("Floodmap Representation of Channels")
    ax.axis("equal")  # To maintain the aspect ratio
    plt.show()
