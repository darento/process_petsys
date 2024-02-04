import math
from matplotlib import pyplot as plt

from src.detector_features import get_electronics_nums


def plot_chan_position(dict_coords: dict, mod_number) -> None:
    # TODO: Save figure per chipID ? Is it useful?
    """
    This function plots the position of the channels on a graph.

    Parameters:
    dict_coords (dict): A dictionary mapping channels to their positions.
                        The keys are the channel names and the values are lists of coordinates [x, y].

    Output:
    None. This function plots a graph and does not return anything.
    """
    sqrt_mod = math.ceil(mod_number**0.5)

    # Create a dictionary to store the count of channels at each position
    mM_dict = {}
    fig_number = 0

    # Plot each channel
    for ch, xy in dict_coords.items():
        portID, slaveID, chipID, channelID = get_electronics_nums(ch)
        if (portID, slaveID, chipID) not in mM_dict:
            mM_dict[(portID, slaveID, chipID)] = fig_number
            fig_number += 1

        x, y = xy

        # chipID figure
        fig = plt.figure(mM_dict[(portID, slaveID, chipID)], figsize=(10, 10))

        plt.scatter(x, y, label=f"Channel {ch}")

        # Adjust the text position based on the count
        plt.text(x, y, str(ch), va="center", ha="center")

        # Similarly plot for J2 and any other ports if needed

        # Set plot properties
        plt.xlabel("X position (mm)")
        plt.ylabel("Y position (mm)")
        plt.title(
            f"Floodmap Representation of Channels for portID {portID}, slaveID {slaveID}, chipID {chipID}"
        )
        plt.axis("equal")  # To maintain the aspect ratio
    plt.show()
