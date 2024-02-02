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

    # Initialize plot
    fig, ax = plt.subplots(sqrt_mod, sqrt_mod, figsize=(10, 10))

    # Create a dictionary to store the count of channels at each position
    mM_dict = {}
    fig_number = 0

    # Plot each channel
    for ch, xy in dict_coords.items():
        portID, slaveID, chipID, channelID = get_electronics_nums(ch)
        if (portID, slaveID, chipID) not in mM_dict:
            mM_dict[(portID, slaveID, chipID)] = fig_number
            fig_number += 1

        # Calculate the row and column indices
        row_index = mM_dict[(portID, slaveID, chipID)] // sqrt_mod
        col_index = mM_dict[(portID, slaveID, chipID)] % sqrt_mod

        x, y = xy
        print(mM_dict[(portID, slaveID, chipID)], row_index, col_index)
        ax[row_index, col_index].scatter(x, y, label=f"Channel {ch}")

        # Adjust the text position based on the count
        ax[row_index, col_index].text(x, y, str(ch), va="center", ha="center")

        # Similarly plot for J2 and any other ports if needed

        # Set plot properties
        ax[row_index, col_index].set_xlabel("X position (mm)")
        ax[row_index, col_index].set_ylabel("Y position (mm)")
        ax[row_index, col_index].set_title("Floodmap Representation of Channels")
        ax[row_index, col_index].axis("equal")  # To maintain the aspect ratio
    plt.show()
