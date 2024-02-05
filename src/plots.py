import math
from matplotlib import pyplot as plt

from src.detector_features import get_electronics_nums


def plot_chan_position(dict_coords: dict) -> None:
    # TODO: Save figure per chipID ? Is it useful?
    """
    This function plots the position of the channels on a graph.

    Parameters:
    dict_coords (dict): A dictionary mapping channels to their positions.
                        The keys are the channel names and the values are lists of coordinates [x, y].

    Output:
    None. This function plots a graph and does not return anything.
    """
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


def plot_floodmap_2D_mM(floodmap: dict, bins: tuple = (200, 200)) -> None:
    """
    This function plots the floodmap of the channels on a 2D graph.

    Parameters:
    floodmap (dict): A dictionary mapping the (sm, mM) to a list of coordinates.
                     The keys are tuples (sm, mM) and the values are lists of coordinates [(x1, y1), (x2, y2), ...].

    Output:
    None. This function plots a graph and does not return anything.
    """
    # Create a dictionary to store the count of channels at each position
    mM_dict = {}
    fig_number = 0

    # Plot each channel
    for (sm, mM), xy_list in sorted(floodmap.items()):
        if (sm, mM) not in mM_dict:
            mM_dict[(sm, mM)] = fig_number
            fig_number += 1

        # chipID figure
        fig = plt.figure(mM_dict[(sm, mM)], figsize=(10, 10))

        # Unpack the x and y coordinates
        x, y = zip(*xy_list)

        plt.hist2d(x, y, bins=bins, range=[[0, 26], [0, 26]], cmap="plasma")

        # Set plot properties
        plt.xlabel("X position (mm)")
        plt.ylabel("Y position (mm)")
        plt.title(f"Floodmap Representation of sm {sm}, mM {mM}")
        plt.axis("equal")  # To maintain the aspect ratio
        plt.xlim([0, 26])  # replace min_x, max_x with your desired values
        plt.ylim([0, 26])
    plt.show()


def plot_energy_spectrum_mM(
    sm_mM_energy: dict, en_min: float = 0, en_max: float = 100
) -> None:
    """
    This function plots the energy spectrum of the channels.

    Parameters:
    sm_mM_energy (dict): A dictionary mapping the (sm, mM) to a list of energies.
                         The keys are tuples (sm, mM) and the values are lists of energies.

    Output:
    None. This function plots a graph and does not return anything.
    """
    # Create a dictionary to store the count of channels at each position
    mM_dict = {}
    fig_number = 0

    # Plot each channel
    for (sm, mM), energy_list in sorted(sm_mM_energy.items()):
        if (sm, mM) not in mM_dict:
            mM_dict[(sm, mM)] = fig_number
            fig_number += 1

        # chipID figure
        fig = plt.figure(mM_dict[(sm, mM)], figsize=(10, 10))

        plt.hist(
            energy_list,
            bins=int(en_max - en_min),
            range=(en_min, en_max),
            alpha=0.7,
            label=f"sm {sm}, mM {mM}",
        )

        # Set plot properties
        plt.xlabel("Energy (a.u.)")
        plt.ylabel("Counts")
        plt.title(f"Energy Spectrum of sm {sm}, mM {mM}")
        plt.legend()
    plt.show()
