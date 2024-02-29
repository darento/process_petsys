import math
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np

from src.utils import get_electronics_nums
from src.fem_handler import FEMBase
from src.fits import fit_gaussian


def plot_chan_position(dict_coords: dict) -> None:
    # TODO: Save figure per chipID ? Is it useful?
    """
    This function plots the position of the channels on a graph.

    Parameters:
        - dict_coords (dict): A dictionary mapping channels to their positions. The keys are the channel names and the values are lists of coordinates [x, y].

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


def plot_floodmap(
    xy_list: list, sm: int, mM: int, bins: tuple = (200, 200), show_fig: bool = False
) -> None:
    """
    This function plots the floodmap of a single channel on a 2D graph.

    Parameters:
        - xy_list (list): A list of coordinates [(x1, y1), (x2, y2), ...] for a single channel.
        - bins (tuple, optional): A tuple specifying the number of bins in the x and y directions.

    Output:
    None. This function plots a graph and does not return anything.
    """
    # Unpack the x and y coordinates
    x, y = zip(*xy_list)

    fig = plt.figure(figsize=(10, 10))
    h, x_edges, y_edges, _ = plt.hist2d(
        x, y, bins=bins, range=[[0, 26], [0, 26]], cmap="plasma"
    )

    # Set plot properties
    plt.xlabel("X position (mm)")
    plt.ylabel("Y position (mm)")
    plt.axis("equal")  # To maintain the aspect ratio
    plt.xlim([0, 26])  # replace min_x, max_x with your desired values
    plt.ylim([0, 26])
    plt.title(f"Floodmap Representation of sm {sm}, mM {mM}")
    if show_fig:
        plt.show()
    else:
        plt.clf()
        plt.close()
    return h, x_edges, y_edges


def plot_floodmap_mM(floodmap: dict, bins: tuple = (200, 200)) -> None:
    """
    This function plots the floodmap of the channels on a 2D graph for each (sm, mM).

    Parameters:
        - floodmap (dict): A dictionary mapping the (sm, mM) to a list of coordinates. The keys are tuples (sm, mM) and the values are lists of coordinates [(x1, y1), (x2, y2), ...].
        - bins (tuple, optional): A tuple specifying the number of bins in the x and y directions.

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

        plot_floodmap(xy_list, sm, mM, bins)


def plot_single_spectrum(
    hist_list: list,
    sm: int,
    mM: int,
    title: str,
    xlabel: str,
    hist_lim: Tuple[float, float] = (0, 100),
    show_fig: bool = False,
    fit_flag: bool = False,
    num_bins: int = 100,
) -> None:
    """
    This function plots the energy spectrum of a single channel.

    Parameters:
        - energy_list (list): A list of energies for a single channel.
        - en_min (float): The lower limit of the energy range.
        - en_max (float): The upper limit of the energy range.
        - sm, mM (int): Channel identifiers.
    """
    # Plot the energy spectrum and the Gaussian fit if the flag is set
    fig = plt.figure(figsize=(10, 10))
    n, bins, _ = plt.hist(
        hist_list,
        bins=num_bins,
        range=(hist_lim[0], hist_lim[1]),
        label=f"sm {sm}, mM {mM}",
    )

    if fit_flag:
        # Fit a Gaussian to the energy spectrum
        x, y, _, _, _ = fit_gaussian(n, bins, cb=8)
        plt.plot(x, y, label="Gaussian fit")

    # Set plot properties
    plt.xlabel(f"{xlabel}")
    plt.ylabel("Counts")
    plt.title(f"{title} Spectrum of sm {sm}, mM {mM}")
    plt.legend()

    if show_fig:
        plt.show()
    else:
        plt.clf()
        plt.close()

    return n, bins


def plot_energy_spectrum_mM(sm_mM_energy, en_min=0, en_max=100):
    """
    This function plots the energy spectrum of the channels.

    Parameters:
        - sm_mM_energy (dict): A dictionary mapping the (sm, mM) to a list of energies. The keys are tuples (sm, mM) and the values are lists of energies.
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

        # Call the function to plot a single energy spectrum
        plot_single_spectrum(
            energy_list, sm, mM, "Energy ", "Energy (a.u.)", (en_min, en_max)
        )

    plt.show()


def plot_event_impact(impact_matrix: np.ndarray) -> None:
    """
    This function plots the impact of the event on the detector.

    Parameters:
        - impact_matrix (np.ndarray): A 2D NumPy array representing the impact of the event on the detector.
    """
    plt.imshow(impact_matrix, cmap="binary", interpolation="nearest")
    plt.colorbar(label="Energy")
    plt.show()


def plot_xy_projection(
    h: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    projection_axis: str = "y",
    show_fig: bool = False,
) -> None:
    """
    This function plots the 1D projection of the 2D histogram for the ROI.

    Parameters:
        - h (np.ndarray): The 2D histogram.
        - x_edges (np.ndarray): The bin edges for the x-axis.
        - y_edges (np.ndarray): The bin edges for the y-axis.
        - projection_axis (str, optional): The axis along which to project the histogram. Defaults to "y".
        - show_fig (bool, optional): A flag to show the figure. Defaults to False.

    Returns:
        - np.ndarray: The 1D projection of the 2D histogram.
        - np.ndarray: The bin centers for the 1D projection.
    """
    # TODO: Maybe ROI directly here instead on the data filter?

    # Find the indices of the y bins that correspond to the ROI
    # y_min_index = np.searchsorted(y_edges, 11.5, side="right") - 1
    # y_max_index = np.searchsorted(y_edges, 14.0, side="right") - 1
    # Find the indices of the x bins that correspond to the ROI
    # x_min_index = np.searchsorted(x_edges, 0, side="right") - 1
    # x_max_index = np.searchsorted(x_edges, 16, side="right") - 1

    # Sum the 2D histogram 'h' over the y-axis within the ROI to get a 1D projection
    # h_projection = np.sum(h[x_min_index:x_max_index, y_min_index:y_max_index], axis=1)

    # The x-values for the projection are the center of the x-bins
    # x_bin_centers = (x_edges[x_min_index:x_max_index]) / 2

    # width = x_bin_centers[1] - x_bin_centers[0]

    if projection_axis == "x":
        h_projection = np.sum(h, axis=1)
        bin_centers = (x_edges[1:] + x_edges[:-1]) / 2
        xy_projection = "X"
    elif projection_axis == "y":
        h_projection = np.sum(h, axis=0)
        bin_centers = (y_edges[1:] + y_edges[:-1]) / 2
        xy_projection = "Y"
    width = bin_centers[1] - bin_centers[0]

    # Now we can plot the 1D projection
    plt.bar(bin_centers, h_projection, width=width, align="center")
    plt.xlabel(f"{xy_projection} position (mm)")
    plt.ylabel("Summed counts in ROI")
    plt.title("1D Projection of 2D Histogram")
    if show_fig:
        plt.show()
    else:
        plt.clf()
        plt.close()
    return bin_centers, h_projection
