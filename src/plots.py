import math
from matplotlib import pyplot as plt
import numpy as np

from src.detector_features import get_electronics_nums
from src.fem_handler import FEMBase
from src.fits import fit_gaussian


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


def plot_floodmap(
    xy_list: list, sm: int, mM: int, bins: tuple = (200, 200), show_fig: bool = False
) -> None:
    """
    This function plots the floodmap of a single channel on a 2D graph.

    Parameters:
    xy_list (list): A list of coordinates [(x1, y1), (x2, y2), ...] for a single channel.
    bins (tuple, optional): A tuple specifying the number of bins in the x and y directions.

    Output:
    None. This function plots a graph and does not return anything.
    """
    # Unpack the x and y coordinates
    x, y = zip(*xy_list)

    plt.hist2d(x, y, bins=bins, range=[[0, 26], [0, 26]], cmap="plasma")

    # Set plot properties
    plt.xlabel("X position (mm)")
    plt.ylabel("Y position (mm)")
    plt.axis("equal")  # To maintain the aspect ratio
    plt.xlim([0, 26])  # replace min_x, max_x with your desired values
    plt.ylim([0, 26])
    plt.title(f"Floodmap Representation of sm {sm}, mM {mM}")
    if show_fig:
        plt.show()


def plot_floodmap_mM(floodmap: dict, bins: tuple = (200, 200)) -> None:
    """
    This function plots the floodmap of the channels on a 2D graph for each (sm, mM).

    Parameters:
    floodmap (dict): A dictionary mapping the (sm, mM) to a list of coordinates.
                     The keys are tuples (sm, mM) and the values are lists of coordinates [(x1, y1), (x2, y2), ...].
    bins (tuple, optional): A tuple specifying the number of bins in the x and y directions.

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


def plot_single_energy_spectrum(
    energy_list: list,
    en_min: float,
    en_max: float,
    sm: int,
    mM: int,
    show_fig: bool = False,
    fit_flag: bool = False,
) -> None:
    """
    This function plots the energy spectrum of a single channel.

    Parameters:
    energy_list (list): A list of energies for a single channel.
    en_min (float): The lower limit of the energy range.
    en_max (float): The upper limit of the energy range.
    sm, mM (int): Channel identifiers.
    """
    # Plot the energy spectrum and the Gaussian fit if the flag is set
    n, bins, _ = plt.hist(
        energy_list, bins=100, range=(en_min, en_max), label=f"sm {sm}, mM {mM}"
    )

    if fit_flag:
        # Fit a Gaussian to the energy spectrum
        x, y, _, _, _ = fit_gaussian(n, bins, cb=16)
        plt.plot(x, y, label="Gaussian fit")

    # Set plot properties
    plt.xlabel("Energy (a.u.)")
    plt.ylabel("Counts")
    plt.title(f"Energy Spectrum of sm {sm}, mM {mM}")
    plt.legend()

    if show_fig:
        plt.show()

    return n, bins


def plot_energy_spectrum_mM(sm_mM_energy, en_min=0, en_max=100):
    """
    This function plots the energy spectrum of the channels.

    Parameters:
    sm_mM_energy (dict): A dictionary mapping the (sm, mM) to a list of energies.
                         The keys are tuples (sm, mM) and the values are lists of energies.
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
        plot_single_energy_spectrum(energy_list, en_min, en_max, sm, mM)

    plt.show()


def plot_event_impact(
    det_event: list[list], local_coord_dict: dict, FEM_instance: FEMBase
) -> None:
    """
    This function plots the impact of the event on the detector.

    Parameters:
    det_event (list): The event data.
    local_coord_dict (dict): The local coordinates of the channels.
    FEM_instance (FEMBase): The FEM instance.
    """
    num_ASIC_ch = FEM_instance.channels / FEM_instance.num_ASICS
    num_boxes_side = int(math.sqrt(num_ASIC_ch))

    # Crear una matriz para almacenar la energía en cada canal
    energy_matrix = np.zeros((num_boxes_side, num_boxes_side))

    # Llenar la matriz de energía con los datos del evento
    for hit in det_event:
        channel, energy = (
            hit[2],
            hit[1],
        )  # Asumiendo que el canal y la energía están en estas posiciones
        x, y = local_coord_dict[
            channel
        ]  # Asumiendo que las coordenadas locales son una tupla (x, y)

        # Convertir las coordenadas del centro de la caja a índices de matriz
        x_index = int(x / FEM_instance.x_pitch)
        y_index = int(y / FEM_instance.x_pitch)

        energy_matrix[
            y_index, x_index
        ] = energy  # Asumiendo que y_index, x_index están en el rango correcto

    # Dibujar la matriz de energía
    plt.imshow(energy_matrix, cmap="binary", interpolation="nearest")
    plt.colorbar(label="Energy")
    plt.show()
