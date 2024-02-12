#!/usr/bin/env python3

"""Run the basic analysis for PETsys. 
Usage: main.py YAMLCONF INFILE

Arguments:
    YAMLCONF  File with all parameters to take into account in the scan.
    INFILE    Input file to be processed. Must be a compact binary file from PETsys.

Options:
    -h --help     Show this screen.    
"""


import os
import time
from typing import Callable
from docopt import docopt
import numpy as np
import yaml

from src.fem_handler import FEMBase
from src.read_compact import read_binary_file
from src.filters import filter_total_energy, filter_min_ch, filter_single_mM
from src.detector_features import (
    calculate_centroid,
    calculate_impact_hits,
    calculate_total_energy,
    calculate_DOI,
)
from src.mapping_generator import ChannelType, map_factory
from src.plots import (
    plot_floodmap,
    plot_single_spectrum,
    plot_event_impact,
)
from src.fits import fit_gaussian
from src.utils import get_num_eng_channels, get_max_en_channel

# Total number of eevents
EVT_COUNT_T = 0
# Events passing the filter
EVT_COUNT_F = 0


def increment_total():
    """
    Increments the total event count.

    This function is used to keep track of the total number of events processed.
    """
    global EVT_COUNT_T
    EVT_COUNT_T += 1


def increment_pf():
    """
    Increments the count of events that pass the filter.

    This function is used to keep track of the number of events that pass the filter.
    """
    global EVT_COUNT_F
    EVT_COUNT_F += 1


def data_and_en_dict(
    read_det_evt: Callable,
    local_coord_dict: dict,
    chtype_map: dict,
    sm_mM_map: dict,
    FEM_instance: FEMBase,
    min_ch: int,
) -> tuple[dict, np.ndarray]:
    """
    This function processes detector events, filters them based on energy and minimum channel,
    calculates various parameters for each event, and stores the results in a dictionary and a matrix.

    Parameters:
    read_det_evt (Callable): A generator or iterable that yields detector events.
    local_coord_dict (dict): A dictionary mapping detector channels to local coordinates.
    chtype_map (dict): A dictionary mapping the channel type to the channel number.
    FEM_instance (FEMBase): A dictionary representing a Finite Element Model instance.
    min_ch (int): The minimum channel number for the filter.

    Returns:
    tuple: A tuple containing two elements:
        - data_dict (dict): A dictionary where each key is an event count and each value is another dictionary
          with keys for various calculated parameters ("det1_en", "det2_en", etc.) and their corresponding values.
        - en_matrix (np.ndarray): A 2D NumPy array where each row corresponds to a detector event and the columns
          are the energies of detector 1 and detector 2.
    """
    event_count = 0
    data_dict = {}
    en_list = []
    start_time = time.time()
    for event in read_det_evt:
        increment_total()
        det1, det2 = event
        det1_en = calculate_total_energy(det1, chtype_map)
        det2_en = calculate_total_energy(det2, chtype_map)

        min_ch_filter1 = filter_min_ch(det1, min_ch, chtype_map)
        min_ch_filter2 = filter_min_ch(det2, min_ch, chtype_map)

        if min_ch_filter1 and min_ch_filter2:
            increment_pf()

            det1_doi = calculate_DOI(det1, local_coord_dict)
            det2_doi = calculate_DOI(det2, local_coord_dict)
            impact_matrix = calculate_impact_hits(det1, local_coord_dict, FEM_instance)
            x_det1, y_det1 = calculate_centroid(
                det1, local_coord_dict, x_rtp=1, y_rtp=2
            )

            x_det2, y_det2 = calculate_centroid(
                det2, local_coord_dict, x_rtp=1, y_rtp=2
            )
            en_list.append((det1_en, det2_en))
            data_dict[event_count] = {
                "det1_en": det1_en,
                "det2_en": det2_en,
                "det1_doi": det1_doi,
                "det2_doi": det2_doi,
                "x_det1": x_det1,
                "y_det1": y_det1,
                "x_det2": x_det2,
                "y_det2": y_det2,
                "impact_matrix": impact_matrix,
            }
        event_count += 1
        if event_count % 10000 == 0:
            print(f"Events processed: {event_count}")
        pass
    print("---------------------")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total events: {EVT_COUNT_T}")
    print(f"Events passing the filter: {EVT_COUNT_F}")
    en_matrix = np.array(en_list)
    return data_dict, en_matrix


def extract_photopeak_limits(
    en_matrix: np.ndarray, percentage: float = 0.2
) -> tuple[float, float]:
    """
    This function extracts the photopeak limits from the energy matrix.

    Parameters:
    en_matrix (np.ndarray): A 2D NumPy array where each row corresponds to a detector event and the columns
                            are the energies of detector 1 and detector 2.
    percentage (float, optional): The percentage of the photopeak limits to consider. Defaults to 0.2.

    Returns:
    tuple: A tuple of two floats:
        - en_peak_min (float): The lower limit of the photopeak.
        - en_peak_max (float): The upper limit of the photopeak.
    """
    n, bins = plot_single_spectrum(
        en_matrix[:, 0], 0, 200, 0, 0, "Detector 1 energy", "Energy (a.u.)"
    )

    _, _, pars, _, _ = fit_gaussian(n, bins)
    mu, sigma = pars[1], pars[2]
    en_peak_min = mu * (1 - percentage)
    en_peak_max = mu * (1 + percentage)
    return en_peak_min, en_peak_max


def extract_impact_info(
    data_dict: dict,
    n: int,
    en_min_peak: float = 0,
    en_max_peak: float = 1000,
):
    """
    This function extracts the average of the top n maximum values from the impact_matrix in each event.

    Parameters:
    data_dict (dict): A dictionary containing the event data.
    n (int): The number of top maximum values to consider.
    en_min_peak (float): The lower limit of the photopeak.
    en_max_peak (float): The upper limit of the photopeak.

    Returns:
    list: A list of length n where the i-th element is the average of the i-th maximum values.
    """
    max_values = np.zeros(n)
    events_passing_filter = 0
    max_value_av = 0
    for event, data in data_dict.items():
        det1_en = data["det1_en"]
        if not filter_total_energy(det1_en, en_min_peak, en_max_peak):
            continue
        impact_matrix = data["impact_matrix"]
        sorted_values = np.sort(impact_matrix, axis=None)[::-1]
        max_values[: len(sorted_values)] += sorted_values[:n] / np.max(sorted_values)
        max_value_av += sorted_values[0]
        events_passing_filter += 1
        # plot_event_impact(impact_matrix)
    return np.round(max_values / events_passing_filter, 3), np.round(
        max_value_av / events_passing_filter, 3
    )


def main():
    # Read the YAML configuration file
    args = docopt(__doc__)

    # Read the binary file
    binary_file_path = args["INFILE"]

    file_name = os.path.basename(binary_file_path)
    file_name = file_name.replace(".ldat", "_impactArray.txt")

    with open(args["YAMLCONF"], "r") as f:
        config = yaml.safe_load(f)

    # Read the mapping file
    map_file = config["map_file"]

    # Get the coordinates of the channels
    local_coord_dict, sm_mM_map, chtype_map, FEM_instance = map_factory(map_file)

    # Plot the coordinates of the channels
    # plot_chan_position(local_coord_dict)

    # Read the energy range
    en_min = float(config["energy_range"][0])
    en_max = float(config["energy_range"][1])
    print(f"Energy range: {en_min} - {en_max}")

    min_ch = int(config["min_ch"])
    print(f"Minimum number of channels: {min_ch}")

    en_min_ch = float(config["en_min_ch"])
    print(f"Minimum energy per channel: {en_min_ch}")

    reader = read_binary_file(binary_file_path, en_min_ch)

    data_dict, en_matrix = data_and_en_dict(
        reader, local_coord_dict, chtype_map, sm_mM_map, FEM_instance, min_ch
    )

    en_min_peak, en_max_peak = extract_photopeak_limits(en_matrix, percentage=0.2)

    av_impact_matrix, av_max = extract_impact_info(
        data_dict, 64, en_min_peak, en_max_peak
    )

    # Writing the av_impact_matrix into file
    with open(file_name, "w") as f:
        f.write(str(av_max) + "\n")
        for value in av_impact_matrix:
            f.write(str(value) + "\n")

    # Plot the energy spectrum
    # plot_single_spectrum(
    #    en_matrix[:, 1], 0, 200, 0, 0, "Detector 1 energy", "Energy (a.u.)", True
    # )


if __name__ == "__main__":
    main()
