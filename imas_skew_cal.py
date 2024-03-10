#!/usr/bin/env python3

"""Extracts the photopeak from the energy per minimodule and saves it to a file.
Usage: imas_skew_cal YAMLCONF INFILE SLAB_EN_MAP

Arguments:
    YAMLCONF       File with all parameters to take into account in the scan.
    INFILE         Input file to be processed. Must be a compact binary file from PETsys.
    SLAB_EN_MAP    File with the energy per slab.

Options:
    -h --help     Show this screen.    
"""


from collections import defaultdict
import os
import time
from typing import Callable
from docopt import docopt
from matplotlib import pyplot as plt
import numpy as np
import yaml

from src.fem_handler import FEMBase
from src.read_compact import read_binary_file
from src.filters import (
    filter_max_sm,
    filter_total_energy,
    filter_min_ch,
    filter_single_mM,
)
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
from src.utils import get_maxEnergy_sm_mM, get_num_eng_channels, get_max_en_channel

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


def extract_data_dict(
    read_det_evt: Callable,
    local_coord_map: dict,
    chtype_map: dict,
    sm_mM_map: dict,
    FEM_instance: FEMBase,
    min_ch: int,
) -> tuple[dict, np.ndarray]:
    """
    This function processes the detector events and calculates the energy per minimodule.

    Parameters:
    read_det_evt (Callable): A generator or iterable that yields detector events.
    local_coord_dict (dict): A dictionary mapping detector channels to local coordinates.
    chtype_map (dict): A dictionary mapping the channel type to the channel number.
    FEM_instance (FEMBase): A dictionary representing a Finite Element Model instance.
    min_ch (int): The minimum channel number for the filter.

    Returns:
    dict: A dictionary containing the energy per minimodule.
    """
    event_count = 0
    start_time = time.time()
    sm_mm_dict_count = defaultdict(int)
    sm_mm_dict_energy = defaultdict(list)
    for event in read_det_evt:
        increment_total()
        det1, det2 = event
        min_ch_filter1 = filter_min_ch(det1, min_ch, chtype_map)
        min_ch_filter2 = filter_min_ch(det2, min_ch, chtype_map)
        event_count += 1
        if event_count % 100000 == 0:
            count_time = time.time()
            print(
                f"Events processed: {event_count}\tTime taken: {round(count_time - start_time,1)} seconds"
            )
        if not (min_ch_filter1 and min_ch_filter2):
            continue
        max_det1, energy_det1 = get_maxEnergy_sm_mM(det1, sm_mM_map, chtype_map)
        max_det2, energy_det2 = get_maxEnergy_sm_mM(det2, sm_mM_map, chtype_map)
        max_sm_det1 = sm_mM_map[max_det1[0][2]][0]
        max_mM_det1 = sm_mM_map[max_det1[0][2]][1]
        max_sm_det2 = sm_mM_map[max_det2[0][2]][0]
        max_mM_det2 = sm_mM_map[max_det2[0][2]][1]
        sm_mm_dict_count[(max_sm_det1, max_mM_det1)] += 1
        sm_mm_dict_count[(max_sm_det2, max_mM_det2)] += 1
        sm_mm_dict_energy[(max_sm_det1, max_mM_det1)].append(
            energy_det1
        )  # Change this line
        sm_mm_dict_energy[(max_sm_det2, max_mM_det2)].append(
            energy_det2
        )  # Add this line
        increment_pf()

    print("---------------------")
    end_time = time.time()
    print(len(sm_mm_dict_count))
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total events: {EVT_COUNT_T}")
    print(f"Events passing the filter: {EVT_COUNT_F}")
    return sm_mm_dict_energy


def extract_photopeak_mm(sm_mm_dict_energy: dict) -> dict:
    """
    This function extracts the photopeak from the energy per minimodule.

    Parameters:
    sm_mm_dict_energy (dict): A dictionary containing the energy per minimodule.
    sm_mm_dict_count (dict): A dictionary containing the count per minimodule.

    Returns:
    dict: A dictionary containing the photopeak per minimodule.
    """
    sm_mm_dict_photopeak = {}
    for key, value in sm_mm_dict_energy.items():
        if len(value) < 100:
            sm_mm_dict_photopeak[key] = (0, 0)
            continue
        n, bins = np.histogram(value, bins=200, range=(0, 200))
        try:
            x, y, pars, _, _ = fit_gaussian(n, bins)
            mu, sigma = pars[1], pars[2]
        except RuntimeError:
            mu, sigma = 0, 0
        sm_mm_dict_photopeak[key] = (mu, sigma)
    return sm_mm_dict_photopeak


def write_mm_cal(sm_mm_dict_photopeak: dict):
    """
    This function writes the photopeak per minimodule to a file.

    Parameters:
    sm_mm_dict_photopeak (dict): A dictionary containing the photopeak per minimodule.
    """
    file_name = "mm_en_cal.txt"
    with open(file_name, "w") as f:
        f.write("sm\tmM\tmu\tsigma\n")
        for key, value in sorted(sm_mm_dict_photopeak.items()):
            f.write(f"{key[0]}\t{key[1]}\t{round(value[0],3)}\t{round(value[1],3)}\n")
    print(f"File {file_name} written.")


def main():
    # Read the YAML configuration file
    args = docopt(__doc__)

    # Data binary file
    binary_file_path = args["INFILE"]
    # File with the energy per slab
    slab_file_path = args["SLAB_EN_MAP"]

    file_name = os.path.basename(binary_file_path)
    file_name = file_name.replace(".ldat", "_impactArray.txt")

    with open(args["YAMLCONF"], "r") as f:
        config = yaml.safe_load(f)

    # Read the mapping file
    map_file = config["map_file"]

    # Get the coordinates of the channels
    local_coord_map, sm_mM_map, chtype_map, FEM_instance = map_factory(map_file)

    # Read the energy range
    en_min = float(config["energy_range"][0])
    en_max = float(config["energy_range"][1])
    print(f"Energy range: {en_min} - {en_max}")

    min_ch = int(config["min_ch"])
    print(f"Minimum number of channels: {min_ch}")

    en_min_ch = float(config["en_min_ch"])
    print(f"Minimum energy per channel: {en_min_ch}")

    reader = read_binary_file(binary_file_path, en_min_ch)

    slab_en_map = read_slab_energy_map(slab_file_path)

    sm_mm_dict_energy = extract_data_dict(
        reader, local_coord_map, chtype_map, sm_mM_map, FEM_instance, min_ch
    )

    sm_mm_dict_photopeak = extract_photopeak_mm(sm_mm_dict_energy)
    write_mm_cal(sm_mm_dict_photopeak)


if __name__ == "__main__":
    main()
