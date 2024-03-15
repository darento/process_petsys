#!/usr/bin/env python3

"""Extracts the photopeak from the energy per minimodule and saves it to a file.
Usage: imas_slab_en_cal YAMLCONF INFILES ...

Arguments:
    YAMLCONF  File with all parameters to take into account in the scan.
    INFILE    Input file to be processed. Must be a compact binary file from PETsys.

Options:
    -h --help     Show this screen.    
"""


from collections import defaultdict
from itertools import chain
from multiprocessing import cpu_count, get_context
import os
import time
from typing import Callable
from docopt import docopt
from matplotlib import pyplot as plt
import numpy as np
import yaml
from src.detector_features import calculate_total_energy

from src.read_compact import read_binary_file
from src.filters import filter_min_ch


from src.mapping_generator import ChannelType, map_factory

from src.fits import fit_gaussian
from src.utils import get_max_en_channel, get_maxEnergy_sm_mM

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
    chtype_map: dict,
    sm_mM_map: dict,
    min_ch: int,
    sum_rows_cols: bool,
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
    slab_dict_count = defaultdict(int)
    slab_dict_energy = defaultdict(list)
    for event in read_det_evt:
        increment_total()
        det1, det2 = event
        min_ch_filter1 = filter_min_ch(det1, min_ch, chtype_map, sum_rows_cols)
        min_ch_filter2 = filter_min_ch(det2, min_ch, chtype_map, sum_rows_cols)
        if EVT_COUNT_T % 100000 == 0:
            count_time = time.time()
            print(
                f"Events processed / passing filter: {EVT_COUNT_T} / {EVT_COUNT_F}\tTime taken: {round(count_time - start_time,1)} seconds"
            )
        if not (min_ch_filter1 and min_ch_filter2):
            continue
        max_det1, energy_det1 = get_maxEnergy_sm_mM(det1, sm_mM_map, chtype_map)
        max_det2, energy_det2 = get_maxEnergy_sm_mM(det2, sm_mM_map, chtype_map)
        min_ch_maxdet1 = filter_min_ch(max_det1, min_ch, chtype_map, sum_rows_cols)
        min_ch_maxdet2 = filter_min_ch(max_det2, min_ch, chtype_map, sum_rows_cols)
        if not (min_ch_maxdet1 and min_ch_maxdet2):
            continue
        slab_det1 = get_max_en_channel(max_det1, chtype_map, ChannelType.TIME)[2]
        slab_det2 = get_max_en_channel(max_det2, chtype_map, ChannelType.TIME)[2]
        slab_dict_count[slab_det1] += 1
        slab_dict_count[slab_det2] += 1
        slab_dict_energy[slab_det1].append(energy_det1)
        slab_dict_energy[slab_det2].append(energy_det2)
        increment_pf()

    print("---------------------")
    end_time = time.time()
    print(len(slab_dict_count))
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total events: {EVT_COUNT_T}")
    print(f"Events passing the filter: {EVT_COUNT_F}")
    return slab_dict_energy


def extract_photopeak_slab(slab_dict_energy: dict) -> dict:
    """
    This function extracts the photopeak from the energy per minimodule.

    Parameters:
    slab_dict_energy (dict): A dictionary containing the energy per minimodule.
    slab_dict_photopeak (dict): A dictionary containing the count per minimodule.

    Returns:
    dict: A dictionary containing the photopeak per minimodule.
    """
    slab_dict_photopeak = {}
    for key, value in slab_dict_energy.items():
        n, bins = np.histogram(value, bins=200, range=(0, 200))
        # n, bins, patches = plt.hist(value, bins=200, range=(0, 200))
        try:
            x, y, pars, _, _ = fit_gaussian(n, bins)
            mu, sigma = pars[1], pars[2]
        except RuntimeError:
            mu, sigma = 0, 0
            # plt.legend([f"Slab: {key}\nError fitting the Gaussian"])
            # plt.show()
        slab_dict_photopeak[key] = (mu, sigma)
        # plt.plot(x, y, "-r", label="fit")
        # plt.legend([f"Slab: {key}\nEnergy res: {round(2.35*sigma/mu*100,2)}%"])
        # plt.show()
    return slab_dict_photopeak


def write_mm_cal(slab_dict_photopeak: dict, chtype_map: dict):
    """
    This function writes the photopeak per minimodule to a file.

    Parameters:
    slab_dict_photopeak (dict): A dictionary containing the photopeak per minimodule.
    """
    time_channels = [
        ch for ch in chtype_map.keys() if ChannelType.TIME in chtype_map[ch]
    ]
    file_name = "slab_en_cal.txt"
    with open(file_name, "w") as f:
        f.write("ID\tmu\tsigma\n")
        for tch in sorted(time_channels):
            if tch in slab_dict_photopeak.keys():
                f.write(
                    f"{tch}\t{round(slab_dict_photopeak[tch][0],3)}\t{round(slab_dict_photopeak[tch][1],3)}\n"
                )
            else:
                f.write(f"{tch}\t0\t0\n")
    print(f"File {file_name} written.")


def process_file(
    binary_file_path: str,
    chtype_map: dict,
    sm_mM_map: dict,
    min_ch: int,
    sum_rows_cols: bool,
) -> dict:
    """
    This function processes the binary file and returns a dictionary of energy list values for each slab.
    """

    print(f"Processing file: {binary_file_path}")
    # Read the binary file
    reader = read_binary_file(binary_file_path)
    # Extract the data dictionary
    slab_dict_energy = extract_data_dict(
        reader, chtype_map, sm_mM_map, min_ch, sum_rows_cols
    )
    return slab_dict_energy


def main():
    # Read the YAML configuration file
    args = docopt(__doc__)

    # Read the binary file
    binary_file_paths = args["INFILES"]

    with open(args["YAMLCONF"], "r") as f:
        config = yaml.safe_load(f)

    # Read the mapping file
    map_file = config["map_file"]

    # Get the coordinates of the channels
    _, sm_mM_map, chtype_map, FEM_instance = map_factory(map_file)

    sum_rows_cols = FEM_instance.sum_rows_cols

    min_ch = int(config["min_ch"])
    print(f"Minimum number of channels: {min_ch}")

    en_min_ch = float(config["en_min_ch"])
    print(f"Minimum energy per channel: {en_min_ch}")

    with get_context("spawn").Pool(processes=cpu_count()) as pool:
        args_list = [
            (binary_file_path, chtype_map, sm_mM_map, min_ch, sum_rows_cols)
            for binary_file_path in binary_file_paths
        ]
        results = pool.starmap(process_file, args_list)

    slab_dict_energy = defaultdict(list)
    for result in results:
        if isinstance(result, Exception):
            print(f"Exception in child process: {result}")
        else:
            for key, value in result.items():
                slab_dict_energy[key].extend(value)

    slab_dict_photopeak = extract_photopeak_slab(slab_dict_energy)
    write_mm_cal(slab_dict_photopeak, chtype_map)


if __name__ == "__main__":
    main()
