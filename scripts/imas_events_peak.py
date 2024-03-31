#!/usr/bin/env python3

"""Extracts the photopeak from the energy per minimodule and saves it to a file.
Usage: imas_quality_control YAMLCONF SLAB_EN_MAP INFILES ...

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
import pandas as pd
import yaml
from tqdm import tqdm


from src.read_compact import read_binary_file
from src.filters import filter_min_ch, filter_total_energy
from src.mapping_generator import ChannelType, map_factory
from src.fits import fit_gaussian
from src.utils import KevConverter, get_max_en_channel, get_maxEnergy_sm_mM

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
    en_min: float,
    en_max: float,
    sum_rows_cols: bool,
    slab_kev_fn: Callable,
) -> tuple[dict, np.ndarray]:
    """
    This function processes the detector events and count number of events in photopeak.
    """
    start_time = time.time()
    total_energy_filtered_kev = []
    total_energy_kev = []
    for event in read_det_evt:
        increment_total()
        # break to limit the number of events
        if EVT_COUNT_T > 1000000:
            break
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
        # Get the mm with the maximum energy for each detector
        max_det1, energy_det1 = get_maxEnergy_sm_mM(det1, sm_mM_map, chtype_map)
        max_det2, energy_det2 = get_maxEnergy_sm_mM(det2, sm_mM_map, chtype_map)
        min_ch_maxdet1 = filter_min_ch(max_det1, min_ch, chtype_map, sum_rows_cols)
        min_ch_maxdet2 = filter_min_ch(max_det2, min_ch, chtype_map, sum_rows_cols)
        if not (min_ch_maxdet1 and min_ch_maxdet2):
            continue
        # Get the slab index (time channel) with the maximum energy in each detector
        slab_det1 = get_max_en_channel(max_det1, chtype_map, ChannelType.TIME)[2]
        slab_det2 = get_max_en_channel(max_det2, chtype_map, ChannelType.TIME)[2]
        energy_det1_kev = slab_kev_fn(slab_det1) * energy_det1
        energy_det2_kev = slab_kev_fn(slab_det2) * energy_det2
        total_energy_kev.extend((energy_det1_kev, energy_det2_kev))
        en_filter1 = filter_total_energy(energy_det1_kev, en_min, en_max)
        en_filter2 = filter_total_energy(energy_det2_kev, en_min, en_max)
        if not (en_filter1 and en_filter2):
            continue
        increment_pf()
        total_energy_filtered_kev.extend((energy_det1_kev, energy_det2_kev))

    print("---------------------")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total events: {EVT_COUNT_T}")
    print(f"Events passing the filter: {EVT_COUNT_F}")
    return EVT_COUNT_T, EVT_COUNT_F, total_energy_kev


def process_file(
    binary_file_path: str,
    chtype_map: dict,
    sm_mM_map: dict,
    min_ch: int,
    en_min: float,
    en_max: float,
    sum_rows_cols: bool,
    slab_kev_fn: Callable,
) -> dict:
    """
    This function processes the binary file and returns a dictionary of energy list values for each slab.
    """

    print(f"Processing file: {binary_file_path}")
    # Read the binary file
    reader = read_binary_file(binary_file_path)
    # Extract the data dictionary
    slab_dict_energy = extract_data_dict(
        reader,
        chtype_map,
        sm_mM_map,
        min_ch,
        en_min,
        en_max,
        sum_rows_cols,
        slab_kev_fn,
    )
    return slab_dict_energy


def plot_energy_kev(total_energy_kev: list, en_min: float, en_max: float) -> None:
    """
    This function plots the total energy of the system.
    """
    plt.hist(total_energy_kev, bins=1500, range=(0, 1500))
    plt.axvline(x=en_min, color="r", linestyle="--")
    plt.axvline(x=en_max, color="r", linestyle="--")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Number of events")
    plt.title("Energy per channel")
    plt.show()


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
    local_map, sm_mM_map, chtype_map, FEM_instance = map_factory(map_file)

    sum_rows_cols = FEM_instance.sum_rows_cols

    min_ch = int(config["min_ch"])
    print(f"Minimum number of channels: {min_ch}")

    en_min_ch = float(config["en_min_ch"])
    print(f"Minimum energy per channel: {en_min_ch}")

    # Read the energy range
    en_min = float(config["energy_range"][0])
    en_max = float(config["energy_range"][1])
    print(f"Energy range: {en_min} - {en_max}")

    slab_file_path = args["SLAB_EN_MAP"]
    kev_converter = KevConverter(slab_file_path)

    with get_context("spawn").Pool(processes=cpu_count()) as pool:
        args_list = [
            (
                binary_file_path,
                chtype_map,
                sm_mM_map,
                min_ch,
                en_min,
                en_max,
                sum_rows_cols,
                kev_converter.convert,
            )
            for binary_file_path in binary_file_paths
        ]
        results = pool.starmap(process_file, args_list)

    total_energy_kev = []
    total_number_events = 0
    total_pk_events = 0
    print(f"Processing results. Adding to the final dictionary, please wait...")
    for result in tqdm(results):
        if isinstance(result, Exception):
            print(f"Exception in child process: {result}")
        else:
            total_evt, pk_evt, energy_kev = result
            total_number_events += total_evt
            total_pk_events += pk_evt
            total_energy_kev.extend(energy_kev)

    print(f"Total number of events processed: {total_number_events}")
    print(f"Total number of events in photopeak: {total_pk_events}")
    print(f"Photopeak efficiency: {round(total_pk_events/total_number_events*100,2)}%")

    plot_energy_kev(total_energy_kev, en_min, en_max)


if __name__ == "__main__":
    main()
