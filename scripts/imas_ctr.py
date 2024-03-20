#!/usr/bin/env python3

"""Extracts the photopeak from the energy per minimodule and saves it to a file.
Usage: imas_ctr YAMLCONF [--sk SKEW] SLAB_EN_MAP INFILE

Arguments:
    YAMLCONF       File with all parameters to take into account in the scan.
    SLAB_EN_MAP    File with the energy per slab.
    SKEW_MAP       File with the skew values.
    INFILE         Input file to be processed. Must be a compact binary file from PETsys.

Options:
    --sk SKEW      File with the skew values.
    -h --help      Show this screen.   
"""


from collections import defaultdict
import os
import pickle
import re
import time
from typing import Callable
from docopt import docopt
from matplotlib import pyplot as plt
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.constants import c as c_vac
from multiprocessing import cpu_count, get_context

from src.detector_features import calculate_centroid_sum

from src.fem_handler import FEMBase
from src.read_compact import read_binary_file
from src.filters import (
    filter_total_energy,
    filter_min_ch,
)

from src.mapping_generator import ChannelType, map_factory

from src.fits import fit_gaussian, mean_around_max, shift_to_centres
from src.utils import (
    KevConverter,
    get_maxEnergy_sm_mM,
    get_max_en_channel,
    read_skew,
)

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
    FEM_instance: FEMBase,
    chtype_map: dict,
    sm_mM_map: dict,
    skew_map: dict,
    slab_kev_fn: Callable,
    min_ch: int,
    en_min: float,
    en_max: float,
) -> tuple[dict, np.ndarray]:
    """
    This function processes the detector events and returns a dictionary of dt values for each slab
    after comparing the time difference between the two detectors with the theoretical time difference.
    """
    global EVT_COUNT_T
    global EVT_COUNT_F
    EVT_COUNT_T = 0
    EVT_COUNT_F = 0
    start_time = time.time()
    dt_list = []
    sum_rows_cols = FEM_instance.sum_rows_cols
    total_energy = []
    for event in read_det_evt:
        increment_total()
        det1, det2 = event
        min_ch_filter1 = filter_min_ch(det1, min_ch, chtype_map, sum_rows_cols)
        min_ch_filter2 = filter_min_ch(det2, min_ch, chtype_map, sum_rows_cols)
        if EVT_COUNT_T > 100000000:
            break
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
        tch_det1 = get_max_en_channel(max_det1, chtype_map, ChannelType.TIME)
        tch_det2 = get_max_en_channel(max_det2, chtype_map, ChannelType.TIME)
        slab_det1 = tch_det1[2]
        slab_det2 = tch_det2[2]
        energy_det1_kev = slab_kev_fn(slab_det1) * energy_det1
        energy_det2_kev = slab_kev_fn(slab_det2) * energy_det2
        en_filter1 = filter_total_energy(energy_det1_kev, en_min, en_max)
        en_filter2 = filter_total_energy(energy_det2_kev, en_min, en_max)
        total_energy.extend((energy_det1_kev, energy_det2_kev))
        if not (en_filter1 and en_filter2):
            continue
        skew_det1 = skew_map[slab_det1]
        skew_det2 = skew_map[slab_det2]
        dt = (tch_det1[0] - skew_det1) - (tch_det2[0] - skew_det2)
        dt_list.append(dt)
        increment_pf()
    n, bins, _ = plt.hist(total_energy, bins=1000, range=(0, 1000))
    x, y, pars, _, _ = fit_gaussian(n, bins, cb=16)
    mu, sigma = pars[1], pars[2]
    plt.plot(x, y, "-r", label="fit")
    plt.legend([f"Energy res: {round(2.35*sigma/mu*100,2)}%"])
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.title("Total energy")
    plt.show()

    print("---------------------")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total events: {EVT_COUNT_T}")
    print(f"Events passing the filter: {EVT_COUNT_F}")
    return dt_list


def read_skew_init(chtype_map: dict):
    """
    This function reads the skew init file and returns a dictionary with the skew values with a relax factor.
    """
    relax_factor = 1.0
    skew_init_dict = {}
    with open("skew_1ASIC_init.txt", "r") as f:
        for line in f:
            ch, skew = line.split()
            skew_init_dict[int(ch)] = float(skew) * relax_factor
    skew_dict = {}
    for ch, ch_type in chtype_map.items():
        if ChannelType.TIME in ch_type:
            ch_init = ch % 64
            skew_dict[ch] = skew_init_dict[ch_init]
    return skew_dict


def main():
    # Read the YAML configuration file
    args = docopt(__doc__)

    # Data binary file
    input_file_path = args["INFILE"]
    file_name = os.path.basename(input_file_path)
    # File with the energy per slab
    slab_file_path = args["SLAB_EN_MAP"]

    with open(args["YAMLCONF"], "r") as f:
        config = yaml.safe_load(f)

    # Read the mapping file
    map_file = config["map_file"]

    # Get the coordinates of the channels
    local_map, sm_mM_map, chtype_map, FEM_instance = map_factory(map_file)

    # Read the energy range
    en_min = float(config["energy_range"][0])
    en_max = float(config["energy_range"][1])
    print(f"Energy range: {en_min} - {en_max}")

    min_ch = int(config["min_ch"])
    print(f"Minimum number of Energy channels (+1 Time): {min_ch}")

    en_min_ch = float(config["en_min_ch"])
    print(f"Minimum energy per channel: {en_min_ch}")

    kev_converter = KevConverter(slab_file_path)

    reader = read_binary_file(input_file_path)
    print(f"Skew file: {args['--sk']}")

    if args["--sk"] == None:
        skew_map = {
            ch: 0 for ch, ch_type in chtype_map.items() if ChannelType.TIME in ch_type
        }
    else:
        if args["--sk"] == "skew_1ASIC.txt":
            skew_map = read_skew_init(chtype_map)
        else:
            skew_map = read_skew(args["--sk"])
    dt_list = extract_data_dict(
        reader,
        FEM_instance,
        chtype_map,
        sm_mM_map,
        skew_map,
        kev_converter.convert,
        min_ch,
        en_min,
        en_max,
    )

    n, bins, _ = plt.hist(
        dt_list,
        bins=200,
        range=(-10000, 10000),
        label=f"Point source dt distribution",
    )
    plt.xlabel("Time difference (ps)")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
