#!/usr/bin/env python3

"""Extracts the photopeak from the energy per minimodule and saves it to a file.
Usage: imas_slab_en_cal YAMLCONF INFILE

Arguments:
    YAMLCONF  File with all parameters to take into account in the scan.
    INFILE    Input file to be processed. Must be a compact binary file from PETsys.

Options:
    -h --help     Show this screen.    
"""


from collections import defaultdict
from itertools import chain
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
        min_ch_filter1 = filter_min_ch(det1, min_ch, chtype_map)
        min_ch_filter2 = filter_min_ch(det2, min_ch, chtype_map)
        if EVT_COUNT_T % 100000 == 0:
            count_time = time.time()
            print(
                f"Events processed / passing filter: {EVT_COUNT_T} / {EVT_COUNT_F}\tTime taken: {round(count_time - start_time,1)} seconds"
            )
        if not (min_ch_filter1 and min_ch_filter2):
            continue
        max_det1, energy_det1 = get_maxEnergy_sm_mM(det1, sm_mM_map, chtype_map)
        max_det2, energy_det2 = get_maxEnergy_sm_mM(det2, sm_mM_map, chtype_map)
        min_ch_maxdet1 = filter_min_ch(max_det1, min_ch, chtype_map)
        min_ch_maxdet2 = filter_min_ch(max_det2, min_ch, chtype_map)
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


def write_mm_cal(slab_dict_photopeak: dict):
    """
    This function writes the photopeak per minimodule to a file.

    Parameters:
    slab_dict_photopeak (dict): A dictionary containing the photopeak per minimodule.
    """
    file_name = "slab_en_cal.txt"
    with open(file_name, "w") as f:
        f.write("ID\tmu\tsigma\n")
        for key, value in sorted(slab_dict_photopeak.items()):
            f.write(f"{key}\t{round(value[0],3)}\t{round(value[1],3)}\n")
    print(f"File {file_name} written.")


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
    _, sm_mM_map, chtype_map, _ = map_factory(map_file)

    # Read the energy range
    en_min = float(config["energy_range"][0])
    en_max = float(config["energy_range"][1])
    print(f"Energy range: {en_min} - {en_max}")

    min_ch = int(config["min_ch"])
    print(f"Minimum number of channels: {min_ch}")

    en_min_ch = float(config["en_min_ch"])
    print(f"Minimum energy per channel: {en_min_ch}")

    reader = read_binary_file(binary_file_path, en_min_ch)

    slab_dict_energy = extract_data_dict(reader, chtype_map, sm_mM_map, min_ch)

    slab_dict_energy = extract_photopeak_slab(slab_dict_energy)
    write_mm_cal(slab_dict_energy)


if __name__ == "__main__":
    main()
