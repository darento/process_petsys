#!/usr/bin/env python3

"""Run the basic analysis for PETsys. 
Usage: main.py YAMLCONF INFILE ...

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
from matplotlib import pyplot as plt
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
from src.utils import (
    get_max_num_ch,
    get_neighbour_channels,
    get_num_eng_channels,
    get_max_en_channel,
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
    
def reset_total():
    global EVT_COUNT_T
    EVT_COUNT_T = 0
    
def reset_pf():
    global EVT_COUNT_F
    EVT_COUNT_F = 0


def count_events(
    read_det_evt: Callable,
    chtype_map: dict,    
    min_ch: int,
) -> tuple[dict, np.ndarray]:
    """
    This function processes detector events, filters them based on energy and minimum channel,
    calculates various parameters for each event, and stores the results in a dictionary and a matrix.

    Parameters:
    read_det_evt (Callable): A generator or iterable that yields detector events.
    min_ch (int): The minimum channel number for the filter.

    Returns:    
    """
    event_count = 0
    data_dict = {}
    en_list = []
    start_time = time.time()
    for event in read_det_evt:
        increment_total()
        det1, det2 = event       

        min_ch_filter1 = filter_min_ch(det1, min_ch, chtype_map)
        min_ch_filter2 = filter_min_ch(det2, min_ch, chtype_map)

        if min_ch_filter1 and min_ch_filter2:
            increment_pf()
        event_count += 1            
        if event_count % 10000 == 0:
            print(f"Events processed: {event_count}")
        pass
    print("---------------------")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total events: {EVT_COUNT_T}")
    print(f"Events passing the filter: {EVT_COUNT_F}")
    return EVT_COUNT_F


def main():
    # Read the YAML configuration file
    args = docopt(__doc__)

    infiles = args["INFILE"]

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

    # Initialize an empty dictionary for the accumulated data
    accumulated_data_dict = {}
    accumulated_en_matrix = []
    evt_count = 0


    pos_list = []
    ev_count_list = []
    for infile in infiles:
        print(infile)

        # Read the binary file
        binary_file_path = infile

        file_name = os.path.basename(binary_file_path)
        file_name = file_name.replace(".ldat", "_impactArray.txt")

        reader = read_binary_file(binary_file_path, en_min_ch)
        
        pos_num = int([split for split in file_name.split("_") if "pos" in split][0].replace("pos",""))
        ev_counts = count_events(reader, chtype_map, min_ch)
        
        pos_list.append(pos_num)
        ev_count_list.append(ev_counts)
        reset_total()
        reset_pf()
    
    plt.plot(pos_list, ev_count_list, "o")
    plt.xlabel("Position number")
    plt.ylabel("Number of counts")
    plt.show()
	
	



if __name__ == "__main__":
    main()

