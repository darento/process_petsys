#!/usr/bin/env python3

"""Run the basic analysis for PETsys. 
Usage: main.py YAMLCONF INFILE

Arguments:
    YAMLCONF  File with all parameters to take into account in the scan.
    INFILE    Input file to be processed. Must be a compact binary file from PETsys.

Options:
    -h --help     Show this screen.    
"""


import time
from docopt import docopt
import yaml

from src.read_compact import read_binary_file
from src.filters import filter_total_energy
import src.filters as filters

from src.detector_features import calculate_centroid, calculate_total_energy
from src.mapping_generator import map_factory
from src.plots import plot_chan_position


def main():
    # Read the YAML configuration file
    args = docopt(__doc__)

    # Read the binary file
    binary_file_path = args["INFILE"]

    with open(args["YAMLCONF"], "r") as f:
        config = yaml.safe_load(f)

    # Read the mapping file
    map_file = config["map_file"]

    # Get the coordinates of the channels
    local_coord_dict, num_ASICs = map_factory(map_file)

    # Plot the coordinates of the channels
    # plot_chan_position(local_coord_dict, num_ASICs)

    # Read the energy range
    en_min = float(config["energy_range"][0])
    en_max = float(config["energy_range"][1])
    print(f"Energy range: {en_min} - {en_max}")

    min_ch = int(config["min_ch"])
    print(f"Minimum number of channels: {min_ch}")

    en_min_ch = float(config["en_min_ch"])
    print(f"Minimum energy per channel: {en_min_ch}")

    # test the time taken to read the entire file using read_binary_file function
    start_time = time.time()
    event_count = 0
    for event in read_binary_file(binary_file_path, min_ch, en_min_ch):
        det1, det2 = event
        det1_en = calculate_total_energy(det1)
        en_filter = filter_total_energy(det1_en, en_min, en_max)
        # print(det1, det1_en, det2, en_filter)
        if en_filter:
            calculate_centroid(local_coord_dict, det1, x_rtp=1, y_rtp=1)
        # print(f"En filter: {filter_total_energy(det1, 50)}")
        # print(f"Lenghts: {len(det1)}, {len(det2)}")
        # time.sleep(1)
        event_count += 1
        if event_count % 10000 == 0:
            print(f"Events processed: {event_count}")
        pass
    print("---------------------")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total events: {filters.EVT_COUNT_T}")
    print(f"Events passing the filter: {filters.EVT_COUNT_F}")


if __name__ == "__main__":
    main()
