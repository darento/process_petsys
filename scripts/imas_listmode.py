#!/usr/bin/env python3

"""Extracts the photopeak from the energy per minimodule and saves it to a file.
Usage: imas_quality_control [-d] YAMLCONF SLAB_EN_MAP INFILES ...

Arguments:
    YAMLCONF     File with all parameters to take into account in the scan.
    SLAB_EN_MAP  File with the energy per slab.
    INFILE       Input file to be processed. Must be a compact binary file from PETsys.

Options:
    -d --debug    Debug mode. 
    -h --help     Show this screen.    
"""


from collections import defaultdict
import logging
from multiprocessing import cpu_count, get_context
import os
import time
from typing import BinaryIO, Callable, Tuple
from docopt import docopt
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from natsort import natsorted
import shutil

from src.listmode import CoincidenceV5, LMHeader
from src.read_compact import read_binary_file
from src.filters import filter_min_ch, filter_total_energy
from src.mapping_generator import ChannelType, map_factory
from src.fits import fit_gaussian
from src.utils import KevConverter, get_max_en_channel, get_maxEnergy_sm_mM
from src.slab_nn import neural_net_pcalc, read_category_file


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


def reset_globals():
    global EVT_COUNT_T
    global EVT_COUNT_F
    EVT_COUNT_T = 0
    EVT_COUNT_F = 0


def extract_pair_map(pair_path: str) -> dict:
    pair_cols = ["p", "s0", "s1"]
    pair_map = (
        pd.read_csv(pair_path, sep="\t", names=pair_cols)
        .set_index(pair_cols[1:])
        .to_dict()["p"]
    )
    return pair_map


def get_pixels(
    pos_x: float, pos_y: float, bins_x: np.ndarray, bins_y: np.ndarray
) -> Tuple[int, int]:
    return (
        np.searchsorted(bins_x, pos_x, side="right") - 1,
        np.searchsorted(bins_y, pos_y, side="right") - 1,
    )


def write_header(
    bin_out: BinaryIO,
    xpixels: np.ndarray,
    ypixels: np.ndarray,
) -> None:
    """
    Write the LMHeader object to file.
    """
    sec = "header"
    header = LMHeader(
        identifier="IMAS".encode("utf-8"),
        acqTime=10,
        isotope="Na22".encode("utf-8"),
        detectorSizeX=103.22,
        detectorSizeY=103.22,
        startTime=0,
        measurementTime=10,
        moduleNumber=120,
        ringNumber=5,
        ringDistance=820,
        detectorPixelSizeX=np.diff(xpixels)[0],
        detectorPixelSizeY=np.diff(ypixels)[0],
        version=(9, 5),
        detectorPixelsX=xpixels.size - 1,
        detectorPixelsY=ypixels.size - 1,
    )
    bin_out.write(header)


def nn_lm_loop(
    read_det_evt: Callable,
    chtype_map: dict,
    sm_mM_map: dict,
    local_map: dict,
    pair_map: dict,
    min_ch: int,
    en_min: float,
    en_max: float,
    sum_rows_cols: bool,
    positions_pred: Callable,
    cat_map_XY: dict,
    cat_map_DOI: dict,
    slab_kev_fn: Callable,
    xpixels: np.ndarray,
    ypixels: np.ndarray,
    lm_file_io: BinaryIO,
    debug_flag: bool,
) -> tuple[dict, np.ndarray]:
    """
    This function processes the detector events and generates all the necessary dictionaries for the quality control.
    """
    bunch_size = 10000
    start_time = time.time()

    infer_type = np.dtype([("slab_idx", np.int32), ("Esignals", np.float32, 8)])
    channel_energies = np.zeros(bunch_size, infer_type)

    cat_type = np.dtype([("Y", np.float32), ("DOI", np.float32)])
    cat_events = np.zeros(bunch_size, cat_type)

    coincidences = np.zeros(bunch_size // 2, CoincidenceV5)

    swap_list = []

    for coincidence in coincidences:
        coincidence["amount"] = 1.0

    chan_en_cnt = 0
    if debug_flag:
        total_energy_kev = []
        sm_flood = defaultdict(list)
    for event in read_det_evt:
        increment_total()
        # break to limit the number of events
        if EVT_COUNT_T > 500000 and debug_flag:
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
        time_info_det1 = get_max_en_channel(max_det1, chtype_map, ChannelType.TIME)
        time_info_det2 = get_max_en_channel(max_det2, chtype_map, ChannelType.TIME)
        slab_det1 = time_info_det1[2]
        slab_det2 = time_info_det2[2]
        energy_det1_kev = slab_kev_fn(slab_det1) * energy_det1
        energy_det2_kev = slab_kev_fn(slab_det2) * energy_det2
        en_filter1 = filter_total_energy(energy_det1_kev, en_min, en_max)
        en_filter2 = filter_total_energy(energy_det2_kev, en_min, en_max)
        if not (en_filter1 and en_filter2):
            continue
        sm_det1 = sm_mM_map[slab_det1][0]
        sm_det2 = sm_mM_map[slab_det2][0]
        try:
            pair_sms = (sm_det1, sm_det2)
            pair = pair_map[pair_sms]
            swap = False
        except KeyError:
            try:
                pair_sms = (sm_det2, sm_det1)
                pair = pair_map[pair_sms]
                swap = True
            except KeyError:
                continue
        increment_pf()

        time_min = min(time_info_det1[0], time_info_det2[0])

        # Fill the category per each of of the 10k events in the bunch
        try:
            cat_events[chan_en_cnt * 2]["Y"] = cat_map_XY[slab_det1]
        except KeyError:
            cat_events[chan_en_cnt * 2]["Y"] = 0

        try:
            cat_events[chan_en_cnt * 2]["DOI"] = cat_map_DOI[slab_det1]
        except KeyError:
            cat_events[chan_en_cnt * 2]["DOI"] = 0

        try:
            cat_events[chan_en_cnt * 2 + 1]["Y"] = cat_map_XY[slab_det2]
        except KeyError:
            cat_events[chan_en_cnt * 2 + 1]["Y"] = 0

        try:
            cat_events[chan_en_cnt * 2 + 1]["DOI"] = cat_map_DOI[slab_det2]
        except KeyError:
            cat_events[chan_en_cnt * 2 + 1]["DOI"] = 0

        en_ch_det1 = filter(lambda x: ChannelType.ENERGY in chtype_map[x[2]], max_det1)
        en_ch_det2 = filter(lambda x: ChannelType.ENERGY in chtype_map[x[2]], max_det2)

        # Fill the energy signals per each of the 10k events in the bunch
        for hit in en_ch_det1:
            pos = local_map[hit[2]][2]
            channel_energies[chan_en_cnt * 2]["slab_idx"] = slab_det1
            channel_energies[chan_en_cnt * 2]["Esignals"][pos] = hit[1]
        for hit in en_ch_det2:
            pos = local_map[hit[2]][2]
            channel_energies[chan_en_cnt * 2 + 1]["slab_idx"] = slab_det2
            channel_energies[chan_en_cnt * 2 + 1]["Esignals"][pos] = hit[1]

        coincidences[chan_en_cnt]["time"] = time_min
        coincidences[chan_en_cnt]["pair"] = pair

        if not swap:
            coincidences[chan_en_cnt]["energy1"] = round(energy_det1_kev)
            coincidences[chan_en_cnt]["energy2"] = round(energy_det2_kev)
            coincidences[chan_en_cnt]["dt"] = round(
                time_info_det1[0] - time_info_det2[0]
            )
        else:
            coincidences[chan_en_cnt]["energy1"] = round(energy_det2_kev)
            coincidences[chan_en_cnt]["energy2"] = round(energy_det1_kev)
            coincidences[chan_en_cnt]["dt"] = round(
                time_info_det2[0] - time_info_det1[0]
            )
        swap_list.append(swap)

        # If the bunch is full, predict the positions
        if chan_en_cnt == bunch_size / 2 - 1:
            predicted_xy, predicted_doi = positions_pred(
                channel_energies["slab_idx"],
                channel_energies,
                cat_xy=cat_events["Y"],
                cat_doi=cat_events["DOI"],
            )
            if debug_flag:
                for (
                    slab_id,
                    xy,
                ) in zip(channel_energies["slab_idx"], predicted_xy):
                    sm = sm_mM_map[slab_id][0]
                    sm_flood[sm].append((xy[0], xy[1]))
            for coin in range(chan_en_cnt):
                pixels = tuple(
                    map(
                        lambda xy: get_pixels(*xy, xpixels, ypixels),
                        predicted_xy[2 * coin : 2 * coin + 2],
                    )
                )
                if not swap_list[coin]:
                    coincidences[coin]["xPosition1"] = pixels[0][0]
                    coincidences[coin]["yPosition1"] = pixels[0][1]
                    coincidences[coin]["zPosition1"] = predicted_doi[2 * coin]
                    coincidences[coin]["xPosition2"] = pixels[1][0]
                    coincidences[coin]["yPosition2"] = pixels[1][1]
                    coincidences[coin]["zPosition2"] = predicted_doi[2 * coin + 1]
                else:
                    coincidences[coin]["xPosition1"] = pixels[1][0]
                    coincidences[coin]["yPosition1"] = pixels[1][1]
                    coincidences[coin]["zPosition1"] = predicted_doi[2 * coin + 1]
                    coincidences[coin]["xPosition2"] = pixels[0][0]
                    coincidences[coin]["yPosition2"] = pixels[0][1]
                    coincidences[coin]["zPosition2"] = predicted_doi[2 * coin]

                # print(coincidences[coin])
                # exit(0)
                lm_file_io.write(coincidences[coin])

            # Reset the bunch
            channel_energies = np.zeros(bunch_size, infer_type)
            cat_events = np.zeros(bunch_size, cat_type)
            coincidences = np.zeros(bunch_size // 2, CoincidenceV5)
            swap_list = []
            chan_en_cnt = 0
        chan_en_cnt += 1
        if debug_flag:
            total_energy_kev.extend((energy_det1_kev, energy_det2_kev))

    if debug_flag:
        debug_plots(total_energy_kev, sm_flood)

    print("---------------------")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total events: {EVT_COUNT_T}")
    print(f"Events passing the filter: {EVT_COUNT_F}")
    return EVT_COUNT_F


def debug_plots(total_energy_kev: list, sm_flood: dict):
    n, bins, _ = plt.hist(total_energy_kev, bins=1500, range=(0, 1500))
    x, y, pars, _, _ = fit_gaussian(n, bins, cb=16)
    mu, sigma = pars[1], pars[2]
    plt.plot(x, y, "-r", label="fit")
    plt.legend([f"Energy res: {round(2.35*sigma/mu*100,2)}%"])
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.title("Total energy")
    plt.show()

    for sm, flood in sm_flood.items():
        plt.hist2d(
            *zip(*flood),
            bins=(200, 200),
            range=[[0, 108], [0, 108]],
            cmap="jet",
            norm=LogNorm(),
        )
        plt.colorbar()
        plt.title(f"SM {sm}")
        plt.show()


def initialize_nn(local_map: dict) -> Callable:
    """
    This function initializes the neural network for the position prediction.
    """
    nn_yfile = "/scratch/imas_files_cal/IMASY.h5"
    nn_doifile = "/scratch/imas_files_cal/IMASDOI.h5"
    positions_pred = neural_net_pcalc("IMAS", nn_yfile, nn_doifile, local_map)
    return positions_pred


def process_file(
    binary_file_path: str,
    chtype_map: dict,
    sm_mM_map: dict,
    local_map: dict,
    pair_map: dict,
    min_ch: int,
    en_min: float,
    en_max: float,
    sum_rows_cols: bool,
    positions_pred: Callable,
    cat_map_XY: dict,
    cat_map_DOI: dict,
    slab_kev_fn: Callable,
    xpixels: np.ndarray,
    ypixels: np.ndarray,
    lm_dir: str,
    debug_flag: bool,
) -> dict:
    """
    This function processes the binary file and returns a dictionary of energy list values for each slab.
    """
    reset_globals()
    print(f"Processing file: {binary_file_path}")
    # Read the binary file
    reader = read_binary_file(binary_file_path)

    lm_file = lm_dir + os.path.basename(binary_file_path).replace(".ldat", ".lm")
    lm_file_io = open(lm_file, "wb")
    # Extract the data dictionary
    slab_dict_energy = nn_lm_loop(
        reader,
        chtype_map,
        sm_mM_map,
        local_map,
        pair_map,
        min_ch,
        en_min,
        en_max,
        sum_rows_cols,
        positions_pred,
        cat_map_XY,
        cat_map_DOI,
        slab_kev_fn,
        xpixels,
        ypixels,
        lm_file_io,
        debug_flag,
    )
    lm_file_io.close()
    return slab_dict_energy, lm_file


def main():
    # Read the YAML configuration file
    args = docopt(__doc__)

    # Read the binary file
    binary_file_paths = args["INFILES"]

    with open(args["YAMLCONF"], "r") as f:
        config = yaml.safe_load(f)

    # Debug mode
    if args["--debug"]:
        debug_flag = True
        print("Debug mode enabled")
    else:
        debug_flag = False
        print("Debug mode disabled")

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

    pair_path = "/scratch/imas_files_cal/1DAQ/pares_5SR.txt"
    pair_map = extract_pair_map(pair_path)

    cat_map_XY = read_category_file("/scratch/imas_files_cal/1DAQ/CategoryMapRTPY.bin")
    cat_map_DOI = read_category_file(
        "/scratch/imas_files_cal/1DAQ/CategoryMapRTPDOI.bin"
    )
    print(f"Category maps read.")

    pixels_x = (0, 103.22, 101)
    pixels_y = (0, 103.22, 101)
    xpixels = np.linspace(*pixels_x[:2], int(pixels_x[2]))
    ypixels = np.linspace(*pixels_y[:2], int(pixels_y[2]))

    positions_pred = initialize_nn(local_map)

    slab_file_path = args["SLAB_EN_MAP"]
    kev_converter = KevConverter(slab_file_path)

    lm_dir = "/mnt/data/000_samba_lin/LaFe_acquisitions_win/LMs/"
    if not os.path.exists(lm_dir):
        os.makedirs(lm_dir)
    num_cpu_used = 13
    with get_context("spawn").Pool(processes=num_cpu_used) as pool:
        args_list = [
            (
                binary_file_path,
                chtype_map,
                sm_mM_map,
                local_map,
                pair_map,
                min_ch,
                en_min,
                en_max,
                sum_rows_cols,
                positions_pred,
                cat_map_XY,
                cat_map_DOI,
                kev_converter.convert,
                xpixels,
                ypixels,
                lm_dir,
                debug_flag,
            )
            for binary_file_path in binary_file_paths
        ]
        results = pool.starmap(process_file, args_list)

    total_number_events = 0
    lm_files = []
    print(f"Compiling all lms into a single one, please wait...")
    for result in tqdm(results):
        if isinstance(result, Exception):
            print(f"Exception in child process: {result}")
        else:
            evt_filtered, lm_path = result
            total_number_events += evt_filtered
            lm_files.append(lm_path)
    # Sort the filenames in natural order
    lm_files = natsorted(lm_files)
    final_lm_name = "_".join(lm_files[0].split("_")[:-1]) + "_all.lm"
    log_lm_name = "_".join(lm_files[0].split("_")[:-1]) + "_log.txt"
    logging.basicConfig(filename=log_lm_name, level=logging.INFO)

    # Open the final output file
    with open(final_lm_name, "wb") as outfile:
        # Write header
        write_header(outfile, xpixels, ypixels)
        # Iterate over the sorted filenames
        for filename in lm_files:
            # Log the input file name and size
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            logging.info(f"Input file: {filename}, size: {file_size_mb:.2f} MB")
            # Open each file and copy its contents to the final output file
            with open(filename, "rb") as infile:
                shutil.copyfileobj(infile, outfile)
            # Delete the file
            os.remove(filename)

    # Log the final output file name
    final_file_size_mb = os.path.getsize(final_lm_name) / (1024 * 1024)
    logging.info(
        f"Final output file: {final_lm_name}, size: {final_file_size_mb:.2f} MB"
    )
    print(f"Total number of events processed to LM: {total_number_events}")


if __name__ == "__main__":
    main()
