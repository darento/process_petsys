#!/usr/bin/env python3

"""Extracts the photopeak from the energy per minimodule and saves it to a file.
Usage: imas_skew_cal [--it NITER] [--firstit FITER] [--sk SKEW] [-r] YAMLCONF SLAB_EN_MAP INFILES ...

Arguments:
    YAMLCONF       File with all parameters to take into account in the scan.
    SLAB_EN_MAP    File with the energy per slab.
    INFILE         Input file to be processed. Must be a compact binary file from PETsys.

Options:
    --it NITER       Number of iterations to perform [default: 3].
    --firstit FITER  First iteration to be used in the calculation [default: 0].
    --sk SKEW        Existing skew file to be used if FITER != 0.
    -r               Recalculate skew from it 0 assuming dt pickle file available.
    -h --help        Show this screen.   
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

# Total number of events
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


def sm_map_gen(ring_r: float, ring_z: list, ring_yx: dict) -> dict:
    """
    This function generates a map of the centre position of the supermodules in polar coordinates.
    """
    sm_center_map = {}
    for i, rz in enumerate(ring_z):
        for sm, yx in ring_yx.items():
            sm_total = sm + i * len(ring_yx)
            sm_center_map[sm_total] = {}
            sm_r, sm_ang = sm_centre_pos(ring_r, yx)
            sm_center_map[sm_total]["sm_r"] = sm_r
            sm_center_map[sm_total]["sm_ang"] = sm_ang
            sm_center_map[sm_total]["sm_z"] = rz
    return sm_center_map


def sm_centre_pos(sm_r: float, sm_yx: list[float, float]) -> Callable:
    """
    This function returns the centre radial and angular position of the center of the supermodule.
    """
    return sm_r, np.arctan2(*sm_yx)


def local_to_global(x, y, sm_glob_par):
    """
    This function converts local coordinates to global coordinates.
    """
    x_center, y_center = 48, 48
    x -= x_center
    y -= sm_glob_par["sm_z"] + y_center
    sm_rot = R.from_euler("z", sm_glob_par["sm_ang"])
    local_coords = np.array([sm_glob_par["sm_r"], x, -y])
    global_coords = sm_rot.apply(local_coords)

    return global_coords[0], global_coords[1], global_coords[2]


# Define the cylinder intersection check function
def cylinder_intersection_check(
    line_start: np.ndarray, line_end: np.ndarray, radius: float
) -> bool:
    """
    This function checks if a line intersects the virtual cylinder along the pipe rotation
    """
    c_mm_per_ps = c_vac * 1000 / 1e12
    line_length = np.linalg.norm(line_end - line_start)
    line_vec = line_end - line_start
    A = np.dot(line_vec[:2], line_vec[:2])
    B = 2 * np.dot(line_start[:2], line_vec[:2])
    C = np.dot(line_start[:2], line_start[:2]) - radius**2
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return False, 0, 0

    # Assuming scanner_center is a numpy array representing the coordinates of the scanner center
    scanner_center = np.array([0, 0, 0])

    # Calculate vectors
    vec_center_to_start = line_start - scanner_center
    vec_center_to_end = line_end - scanner_center

    # Normalize vectors
    vec_center_to_start /= np.linalg.norm(vec_center_to_start)
    vec_center_to_end /= np.linalg.norm(vec_center_to_end)

    # Calculate angle in radians
    angle_rad = np.arccos(
        np.clip(np.dot(vec_center_to_start, vec_center_to_end), -1.0, 1.0)
    )

    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)

    # print(f"Angle deviation: {angle_deg} degrees")

    ang_acceptance = 20

    if angle_deg < 180 - ang_acceptance:
        return False, 0, 0

    t1 = (-B - np.sqrt(discriminant)) / (2 * A)
    t2 = (-B + np.sqrt(discriminant)) / (2 * A)

    tof1 = int(line_length * (2 * t1 - 1) / c_mm_per_ps)
    tof2 = int(line_length * (2 * t2 - 1) / c_mm_per_ps)

    return (0 <= t1 <= 1) or (0 <= t2 <= 1), tof1, tof2


def crossing_visualization(
    intersection: bool,
    pipe_radius: float,
    line_start: np.ndarray,
    line_end: np.ndarray,
):
    height = 720.0

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Cylinder
    z = np.linspace(-height / 2, height / 2, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = pipe_radius * np.cos(theta_grid)
    y_grid = pipe_radius * np.sin(theta_grid)
    ax.plot_surface(
        x_grid, y_grid, z_grid, alpha=0.5, color="blue" if intersection else "red"
    )
    # Line
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        [line_start[2], line_end[2]],
        color="black",
        linewidth=3,
        label="Line",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f'Line Intersection with Cylinder: {"Yes" if intersection else "No"}')
    ax.legend()
    plt.show()


def extract_data_dict(
    read_det_evt: Callable,
    FEM_instance: FEMBase,
    chtype_map: dict,
    sm_mM_map: dict,
    local_map: dict,
    sm_center_map: dict,
    slab_kev_fn: Callable,
    min_ch: int,
    en_min: float,
    en_max: float,
    pipe_radius: float,
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
    dt_evt_dict = defaultdict(list)
    sum_rows_cols = FEM_instance.sum_rows_cols
    total_energy = []
    for event in read_det_evt:
        increment_total()
        det1, det2 = event
        min_ch_filter1 = filter_min_ch(det1, min_ch, chtype_map, sum_rows_cols)
        min_ch_filter2 = filter_min_ch(det2, min_ch, chtype_map, sum_rows_cols)
        # if EVT_COUNT_T > 300000000:
        #     break
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
        # Time channel maximum energy det1 (tstp, en, chID)
        tch_det1 = get_max_en_channel(max_det1, chtype_map, ChannelType.TIME)
        # Time channel maximum energy det2 (tstp, en, chID)
        tch_det2 = get_max_en_channel(max_det2, chtype_map, ChannelType.TIME)
        slab_det1 = tch_det1[2]  # Time channel ID det1
        slab_det2 = tch_det2[2]  # Time channel ID det2
        energy_det1_kev = slab_kev_fn(slab_det1) * energy_det1
        energy_det2_kev = slab_kev_fn(slab_det2) * energy_det2
        en_filter1 = filter_total_energy(energy_det1_kev, en_min, en_max)
        en_filter2 = filter_total_energy(energy_det2_kev, en_min, en_max)
        # total_energy.extend((energy_det1_kev, energy_det2_kev))
        if not (en_filter1 and en_filter2):
            continue

        x_det1, y_det1 = calculate_centroid_sum(
            max_det1, local_map, chtype_map, x_rtp=1, y_rtp=2
        )
        x_det2, y_det2 = calculate_centroid_sum(
            max_det2, local_map, chtype_map, x_rtp=1, y_rtp=2
        )
        sm1 = sm_mM_map[slab_det1][0]
        sm2 = sm_mM_map[slab_det2][0]
        # print(f"tch_det1 {tch_det1} - tch_det2 {tch_det2}")
        # print(
        #     f"sm_center_map[sm1] {sm_center_map[sm1]} - sm_center_map[sm2] {sm_center_map[sm2]}"
        # )
        # print(f"x_det1 {x_det1}, y_det1 {y_det1} - x_det2 {x_det2}, y_det2 {y_det2}")
        x_glob1, y_glob1, z_glob1 = local_to_global(x_det1, y_det1, sm_center_map[sm1])
        x_glob2, y_glob2, z_glob2 = local_to_global(x_det2, y_det2, sm_center_map[sm2])
        line_start = np.array([x_glob1, y_glob1, z_glob1])
        line_end = np.array([x_glob2, y_glob2, z_glob2])
        intersection, dt1_theoric, dt2_theoric = cylinder_intersection_check(
            line_start,
            line_end,
            pipe_radius,
        )

        if not intersection:
            continue

        dt = tch_det1[0] - tch_det2[0]
        dt1_error = dt - dt1_theoric
        dt2_error = dt - dt2_theoric

        dt_evt_dict[slab_det1].append((slab_det2, dt1_error, True))
        dt_evt_dict[slab_det1].append((slab_det2, dt2_error, True))
        dt_evt_dict[slab_det2].append((slab_det1, -dt1_error, False))
        dt_evt_dict[slab_det2].append((slab_det1, -dt2_error, False))

        # print(f"dt_system: {dt}")
        # print(f"dt1_teoric: {dt1_theoric} - dt2_teoric: {dt2_theoric}")
        # print(f"dt1_error: {dt1_error} - dt2_error: {dt2_error}")
        # print(f"slab_det1: {slab_det1} - slab_det2: {slab_det2}")
        # print(
        #     f"energy_det1_kev: {energy_det1_kev} - energy_det2_kev: {energy_det2_kev}"
        # )
        # print(f"sm1: {sm1} - sm2: {sm2}")
        # print(f"line_start: {line_start} - line_end: {line_end}")
        # crossing_visualization(intersection, pipe_radius, line_start, line_end)
        # print("---------------------")
        # total_energy.extend((energy_det1_kev, energy_det2_kev))
        increment_pf()
    # n, bins, _ = plt.hist(total_energy, bins=1000, range=(0, 1000))
    # x, y, pars, _, _ = fit_gaussian(n, bins, cb=16)
    # mu, sigma = pars[1], pars[2]
    # plt.plot(x, y, "-r", label="fit")
    # plt.legend([f"Energy res: {round(2.35*sigma/mu*100,2)}%"])
    # plt.xlabel("Energy (keV)")
    # plt.ylabel("Counts")
    # plt.title("Total energy")
    # plt.show()

    print("---------------------")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total events: {EVT_COUNT_T}")
    print(f"Events passing the filter: {EVT_COUNT_F}")
    return dt_evt_dict


def process_file(
    binary_file_path: str,
    FEM_instance: FEMBase,
    chtype_map: dict,
    sm_mM_map: dict,
    local_map: dict,
    sm_center_map: dict,
    kev_converter: Callable,
    min_ch: int,
    en_min: float,
    en_max: float,
    pipe_radius: float,
) -> dict:
    """
    This function processes the binary file and returns a dictionary of dt values for each slab.
    """

    print(f"Processing file: {binary_file_path}")
    # Read the binary file
    reader = read_binary_file(binary_file_path)
    slab_kev_fn = kev_converter.convert
    # Extract the data dictionary
    dt_evt_dict = extract_data_dict(
        reader,
        FEM_instance,
        chtype_map,
        sm_mM_map,
        local_map,
        sm_center_map,
        slab_kev_fn,
        min_ch,
        en_min,
        en_max,
        pipe_radius,
    )
    return dt_evt_dict


def write_dt_to_bin(dt_evt_dict: dict, bin_name: str):
    """
    This function writes the slab_det1, slab_det2, and dt1_error values to a binary file.
    """
    with open(bin_name, "wb") as f:
        pickle.dump(dt_evt_dict, f)


def read_bin_to_dt(bin_name: str):
    """
    This function reads the slab_det1, slab_det2, and dt1_error values from a binary file.
    """
    start_time = time.time()
    with open(bin_name, "rb") as f:
        dt_evt_dict = pickle.load(f)
    end_time = time.time()
    print(f"Time taken to read bin file: {end_time - start_time} seconds")
    return dt_evt_dict


def read_skew_init(chtype_map: dict, relax_factor: float = 1.0):
    """
    This function reads the skew init file and returns a dictionary with the skew values with a relax factor.
    """
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


def skew_calc(
    dt_evt_dict: dict,
    skew_map: dict,
    niter: int,
    firstit: int,
    mon_id: list,
    relax_factor: float,
):
    """
    This function calculates the skew for each channel.
    """
    print(f"Number of keys: {len(dt_evt_dict)}")
    mon_path = "skew_mon/"
    if not os.path.exists(mon_path):
        os.makedirs(mon_path)
    freq_num_ev = defaultdict(int)
    for it in range(firstit, niter):
        print(f"Iteration: {it}")
        data_ch_numCoincCh = defaultdict(int)
        for ch, coinc_ch_dt_list in dt_evt_dict.items():
            # Check how many channels are coincident with each channel and store the number
            # data_ch_numCoincCh[ch] = defaultdict(int)
            # freq_num_ev[ch] = len(coinc_ch_dt_list)
            # for coinc_ch, _, _ in coinc_ch_dt_list:
            #     data_ch_numCoincCh[ch][coinc_ch] += 1
            ref_skew = skew_map[ch]

            # Plotting before iterations (optional)
            if ch in mon_id and it == firstit:
                dt_values = np.array([dt for _, dt, _ in coinc_ch_dt_list])
                n, bins = np.histogram(dt_values, bins=100, range=(-10000, 10000))
                bin_centers = (bins[:-1] + bins[1:]) / 2  # compute bin centers
                plt.bar(
                    bin_centers,
                    n,
                    width=np.diff(bins),
                    label=f"Slab {ch} monitoring\nNum. counts: {len(dt_values)}",
                )
                plt.xlabel("Time difference (ps)")
                plt.ylabel("Counts")
                plt.title(f"Time ch {ch} before iterations")
                plt.legend()
                plt.savefig(f"{mon_path}skew_{it}_ch{ch}_before.png")
                plt.clf()

            coinc_ch_dt_list = [
                (
                    coinc_ch,
                    (
                        dt + ref_skew - skew_map[coinc_ch]
                        if flag
                        else dt - ref_skew + skew_map[coinc_ch]
                    ),
                    flag,
                )
                for coinc_ch, dt, flag in coinc_ch_dt_list
            ]
            dt_evt_dict[ch] = coinc_ch_dt_list

            # Calculate histogram
            dt_values = np.array([dt for _, dt, _ in coinc_ch_dt_list])
            n, bins = np.histogram(dt_values, bins=100, range=(-10000, 10000))

            # Fit Gaussian
            try:
                x, y, pars, _, _ = fit_gaussian(n, bins, cb=8)
                mu, sigma = pars[1], pars[2]
            except RuntimeError:
                peak_mean, *_ = mean_around_max(n, shift_to_centres(bins), 6)
                mu = peak_mean if peak_mean else 0
            # peak_mean, *_ = mean_around_max(n, bins, 6)
            # mu = peak_mean if peak_mean else 0
            # Update skew_map
            skew_map[ch] += mu * (-relax_factor)

            # Plotting (optional)
            if ch in mon_id:
                bin_centers = (bins[:-1] + bins[1:]) / 2  # compute bin centers
                plt.bar(
                    bin_centers,
                    n,
                    width=np.diff(bins),
                    label=f"Slab {ch} monitoring\nNum. counts: {len(dt_values)}",
                )
                # plt.plot(x, y, "-r", label="fit")
                # plt.legend([f"Centroid {round(mu,2)}"])
                plt.axvline(
                    mu, color="r", linestyle="--", label=f"Centroid {round(mu,2)}"
                )
                plt.xlabel("Time difference (ps)")
                plt.ylabel("Counts")
                plt.title(f"Time ch {ch}")
                plt.legend()
                plt.savefig(f"{mon_path}skew_{it}_ch{ch}.png")
                plt.clf()

        # freq_dict_num_ch = defaultdict(int)
        # freq_dict_num_ev_per_ch = []
        # for ch, coinc_ch in data_ch_numCoincCh.items():
        #     freq_dict_num_ch[ch] = len(coinc_ch)
        #     freq_dict_num_ev_per_ch.extend([num_ev for num_ev in coinc_ch.values()])

        # total_num_ev = sum(freq_dict_num_ev_per_ch)
        # print(f"Total number of events: {total_num_ev}")

        # plt.hist(freq_num_ev.values(), bins=100)
        # plt.xlabel("Total number of events per channel")
        # plt.ylabel("Counts")
        # plt.show()

        # plt.hist(freq_dict_num_ch.values(), bins=100)
        # plt.xlabel("Number of coincident channels")
        # plt.ylabel("Counts")
        # plt.show()

        # plt.hist(freq_dict_num_ev_per_ch, bins=11, range=(0, 10))
        # plt.xlabel("Number of events per coincident channel")
        # plt.ylabel("Counts")
        # plt.show()

        with open(f"{mon_path}skew_{it}.txt", "w") as f:
            for ch, skew in sorted(skew_map.items()):
                f.write(f"{ch} {skew}\n")


def main():
    # Read the YAML configuration file
    args = docopt(__doc__)
    niter = int(args["--it"])
    print(f"Number of iterations: {niter}")
    firstit = int(args["--firstit"])

    # Data binary file
    input_file_paths = args["INFILES"]
    file_name = os.path.basename(input_file_paths[0])
    bin_name = re.sub(r"_\d+\.ldat$", "_dt.pickle", file_name)
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

    relax_factor = float(config["relax_factor"])
    print(f"Relax factor: {relax_factor}")

    mon_id = config["mon_id"]
    print(f"Monitoring channels: {mon_id}")

    # Pipe rotation information
    pipe_radius = float(config["pipe_radius"])
    print(f"Pipe radius: {pipe_radius}")

    # imas_radius
    ring_r = float(config["ring_r"])
    ring_z = config["ring_z"]
    ring_yx = config["ring_yx"]
    sm_center_map = sm_map_gen(ring_r, ring_z, ring_yx)

    kev_converter = KevConverter(slab_file_path)

    if firstit != 0:
        skew_map = read_skew(args["--sk"])
    else:
        if args["--sk"] == "skew_1ASIC.txt":
            skew_map = read_skew_init(chtype_map, relax_factor)
        else:
            skew_map = {
                ch: 0
                for ch, ch_type in chtype_map.items()
                if ChannelType.TIME in ch_type
            }
        if not args["-r"]:
            with get_context("spawn").Pool(processes=cpu_count()) as pool:
                args_list = [
                    (
                        binary_file_path,
                        FEM_instance,
                        chtype_map,
                        sm_mM_map,
                        local_map,
                        sm_center_map,
                        kev_converter,
                        min_ch,
                        en_min,
                        en_max,
                        pipe_radius,
                    )
                    for binary_file_path in input_file_paths
                ]
                results = pool.starmap(process_file, args_list)
            dt_evt_dict = defaultdict(list)
            for result in results:
                if isinstance(result, Exception):
                    print(f"Exception in child process: {result}")
                else:
                    for key, value in result.items():
                        dt_evt_dict[key].extend(value)
            write_dt_to_bin(dt_evt_dict, bin_name)
        else:
            dt_evt_dict = read_bin_to_dt(bin_name)
    skew_calc(dt_evt_dict, skew_map, niter, firstit, mon_id, relax_factor)


if __name__ == "__main__":
    main()
