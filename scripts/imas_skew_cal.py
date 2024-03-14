#!/usr/bin/env python3

"""Extracts the photopeak from the energy per minimodule and saves it to a file.
Usage: imas_skew_cal YAMLCONF SLAB_EN_MAP INFILE

Arguments:
    YAMLCONF       File with all parameters to take into account in the scan.
    SLAB_EN_MAP    File with the energy per slab.
    INFILE         Input file to be processed. Must be a compact binary file from PETsys.

Options:
    -h --help     Show this screen.    º
"""


from collections import defaultdict
import os
import time
from typing import Callable
from docopt import docopt
from matplotlib import pyplot as plt
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.constants import c as c_vac

from src.detector_features import calculate_centroid, calculate_centroid_sum

from src.fem_handler import FEMBase
from src.read_compact import read_binary_file
from src.filters import (
    filter_max_sm,
    filter_total_energy,
    filter_min_ch,
    filter_single_mM,
)

from src.mapping_generator import ChannelType, map_factory

from src.fits import fit_gaussian
from src.utils import (
    convert_to_kev,
    get_maxEnergy_sm_mM,
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


def read_slab_energy_map(slab_file_path: str) -> dict:
    """
    This function reads the energy per slab from a file and returns a Callable to convert the energy to keV.

    Parameters:
    slab_file_path (str): The path to the file containing the energy per slab.

    Returns:
    Callable: A function that takes the slab number and returns the energy in keV.
    """
    kev_map = convert_to_kev(slab_file_path)
    return kev_map


def sm_map_gen(ring_r: float, ring_z: list, ring_yx: dict) -> dict:
    """
    This function generates the centre position of the minimodules in polar coordinates.

    Parameters:
        - ring_r (float): The radius of the ring.
        - ring_z (float): The z coordinate of the ring.
        - ring_yx (dict): A dictionary containing the y and x coordinates of the ring.

    Returns:
        dict: A dictionary containing the centre position of the minimodules in polar coordinates.
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
    This function returns the centre position of a minimodule in polar coordinates.

    Parameters:
        - sm (int): The minimodule number.
        - sm_r (float): The radius of the minimodule.
        - sm_yx (dict): A dictionary containing the y and x coordinates of the minimodule.

    Returns:
        - sm_r (float): The radius of the minimodule.
        - sm_yx[sm] (np.ndarray): The y and x coordinates of the minimodule.
    """
    return sm_r, np.arctan2(*sm_yx)


def local_to_global(x, y, sm_glob_par):

    x_center, y_center = 48, 48
    x -= x_center
    y -= sm_glob_par["sm_z"] + y_center
    # print(f"x: {x}, y: {y}")
    sm_rot = R.from_euler("z", sm_glob_par["sm_ang"])
    local_coords = np.array([sm_glob_par["sm_r"], x, -y])
    # print(f"Local: {local_coords}")
    global_coords = sm_rot.apply(local_coords)
    # print(f"Global: {global_coords}")

    return global_coords[0], global_coords[1], global_coords[2]


# Define the cylinder intersection check function
def cylinder_intersection_check(
    line_start: np.ndarray, line_end: np.ndarray, radius: float
) -> bool:
    """
    This function checks if a line intersects a cylinder.

    Parameters:
        - line_start (np.ndarray): The starting point of the line.
        - line_end (np.ndarray): The ending point of the line.
        - radius (float): The radius of the cylinder.

    Returns:
        bool: True if the line intersects the cylinder, False otherwise.
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
    t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    t2 = (-B - np.sqrt(discriminant)) / (2 * A)

    dt1 = line_length * (2 * t1 - 1) / c_mm_per_ps
    dt2 = line_length * (2 * t2 - 1) / c_mm_per_ps
    # print(f"line_length: {line_length}")
    # print(f"dt_t1: {dt_t1} - dt_t2: {dt_t2}")
    # print(f"t1: {t1}, t2: {t2}")
    # print(f" t1 + t2 = {t1 + t2}")

    return (0 <= t1 <= 1) or (0 <= t2 <= 1), dt1, dt2


def crossing_visualization(
    intersection: bool,
    pipe_radius: float,
    line_start: np.ndarray,
    line_end: np.ndarray,
):
    height = 720.0
    # Visualization
    fig = plt.figure(figsize=(10, 8))
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
    # Setting labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Title
    ax.set_title(f'Line Intersection with Cylinder: {"Yes" if intersection else "No"}')
    # Legend
    ax.legend()
    # Show plot
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
    This function processes the detector events and calculates the energy per minimodule.

    Parameters:
        - read_det_evt (Callable): A generator or iterable that yields detector events.
        - FEM_instance (FEMBase): A dictionary representing a Finite Element Model instance.
        - chtype_map (dict): A dictionary mapping the channel type to the channel number.
        - sm_mM_map (dict): A dictionary mapping the channel number to the minimodule number.
        - local_map (dict): A dictionary mapping the channel number to the local coordinates.
        - sm_center_map (dict): A dictionary containing the centre position of the minimodules in polar coordinates.
        - slab_kev_fn (Callable): A function that takes the slab number and returns the energy in keV.
        - min_ch (int): The minimum channel number for the filter.
        - en_min (float): The minimum energy for the filter.
        - en_max (float): The maximum energy for the filter.
        - pipe_radius (float): The radius of the pipe.

    Returns:
        NONE
    """
    start_time = time.time()
    dt_ch_dict = defaultdict(list)
    sum_rows_cols = FEM_instance.sum_rows_cols
    for event in read_det_evt:
        increment_total()
        det1, det2 = event
        min_ch_filter1 = filter_min_ch(det1, min_ch, chtype_map, sum_rows_cols)
        min_ch_filter2 = filter_min_ch(det2, min_ch, chtype_map, sum_rows_cols)
        if EVT_COUNT_T > 1000000:
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
        # slab_dict_count[slab_det1] += 1
        # slab_dict_count[slab_det2] += 1
        energy_det1_kev = slab_kev_fn(slab_det1) * energy_det1
        energy_det2_kev = slab_kev_fn(slab_det2) * energy_det2
        en_filter1 = filter_total_energy(energy_det1_kev, en_min, en_max)
        en_filter2 = filter_total_energy(energy_det2_kev, en_min, en_max)
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
        # print(f"sm1 {sm1} - sm2 {sm2}")
        # print(
        #     f"sm_center_map[sm1] {sm_center_map[sm1]} - sm_center_map[sm2] {sm_center_map[sm2]}"
        # )
        # print(f"x_det1 {x_det1}, y_det1 {y_det1} - x_det2 {x_det2}, y_det2 {y_det2}")
        x_glob1, y_glob1, z_glob1 = local_to_global(x_det1, y_det1, sm_center_map[sm1])
        x_glob2, y_glob2, z_glob2 = local_to_global(x_det2, y_det2, sm_center_map[sm2])
        line_start = np.array([x_glob1, y_glob1, z_glob1])
        line_end = np.array([x_glob2, y_glob2, z_glob2])
        # print(
        #     f"x_glob1 {x_glob1}, y_glob1 {y_glob1}, z_glob1 {z_glob1} - x_glob2 {x_glob2}, y_glob2 {y_glob2}, z_glob2 {z_glob2}"
        # )

        intersection, dt1_teoric, dt2_teoric = cylinder_intersection_check(
            line_start,
            line_end,
            pipe_radius,
        )
        if not intersection:
            continue
        dt = tch_det1[0] - tch_det2[0]
        dt1_error = dt - dt1_teoric
        dt2_error = dt - dt2_teoric

        dt_ch_dict[slab_det1].append(dt1_error)
        # dt_ch_dict[slab_det2].append(-dt1_error)
        dt_ch_dict[slab_det1].append(dt2_error)
        # dt_ch_dict[slab_det2].append(-dt2_error)

        # print(f"dt_system: {dt}")
        # crossing_visualization(intersection, pipe_radius, line_start, line_end)
        # print("---------------------")
        # slab_dict_energy[slab_det1].append(energy_det1_kev)
        # slab_dict_energy[slab_det2].append(energy_det2_kev)
        # total_energy.extend((energy_det1_kev, energy_det2_kev))
        # max_sm_det1 = sm_mM_map[max_det1[0][2]][0]
        # max_sm_det2 = sm_mM_map[max_det2[0][2]][0]
        # flood_dict[max_sm_det1].append((x_det1, y_det1))
        # flood_dict[max_sm_det2].append((x_det2, y_det2))
        increment_pf()

    # for key, value in flood_dict.items():
    #     x, y = zip(*value)
    #     plt.hist2d(x, y, bins=[500, 500], range=[[0, 105], [0, 105]], cmap="plasma")
    #     plt.colorbar()
    #     plt.title(f"SM: {key}")
    #     plt.show()

    print("---------------------")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total events: {EVT_COUNT_T}")
    print(f"Events passing the filter: {EVT_COUNT_F}")
    return dt_ch_dict


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
    local_map, sm_mM_map, chtype_map, FEM_instance = map_factory(map_file)

    # Read the energy range
    en_min = float(config["energy_range"][0])
    en_max = float(config["energy_range"][1])
    print(f"Energy range: {en_min} - {en_max}")

    min_ch = int(config["min_ch"])
    print(f"Minimum number of Energy channels (+1 Time): {min_ch}")

    en_min_ch = float(config["en_min_ch"])
    print(f"Minimum energy per channel: {en_min_ch}")

    # Pipe rotation information
    pipe_radius = float(config["pipe_radius"])
    print(f"Pipe radius: {pipe_radius}")

    # imas_radius
    ring_r = float(config["ring_r"])
    ring_z = config["ring_z"]
    ring_yx = config["ring_yx"]
    sm_center_map = sm_map_gen(ring_r, ring_z, ring_yx)

    reader = read_binary_file(binary_file_path, en_min_ch)

    slab_kev_fn = read_slab_energy_map(slab_file_path)

    dt_ch_dict = extract_data_dict(
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

    for ch, dt_list in dt_ch_dict.items():
        n, bins, _ = plt.hist(
            dt_list, bins=250, range=(-5000, 5000), label=f"Slab {ch}"
        )
        # x, y, pars, _, _ = fit_gaussian(n, bins, cb=4)
        # mu, sigma = pars[1], pars[2]
        # plt.plot(x, y, "-r", label="fit")
        # plt.legend([f"Energy res: {round(2.35*sigma/mu*100,2)}%"])
        plt.xlabel("Time difference (ps)")
        plt.ylabel("Counts")
        plt.title(f"Time ch {ch}")
        plt.show()


if __name__ == "__main__":
    main()
