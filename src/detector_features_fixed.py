import numpy as np
from typing import Union, Tuple, Dict, Optional
from src.utils_fixed import (
    get_energy_channel_mask,
    get_time_channel_mask,
)


def calculate_centroid_vectorized(
    chunk: np.ndarray,
    max_mm_ids: np.ndarray,
    sm_mM_map_arr: np.ndarray,
    local_map_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    chtype_map: Union[Dict, Tuple[np.ndarray, np.ndarray]],
    x_rtp: float,
    y_rtp: float,
    sum_rows_cols: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized centroid calculation.

    Args:
        chunk: Structured array of events (filtered chunk)
        max_mm_ids: Array of max minimodule IDs for each event (N,)
        sm_mM_map_arr: Array mapping channel ID to MM ID
        local_map_arrays: Tuple of (x_map, y_map, pos_map)
        chtype_map: Dictionary of channel types or Tuple of (time_mask, energy_mask)
        x_rtp: Power for X weighting
        y_rtp: Power for Y weighting
        sum_rows_cols: If True, calculates X from TIME channels and Y from ENERGY channels.

    Returns:
        centroids_x, centroids_y: (N,) arrays of calculated centroids
    """
    # Unpack maps
    x_map, y_map, _ = local_map_arrays

    # Get data from chunk
    channel_ids = chunk["hits"]["channelID"]
    energies = chunk["hits"]["energy"]

    # Valid hits mask
    valid_mask = channel_ids != -1
    safe_ids = np.where(valid_mask, channel_ids, 0)

    # Filter hits belonging to max_mm
    mm_ids = sm_mM_map_arr[safe_ids]
    is_max_mm = (mm_ids == max_mm_ids[:, None]) & valid_mask

    # Offsets and powers
    offsets = [0.00001, 0.00001]
    powers = [x_rtp, y_rtp]

    if sum_rows_cols:
        # Logic: X centroid from TIME channels, Y centroid from ENERGY channels

        # Resolve masks
        if isinstance(chtype_map, dict):
            max_ch = len(sm_mM_map_arr)
            time_mask = get_time_channel_mask(chtype_map, max_ch)
            energy_mask = get_energy_channel_mask(chtype_map, max_ch)
        elif isinstance(chtype_map, tuple):
            time_mask, energy_mask = chtype_map
        else:
            raise ValueError(
                "chtype_map must be dict or (time_mask, energy_mask) tuple"
            )

        is_time = time_mask[safe_ids] & is_max_mm
        is_energy = energy_mask[safe_ids] & is_max_mm

        # --- X Calculation (TIME channels) ---
        w_x = np.power(energies + offsets[0], powers[0])
        w_x_valid = np.where(is_time, w_x, 0.0)

        term_x = w_x_valid * x_map[safe_ids]
        sum_x = np.sum(term_x, axis=1)
        weights_x = np.sum(w_x_valid, axis=1)

        # --- Y Calculation (ENERGY channels) ---
        w_y = np.power(energies + offsets[1], powers[1])
        w_y_valid = np.where(is_energy, w_y, 0.0)

        term_y = w_y_valid * y_map[safe_ids]
        sum_y = np.sum(term_y, axis=1)
        weights_y = np.sum(w_y_valid, axis=1)

    else:
        # Logic: Every hit contributes to both X and Y
        w_x = np.power(energies + offsets[0], powers[0])
        w_x_valid = np.where(is_max_mm, w_x, 0.0)

        w_y = np.power(energies + offsets[1], powers[1])
        w_y_valid = np.where(is_max_mm, w_y, 0.0)

        term_x = w_x_valid * x_map[safe_ids]
        term_y = w_y_valid * y_map[safe_ids]

        sum_x = np.sum(term_x, axis=1)
        sum_y = np.sum(term_y, axis=1)
        weights_x = np.sum(w_x_valid, axis=1)
        weights_y = np.sum(w_y_valid, axis=1)

    # Calculate centroids (avoid division by zero)
    centroid_x = np.divide(
        sum_x, weights_x, out=np.zeros_like(sum_x), where=weights_x != 0
    )
    centroid_y = np.divide(
        sum_y, weights_y, out=np.zeros_like(sum_y), where=weights_y != 0
    )

    return centroid_x, centroid_y


def calculate_DOI_vectorized(
    chunk: np.ndarray,
    max_mm_ids: np.ndarray,
    sm_mM_map_arr: np.ndarray,
    local_map_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    chtype_map: Union[Dict, Tuple[np.ndarray, np.ndarray], np.ndarray],
    sum_rows_cols: bool = False,
    slab_orientation: str = "x",
) -> np.ndarray:
    """
    Vectorized DOI calculation.

    Args:
        chunk: Structured array of events
        max_mm_ids: Array of max minimodule IDs (N,)
        sm_mM_map_arr: Array mapping channel ID to MM ID
        local_map_arrays: Tuple of (x_map, y_map, pos_map)
        chtype_map: Dictionary, Tuple of masks, or Energy mask array
        sum_rows_cols: If True, uses max energy of any channel. If False, projects energies.
        slab_orientation: 'x' or 'y', used for projection in non-sum_rows_cols mode.

    Returns:
        doi: (N,) array of calculated DOI values
    """
    # Unpack maps
    x_map, y_map, _ = local_map_arrays

    # Get data
    channel_ids = chunk["hits"]["channelID"]
    energies = chunk["hits"]["energy"]

    valid_mask = channel_ids != -1
    safe_ids = np.where(valid_mask, channel_ids, 0)

    mm_ids = sm_mM_map_arr[safe_ids]
    is_max_mm = (mm_ids == max_mm_ids[:, None]) & valid_mask

    # Handle masks
    max_ch = len(sm_mM_map_arr)
    if isinstance(chtype_map, dict):
        energy_mask = get_energy_channel_mask(chtype_map, max_ch)
    elif isinstance(chtype_map, tuple):
        _, energy_mask = chtype_map
    elif isinstance(chtype_map, np.ndarray):
        energy_mask = chtype_map
    else:
        raise ValueError(
            "chtype_map must be dict, (time_mask, energy_mask) tuple, or energy_mask array"
        )

    # Filter for ENERGY channels in the max MM
    is_valid_energy = energy_mask[safe_ids] & is_max_mm
    valid_energies = np.where(is_valid_energy, energies, 0.0)

    # Calculate sum_energy (sum of all ENERGY channels in cluster)
    sum_energy = np.sum(valid_energies, axis=1)

    if sum_rows_cols:
        # Max energy of any single channel in the cluster
        max_energy = np.max(valid_energies, axis=1)

    else:
        # Project energies onto X or Y axis and take max of sums
        if slab_orientation == "x":
            coords = x_map[safe_ids]
        else:
            coords = y_map[safe_ids]

        # We need to sum energies for hits with same coordinate
        # Broadcast to (N, 16, 16) for adjacency check
        coords_col = coords[:, :, None]
        coords_row = coords[:, None, :]

        mask_col = is_valid_energy[:, :, None]
        mask_row = is_valid_energy[:, None, :]

        # Adjacency based on coordinate equality (with small epsilon for floats)
        match = (np.abs(coords_col - coords_row) < 1e-5) & mask_col & mask_row

        # Sum energies for each group: sum over rows (axis 2)
        energies_broadcast = valid_energies[:, None, :]  # (N, 1, 16)
        proj_sums = np.sum(match * energies_broadcast, axis=2)  # (N, 16)

        # Take max over all projections
        max_energy = np.max(proj_sums, axis=1)

    # Calculate DOI (avoid division by zero)
    doi = np.divide(
        sum_energy, max_energy, out=np.zeros_like(sum_energy), where=max_energy != 0
    )

    return doi
