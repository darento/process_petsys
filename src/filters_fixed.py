import numpy as np
from typing import Union, Dict
from src.utils_fixed import get_energy_channel_mask, create_mm_map_array


def filter_min_ch_vectorized(
    chunk: np.ndarray,
    chtype_map: Union[Dict, np.ndarray],
    min_ch: int,
    sum_rows_cols: bool = False,
) -> np.ndarray:
    """
    Vectorized filter for minimum number of energy channels.

    Args:
        chunk: Structured array of events
        chtype_map: Dictionary or pre-computed boolean mask array (preferred for speed)
        min_ch: Minimum number of energy channels required
        sum_rows_cols: If True, also checks that num_energy_ch < total_hits

    Returns:
        Boolean mask of passing events
    """
    # Resolve energy mask
    if isinstance(chtype_map, dict):
        energy_mask = get_energy_channel_mask(chtype_map)
    else:
        energy_mask = chtype_map

    # Get channel IDs
    channel_ids = chunk["hits"]["channelID"]

    # Mask for valid hits (not padding)
    valid_hits_mask = channel_ids != -1

    # Safe lookup (handle -1 padding)
    safe_channel_ids = np.where(valid_hits_mask, channel_ids, 0)

    # Identify energy channels
    is_energy = energy_mask[safe_channel_ids] & valid_hits_mask

    # Count energy hits per event
    num_eng_ch = np.sum(is_energy, axis=1)

    if not sum_rows_cols:
        return num_eng_ch >= min_ch
    else:
        n_hits = chunk["header"]
        return (num_eng_ch >= min_ch) & (num_eng_ch < n_hits)


def filter_total_energy_vectorized(
    chunk: np.ndarray, en_min: float = 10.0, en_max: float = 100.0
) -> np.ndarray:
    """
    Vectorized filter for total energy range.

    Args:
        chunk: Structured array of events
        en_min: Minimum total energy threshold
        en_max: Maximum total energy threshold

    Returns:
        Boolean mask of passing events
    """
    energies = chunk["hits"]["energy"]
    total_energy = np.sum(energies, axis=1)

    return (total_energy > en_min) & (total_energy < en_max)


def filter_single_mM_vectorized(
    chunk: np.ndarray, sm_mM_map: Union[Dict, np.ndarray]
) -> np.ndarray:
    """
    Vectorized filter for events with all hits in a single minimodule.

    Args:
        chunk: Structured array of events
        sm_mM_map: Dictionary or pre-computed MM ID array (preferred)

    Returns:
        Boolean mask of passing events
    """
    if isinstance(sm_mM_map, dict):
        mm_map = create_mm_map_array(sm_mM_map)
    else:
        mm_map = sm_mM_map

    channel_ids = chunk["hits"]["channelID"]
    valid_mask = channel_ids != -1

    # Safe lookup
    safe_ids = np.where(valid_mask, channel_ids, 0)
    mm_ids = mm_map[safe_ids]

    # Check if min MM ID == max MM ID for each event
    # Use sentinel values for padding to ignore them in min/max
    mm_ids_for_min = np.where(valid_mask, mm_ids, np.iinfo(np.int32).max)
    mm_ids_for_max = np.where(valid_mask, mm_ids, np.iinfo(np.int32).min)

    min_mm = np.min(mm_ids_for_min, axis=1)
    max_mm = np.max(mm_ids_for_max, axis=1)

    # Require at least one hit and all hits in same MM
    has_hits = chunk["header"] > 0

    return (min_mm == max_mm) & has_hits
