import numpy as np
from typing import Dict, Tuple, Union, Optional
from src.mapping_generator import ChannelType


def get_max_channel_id(chtype_map: Dict) -> int:
    """Returns the maximum channel ID in the map + 1."""
    if not chtype_map:
        return 0
    return max(chtype_map.keys()) + 1


def create_channel_type_mask(
    chtype_map: Dict, channel_type: ChannelType, max_ch: Optional[int] = None
) -> np.ndarray:
    """
    Creates a boolean mask array where mask[ch] is True if channel has the specified type.

    Args:
        chtype_map: Dictionary mapping channel ID to list of ChannelTypes
        channel_type: Type of channel to create mask for
        max_ch: Maximum channel ID + 1. If None, computed from chtype_map

    Returns:
        Boolean array indexed by channel ID
    """
    if max_ch is None:
        max_ch = get_max_channel_id(chtype_map)

    mask = np.zeros(max_ch, dtype=bool)
    for ch, types in chtype_map.items():
        if channel_type in types:
            mask[ch] = True
    return mask


def create_mm_map_array(sm_mM_map: Dict, max_ch: Optional[int] = None) -> np.ndarray:
    """
    Creates an array mapping channel ID to a unique minimodule ID.

    Encodes (sm, mm) as sm * 1000 + mm.
    Returns -1 for channels not in the map.

    Args:
        sm_mM_map: Dictionary mapping channel ID to (supermodule, minimodule) tuple
        max_ch: Maximum channel ID + 1. If None, computed from sm_mM_map

    Returns:
        Int32 array indexed by channel ID containing encoded MM IDs
    """
    if max_ch is None:
        if not sm_mM_map:
            return np.zeros(0, dtype=np.int32)
        max_ch = max(sm_mM_map.keys()) + 1

    mm_map = np.full(max_ch, -1, dtype=np.int32)
    for ch, (sm, mm) in sm_mM_map.items():
        mm_map[ch] = sm * 1000 + mm
    return mm_map


def create_local_map_arrays(
    local_map: Dict, max_ch: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates arrays for local coordinates (x, y, z/pos).

    Args:
        local_map: Dictionary mapping channel ID to (x, y, pos) coordinates
        max_ch: Maximum channel ID + 1. If None, computed from local_map

    Returns:
        Tuple of (x_map, y_map, pos_map) arrays indexed by channel ID
    """
    if max_ch is None:
        if not local_map:
            return (np.zeros(0), np.zeros(0), np.zeros(0))
        max_ch = max(local_map.keys()) + 1

    x_map = np.zeros(max_ch, dtype=np.float32)
    y_map = np.zeros(max_ch, dtype=np.float32)
    pos_map = np.zeros(max_ch, dtype=np.int32)

    for ch, coords in local_map.items():
        # coords is (x, y, pos)
        x_map[ch] = coords[0]
        y_map[ch] = coords[1]
        pos_map[ch] = coords[2]

    return x_map, y_map, pos_map


def get_energy_channel_mask(
    chtype_map: Dict, max_ch: Optional[int] = None
) -> np.ndarray:
    """Wrapper to get mask for ENERGY channels."""
    return create_channel_type_mask(chtype_map, ChannelType.ENERGY, max_ch)


def get_time_channel_mask(chtype_map: Dict, max_ch: Optional[int] = None) -> np.ndarray:
    """Wrapper to get mask for TIME channels."""
    return create_channel_type_mask(chtype_map, ChannelType.TIME, max_ch)


def create_region_lookup_arrays(
    region_map: Dict, max_mm_id: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create lookup arrays for region and offsets from region_map.
    Returns (region_arr, offset_x_arr, offset_y_arr) indexed by encoded MM ID.

    Args:
        region_map: Dictionary mapping (sm, mm) to (region, (offset_x, offset_y))
        max_mm_id: Maximum encoded MM ID + 1. If None, computed from region_map

    Returns:
        Tuple of (region_arr, offset_x_arr, offset_y_arr) indexed by encoded MM ID
    """
    # Calculate max MM ID from region_map keys
    if max_mm_id is None:
        max_mm_id = max(sm * 1000 + mm for (sm, mm) in region_map.keys()) + 1

    region_arr = np.full(max_mm_id, -1, dtype=np.int32)
    offset_x_arr = np.zeros(max_mm_id, dtype=np.float32)
    offset_y_arr = np.zeros(max_mm_id, dtype=np.float32)

    for (sm, mm), (region, (offset_x, offset_y)) in region_map.items():
        mm_id = sm * 1000 + mm
        region_arr[mm_id] = region
        offset_x_arr[mm_id] = offset_x
        offset_y_arr[mm_id] = offset_y

    return region_arr, offset_x_arr, offset_y_arr


def create_pair_lookup_array(pair_map: Dict, max_region: int = 100) -> np.ndarray:
    """
    Create lookup array for pair IDs from region pairs.

    Args:
        pair_map: Dictionary mapping (region1, region2) to pair_id
        max_region: Maximum region ID to allocate array for

    Returns:
        2D array indexed by [region1, region2] containing pair IDs (-1 if not found)
    """
    pair_arr = np.full((max_region, max_region), -1, dtype=np.int32)

    for (region1, region2), pair_id in pair_map.items():
        pair_arr[region1, region2] = pair_id

    return pair_arr


def create_dec_lookup_arrays(
    dec_map: Dict, max_ch: int, max_slab: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lookup arrays for decompression parameters.

    Args:
        dec_map: Dictionary mapping (time_ch, slab) to (y_min, y_max)
        max_ch: Maximum channel ID to allocate arrays for
        max_slab: Maximum slab ID to allocate arrays for

    Returns:
        Tuple of (y_min_arr, y_max_arr) 2D arrays indexed by [channel_id, slab_id]
    """
    y_min_arr = np.zeros((max_ch, max_slab), dtype=np.float32)
    y_max_arr = np.zeros((max_ch, max_slab), dtype=np.float32)

    for (time_ch, slab), (y_min, y_max) in dec_map.items():
        if time_ch < max_ch and slab < max_slab:
            y_min_arr[time_ch, slab] = y_min
            y_max_arr[time_ch, slab] = y_max

    return y_min_arr, y_max_arr


def get_maxEnergy_sm_mM_vectorized(
    chunk: np.ndarray,
    sm_mM_map: Union[Dict, np.ndarray],
    energy_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version of get_maxEnergy_sm_mM.

    Args:
        chunk: Structured array of events
        sm_mM_map: Dictionary or pre-computed MM ID array
        energy_mask: Boolean mask indicating energy channels

    Returns:
        Tuple of (max_energy, max_mm_ids) arrays
    """
    # Handle inputs
    if isinstance(sm_mM_map, dict):
        mm_map = create_mm_map_array(sm_mM_map)
    else:
        mm_map = sm_mM_map

    channel_ids = chunk["hits"]["channelID"]
    energies = chunk["hits"]["energy"]

    valid_mask = channel_ids != -1

    # Safe lookup
    safe_ids = np.where(valid_mask, channel_ids, 0)

    # Identify energy channels
    is_energy = energy_mask[safe_ids] & valid_mask

    # Get MM IDs
    mm_ids = mm_map[safe_ids]

    # Calculate energy sum per MM for each event
    # We use adjacency matrix approach for small hit_limit (16)
    mm_ids_col = mm_ids[:, :, None]  # (N, 16, 1)
    mm_ids_row = mm_ids[:, None, :]  # (N, 1, 16)

    # Mask to ignore padding in adjacency check
    # (padding MM ID is -1, so they would match each other, but we mask energy)
    adjacency = mm_ids_col == mm_ids_row

    # Energies to sum (only energy channels)
    valid_energies = np.where(is_energy, energies, 0.0)

    # Sum energies: (N, 16, 16) * (N, 1, 16) -> sum(axis=2) -> (N, 16)
    # Result[i, j] = sum of energy of all hits belonging to same MM as hit j
    mm_total_energies = np.sum(adjacency * valid_energies[:, None, :], axis=2)

    # Find max energy
    max_energy = np.max(mm_total_energies, axis=1)
    max_idx = np.argmax(mm_total_energies, axis=1)

    # Get the MM ID corresponding to the max energy
    batch_indices = np.arange(len(channel_ids))
    max_mm_ids = mm_ids[batch_indices, max_idx]

    return max_energy, max_mm_ids


def get_timestamp_of_max_energy_hit_vectorized(
    hits: np.ndarray,
    sm_mM_map_arr: np.ndarray,
    time_mask: np.ndarray,
    target_mm_ids: np.ndarray,
) -> np.ndarray:
    """
    Vectorized extraction of timestamp from the hit with max energy
    among time channels belonging to the target minimodule.

    Args:
        hits: Structured array of hits (N, hit_limit)
        sm_mM_map_arr: Array mapping channel ID to MM ID
        time_mask: Boolean array indicating time channels
        target_mm_ids: Array of target MM IDs for each event (N,)

    Returns:
        Array of timestamps (N,), -1 for invalid events
    """
    channel_ids = hits["channelID"]
    energies = hits["energy"]
    timestamps = hits["time"]

    valid_mask = channel_ids != -1
    safe_ids = np.where(valid_mask, channel_ids, 0)

    # 1. Filter for time channels
    is_time = time_mask[safe_ids] & valid_mask

    # 2. Filter for target MM
    # sm_mM_map_arr[safe_ids] gives (N, hit_limit)
    # target_mm_ids[:, None] gives (N, 1)
    # Broadcasting results in (N, hit_limit) boolean mask
    hit_mm_ids = sm_mM_map_arr[safe_ids]
    is_target_mm = hit_mm_ids == target_mm_ids[:, None]

    # Combined mask
    mask = is_time & is_target_mm

    # 3. Find max energy among filtered hits
    # Use -1.0 for masked values so they lose against any valid energy (assuming energy >= 0)
    masked_energies = np.where(mask, energies, -1.0)

    # Find index of max energy hit for each event
    max_idx = np.argmax(masked_energies, axis=1)

    # Check if we actually found any valid hits
    # If max energy is -1.0, it means no hits satisfied the mask
    max_energies = np.max(masked_energies, axis=1)
    valid_events = max_energies >= 0

    # Extract timestamps
    batch_indices = np.arange(len(hits))
    result_timestamps = timestamps[batch_indices, max_idx]

    # Set invalid events to -1
    result_timestamps = np.where(valid_events, result_timestamps, -1)

    return result_timestamps


def get_slab_cornell_vectorized(
    chunk: np.ndarray,
    max_mm_ids: np.ndarray,
    sm_mM_map: Union[Dict, np.ndarray],
    time_mask: np.ndarray,
    local_map: Union[Dict, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized version of get_slab_cornell.

    Args:
        chunk: Structured array of events
        max_mm_ids: Array of max minimodule IDs
        sm_mM_map: Dictionary or pre-computed MM ID array
        time_mask: Boolean mask indicating time channels
        local_map: Dictionary or tuple of (x_map, y_map, pos_map)

    Returns:
        Tuple of (slab_ids, flags, x_positions, max_time_ch)
        slab_id -1 indicates invalid/None
    """
    # Handle inputs
    if isinstance(sm_mM_map, dict):
        mm_map = create_mm_map_array(sm_mM_map)
    else:
        mm_map = sm_mM_map

    if isinstance(local_map, dict):
        x_map, _, pos_map = create_local_map_arrays(local_map)
    else:
        x_map, _, pos_map = local_map

    channel_ids = chunk["hits"]["channelID"]
    energies = chunk["hits"]["energy"]
    valid_mask = channel_ids != -1
    safe_ids = np.where(valid_mask, channel_ids, 0)

    # 1. Identify hits belonging to the max MM
    mm_ids = mm_map[safe_ids]
    # Broadcast max_mm_ids to (N, 16)
    is_max_mm = (mm_ids == max_mm_ids[:, None]) & valid_mask

    # 2. Identify TIME channels among them
    is_time = time_mask[safe_ids] & is_max_mm

    # 3. Find max energy TIME channel
    # Use -1.0 for non-time channels so they aren't selected
    time_energies = np.where(is_time, energies, -1.0)
    max_time_idx = np.argmax(time_energies, axis=1)

    batch_indices = np.arange(len(channel_ids))
    max_time_ch = channel_ids[batch_indices, max_time_idx]

    # Check if we actually found a time channel (energy > -1)
    found_time = time_energies[batch_indices, max_time_idx] > -1.0

    # Get properties of max time channel
    # Use safe lookup for pos/x (handle invalid channels if found_time is false)
    safe_max_time_ch = np.where(found_time, max_time_ch, 0)
    max_time_pos = pos_map[safe_max_time_ch]
    max_time_x = x_map[safe_max_time_ch]

    # Count time hits
    num_time_hits = np.sum(is_time, axis=1)

    # Initialize results
    slab_ids = np.full(len(channel_ids), -1, dtype=np.int32)
    flags = np.zeros(len(channel_ids), dtype=np.int32)
    x_positions = np.zeros(len(channel_ids), dtype=np.float32)

    half_slab_width = 0.8

    # --- Logic Implementation ---

    # Case 1: Edges (pos 0 or 7)
    is_edge_0 = max_time_pos == 0
    is_edge_7 = max_time_pos == 7
    is_edge = (is_edge_0 | is_edge_7) & found_time

    # Edge 0
    mask_e0_2hits = is_edge_0 & (num_time_hits >= 2)
    mask_e0_other = is_edge_0 & (num_time_hits < 2)

    # Slab calculation: pos * 2 + offset
    # Edge 0: pos=0. Slabs 0, 1.
    # If >= 2 hits: slab 1 (index 1), x + 0.8
    # Else: slab 0 (index 0), x - 0.8

    slab_ids[mask_e0_2hits] = 1
    x_positions[mask_e0_2hits] = max_time_x[mask_e0_2hits] + half_slab_width

    slab_ids[mask_e0_other] = 0
    x_positions[mask_e0_other] = max_time_x[mask_e0_other] - half_slab_width

    # Edge 7
    mask_e7_2hits = is_edge_7 & (num_time_hits >= 2)
    mask_e7_other = is_edge_7 & (num_time_hits < 2)

    # Edge 7: pos=7. Slabs 14, 15.
    # If >= 2 hits: slab 14 (index 0), x - 0.8
    # Else: slab 15 (index 1), x + 0.8

    slab_ids[mask_e7_2hits] = 14
    x_positions[mask_e7_2hits] = max_time_x[mask_e7_2hits] - half_slab_width

    slab_ids[mask_e7_other] = 15
    x_positions[mask_e7_other] = max_time_x[mask_e7_other] + half_slab_width

    # Case 2: Non-edges
    is_non_edge = (~is_edge) & found_time

    # Subcase 2a: 1 hit
    mask_ne_1hit = is_non_edge & (num_time_hits == 1)

    # Random selection 0 or 1
    rand_sel = np.random.randint(0, 2, size=len(channel_ids))
    # slab = pos*2 + rand
    slab_ids[mask_ne_1hit] = max_time_pos[mask_ne_1hit] * 2 + rand_sel[mask_ne_1hit]
    flags[mask_ne_1hit] = 1

    # x pos: if rand=0 -> -0.8, if rand=1 -> +0.8
    ad_factors = np.where(rand_sel == 0, -half_slab_width, half_slab_width)
    x_positions[mask_ne_1hit] = max_time_x[mask_ne_1hit] + ad_factors[mask_ne_1hit]

    # Subcase 2b: >= 2 hits
    mask_ne_multi = is_non_edge & (num_time_hits >= 2)

    # Find 2nd max time channel
    # Mask out the max hit in time_energies
    time_energies_2nd = time_energies.copy()
    time_energies_2nd[batch_indices, max_time_idx] = -1.0

    sec_max_idx = np.argmax(time_energies_2nd, axis=1)
    sec_max_ch = channel_ids[batch_indices, sec_max_idx]

    # Safe lookup
    safe_sec_max_ch = np.where(mask_ne_multi, sec_max_ch, 0)
    sec_max_pos = pos_map[safe_sec_max_ch]

    diff = max_time_pos - sec_max_pos

    # Logic:
    # abs(diff) > 1: None (slab_id -1, flag 2)
    mask_diff_gt1 = mask_ne_multi & (np.abs(diff) > 1)
    flags[mask_diff_gt1] = 2
    # slab_ids already -1

    # diff == 1: slab index 1, x + 0.8
    mask_diff_1 = mask_ne_multi & (diff == 1)
    slab_ids[mask_diff_1] = max_time_pos[mask_diff_1] * 2 + 1
    x_positions[mask_diff_1] = max_time_x[mask_diff_1] + half_slab_width
    flags[mask_diff_1] = 3

    # diff != 1 (implies -1 or 0? 0 shouldn't happen if pos distinct, but logic says else)
    # Original: else -> slab index 0, x - 0.8
    mask_diff_other = mask_ne_multi & (np.abs(diff) <= 1) & (diff != 1)
    slab_ids[mask_diff_other] = max_time_pos[mask_diff_other] * 2 + 0
    x_positions[mask_diff_other] = max_time_x[mask_diff_other] - half_slab_width
    flags[mask_diff_other] = 3

    return slab_ids, flags, x_positions, safe_max_time_ch


def get_slab_imas_vectorized(
    chunk: np.ndarray,
    max_mm_ids: np.ndarray,
    sm_mM_map_arr: np.ndarray,
    time_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized slab extraction for IMAS system.

    In IMAS, the slab is simply the channel ID of the TIME channel with maximum energy
    within the maximum energy minimodule. Much simpler than Cornell's approach.

    Args:
        chunk: Structured array of events
        max_mm_ids: Array of max minimodule IDs for each event (N,)
        sm_mM_map_arr: Array mapping channel ID to MM ID
        time_mask: Boolean mask indicating time channels

    Returns:
        Tuple of (slab_channels, timestamps, energies) arrays (N,)
            - slab_channels: Channel IDs of max energy TIME channels
            - timestamps: Timestamps of those channels
            - energies: Energies of those channels
    """
    channel_ids = chunk["hits"]["channelID"]
    energies = chunk["hits"]["energy"]
    timestamps = chunk["hits"]["time"]

    valid_mask = channel_ids != -1
    safe_ids = np.where(valid_mask, channel_ids, 0)

    # 1. Filter for hits belonging to the max MM
    mm_ids = sm_mM_map_arr[safe_ids]
    is_max_mm = (mm_ids == max_mm_ids[:, None]) & valid_mask

    # 2. Filter for TIME channels among them
    is_time = time_mask[safe_ids] & is_max_mm

    # 3. Find max energy TIME channel
    # Use -1.0 for non-time channels so they aren't selected
    time_energies = np.where(is_time, energies, -1.0)
    max_time_idx = np.argmax(time_energies, axis=1)

    batch_indices = np.arange(len(channel_ids))

    # Extract channel ID (slab), timestamp, and energy
    slab_channels = channel_ids[batch_indices, max_time_idx]
    slab_timestamps = timestamps[batch_indices, max_time_idx]
    slab_energies = time_energies[batch_indices, max_time_idx]

    # Check if we actually found a time channel (energy > -1)
    found_time = slab_energies > -1.0

    # Set invalid events to -1
    slab_channels = np.where(found_time, slab_channels, -1)
    slab_timestamps = np.where(found_time, slab_timestamps, -1)
    slab_energies = np.where(found_time, slab_energies, 0.0)

    return slab_channels, slab_timestamps, slab_energies
