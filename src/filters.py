from itertools import chain


def filter_total_energy(
    en_total: float, en_min: float = 10, en_max: float = 100
) -> list:
    """
    Filters the event based on the total energy of the event.

    The function checks if the total energy of the event is within a specified range (en_min, en_max).

    Parameters:
    en_total (float): The total energy of the event.
    en_min (float, optional): The minimum energy threshold. Defaults to 10.
    en_max (float, optional): The maximum energy threshold. Defaults to 100.

    Returns:
    bool: True if the total energy is within the range (en_min, en_max), False otherwise.
    """
    if (en_total > en_min) & (en_total < en_max):
        return True
    else:
        return False


def filter_min_ch(det_list: list[list], min_ch: int) -> bool:
    """
    Filters the event based on the minimum number of channels.

    The function checks if the number of channels in the detector is greater than a specified threshold (min_ch).

    Parameters:
    det_list (list): The event data.
    min_ch (int): The minimum number of channels.

    Returns:
    bool: True if the number of channels is greater than min_ch, False otherwise.
    """
    if len(det_list) >= min_ch:
        return True
    else:
        return False


def filter_single_mM(det_list: list[list], sm_mM_map: dict) -> bool:
    """
    This function filters out super modules that have more than one mini module hit.

    Parameters:
    det_list (list[list]): A nested list where each sublist represents a hit.
                           Each sublist contains information about the hit.
    sm_mM_map (dict): A dictionary mapping the super module (sm) to the mini module (mM).

    Returns:
    bool: True if the super module has exactly one mini module hit, False otherwise.
    """
    n_mm = len(set(sm_mM_map[x[2]] for x in det_list))
    return n_mm == 1


def filter_max_sm(
    det1_list: list[list], det2_list: list[list], max_sm: int, sm_mM_map: dict
) -> bool:
    """
    Filters events based on the number of supermodules present.

    Parameters:
    det1_list (list[list]): The first list of detections.
    det2_list (list[list]): The second list of detections.
    max_sm (int): The maximum number of supermodules allowed.
    sm_mM_map (dict): A mapping from module to supermodule.

    Returns:
    bool: True if the number of unique supermodules in the event does not exceed max_sm, False otherwise.

    Note: This function is only valid for coinc mode.
    """
    sm_set = set()
    for hit in chain(det1_list, det2_list):
        sm_set.add(sm_mM_map(hit[2]))
        if len(sm_set) > max_sm:
            return False
    return True
