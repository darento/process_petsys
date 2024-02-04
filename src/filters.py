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


def filter_min_ch(det_list: list, min_ch: int) -> bool:
    """
    Filters the event based on the minimum number of channels.

    The function checks if the number of channels in the detector is greater than a specified threshold (min_ch).

    Parameters:
    det_list (list): The event data.
    min_ch (int): The minimum number of channels.

    Returns:
    bool: True if the number of channels is greater than min_ch, False otherwise.
    """
    if len(det_list) > min_ch:
        return True
    else:
        return False
