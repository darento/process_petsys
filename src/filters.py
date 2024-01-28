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
    increment_total()
    if (en_total > en_min) & (en_total < en_max):
        increment_pf()
        return True
    else:
        return False
