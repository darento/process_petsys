def calculate_total_energy(det_event: list) -> float:
    """
    Calculate the total energy of the event.

    Parameters:
    event: The event data.

    Returns:
    float: The total energy of the event.
    """
    return sum([ch[1] for ch in det_event])
