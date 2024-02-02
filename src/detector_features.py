def get_electronics_nums(channel_id: int) -> tuple[int, int, int, int]:
    """
    Calculates the electronics numbers:
    portID, slaveID, chipID, channelID
    """
    portID = channel_id // 131072
    slaveID = (channel_id % 131072) // 4096
    chipID = ((channel_id % 131072) % 4096) // 64
    channelID = channel_id % 64
    return portID, slaveID, chipID, channelID


def get_absolute_id(portID: int, slaveID: int, chipID: int, channelID: int) -> int:
    """
    Calculates absolute channel id from
    electronics numbers.
    """
    return 131072 * portID + 4096 * slaveID + 64 * chipID + channelID


def calculate_total_energy(det_event: list) -> float:
    """
    Calculate the total energy of the event.

    Parameters:
    event: The event data.

    Returns:
    float: The total energy of the event.
    """
    return sum([ch[1] for ch in det_event])


def calculate_centroid(det_event: list, x_rtp: int, y_rtp: int) -> tuple:
    """
    Calculate the centroid of the event.

    Parameters:
    event: The event data.
    offset_x: The offset in the x direction.
    offset_y: The offset in the y direction.

    Returns:
    tuple: The centroid of the event.
    """
    powers = [x_rtp, y_rtp]
    offsets = [0.00001, 0.00001]
    print(det_event)
    exit(0)


def centroid_calculation(
    plot_pos: dict, offset_x: float = 0.00001, offset_y: float = 0.00001
) -> None:
    """
    Calculates the centroid of a set of module
    data according to a centroid map.
    """
    powers = [1, 2]
    offsets = [offset_x, offset_y]
    plot_ax = ["local_x", "local_y"]

    def centroid(data: list[list]) -> tuple[float, float, float]:
        """
        Calculate the average position of the time
        and energy channels and return them plus
        the total energy channel deposit.
        """
        sums = [0.0, 0.0]
        weights = [0.0, 0.0]
        for imp in data:
            en_t = imp[1].value - 1
            pos = plot_pos[imp[0]][plot_ax[en_t]]
            weight = (imp[3] + offsets[en_t]) ** powers[en_t]
            sums[en_t] += weight * pos
            weights[en_t] += weight
        return (
            sums[0] / weights[0] if weights[0] else 0.0,
            sums[1] / weights[1] if weights[1] else 0.0,
            weights[1],
        )

    return centroid
