import struct
from typing import Iterator, BinaryIO


def read_detector_evt(
    f: BinaryIO, data_format: str, data_size: int, num_lines: int, en_filter: float
) -> list:
    """
    Reads and unpacks data from a binary file for the lines corresponding to the hits
    at detector level.

    Parameters:
    f (file): The binary file to read from.
    data_format (str): The format string for struct.unpack to parse the data.
    data_size (int): The size of each data line in bytes.
    num_lines (int): The number of data lines to read.
    en_filter (float): The energy filter threshold.

    Returns:
    list: A list of tuples containing the unpacked data.
    """
    try:
        data = [struct.unpack(data_format, f.read(data_size)) for _ in range(num_lines)]
    except struct.error:
        pass

    return [evt_ch for evt_ch in data if evt_ch[1] >= en_filter]


def read_binary_file(
    file_path: str, en_filter: float = 0, group_events: bool = False
) -> Iterator[tuple]:
    """
    Generates events from a binary file.

    Parameters:
    file_path (str): The path to the binary file to read.
    en_filter (float, optional): The energy filter threshold. Defaults to 0.
    group_events (bool, optional): Whether to group events. Defaults to True.

    Returns:
    generator: A generator that yields tuples containing the data for each event.
               det1: list[list] -> [[tstp_n, energy_n, chid_n]] for n in number of hits in det1
               det2: list[list] -> [[tstp_n, energy_n, chid_n]] for n in number of hits in det2
    """
    # Define the struct formats and sizes
    header_format = "B" if group_events else "2B"  # Format for the header
    data_format = "qfi"  # Format for the data (long long, float, int)
    header_size = struct.calcsize(header_format)
    data_size = struct.calcsize(data_format)

    with open(file_path, "rb") as f:
        while True:
            header_data = f.read(header_size)
            if not header_data:
                break
            header = struct.unpack(header_format, header_data)

            det1 = read_detector_evt(f, data_format, data_size, header[0], en_filter)
            det2 = (
                read_detector_evt(f, data_format, data_size, header[1], en_filter)
                if not group_events
                else []
            )

            yield det1, det2


if __name__ == "__main__":
    pass
