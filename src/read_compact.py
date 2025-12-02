import os
import struct
from typing import Iterator, BinaryIO

from tqdm import tqdm


def read_detector_evt(
    f: BinaryIO, data_format: str, data_size: int, num_lines: int, en_filter: float
) -> list:
    """
    Reads and unpacks data from a binary file for the lines corresponding to the hits
    at detector level.

    Parameters:
        - f (file): The binary file to read from.
        - data_format (str): The format string for struct.unpack to parse the data.
        - data_size (int): The size of each data line in bytes.
        - num_lines (int): The number of data lines to read.
        - en_filter (float): The energy filter threshold.

    Returns:
    list: A list of tuples containing the unpacked data.
    """
    try:
        data = [struct.unpack(data_format, f.read(data_size)) for _ in range(num_lines)]
    except struct.error:
        print("Error reading data")
        return []

    return [evt_ch for evt_ch in data if evt_ch[1] >= en_filter]


def read_binary_file(
    file_path: str, en_filter: float = 0, group_events: bool = False
) -> Iterator[tuple]:
    """
    Generates events from a binary file.

    Args:
        file_path: The path to the binary file to read
        en_filter: The energy filter threshold. Defaults to 0
        group_events: Whether to group events. Defaults to False

    Yields:
        Tuple of (det1, det2) where:
            - det1: list of [[timestamp, energy, channel_id]] for detector 1
            - det2: list of [[timestamp, energy, channel_id]] for detector 2
                   (empty list if group_events is True)
    """
    # Define the struct formats and sizes
    header_format = "B" if group_events else "2B"  # Format for the header
    data_format = "qfi"  # Format for the data (long long, float, int)
    header_size = struct.calcsize(header_format)
    data_size = struct.calcsize(data_format)

    total_size = os.path.getsize(file_path)
    read_size = 0

    with open(file_path, "rb") as f:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="File read progress"
        ) as pbar:
            while True:
                header_data = f.read(header_size)
                if not header_data:
                    break
                header = struct.unpack(header_format, header_data)

                # Update the read size
                read_size += header_size
                pbar.update(header_size)

                det1 = read_detector_evt(
                    f, data_format, data_size, header[0], en_filter
                )
                det2 = (
                    read_detector_evt(f, data_format, data_size, header[1], en_filter)
                    if not group_events
                    else []
                )
                # Update the read size
                read_size += header[0] * data_size + (
                    header[1] * data_size if not group_events else 0
                )

                pbar.update(
                    header[0] * data_size
                    + (header[1] * data_size if not group_events else 0)
                )

                yield det1, det2


if __name__ == "__main__":
    pass
