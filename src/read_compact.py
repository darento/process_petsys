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

    Raises:
    ValueError: If the file ends before all ``num_lines`` hits are read.
    """
    expected = data_size * num_lines
    raw = f.read(expected)
    if len(raw) < expected:
        raise ValueError(f"expected {num_lines} hits ({expected} bytes), found {len(raw)} bytes")
    data = [struct.unpack_from(data_format, raw, i * data_size) for i in range(num_lines)]

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

    Raises:
        ValueError: If the last record is truncated, after every complete record
            has been yielded; a partial record never yields hits.
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
                record_start = read_size
                header_data = f.read(header_size)
                if not header_data:
                    break
                if len(header_data) < header_size:
                    raise ValueError(
                        f"{file_path}: truncated record header at byte {record_start}"
                    )
                header = struct.unpack(header_format, header_data)

                # Update the read size
                read_size += header_size
                pbar.update(header_size)

                try:
                    det1 = read_detector_evt(
                        f, data_format, data_size, header[0], en_filter
                    )
                    det2 = (
                        read_detector_evt(f, data_format, data_size, header[1], en_filter)
                        if not group_events
                        else []
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{file_path}: truncated record at byte {record_start}: {exc}"
                    ) from None
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
