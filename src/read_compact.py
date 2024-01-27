import struct
import time
from typing import Callable, Iterator
import numpy as np


def read_header(
    mmapped_file: np.memmap, file_pos: int, header_format: str, header_size: int
) -> tuple:
    """
    Reads the header for the current event from a memory-mapped file.

    Parameters:
    mmapped_file (mmap.mmap): The memory-mapped file to read from.
    file_pos (int): The current position in the file.
    header_format (str): The format string for struct.unpack to parse the header.
    header_size (int): The size of the header in bytes.

    Returns:
    tuple: A tuple containing the header values and the new file position.
    """
    header = mmapped_file[file_pos : file_pos + header_size]
    header_values = struct.unpack(header_format, header)
    file_pos += header_size
    return header_values, file_pos


def read_event(
    mmapped_file: np.memmap,
    file_pos: int,
    data_format: str,
    data_size: int,
    num_lines: int,
    en_filter: int = 0,
) -> tuple:
    """
    Reads the data for a single event from a memory-mapped file.

    Parameters:
    mmapped_file (mmap.mmap): The memory-mapped file to read from.
    file_pos (int): The current position in the file.
    data_format (str): The format string for struct.unpack to parse the data.
    data_size (int): The size of each data line in bytes.
    num_lines (int): The number of data lines to read.
    en_filter (float, optional): The energy filter threshold. Defaults to 0.

    Returns:
    tuple: A tuple containing the list of data lines read from the file and the new file position.
    """
    data = []
    for _ in range(num_lines):
        if file_pos >= len(mmapped_file) - data_size:
            break
        line = (
            mmapped_file[file_pos : file_pos + data_size].view(dtype=np.uint8).tobytes()
        )
        evt_data = struct.unpack(data_format, line)

        # Filter events with energy below the threshold
        if evt_data[1] < en_filter:
            file_pos += data_size
            continue

        data.append(evt_data)
        file_pos += data_size
    return data, file_pos


def event_generator(
    file_path: str, min_ch: int = 0, en_filter: float = 0, group_events: bool = True
) -> Iterator[tuple]:
    """
    Generates events from a binary file.

    Parameters:
    file_path (str): The path to the binary file to read.
    min_ch (int, optional): The minimum number of channels. Defaults to 0.
    en_filter (float, optional): The energy filter threshold. Defaults to 0.
    group_events (bool, optional): Whether to group events. Defaults to True.

    Returns:
    generator: A generator that yields tuples containing the data for each event.
    """
    # Define the struct formats and sizes
    header_format = "B" if group_events else "2B"  # Format for the header
    data_format = "qfi"  # Format for the data (long long, float, int)
    header_size = struct.calcsize(header_format)
    data_size = struct.calcsize(data_format)

    # Create a memory-mapped file
    mmapped_file = np.memmap(file_path, dtype="uint8", mode="r")

    # Current position in the file
    file_pos = 0

    # Iterate over the file using the memory-mapped array
    while file_pos < len(mmapped_file) - header_size:
        # Read the header for the current event
        header_values, file_pos = read_header(
            mmapped_file, file_pos, header_format, header_size
        )

        # Check if the number of lines for each detector is less than the minimum
        if header_values[0] < min_ch or (
            not group_events and header_values[1] < min_ch
        ):
            # Calculate the size of the event data and add it to file_pos
            event_data_size = (header_values[0] + header_values[1]) * data_size
            file_pos += event_data_size
            continue
        # Read data for each detector
        det1_evt, file_pos = read_event(
            mmapped_file,
            file_pos,
            data_format,
            data_size,
            header_values[0],
            en_filter,
        )
        det2_evt = []
        if not group_events:
            det2_evt, file_pos = read_event(
                mmapped_file,
                file_pos,
                data_format,
                data_size,
                header_values[1],
                en_filter,
            )

        # Yield the data for the current event
        yield det1_evt, det2_evt


if __name__ == "__main__":
    # Path to the binary file
    binary_file_path = "P:/Valencia/I3M/Proyectos/ForTheGroup/event_petsys/data/ERC/david_data_coinc.ldat"

    # Let's test the corrected function with the first few events
    corrected_event_gen = event_generator(binary_file_path)

    start_time = time.time()
    event_count = 0
    try:
        while True:
            det1, det2 = next(corrected_event_gen)
            print(f"Length of detector 1 and detector 2: {len(det1)} - {len(det2)}")
            time.sleep(0.1)
            event_count += 1
            if event_count % 10000 == 0:
                print(f"Events processed: {event_count}")
    except StopIteration:
        pass
    end_time = time.time()

    total_time = end_time - start_time
    events_per_second = event_count / total_time

    print(f"Total time: {total_time} seconds")
    print(f"Total events: {event_count}")
    print(f"Events per second: {events_per_second}")
