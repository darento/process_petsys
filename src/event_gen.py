import struct
import time
from typing import Callable, Iterator
import numpy as np


def event_generator(file_path):
    # Define the struct formats and sizes
    header_format = "2B"  # Format for the header (two uint8_t integers)
    data_format = "qfi"  # Format for the data (long long, float, int)
    header_size = struct.calcsize(header_format)
    data_size = struct.calcsize(data_format)

    # Create a memory-mapped file
    mmapped_file = np.memmap(file_path, dtype="uint8", mode="r")

    # Current position in the file
    position = 0

    # Iterate over the file using the memory-mapped array
    while position < len(mmapped_file) - header_size:
        # Read the header for the current event
        header = mmapped_file[position : position + header_size]
        num_lines_detector1, num_lines_detector2 = struct.unpack(header_format, header)
        position += header_size

        # Prepare data for each detector
        detector1_data = []
        detector2_data = []

        # Read data for Detector 1
        for _ in range(num_lines_detector1):
            if position >= len(mmapped_file) - data_size:
                break
            data = (
                mmapped_file[position : position + data_size]
                .view(dtype=np.uint8)
                .tobytes()
            )
            detector1_data.append(struct.unpack(data_format, data))
            position += data_size

        # Read data for Detector 2
        for _ in range(num_lines_detector2):
            if position >= len(mmapped_file) - data_size:
                break
            data = (
                mmapped_file[position : position + data_size]
                .view(dtype=np.uint8)
                .tobytes()
            )
            detector2_data.append(struct.unpack(data_format, data))
            position += data_size

        # Yield the data for the current event
        yield detector1_data, detector2_data

    ## TODO: Implement the Andrew's yield function to check the performance

    def read_petsys_filebyfile(
        type_dict: dict, sm_filter: Callable = lambda x, y: True, singles: bool = False
    ) -> Callable:
        """
        Reader for petsys output for a list of input files.
        type_dict : Lookup for the channel id
                    to ChannelType.
        singles   : Is the file singles mode? default False.
        sm_filter : Function taking a tuple of smto filter the module data.
        returns
        petsys_event: Fn, loops over input file list and yields
                          event information.
        """

        def petsys_event(file_name: str) -> Iterator:
            """
            Read a single file:
            file_name  : String
                         The path to the file to be read.
            """
            yield from _read_petsys_file(file_name, type_dict, sm_filter, singles)

        return petsys_event

    return petsys_event


def _read_petsys_file(
    file_name: str, type_dict: dict, sm_filter: Callable, singles: bool = False
) -> Iterator:
    """
    Read all events from a single petsys
    file yielding those meeting sm_filter
    conditions.
    """
    # line_struct = '<BBqfi'         if singles else '<BBqfiBBqfi'
    line_struct = "B, B, i8, f4, i" if singles else "B, B, i8, f4, i, B, B, i8, f4, i"
    # evt_loop    = singles_evt_loop if singles else coinc_evt_loop
    evt_loop = singles_evt_loop if singles else coincidences_evt_loop
    # with open(file_name, 'rb') as fbuff:
    #     b_iter = struct.iter_unpack(line_struct, fbuff.read())
    #     for first_line in b_iter:
    #         sm1, sm2 = evt_loop(first_line, b_iter, type_dict)
    #         if sm_filter(sm1, sm2):
    #             yield sm1, sm2
    b_iter = np.nditer(np.memmap(file_name, np.dtype(line_struct), mode="r"))
    for first_line in b_iter:
        sm1, sm2 = evt_loop(first_line, b_iter, type_dict)
        if sm_filter(sm1, sm2):
            yield sm1, sm2


def singles_evt_loop(first_line, line_it, type_dict):
    # def singles_evt_loop(line_it, first_indx, type_dict):
    """
    Loop through the lines for an event
    of singles data.
    Needs to be optimised/tested
    Should be for what PETSys calls 'grouped'
    which seems more like a PET single.
    """
    # evt_end = first_indx + line_it[first_indx][0]
    # return evt_end, list(map(unpack_supermodule, line_it[first_indx:evt_end], repeat(type_dict))), []
    nlines = first_line[0] - 1
    return (
        list(
            map(
                unpack_supermodule,
                chain([first_line], islice(line_it, nlines)),
                repeat(type_dict),
            )
        ),
        [],
    )


def coincidences_evt_loop(first_line, line_it, type_dict):
    """
    Loop through the lines for an event
    of coincidence data.
    """
    sm1 = []
    sm2 = []
    ch_sm1 = set()
    ch_sm2 = set()
    nlines = first_line[0] + first_line[5] - 2
    for evt in chain([first_line], islice(line_it, nlines)):
        if evt[4] not in ch_sm1:
            sm1.append(unpack_supermodule(evt[:5], type_dict))
            ch_sm1.add(evt[4])
        if evt[-1] not in ch_sm2:
            sm2.append(unpack_supermodule(evt[5:], type_dict))
            ch_sm2.add(evt[-1])
    return sm1, sm2


if __name__ == "__main__":
    # Path to the binary file
    binary_file_path = "P:/Valencia/I3M/Proyectos/ForTheGroup/event_petsys/data/ERC/david_data_coinc.ldat"

    # Let's test the corrected function with the first few events
    corrected_event_gen = event_generator(binary_file_path)

    start_time = time.time()
    event_count = 0
    try:
        while True:
            next(corrected_event_gen)
            event_count += 1
    except StopIteration:
        pass
    end_time = time.time()

    total_time = end_time - start_time
    events_per_second = event_count / total_time

    print(f"Total time: {total_time} seconds")
    print(f"Total events: {event_count}")
    print(f"Events per second: {events_per_second}")
