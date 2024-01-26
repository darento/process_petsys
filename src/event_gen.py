import struct
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


if __name__ == "__main__":
    # Path to the binary file
    binary_file_path = "P:/Valencia/I3M/Proyectos/ForTheGroup/event_petsys/data/ERC/david_data_coinc.ldat"

    # Let's test the corrected function with the first few events
    corrected_event_gen = event_generator(binary_file_path)
    for _ in range(3):
        print(next(corrected_event_gen))
