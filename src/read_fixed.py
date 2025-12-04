import numpy as np
import os
from typing import Generator
from tqdm import tqdm


def read_fixed_file_numpy(
    file_path: str, batch_size: int = 10000, group_events: bool = False
) -> Generator[np.ndarray, None, None]:
    """
    Reads a fixed-size binary file using numpy for high performance.

    File Format:
        - Header: int32 (4 bytes) -> hit_limit

        Group format per event:
            - Header: uint8 (1 byte) -> n_hits
            - Hits: hit_limit * Event (16 bytes each)

        Coincidence format per event:
            - Header: 2x uint8 (2 bytes) -> (n_hits_side1, n_hits_side2)
            - Side 1: hit_limit * Event (16 bytes each)
            - Side 2: hit_limit * Event (16 bytes each)

        Event structure (16 bytes):
            - time: int64 (8 bytes)
            - energy: float32 (4 bytes)
            - channelID: int32 (4 bytes)

    Args:
        file_path: Path to the binary file
        batch_size: Number of events to read per batch
        group_events: If True, reads Group format (single side).
                     If False, reads Coincidence format (two sides)

    Yields:
        Numpy structured arrays of events
    """

    event_struct_dtype = np.dtype(
        [("time", "i8"), ("energy", "f4"), ("channelID", "i4")]
    )

    # Get total file size for progress bar
    total_size = os.path.getsize(file_path)

    with open(file_path, "rb") as f:
        # Read hit_limit from header (int32)
        hit_limit_data = f.read(4)
        if len(hit_limit_data) < 4:
            return
        hit_limit = np.frombuffer(hit_limit_data, dtype=np.int32)[0]

        # Define the full dtype based on format type
        if group_events:
            # Group format: 1 byte header + hits
            event_size_bytes = 1 + hit_limit * 16
            full_dtype = np.dtype(
                [
                    ("header", "u1"),  # Single uint8 for nHits
                    ("hits", event_struct_dtype, (hit_limit,)),
                ]
            )
        else:
            # Coincidence format: 2 byte header + side1 + side2
            event_size_bytes = 2 + 2 * hit_limit * 16
            full_dtype = np.dtype(
                [
                    ("header", "u1", (2,)),  # Two uint8 for (nHits1, nHits2)
                    ("side1", event_struct_dtype, (hit_limit,)),
                    ("side2", event_struct_dtype, (hit_limit,)),
                ]
            )

        # Verify dtype size matches expected bytes
        assert (
            full_dtype.itemsize == event_size_bytes
        ), f"Calculated dtype size {full_dtype.itemsize} != expected {event_size_bytes}"

        # Create progress bar
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="File read progress"
        ) as pbar:
            # Update progress for header
            pbar.update(4)

            while True:
                # Read a batch of events
                chunk = np.fromfile(f, dtype=full_dtype, count=batch_size)

                if len(chunk) == 0:
                    break

                # Update progress bar
                bytes_read = len(chunk) * event_size_bytes
                pbar.update(bytes_read)

                yield chunk
