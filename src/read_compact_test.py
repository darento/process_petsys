import struct
import os
from .read_compact import read_detector_evt, read_binary_file


def test_read_detector_evt():
    with open("test_data.bin", "wb") as f:
        data_format = "iif"
        data = [(1, 2, 3.0), (4, 5, 6.0), (7, 8, 9.0)]
        for d in data:
            f.write(struct.pack(data_format, *d))

    with open("test_data.bin", "rb") as f:
        result = read_detector_evt(
            f, data_format, struct.calcsize(data_format), len(data), 5.0
        )

        assert result == [(4, 5, 6.0), (7, 8, 9.0)]
        assert not result == [(1, 2, 3.0), (4, 5, 6.0), (7, 8, 9.0)]
        assert isinstance(result, list)

    with open("empty_test_data.bin", "wb") as f:
        result = read_detector_evt(f, data_format, struct.calcsize(data_format), 0, 5.0)
        assert result == []


def test_read_binary_file():
    # Create a binary file in memory
    folder_path = "test_binary_path"
    file_path_1 = "test_binary_file_1.bin"
    file_path_2 = "test_binary_file_2.bin"

    os.makedirs(folder_path, exist_ok=True)

    # Write some test header and detector data to the file
    header_format = "2B"
    data_format = "qfi"

    header_data = [(1, 2), (3, 4)]
    detector_data_1 = [
        (1, 2.0, 3),
        (4, 5.0, 6),
        (7, 8.0, 9),
        (10, 11.0, 12),
        (1, 33.0, 3),
        (4, 15.0, 6),
        (7, 28.0, 9),
        (1, 21.0, 3),
        (4, 5.0, 6),
        (7, 5.5, 9),
    ]
    detector_data_2 = [
        (1, 2.0, 3),
        (4, 4.0, 6),
        (7, 5.5, 9),
        (10, 11.0, 12),
        (1, 33.0, 3),
        (4, 15.0, 6),
        (7, 28.0, 9),
        (1, 21.0, 3),
        (4, 5.0, 6),
        (7, 5.5, 9),
    ]
    with open(os.path.join(folder_path, file_path_1), "wb") as f:
        for header, data in zip(header_data, detector_data_1):
            f.write(struct.pack(header_format, *header))
            f.write(struct.pack(data_format, *data))

    with open(os.path.join(folder_path, file_path_2), "wb") as f:
        for header, data in zip(header_data, detector_data_2):
            f.write(struct.pack(header_format, *header))
            f.write(struct.pack(data_format, *data))

    result_1 = list(
        read_binary_file(os.path.join(folder_path, file_path_1), en_filter=6.0)
    )
    result_2 = list(
        read_binary_file(os.path.join(folder_path, file_path_2), en_filter=24.0)
    )

    # Check that the function returned the expected result
    not_expected_result_1 = [
        ([(1, 2.0, 3)], []),
        ([(4, 5.0, 6)], []),
        ([(7, 8.0, 9)], []),
        ([(10, 11.0, 12)], []),
    ]

    not_expected_result_2 = [
        ([(17, 18.0, 19)], []),
        ([(20, 21.0, 22)], []),
        ([(23, 24.0, 25)], []),
        ([(26, 27.0, 28)], []),
    ]

    expected_result_1 = [
        ([(4, 5.0, 6)], []),
        ([(7, 8.0, 9)], []),
        ([(10, 11.0, 12)], []),
    ]

    expected_result_2 = [
        ([(23, 24.0, 25)], []),
        ([(26, 27.0, 28)], []),
    ]

    assert not result_1 == not_expected_result_1
    assert not result_2 == not_expected_result_2
    assert result_1 == expected_result_1
    assert result_2 == expected_result_2
