import struct
from read_compact import read_detector_evt, read_binary_file


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
    file_path = "test_binary_file.bin"

    # Write some test header and detector data to the file
    header_format = "2B"
    data_format = "qfi"
    header_data = [(1, 2), (3, 4), (5, 6), (7, 8)]
    detector_data = [(1, 2.0, 3), (4, 5.0, 6), (7, 8.0, 9), (10, 11.0, 12)]
    with open(file_path, "wb") as f:
        for header, data in zip(header_data, detector_data):
            f.write(struct.pack(header_format, *header))
            f.write(struct.pack(data_format, *data))

    # Call function with the test data
    result = list(read_binary_file(f, en_filter=5.0))

    # Check that the function returned the expected result
    not_expected_result = [
        ([(1, 2.0, 3)], []),
        ([(4, 5.0, 6)], []),
        ([(7, 8.0, 9)], []),
        ([(10, 11.0, 12)], []),
    ]

    expected_result = [
        ([(4, 5.0, 6)], []),
        ([(7, 8.0, 9)], []),
        ([(10, 11.0, 12)], []),
    ]

    assert not result == not_expected_result
    assert result == expected_result
