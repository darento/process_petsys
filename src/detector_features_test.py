from .detector_features import (
    calculate_total_energy,
    calculate_centroid,
    calculate_DOI,
    calculate_impact_hits,
)
from .variables_test import (
    det_list,
    det_list_2,
    chtype_map,
    local_dict,
    local_coord_dict,
)
from .fem_handler import get_FEM_instance

import numpy as np


def test_calculate_total_energy():
    assert calculate_total_energy(det_list, chtype_map) == 118.5
    assert not calculate_total_energy(det_list, chtype_map) == 119.0


def test_centroid():
    assert calculate_centroid(det_list, local_dict, 1, 1) == (
        0.5877637793800458,
        0.6877637793800458,
    )

    assert calculate_centroid(det_list, local_dict, 0, 0) == (
        0.7000000000000001,
        0.7999999999999999,
    )

    assert not calculate_centroid(det_list, local_dict, 1, 2) == (
        0.5877637793800458,
        0.6877637793800458,
    )


def test_calculate_DOI():
    assert calculate_DOI(det_list, local_dict) == 118.5 / 33.0
    assert not calculate_DOI(det_list, local_dict) == 0.1


def test_calculate_impact_hits():
    FEM_instance = get_FEM_instance("FEM256", 1.1, 1.1, 8, False)
    result = calculate_impact_hits(det_list_2, local_coord_dict, FEM_instance)
    energy = np.array([11.0, 33.0, 15.0, 28.0, 21.0, 5.0, 5.5, 23.0])
    diagonal_matrix = np.diag(energy)

    np.testing.assert_array_equal(result, diagonal_matrix)
