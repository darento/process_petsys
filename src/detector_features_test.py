from .detector_features import (
    calculate_total_energy,
    calculate_centroid,
    calculate_DOI,
    calculate_impact_hits,
)
from .variables_test import (
    det_list,
    chtype_map,
    local_dict,
    local_coord_dict,
)
from .fem_handler import get_FEM_instance

import numpy as np


def test_calculate_total_energy():
    assert calculate_total_energy(det_list, chtype_map) == 141.5
    assert not calculate_total_energy(det_list, chtype_map) == 119.0


def test_centroid():
    print(calculate_centroid(det_list, local_dict, 1, 1))
    assert calculate_centroid(det_list, local_dict, 1, 1) == (
        0.7360424389866067,
        0.8360424389866068,
    )
    print(calculate_centroid(det_list, local_dict, 0, 0))
    assert calculate_centroid(det_list, local_dict, 0, 0) == (
        0.8,
        0.8999999999999999,
    )

    assert not calculate_centroid(det_list, local_dict, 1, 2) == (
        0.5877637793800458,
        0.6877637793800458,
    )


def test_calculate_DOI():
    assert calculate_DOI(det_list, local_dict) == 141.5 / 33.0
    assert not calculate_DOI(det_list, local_dict) == 0.1


def test_calculate_impact_hits():
    FEM_instance = get_FEM_instance("FEM256", 1.1, 1.1, 8, False)
    result = calculate_impact_hits(det_list, local_coord_dict, FEM_instance)
    energy = np.array([11.0, 33.0, 15.0, 28.0, 21.0, 5.0, 5.5, 23.0])
    diagonal_matrix = np.diag(energy)

    np.testing.assert_array_equal(result, diagonal_matrix)
