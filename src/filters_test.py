from filters import filter_total_energy, filter_min_ch, filter_single_mM


def test_filter_total_energy():
    assert not filter_total_energy(10)
    assert not filter_total_energy(100)
    assert filter_total_energy(10.01)
    assert filter_total_energy(99.99)
    assert filter_total_energy(50)
    assert not filter_total_energy(5)
    assert not filter_total_energy(105)


def test_filter_min_ch():
    assert filter_min_ch([1, 2, 3], 3)
    assert filter_min_ch([1, 2, 3, 4], 3)
    assert not filter_min_ch([1, 2], 3)
    assert not filter_min_ch([], 1)


def test_filter_single_mM():
    det_list = [[0, 0, "sm1"]]
    sm_mM_map = {"sm1": "mM1"}
    assert filter_single_mM(det_list, sm_mM_map)

    det_list = [[0, 0, "sm1"], [0, 0, "sm1"]]
    sm_mM_map = {"sm1": "mM1"}
    assert filter_single_mM(det_list, sm_mM_map)

    det_list = [[0, 0, "sm1"], [0, 0, "sm2"]]
    sm_mM_map = {"sm1": "mM1", "sm2": "mM2"}
    assert not filter_single_mM(det_list, sm_mM_map)
