"""IMAS side selection: one code per record by first failing cut, boundaries and features (spec 011 T4)."""
import numpy as np
import pytest

from imas_skew_helpers import CONFIG, FULL, T0, channels, mu_file, side, write
from src.imas.skew import select as sel
from src.imas.skew.setup import ImasSetup

pytestmark = pytest.mark.fr("011-FR-3")
A, B = (0, 5), (60, 6)          # SM 0 (ring 0) against SM 60 (ring 2, opposite azimuth)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    probe = ImasSetup.from_files(CONFIG)
    times = probe.time_channels.tolist()
    t_missing, t_invalid = channels(probe, 7, 0)[0][2], channels(probe, 7, 0)[0][3]
    values = {ch: 511.0 for ch in times if ch != t_missing}
    values[t_invalid] = float("nan")
    path = mu_file(tmp_path_factory.mktemp("mu") / "mu.tsv", values)
    return ImasSetup.from_files(CONFIG, path)


@pytest.fixture(scope="module")
def lookup(setup):
    return setup.lookup()


def run(tmp_path, records, lookup, selection, **options):
    path = write(tmp_path / "f.ldat", records)
    batches = list(sel.feature_batches(path, lookup, selection, batch_records=7, **options))
    codes = np.concatenate([b.codes for b in batches])
    ints = np.concatenate([b.ints for b in batches])
    floats = np.concatenate([b.floats for b in batches])
    return codes, ints, floats


def label(code):
    return sel.code_label(code)


def good(setup, dt=250, **kwargs):
    return side(setup, *A, **kwargs), side(setup, *B, stamp=T0 - dt)


def test_one_code_per_record_by_first_failing_cut(tmp_path, setup, lookup):
    t7, e7 = channels(setup, 7, 0)
    window = sel.COINCIDENCE_WINDOW_PS
    production = sel.Selection.production(lookup)
    cases = {
        "passed pre-geometry": good(setup),
        "unmapped channel": (side(setup, *A, extra=[(T0, 5.0, 790_600)]), side(setup, *B)),
        "side min channels": (side(setup, *A, energy=FULL[:4]), side(setup, *B)),
        "minimodule min channels": (side(setup, *A, energy=FULL[:3], extra=[(T0 + 900, 1.0, ch) for ch in
                                                                             channels(setup, 0, 6)[1][:2]]),
                                    side(setup, *B)),
        "coincidence window": good(setup, dt=window),
        "missing energy calibration": (side(setup, 7, 0, time={2: 9.0}), side(setup, *B)),
        "invalid energy calibration": (side(setup, *B), side(setup, 7, 0, time={3: 9.0})),
        "energy window": (side(setup, *A, energy=[200.0, 100.0, 60.0, 40.0, 8.0]), side(setup, *B)),
    }
    codes, ints, _ = run(tmp_path, list(cases.values()), lookup, production)
    assert [label(c) for c in codes] == list(cases)
    assert len(ints) == 1 and ints[0, 0] == 0
    counts = sel.count_codes(tmp_path / "f.ldat", lookup, production)
    assert sum(counts.values()) == len(cases) and set(counts) == set(cases)


def test_window_and_kev_boundaries(tmp_path, setup, lookup):
    window = sel.COINCIDENCE_WINDOW_PS
    production = sel.Selection.production(lookup)
    records = [good(setup, dt=window - 1), good(setup, dt=-window), good(setup, dt=-(window - 1)),
               (side(setup, *A, energy=[200.0, 100.0, 60.0, 40.0, 8.0]), side(setup, *B)),     # 408 keV
               (side(setup, *A, energy=[200.0, 100.0, 60.0, 40.0, 9.0]), side(setup, *B)),     # 409 keV
               (side(setup, *A, energy=[250.0, 200.0, 100.0, 40.0, 23.0]), side(setup, *B)),   # 613 keV
               (side(setup, *A, energy=[250.0, 200.0, 100.0, 40.0, 22.0]), side(setup, *B))]   # 612 keV
    codes, _, floats = run(tmp_path, records, lookup, production)
    assert [label(c) for c in codes] == ["passed pre-geometry", "coincidence window", "passed pre-geometry",
                                         "energy window", "passed pre-geometry", "energy window",
                                         "passed pre-geometry"]
    assert floats[:, 0].tolist() == [511.0, 511.0, 409.0, 612.0]


def test_hit_energy_floor_and_reference_mode(tmp_path, setup, lookup):
    # the fifth ENERGY hit (0.1 a.u.) is below en_min_ch = 0.2: production drops it, the legacy reader keeps it
    low = (side(setup, *A, energy=[200.0, 150.0, 100.0, 61.0, 0.1]), side(setup, *B))
    # zero ENERGY in two minimodules: no positive maximum (one zero minimodule alone is kept, as in legacy)
    zero = (side(setup, *A, energy=[0.0] * 5, extra=[(T0 + 900, 0.0, channels(setup, 0, 6)[1][0])]),
            side(setup, *B))
    wide = good(setup, dt=12_000)
    production = sel.Selection.production(lookup)
    reference = sel.Selection.reference()
    assert [label(c) for c in run(tmp_path, [low, zero, wide], lookup, production)[0]] == [
        "side min channels", "side min channels", "coincidence window"]
    assert [label(c) for c in run(tmp_path, [low, zero, wide], lookup, reference)[0]] == [
        "passed pre-geometry", "no positive minimodule energy", "passed pre-geometry"]
    assert reference.window_ps == 15_000 and production.window_ps == 10_000


def test_slab_tie_and_positions(tmp_path, setup, lookup):
    t_a, e_a = channels(setup, *A)
    t_b, _ = channels(setup, *B)
    other = channels(setup, 0, 6)
    tie_side = side(setup, *A, extra=[(T0 + 700 + i, value, ch) for i, (value, ch) in
                                      enumerate(zip(FULL, other[1]))] + [(T0 + 5, 3.0, other[0][0])])
    records = [(side(setup, *A, time={6: 9.0, 2: 5.0}), side(setup, *B, time={1: 4.0}, stamp=T0 - 300)), (
        tie_side, side(setup, *B))]
    production = sel.Selection.production(lookup)
    codes, ints, floats = run(tmp_path, records, lookup, production)
    assert codes.tolist() == [0, 0]
    names = sel.FEATURE_INT
    row = dict(zip(names, ints[0].tolist()))
    assert row["time_channel_1"] == t_a[6] and row["slab_1"] == 6 and row["time_channel_2"] == t_b[1]
    assert row["slab_2"] == 1 and row["dt_ps"] == 300 and row["module_tie"] == 0
    assert (row["sm_1"], row["mm_1"], row["sm_2"], row["mm_2"]) == (*A, *B)
    # x = slab centre of the top TIME channel; y = squared-charge ENERGY centroid
    weights = (np.array(FULL) + 1e-5) ** 2
    y = float(np.sum(weights * [setup.local_map[ch][1] for ch in e_a[:5]]) / weights.sum())
    assert floats[0, 2] == setup.local_map[t_a[6]][0] and floats[0, 3] == pytest.approx(y, abs=1e-9)
    assert dict(zip(names, ints[1].tolist()))["module_tie"] == 1 and ints[1, 4] == A[1]   # first seen wins
    _, _, legacy = run(tmp_path, records[:1], lookup, sel.Selection.reference())
    expected = (9.00001 * setup.local_map[t_a[6]][0] + 5.00001 * setup.local_map[t_a[2]][0]) / 14.00002
    assert legacy[0, 2] == pytest.approx(expected, abs=1e-9)


def test_calibration_mode_stops_after_window(tmp_path, setup, lookup):
    records = [(side(setup, 7, 0, time={2: 9.0}), side(setup, *B)), good(setup, dt=sel.COINCIDENCE_WINDOW_PS)]
    codes, ints, floats = run(tmp_path, records, lookup, sel.Selection.production(lookup), calibration=True)
    assert [label(c) for c in codes] == ["passed pre-geometry", "coincidence window"]
    assert np.isnan(floats[0, :2]).all() and ints[0, 1] == channels(setup, 7, 0)[0][2]


def test_holdout_codes_replace_selection(tmp_path, setup, lookup):
    from src.timing.kernels import Partition
    records = [good(setup)] * 5
    partition = Partition(0, "validation", -2 ** 62, 2 ** 62, 2)
    codes, ints, _ = run(tmp_path, records, lookup, sel.Selection.production(lookup), partition=partition)
    assert [label(c) for c in codes] == ["energy calibration prefix"] * 2 + ["passed pre-geometry"] * 3
    assert ints[:, 0].tolist() == [2, 3, 4]
