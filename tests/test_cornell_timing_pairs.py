"""Two-branch shell constraints per TIME-channel pair (spec 008 T6, FR-9).

Synthetic truth: events are born uniformly along the true shell segments of
their own LOR (``shell_intervals``), smeared by a Gaussian kernel, shifted by
a known ``d = b_i - b_j`` and mixed with flat background over the legacy
``|dt| < 15 ns`` window. Fits use the same model family without truth labels;
no event is assigned to its nearest branch.
"""
import numpy as np
import pytest

from cornell_timing_helpers import DRAWING_SOURCE
from src.cornell.skew import pairs as tp
from src.cornell.skew.cache import OBSERVATION_DTYPE
from src.cornell.skew.geometry import C_MM_PER_PS, AnnularSource, shell_intervals

pytestmark = pytest.mark.fr("008-FR-9")
SOURCE = AnnularSource(**DRAWING_SOURCE)
MODEL = tp.PairModel(sigma_ps=200.0, background_fraction=0.1)
DIAMETRIC = ((-400.0, 0.0, 0.0), (400.0, 0.0, 0.0))


def simulate(rng, shift, count, *, geometries=(DIAMETRIC,), weights=None, near_fraction=0.5, model=MODEL,
             true_source=SOURCE, fit_source=None, channels=(10, 20), role=0, record0=0):
    """Canonical observation rows for one channel pair with known shift."""
    fit_source = fit_source or true_source
    rows = np.zeros(count, OBSERVATION_DTYPE)
    rows["channel_i"], rows["channel_j"], rows["role"] = channels[0], channels[1], role
    rows["record"] = record0 + np.arange(count)
    pick = rng.choice(len(geometries), count, p=weights)
    background = rng.random(count) < model.background_fraction
    near = rng.random(count) < near_fraction
    for k, (start, end) in enumerate(geometries):
        length = float(np.linalg.norm(np.subtract(end, start)))
        truth = shell_intervals(start, end, true_source)
        fitted = shell_intervals(start, end, fit_source)
        assert len(truth) == 2 and len(fitted) == 2
        mine = pick == k
        segment = np.where(near[mine], 0, 1)
        lo = np.array([s.fraction_start for s in truth])[segment]
        hi = np.array([s.fraction_end for s in truth])[segment]
        fraction = lo + (hi - lo) * rng.random(mine.sum())
        tof = length * (2 * fraction - 1) / C_MM_PER_PS
        rows["dt_ps"][mine] = np.rint(tof + shift + rng.normal(0, model.sigma_ps, mine.sum()))
        rows["shell_tof_ps"][mine] = [fitted[0].tof_start_ps, fitted[0].tof_end_ps,
                                      fitted[1].tof_start_ps, fitted[1].tof_end_ps]
        rows["endpoint_i_mm"][mine], rows["endpoint_j_mm"][mine] = start, end
    half = model.window_ps / 2
    rows["dt_ps"][background] = np.rint(rng.uniform(-half + 1, half - 1, background.sum()))
    return rows


@pytest.fixture(scope="module", autouse=True)
def compiled():
    """Cold numba compilation belongs to setup, not to the serial call-duration check."""
    rows = simulate(np.random.default_rng(0), 0.0, 80)
    tp.fit_pairs(rows, MODEL, free_separation=True)
    tp.estimate_model(rows, min_events=50, iterations=1)


def one(rows, model=MODEL, **options):
    fits = tp.fit_pairs(rows, model, tp.FitOptions(**options))
    assert len(fits) == 1
    return fits[0]


def test_known_shift_with_unequal_branch_populations():
    rng = np.random.default_rng(81)
    truth = rng.uniform(-6000, 6000, 36)
    rows = np.concatenate([
        simulate(rng, d, 80, near_fraction=(0.2, 0.5, 0.85)[k % 3], channels=(k, 100 + k), record0=1000 * k)
        for k, d in enumerate(truth)])
    rng.shuffle(rows)
    fits = tp.fit_pairs(rows, MODEL)
    assert len(fits) == 36 and (fits["status"] == tp.USABLE).all()
    assert (fits["channel_i"] == np.arange(36)).all() and (fits["events"] == 80).all()
    pulls = (fits["shift_ps"] - truth) / fits["shift_sigma_ps"]
    assert abs(pulls.mean()) < 0.5 and 0.6 < pulls.std() < 1.5
    assert np.abs(pulls).max() < 4
    # the fitted branch fraction follows the populations; the shift does not
    for k, expected in enumerate((0.2, 0.5, 0.85)):
        assert abs(np.median(fits["branch_fraction"][k::3]) - expected) < 0.12
    # a count-weighted midpoint average would be biased by (w - 1/2) * separation
    k = 0
    pair = rows[rows["channel_i"] == k]
    pair = pair[np.abs(pair["dt_ps"] - truth[k]) < 3000]
    mid = pair["shell_tof_ps"].mean(axis=1)
    assert abs(np.mean(pair["dt_ps"] - mid) - truth[k]) > 250
    assert abs(fits["shift_ps"][k] - truth[k]) < 4 * fits["shift_sigma_ps"][k]


def test_geometry_variation_and_shell_thickness():
    rng = np.random.default_rng(82)
    oblique = ((-390.0, -60.0, -40.0), (380.0, 120.0, 70.0))
    shifted = ((-250.0, 0.0, 0.0), (550.0, 0.0, 0.0))  # LOR middle 150 mm from the source axis
    # two LOR geometries in one pair: branch midpoints 0 and -1000 ps
    mixed = simulate(rng, 1234.0, 400, geometries=(DIAMETRIC, shifted), weights=(0.5, 0.5))
    fit = one(mixed)
    assert fit["status"] == tp.USABLE
    assert abs(fit["shift_ps"] - 1234.0) < 4 * fit["shift_sigma_ps"]
    # compressing the pair to its average geometry (grouping error) is visibly biased here
    compressed = mixed.copy()
    compressed["shell_tof_ps"] = mixed["shell_tof_ps"].mean(axis=0)
    assert abs(one(compressed)["shift_ps"] - 1234.0) > max(100, 10 * fit["shift_sigma_ps"])
    # thin and thick walls, each with its own finite-segment template
    for wall in (0.5, 25.0):
        source = AnnularSource(**{**DRAWING_SOURCE, "inner_radius_mm": 142.5 - wall})
        rows = simulate(rng, -2500.0, 300, geometries=(DIAMETRIC, oblique), weights=(0.5, 0.5), true_source=source)
        fit = one(rows)
        assert fit["status"] == tp.USABLE
        assert abs(fit["shift_ps"] + 2500.0) < 4 * fit["shift_sigma_ps"]
    # a 25 mm wall fitted with point-like branches is biased in separation, not shift sign
    thick = AnnularSource(**{**DRAWING_SOURCE, "inner_radius_mm": 117.5})
    rows = simulate(rng, 0.0, 600, true_source=thick, fit_source=SOURCE)
    assert abs(one(rows)["shift_ps"]) < 4 * one(rows)["shift_sigma_ps"]


def test_miscentering_and_axial_edges():
    rng = np.random.default_rng(83)
    moved = AnnularSource(**{**DRAWING_SOURCE, "center_mm": (1.5, 0.0, 0.0)})
    rows = simulate(rng, 500.0, 2000, true_source=moved)
    # templates from the measured pose are unbiased
    fit = one(rows)
    assert abs(fit["shift_ps"] - 500.0) < 4 * fit["shift_sigma_ps"]
    # assuming the nominal centre moves the midpoint by 2*dx/c (6.67 ps/mm) along this LOR
    nominal = rows.copy()
    nominal["shell_tof_ps"] = [s for interval in shell_intervals(*DIAMETRIC, SOURCE)
                               for s in (interval.tof_start_ps, interval.tof_end_ps)]
    bias = one(nominal)["shift_ps"] - fit["shift_ps"]
    assert abs(bias - 2 * 1.5 / C_MM_PER_PS) < 3
    # LORs leaving the shell through its axial end: clipped segments of unequal length
    edge = ((-400.0, 0.0, 140.0), (400.0, 0.0, 154.84))
    clipped = shell_intervals(*edge, SOURCE)
    assert len(clipped) == 2
    widths = [s.tof_end_ps - s.tof_start_ps for s in clipped]
    assert widths[1] < 0.5 * widths[0]
    fit = one(simulate(rng, -800.0, 400, geometries=(edge, DIAMETRIC), weights=(0.6, 0.4)))
    assert fit["status"] == tp.USABLE
    assert abs(fit["shift_ps"] + 800.0) < 4 * fit["shift_sigma_ps"]


def test_unresolved_peak_is_unavailable():
    rng = np.random.default_rng(84)
    # near-tangential LOR: branch midpoints only ~480 ps apart, below 2 kernel widths
    TANGENTIAL = ((-400.0, 135.0, 0.0), (400.0, 135.0, 0.0))
    BROAD = tp.PairModel(sigma_ps=300.0, background_fraction=0.1)
    cases = {
        "one branch": simulate(rng, 300.0, 60, near_fraction=1.0, channels=(1, 2)),
        "merged peaks": simulate(rng, 300.0, 400, geometries=(TANGENTIAL,), channels=(3, 4)),
        "too few events": simulate(rng, 300.0, 4, channels=(5, 6)),
        "background only": simulate(rng, 300.0, 40, model=tp.PairModel(200.0, 0.999), channels=(7, 8)),
    }
    fits = tp.fit_pairs(np.concatenate(list(cases.values())), BROAD)
    assert len(fits) == 4
    for (name, _), fit in zip(cases.items(), fits):
        assert fit["status"] != tp.USABLE, name
        assert np.isnan(fit["shift_ps"]) and np.isnan(fit["shift_sigma_ps"]), name
    assert list(fits["status"][:2]) == [tp.ONE_BRANCH, tp.MERGED]
    # a usable pair beside them is unaffected
    good = tp.fit_pairs(np.concatenate([cases["one branch"], simulate(rng, 300.0, 60, channels=(9, 10))]), BROAD)
    assert list(good["status"]) == [tp.ONE_BRANCH, tp.USABLE]
    # an offset beyond the scan range is flagged, not folded back
    edge = one(simulate(rng, 9000.0, 200), scan_ps=8000.0)
    assert edge["status"] in (tp.SCAN_EDGE, tp.ONE_BRANCH) and np.isnan(edge["shift_ps"])


def test_global_kernel_and_background_estimate():
    rng = np.random.default_rng(85)
    truth = tp.PairModel(sigma_ps=230.0, background_fraction=0.18)
    rows = np.concatenate([simulate(rng, d, 120, model=truth, channels=(k, 500 + k), record0=1000 * k)
                           for k, d in enumerate(rng.uniform(-4000, 4000, 40))])
    model, history = tp.estimate_model(rows, start=tp.PairModel(150.0, 0.05), min_events=50)
    assert abs(model.sigma_ps - 230.0) < 12 and abs(model.background_fraction - 0.18) < 0.03
    assert history and history[-1]["pairs"] == 40


def test_free_separation_is_offset_free():
    rng = np.random.default_rng(86)
    wide = AnnularSource(**{**DRAWING_SOURCE, "inner_radius_mm": 143.5, "outer_radius_mm": 148.5})
    for shift in (-3000.0, 2500.0):
        rows = simulate(rng, shift, 1500, true_source=wide, fit_source=SOURCE)
        fit = tp.fit_pairs(rows, MODEL, free_separation=True)[0]
        expected = 2 * 2 * 6.0 / C_MM_PER_PS  # both branches move 6 mm outward
        assert abs(fit["separation_ps"] - expected) < 4 * fit["separation_sigma_ps"]
        assert abs(fit["shift_ps"] - shift) < 4 * fit["shift_sigma_ps"]
    assert np.isnan(tp.fit_pairs(rows, MODEL)[0]["separation_ps"])


def test_partitions_and_roles_stay_separate():
    rng = np.random.default_rng(87)
    train = simulate(rng, 700.0, 60, channels=(1, 2), role=0)
    validation = simulate(rng, 700.0, 60, channels=(1, 2), role=1, record0=10_000)

    class Observations:
        partitions = 2

        def partition(self, p):
            return (train, validation)[p]

    fits = tp.fit_observation_set(Observations(), MODEL)
    assert list(fits["role"]) == [0] and fits["events"][0] == 60
    both = tp.fit_observation_set(Observations(), MODEL, roles=(0, 1))
    assert list(both["role"]) == [0, 1]
    with pytest.raises(ValueError, match="two finite shell segments"):
        broken = train.copy()
        broken["shell_tof_ps"][0, 2:] = np.nan
        tp.fit_pairs(broken, MODEL)
