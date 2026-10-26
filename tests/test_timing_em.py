"""Global EM estimator on IMAS-geometry synthetic sets (spec 011 T9, FR-7, FR-8)."""
import json

import numpy as np
import pytest

from imas_skew_helpers import CONFIG, Columns, channel_table, channels as module_channels, lors as imas_lors,     observation_set, skew_events
from src.imas.skew.geometry import ImasGeometry
from src.imas.skew.setup import ImasSetup
from src.timing import em, lut, solver as ts

pytestmark = pytest.mark.fr("011-FR-7", "011-FR-8")

RING = 2                      # SMs 48-71
MODULES = (5, 10)
WINDOW = 10_000.0


@pytest.fixture(scope="module")
def scanner():
    setup = ImasSetup.from_files(CONFIG)
    table = channel_table(setup, range(24 * RING, 24 * RING + 24), MODULES + (0,))
    return setup, ImasGeometry.nominal(setup), table


def lors(scanner, count, rng, *, pool=None, partner_pool=None):
    _, geometry, table = scanner
    pool = table[np.isin(table["mm"], MODULES)] if pool is None else pool
    return imas_lors(geometry, pool, count, rng, partner_pool=partner_pool)


def truth(table, rng):
    sm_offset = {sm: rng.uniform(-2500, 2500) for sm in np.unique(table["sm"])}
    mm_offset = {(sm, mm): rng.uniform(-300, 300) for sm, mm in zip(table["sm"], table["mm"])}
    return {int(r["ch"]): sm_offset[r["sm"]] + mm_offset[r["sm"], r["mm"]] + rng.uniform(-150, 150) for r in table}


def events(ci, cj, tof, delays, rng, **options):
    return skew_events(ci, cj, tof, delays, rng, window=WINDOW, **options)


def groupings(scanner, channels):
    setup = scanner[0]
    sm = np.array([setup.sm_mM_map[c][0] for c in channels.tolist()])
    mm = np.array([setup.sm_mM_map[c][1] for c in channels.tolist()])
    return (("sm", sm), ("minimodule", sm * 16 + mm))


def aligned(solution, delays, mask):
    rows = solution.channels[mask]
    true = np.array([delays[c] for c in rows["time_channel"].tolist()])
    return rows["delay_ps"] - rows["delay_ps"].mean(), true - true.mean(), rows["sigma_ps"]


def test_noiseless_zero_width_recovers_exactly(scanner):
    rng = np.random.default_rng(1)
    _, _, table = scanner
    delays = truth(table, rng)
    ci, cj, tof = lors(scanner, 40_000, rng)
    ci, cj, dt, tof = events(ci, cj, tof, delays, rng, noise=0.0, background=0.0, zero_width=True)
    data = em.EmData.from_arrays(ci, cj, dt, tof)
    options = em.EmOptions(sigma_ps=100.0, fit_sigma=False, background_fraction=0.0, fit_background=False,
                           tolerance_ps=1e-9, cg_tolerance=1e-13, min_events=1)
    solution = em.solve_em(data, groupings(scanner, data.channels), options)
    applied = solution.channels["apply"]
    assert applied.all() and solution.diagnostics["converged"]
    estimate, true, _ = aligned(solution, delays, applied)
    assert np.abs(estimate - true).max() < 1e-6


@pytest.fixture(scope="module")
def noisy(scanner):
    """Cold start: SM offsets +-2.5 ns, 200 ps noise, 10 % background; plus status probes."""
    rng = np.random.default_rng(7)
    setup, _, table = scanner
    delays = truth(table, rng)
    ci, cj, tof = lors(scanner, 70_000, rng)
    parts = [events(ci, cj, tof, delays, rng)]
    # ambiguous: one extra channel, one partner, one LOR, near branch only
    extra = table[(table["mm"] == 0) & (table["sm"] == 24 * RING)][:1]
    partner = table[np.isin(table["mm"], MODULES) & (table["sm"] == 24 * RING + 12)]
    for _ in range(50):
        a, b, t = lors(scanner, 50, rng, pool=extra, partner_pool=partner)
        if len(a):
            break
    single = np.repeat(t[:1], 40, axis=0)
    parts.append(events(np.repeat(a[:1], 40), np.repeat(b[:1], 40), single, delays, rng, background=0.0,
                        branch="near"))
    # disconnected: an isolated pair (minimodule 15, used nowhere else)
    sides = []
    for sm in (24 * RING, 24 * RING + 12):
        time, energy = module_channels(setup, sm, 15)
        sides.append((time[3], sm, min(setup.local_map[c][1] for c in energy) + 10.0))
    (first, sm_1, y_1), (second, sm_2, y_2) = sorted(sides)
    result = scanner[1].apply(Columns(sm_1=np.full(30, sm_1), x_1=np.full(30, setup.local_map[first][0]),
                                       y_1=np.full(30, y_1), sm_2=np.full(30, sm_2),
                                       x_2=np.full(30, setup.local_map[second][0]), y_2=np.full(30, y_2)))
    assert (result.codes == 0).all()
    delays[first], delays[second] = 100.0, -100.0
    parts.append(events(np.full(30, first), np.full(30, second), result.shell_tof_ps, delays, rng, background=0.0))
    ci, cj, dt, tof = (np.concatenate(values) for values in zip(*parts))
    unobserved = int(table[(table["mm"] == 0) & (table["sm"] == 24 * RING + 5)]["ch"][0])
    listed = np.unique(np.r_[ci, cj, unobserved])
    data = em.EmData.from_arrays(ci, cj, dt, tof, channels=listed)
    solution = em.solve_em(data, groupings(scanner, data.channels), em.EmOptions(window_ps=2 * WINDOW))
    probes = {"ambiguous": int(extra["ch"][0]), "disconnected": (int(first), int(second)), "unobserved": unobserved}
    return data, solution, delays, probes


def test_cold_start_pulls_and_model(noisy):
    data, solution, delays, probes = noisy
    model = solution.diagnostics["model"]
    assert solution.diagnostics["converged"]
    assert 190 < model["sigma_ps"] < 215 and 0.08 < model["background_fraction"] < 0.12
    assert 0.47 < model["branch_fraction"] < 0.53
    main = solution.channels["apply"] & (solution.channels["component"] == 0)
    assert main.sum() >= 380
    estimate, true, sigma = aligned(solution, delays, main)
    pulls = (estimate - true) / sigma
    assert abs(pulls.mean()) < 0.2 and 0.8 < pulls.std() < 1.25, (pulls.mean(), pulls.std())
    assert np.abs(estimate - true).max() < 150          # no alias-shifted channel among the applied ones


def test_statuses_and_no_invented_values(noisy, tmp_path):
    data, solution, _, probes = noisy
    rows = {int(r["time_channel"]): r for r in solution.channels}
    assert rows[probes["ambiguous"]]["status"] == ts.AMBIGUOUS
    for channel in probes["disconnected"]:
        assert rows[channel]["status"] == ts.DISCONNECTED and np.isfinite(rows[channel]["delay_ps"])
    assert rows[probes["unobserved"]]["status"] == ts.UNOBSERVED
    for channel in (probes["ambiguous"], probes["unobserved"]):
        assert not rows[channel]["apply"] and np.isnan(rows[channel]["delay_ps"])
    applied = solution.channels[solution.channels["apply"]]
    assert set(np.unique(applied["status"]).tolist()) <= set(ts.APPLIED_STATUSES)
    assert np.isfinite(applied["delay_ps"]).all() and np.isfinite(applied["sigma_ps"]).all()
    assert (applied["delay_ps"] != 0).all() and (np.abs(applied["delay_ps"]) < 9000).all()
    # the canonical LUT carries only applied channels as corrections
    path = tmp_path / "lut.csv"
    lut.write_lut(path, solution, {"test": "synthetic"})
    table = lut.CanonicalLut.read(path)
    for channel in (probes["unobserved"], probes["ambiguous"], *probes["disconnected"]):
        with pytest.raises(lut.LutError, match="no applicable delay"):
            table.delay(channel)
    assert table.delay(int(applied["time_channel"][0])) == pytest.approx(float(applied["delay_ps"][0]))
    json.dumps(solution.diagnostics)


def test_hutchinson_matches_dense(noisy):
    data, solution, _, _ = noisy
    observed = data.pair_events > 0
    weights = np.zeros(len(data.pair_i))
    final = em._estep_all(data, np.nan_to_num(solution.channels["delay_ps"]), em._Theta(
        **{"sigma": solution.diagnostics["model"]["sigma_ps"],
           "background": solution.diagnostics["model"]["background_fraction"],
           "branch": solution.diagnostics["model"]["branch_fraction"]}), em.EmOptions(window_ps=2 * WINDOW))
    weights = final.spair
    labels = em._components(data.pair_i[observed], data.pair_j[observed], len(data.channels))
    matrix, degree = em._laplacian(data.pair_i[observed], data.pair_j[observed], weights[observed],
                                   len(data.channels))
    rng = np.random.default_rng(3)
    dense, _ = em._pinv_diagonal(matrix, degree, labels, em.EmOptions(), rng)
    stochastic, methods = em._pinv_diagonal(matrix, degree, labels, em.EmOptions(dense_limit=100, probes=400), rng)
    assert "hutchinson" in methods
    big = np.isfinite(dense) & np.isfinite(stochastic) & (np.bincount(labels)[labels] > 100)
    ratio = np.sqrt(stochastic[big] / dense[big])
    assert abs(np.median(ratio) - 1) < 0.05 and np.percentile(np.abs(ratio - 1), 90) < 0.2


def test_observation_set_chunks_equal_arrays(noisy, tmp_path):
    data, _, _, _ = noisy
    pid, dt, tof = data.chunks[0]
    ci, cj = data.channels[data.pair_i[pid]], data.channels[data.pair_j[pid]]
    observations = observation_set(tmp_path / "obs", ci, cj, dt, tof)
    from_set = em.EmData.from_observation_set(observations, data.channels, directory=tmp_path / "em",
                                              dtype=np.float64)
    from_arrays = em.EmData.from_arrays(ci, cj, np.round(dt), tof, channels=data.channels)
    assert isinstance(from_set.chunks[0][1], np.memmap)
    options = em.EmOptions(window_ps=2 * WINDOW)
    b = np.random.default_rng(0).normal(0, 50, len(data.channels))
    theta = em._Theta(200.0, 0.1, 0.5)
    first, second = em._estep_all(from_set, b, theta, options), em._estep_all(from_arrays, b, theta, options)
    assert first.stats["log_likelihood"] == pytest.approx(second.stats["log_likelihood"], rel=1e-12)
    assert np.allclose(first.grad, second.grad, rtol=1e-9, atol=1e-12)
    assert from_set.channel_events.tolist() == from_arrays.channel_events.tolist()
