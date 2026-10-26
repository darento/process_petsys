"""Spec 008 pair estimator and the global EM on IMAS observation sets (spec 011 T10, FR-7, FR-8)."""
import numpy as np
import pytest

from imas_skew_helpers import CONFIG, channel_table, lors, observation_set, skew_events
from src.imas.skew import solve
from src.imas.skew.geometry import ImasGeometry
from src.imas.skew.setup import ImasSetup
from src.timing import pairs as tp, solver as ts

pytestmark = pytest.mark.fr("011-FR-7", "011-FR-8")


@pytest.fixture(scope="module")
def scanner():
    setup = ImasSetup.from_files(CONFIG)
    return setup, ImasGeometry.nominal(setup)


def delays_for(table, rng):
    sm = {s: rng.uniform(-2500, 2500) for s in np.unique(table["sm"])}
    return {int(r["ch"]): sm[r["sm"]] + rng.uniform(-300, 300) for r in table}


@pytest.fixture(scope="module")
def dense(scanner, tmp_path_factory):
    """Few channels, >= 40 events per pair: the spec 008 regime."""
    setup, geometry = scanner
    rng = np.random.default_rng(21)
    table = channel_table(setup, (48, 49, 60, 61), (5,))
    delays = delays_for(table, rng)
    ci, cj, tof = lors(geometry, table, 60_000, rng)
    ci, cj, dt, tof = skew_events(ci, cj, tof, delays, rng)
    keys, inverse, counts = np.unique(ci * 1_000_000 + cj, return_inverse=True, return_counts=True)
    keep = counts[inverse] >= 40
    observations = observation_set(tmp_path_factory.mktemp("dense") / "obs", ci[keep], cj[keep], dt[keep],
                                   tof[keep])
    return observations, delays


def pulls(solution, delays):
    rows = solution.channels[solution.channels["apply"]]
    true = np.array([delays[c] for c in rows["time_channel"].tolist()])
    return (rows["delay_ps"] - rows["delay_ps"].mean() - (true - true.mean())) / rows["sigma_ps"]


def test_pair_estimator_recovers_delays(scanner, dense):
    setup, _ = scanner
    observations, delays = dense
    solution, fits = solve.pair_estimator(observations, setup)
    model = solution.diagnostics["model"]
    assert model["source"].startswith("estimated") and model["window_ps"] == 20_000
    assert 170 < model["sigma_ps"] < 230
    assert (fits["status"] == tp.USABLE).mean() > 0.9
    applied = solution.channels["apply"]
    assert applied.sum() >= 30 and set(solution.channels["time_channel"][applied].tolist()) <= set(delays)
    p = pulls(solution, delays)
    assert np.abs(p).max() < 4 and abs(p.mean()) < 0.5
    # every IMAS TIME channel has a status; the others are unobserved, NaN and not applied
    assert len(solution.channels) == len(setup.time_channels)
    rest = solution.channels[~np.isin(solution.channels["time_channel"], list(delays))]
    assert (rest["status"] == ts.UNOBSERVED).all() and np.isnan(rest["delay_ps"]).all() and not rest["apply"].any()


def test_em_agrees_with_pairs_on_dense_set(scanner, dense, tmp_path):
    setup, _ = scanner
    observations, delays = dense
    pairs, _ = solve.pair_estimator(observations, setup)
    em, data = solve.em_estimator(observations, setup, directory=tmp_path / "em")
    assert np.abs(pulls(em, delays)).max() < 4
    both = pairs.channels["apply"] & em.channels["apply"]
    assert both.sum() >= 30
    difference = em.channels["delay_ps"][both] - pairs.channels["delay_ps"][both]
    difference -= difference.mean()
    combined = np.hypot(em.channels["sigma_ps"][both], pairs.channels["sigma_ps"][both])
    assert np.abs(difference / combined).max() < 4
    assert em.diagnostics["observations"] == data.observations == int(observations.pair_counts()["count"].sum())


def test_sparse_regime_applies_nothing(scanner, tmp_path):
    """~1.5 events per pair (the T7 measurement): every pair unavailable, no channel applied, no zero."""
    setup, geometry = scanner
    rng = np.random.default_rng(5)
    table = channel_table(setup, range(48, 72), (5, 10))
    ci, cj, tof = lors(geometry, table, 5_000, rng)
    ci, cj, dt, tof = skew_events(ci, cj, tof, delays_for(table, rng), rng)
    observations = observation_set(tmp_path / "obs", ci, cj, dt, tof)
    counts = observations.pair_counts()["count"]
    assert counts.mean() < 2 and counts.max() < 30
    solution, fits = solve.pair_estimator(observations, setup)
    assert solution.diagnostics["model"]["source"].startswith("declared")
    assert len(fits) == len(counts) and (fits["status"] != tp.USABLE).all()
    assert np.isnan(fits["shift_ps"]).all()
    assert not solution.channels["apply"].any() and np.isnan(solution.channels["delay_ps"]).all()
    assert solution.diagnostics["status_counts"]["calibrated"] == 0
