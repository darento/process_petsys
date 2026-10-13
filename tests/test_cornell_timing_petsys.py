"""Canonical Cornell TIME-channel delay LUT (spec 008 T7, FR-12/FR-17).

The table covers the selected map's operational and run7 down TIME channels,
stores positive subtractive delays in ps with an explicit zero-mean gauge and
statuses, and corrects a positive injected delay and an endpoint swap once.
"""
import json

import numpy as np
import pytest

from cornell_timing_helpers import make_setup
from src.cornell import timing_pairs as tp
from src.cornell import timing_solver as ts
from src.cornell.timing_ingest import time_channel_classes
from src.cornell.timing_petsys import CONVENTION, CanonicalLut, LutError, write_lut
from src.mapping_generator import ChannelType

pytestmark = pytest.mark.fr("008-FR-12", "008-FR-17")
RUN7_DOWN = ((14, 0), (14, 1), (14, 4), (14, 5))


def test_canonical_lut_identity_units_gauge_and_status(tmp_path):
    setup = make_setup(tmp_path)
    time_channels, operational, down = time_channel_classes(setup, RUN7_DOWN)
    assert (len(operational), len(down)) == (3168, 32)
    rng = np.random.default_rng(74)
    truth = {ch: float(rng.uniform(-5000, 5000)) for ch in operational}
    observed = np.array(operational[:-10])  # the last ten channels see no coincidence
    edges = set(zip(observed[:-1], observed[1:]))
    while len(edges) < 4 * len(observed):
        a, b = rng.choice(observed, 2, replace=False)
        edges.add((min(a, b), max(a, b)))
    edges = sorted(edges)
    rows = np.zeros(len(edges), tp.PAIR_FIT_DTYPE)
    rows["channel_i"], rows["channel_j"] = np.array(edges).T
    rows["status"], rows["events"], rows["shift_sigma_ps"] = tp.USABLE, 20, 2.0
    rows["shift_ps"] = [truth[i] - truth[j] for i, j in edges]
    solution = ts.solve(rows, operational, down)
    path = tmp_path / "run7_time_delays.csv"
    manifest = write_lut(path, solution, {"pair_fits": "synthetic", "observation_set": "none"})
    lut = CanonicalLut.read(path)

    # identity: every operational and down TIME channel exactly once, nothing else
    assert sorted(lut.rows["time_channel"]) == sorted(operational + down)
    assert all(ChannelType.TIME in setup.chtype_map[int(c)] for c in lut.rows["time_channel"])
    energy = next(ch for ch, kinds in setup.chtype_map.items() if ChannelType.TIME not in kinds)
    with pytest.raises(LutError, match="not in table"):
        lut.delay(energy)
    with pytest.raises(ValueError, match="outside the operational"):
        bad = rows.copy()
        bad["channel_j"][0] = energy if energy > bad["channel_i"][0] else bad["channel_j"][0]
        bad["channel_i"][0] = min(bad["channel_i"][0], energy)
        ts.solve(bad, operational, down)

    # units, sign and gauge are explicit in the manifest
    stored = json.loads(path.with_suffix(".json").read_text())
    assert stored == json.loads(json.dumps(manifest))
    assert stored["units"] == "ps" and stored["convention"] == CONVENTION
    applied = lut.rows[lut.rows["apply"]]
    assert len(applied) == len(observed) and abs(applied["delay_ps"].mean()) < 1e-6
    assert abs(stored["gauge"]["applied_mean_delay_ps"]) < 1e-6
    expected = np.array([truth[c] for c in observed]) - np.mean([truth[c] for c in observed])
    order = np.argsort(applied["time_channel"])
    assert np.abs(applied["delay_ps"][order] - expected).max() < 1e-6

    # statuses are explicit for every row
    status = {int(r["time_channel"]): ts.CHANNEL_STATUSES[r["status"]] for r in lut.rows}
    assert {status[c] for c in down} == {"hardware unavailable"}
    assert {status[c] for c in operational[-10:]} == {"unobserved"}
    assert {status[int(c)] for c in observed} == {"calibrated"}
    assert stored["diagnostics"]["status_counts"] == {
        name: sum(s == name for s in status.values()) for name in ts.CHANNEL_STATUSES}

    # a positive injected delay corrects to the physical TOF, once; endpoint swap negates
    i, j = int(observed[5]), int(observed[900])
    tof = 431.0
    base = 10 ** 15
    t_i, t_j = base + round(truth[i] + tof), base + round(truth[j])
    residual = (t_i - t_j) - (truth[i] - truth[j]) - tof  # rounding of the synthetic timestamps
    assert lut.corrected_dt_ps(t_i, t_j, i, j) == pytest.approx(tof + residual, abs=1e-6)
    assert lut.corrected_dt_ps(t_j, t_i, j, i) == pytest.approx(-(tof + residual), abs=1e-6)
    with pytest.raises(LutError, match="hardware unavailable"):
        lut.corrected_dt_ps(t_i, t_j, i, down[0])
