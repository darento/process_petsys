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
from src.cornell.timing_petsys import (CONVENTION, CanonicalLut, LutError, activated_ini, read_native_table,
                                       write_lut, write_native_table)
from src.cornell.timing_petsys import main as native_main
from src.utils import get_absolute_id
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


def small_lut(folder, setup):
    """LUT over a few real TIME channels: applied (calibrated / low statistics) and non-applied rows."""
    _, operational, down = time_channel_classes(setup, RUN7_DOWN)
    chosen = [operational[0], operational[13], operational[700], operational[2500], operational[3000],
              operational[3100], operational[3101], down[0]]
    rows = np.zeros(len(chosen), ts.CHANNEL_DTYPE)
    rows["time_channel"] = chosen
    status = [ts.CALIBRATED, ts.CALIBRATED, ts.LOW_STATISTICS, ts.CALIBRATED, ts.CALIBRATED,
              ts.CHANNEL_STATUSES.index("unobserved"), ts.CHANNEL_STATUSES.index("insufficient statistics"),
              ts.CHANNEL_STATUSES.index("hardware unavailable")]
    rows["status"] = status
    rows["apply"] = np.isin(status, ts.APPLIED_STATUSES)
    rows["delay_ps"] = [-4647.2534, 2436.1, 803.49951, 0.0004, 1642.0, np.nan, np.nan, np.nan]
    rows["sigma_ps"] = rows["raw_sigma_ps"] = np.where(rows["apply"], 6.0, np.nan)
    path = folder / "v3.csv"
    write_lut(path, ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {"status_counts": {}}),
              {"cross_run_outliers": [{"time_channel": int(chosen[1]), "runs": {}}]})
    return path, rows


def test_native_schema_units_and_channel_identity(tmp_path):
    setup = make_setup(tmp_path)
    lut_path, rows = small_lut(tmp_path, setup)
    table = tmp_path / "time_offset_calibration_v3.tsv"
    manifest = write_native_table(lut_path, table)
    lines = [line for line in table.read_text().splitlines() if not line.startswith("#")]
    applied = rows[rows["apply"]]
    assert len(lines) == len(applied) == manifest["table"]["rows"] == 5
    for line, row in zip(lines, applied[np.argsort(applied["time_channel"])]):
        port, slave, chip, channel, offset = line.split("\t")
        # PETsys MAKE_GID: channel | chip << 6 | slave << 12 | port << 17
        assert int(channel) | int(chip) << 6 | int(slave) << 12 | int(port) << 17 == row["time_channel"]
        assert abs(float(offset) - row["delay_ps"]) <= 0.0005  # ps, same positive subtractive sign
    offsets, skipped, duplicates = read_native_table(table)
    assert not skipped and not duplicates and set(offsets) == set(applied["time_channel"].tolist())
    stored = json.loads(table.with_suffix(".json").read_text())
    assert stored == json.loads(json.dumps(manifest)) and stored["units"] == "ps"
    assert stored["source_lut"]["sha256"] == json.loads(lut_path.with_suffix(".json").read_text())["table"]["sha256"]
    assert "must not receive the same LUT" in stored["correction_state"]
    with pytest.raises(LutError, match="exists"):
        write_native_table(lut_path, table)
    lut_path.write_text(lut_path.read_text().replace("-4647.2534", "-4647.0"))
    with pytest.raises(LutError, match="does not match"):
        write_native_table(lut_path, tmp_path / "other.tsv")


def test_invalid_and_other_channel_policy(tmp_path):
    setup = make_setup(tmp_path)
    lut_path, rows = small_lut(tmp_path, setup)
    table = tmp_path / "native.tsv"
    manifest = write_native_table(lut_path, table)
    offsets, _, _ = read_native_table(table)
    for row in rows[~rows["apply"]]:  # unobserved, insufficient, hardware unavailable: PETsys default 0
        assert int(row["time_channel"]) not in offsets
    assert int(rows["time_channel"][1]) in offsets  # listed cross-run outlier stays (owner decision)
    energy = [ch for ch, kinds in setup.chtype_map.items() if ChannelType.TIME not in kinds]
    assert not set(energy) & set(offsets)
    policy = manifest["policy"]
    assert policy["lut_status_counts"]["unobserved"] == 1 and policy["listed_cross_run_outliers"] == 1
    assert "ENERGY" in policy["not_listed"]
    # the reader follows PETsys normalisation; unparsable lines and duplicates are reported
    odd = tmp_path / "odd.tsv"
    odd.write_bytes(b"# header\r\n  0 0 0 5   12.5  # trailing\r\n\r\n0\t0\t0\tx\t1\n0\t0\t0\t5\t-3\n1\t2\t3\n")
    offsets, skipped, duplicates = read_native_table(odd)
    assert offsets == {get_absolute_id(0, 0, 0, 5): -3.0} and skipped == [4, 6] and duplicates == [5]


def test_activated_ini(tmp_path):
    folder = tmp_path / "conf"
    folder.mkdir()
    source = folder / "config.ini"
    original = ("[main]\ntdc_calibration_table = %CDIR%/tdc_calibration.tsv\n"
                "#time_offset_calibration_table = %CDIR%/time_offset_calibration.tsv\n"
                "channel_map = %cdir%/map_channel.tsv\n\n[sw_trigger]\ncoincidence_time_window = 1\n")
    source.write_text(original)
    table = tmp_path / "native.tsv"
    table.write_text("0\t0\t0\t0\t1.0\n")
    out = tmp_path / "config_v3.ini"
    info = activated_ini(source, table, out)
    lines = out.read_text().splitlines()
    cdir = folder.resolve().as_posix()
    assert lines[1] == f"tdc_calibration_table = {cdir}/tdc_calibration.tsv"
    assert lines[3] == f"time_offset_calibration_table = {table.resolve().as_posix()}" and info["line"] == 4
    assert lines[4] == f"channel_map = {cdir}/map_channel.tsv" and lines[-1] == "coincidence_time_window = 1"
    assert sum(line.startswith("time_offset_calibration_table") for line in lines) == 1
    assert source.read_text() == original  # the owner's INI is never edited
    with pytest.raises(LutError, match="exists"):
        activated_ini(source, table, out)
    with pytest.raises(LutError, match="already activates"):
        activated_ini(out, table, tmp_path / "twice.ini")
    # CLI: native table + activated INI in one call
    setup = make_setup(tmp_path)
    lut_path, _ = small_lut(tmp_path, setup)
    assert native_main([str(lut_path), str(tmp_path / "cli.tsv"), "--ini", str(source),
                        "--out-ini", str(tmp_path / "cli.ini")]) == 0
    assert (tmp_path / "cli.json").exists() and "time_offset_calibration_table" in (tmp_path / "cli.ini").read_text()
