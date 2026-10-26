"""IMAS LUT outputs through their consumers: native table, legacy skew file, timestamp correction (spec 011 T12)."""
from dataclasses import replace
import json

import numpy as np
import pytest

from imas_skew_helpers import CONFIG
from src.imas.skew import outputs
from src.imas.skew.setup import ImasSetup
from src.ldat_stream import LdatBatch
from src.timing import lut, solver as ts
from src.utils import read_skew

pytestmark = pytest.mark.fr("011-FR-9")


@pytest.fixture(scope="module")
def setup():
    return ImasSetup.from_files(CONFIG)


def solution(setup, delays, unobserved=()):
    channels = sorted(delays) + sorted(unobserved)
    rows = np.zeros(len(channels), ts.CHANNEL_DTYPE)
    rows["time_channel"] = channels
    for name in ("delay_ps", "sigma_ps", "raw_sigma_ps", "residual_rms"):
        rows[name] = np.nan
    applied = np.isin(rows["time_channel"], list(delays))
    rows["delay_ps"][applied] = [delays[c] for c in rows["time_channel"][applied].tolist()]
    rows["sigma_ps"][applied] = rows["raw_sigma_ps"][applied] = 12.5
    rows["status"] = np.where(applied, ts.LOW_STATISTICS, ts.UNOBSERVED)
    rows["apply"], rows["component"] = applied, np.where(applied, 0, -1)
    return ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {"estimator": "test"})


@pytest.fixture
def written(setup, tmp_path):
    times = setup.time_channels
    # both DAQ boards: low and high global IDs
    delays = {int(times[0]): 1234.5, int(times[1]): 0.25, int(times[7000]): 7000.125, int(times[-1]): 2.0}
    path = tmp_path / "lut.csv"
    outputs.write_imas_lut(path, solution(setup, delays, unobserved=[int(times[3])]), {"test": True}, setup)
    return path, delays, int(times[3])


def test_positive_delays_survive_each_consumer(setup, written, tmp_path):
    path, delays, unobserved = written
    # native table: PETsys parser gives the same global IDs and offsets
    manifest = outputs.write_native(path, tmp_path / "offsets.tsv", setup)
    offsets, skipped, duplicates = lut.read_native_table(tmp_path / "offsets.tsv")
    assert not skipped and not duplicates and set(offsets) == set(delays)
    assert all(abs(offsets[c] - b) < 1e-3 and offsets[c] > 0 for c, b in delays.items())
    assert manifest["scanner"] == "imas" and manifest["table"]["rows"] == len(delays)
    # legacy file: read_skew and t + skew give t - b
    outputs.write_legacy(path, tmp_path / "skew.txt", setup)
    skew = read_skew(str(tmp_path / "skew.txt"))
    assert set(skew) == set(delays) and unobserved not in skew
    t = 10 ** 12
    assert all(t + skew[c] == t - b for c, b in delays.items())
    legacy_manifest = json.loads((tmp_path / "skew.json").read_text(encoding="utf-8"))
    assert legacy_manifest["omitted_time_channels"] == len(setup.time_channels) - len(delays)
    # correct_timestamps from the canonical LUT and from the legacy round trip agree
    for table in (outputs.delay_table(path, setup), outputs.read_legacy(tmp_path / "skew.txt", setup)):
        channels = np.array(list(delays) + [unobserved])
        stamps = np.arange(len(channels), dtype=np.int64) * 1000 + t
        corrected, found = lut.correct_timestamps(stamps, channels, table)
        assert found.tolist() == [True] * len(delays) + [False]
        assert corrected[:-1].tolist() == [s - delays[c] for s, c in zip(stamps[:-1].tolist(), channels[:-1].tolist())]
        assert corrected[-1] == stamps[-1]            # no delay, not shifted
        # endpoint swap negates dt
        i, j = channels[0], channels[2]
        ci, _ = lut.correct_timestamps(np.array([t + 500]), np.array([i]), table)
        cj, _ = lut.correct_timestamps(np.array([t]), np.array([j]), table)
        assert ci[0] - cj[0] == -(cj[0] - ci[0]) == 500 - delays[int(i)] + delays[int(j)]


def test_correct_batch_counts_uncorrected_time_hits(setup, written):
    path, delays, unobserved = written
    table = outputs.delay_table(path, setup)
    time = list(delays)[:2] + [unobserved]
    energy = int(np.setdiff1d(np.arange(1000), setup.time_channels)[0])
    channels = np.array(time + [energy], np.int32)
    batch = LdatBatch(0, 0, np.array([2]), np.array([2]), np.array([0]), np.full(4, 10 ** 9, np.int64),
                      np.ones(4, np.float32), channels)
    corrected, counts = lut.correct_batch(batch, table, setup.time_channels)
    assert counts == {"hits": 4, "time_hits": 3, "corrected": 2, "uncorrected_time_hits": 1}
    assert corrected.tolist() == [10 ** 9 - delays[time[0]], 10 ** 9 - delays[time[1]], 10 ** 9, 10 ** 9]
    with pytest.raises(lut.LutError, match="integers"):
        lut.correct_timestamps(np.array([1.5]), np.array([time[0]]), table)


def test_another_map_or_scanner_is_refused(setup, written, tmp_path):
    path, delays, _ = written
    other = replace(setup, provenance={**setup.provenance, "map_sha256": "0" * 64})
    readers = (lambda s: outputs.read_imas_lut(path, s), lambda s: outputs.delay_table(path, s),
               lambda s: outputs.write_native(path, tmp_path / "n.tsv", s),
               lambda s: outputs.write_legacy(path, tmp_path / "l.txt", s))
    for reader in readers:
        with pytest.raises(lut.LutError, match="another map"):
            reader(other)
    assert not list(tmp_path.glob("n.*")) and not list(tmp_path.glob("l.*"))
    # a canonical LUT without the IMAS scanner tag (e.g. Cornell's) is refused
    plain = tmp_path / "plain.csv"
    lut.write_lut(plain, solution(setup, delays), {"scanner": "cornell"})
    with pytest.raises(lut.LutError, match="not an IMAS LUT"):
        outputs.read_imas_lut(plain, setup)
    # rows or legacy lines that are not TIME channels of this map
    energy = int(np.setdiff1d(np.arange(1000), setup.time_channels)[0])
    with pytest.raises(lut.LutError, match="not TIME channels"):
        outputs.write_imas_lut(tmp_path / "bad.csv", solution(setup, {energy: 5.0}), {}, setup)
    (tmp_path / "bad_skew.txt").write_text(f"{energy} -5.0\n", encoding="utf-8")
    with pytest.raises(lut.LutError, match="outside the map"):
        outputs.read_legacy(tmp_path / "bad_skew.txt", setup)


def test_legacy_9999_is_no_delay(setup, tmp_path):
    times = setup.time_channels
    (tmp_path / "s.txt").write_text(f"{times[0]} -100.5\n{times[1]} 9999\n", encoding="utf-8")
    table = outputs.read_legacy(tmp_path / "s.txt", setup)
    assert table.delays == {int(times[0]): 100.5}
    assert int(times[1]) in table.missing and len(table.missing) == len(times) - 1
    corrected, found = lut.correct_timestamps(np.array([0, 0]), times[:2], table)
    assert corrected.tolist() == [-100.5, 0.0] and found.tolist() == [True, False]


@pytest.mark.real_data
def test_import_production_legacy_lut(setup, real_data_file):
    table = outputs.read_legacy(real_data_file("IMAS/skew_IMAS.txt"), setup)
    assert len(table.delays) == 14_564 and table.provenance["listed_9999"] == 796
    assert len(table.missing) == len(setup.time_channels) - 14_564 == 796
