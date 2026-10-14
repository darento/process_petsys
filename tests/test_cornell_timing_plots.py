"""Per-ASIC diagnostics of a canonical Cornell delay LUT (spec 008 T7b, FR-19)."""
import json

import numpy as np
import pytest

from src.cornell.skew import plots as tpl
from src.cornell.skew import solver as ts
from src.cornell.skew.lut import write_lut
from src.utils import get_absolute_id

pytestmark = pytest.mark.fr("008-FR-19")
# two ASIC families reading different channel subsets, as on the Cornell boards
FAMILY = (np.r_[0:7, 8, 10:34, 48, 52, 54:60], np.r_[0:4, 5, 10, 11, 13, 16:34, 37:45])


def synthetic(rng, *, asics=40, noise=30.0):
    """Shared within-ASIC profile (step at channel 34, spike at 52) + ASIC offsets + noise."""
    profile = np.where(np.arange(64) < 34, -3.0 * np.arange(64), 500.0 + 8.0 * (np.arange(64) - 34))
    profile[52] += 400.0
    channels, delays, offsets = [], [], {}
    for a in range(asics):
        port, slave, chip = (0, 1, 4)[a % 3], a // 12, a % 12
        offsets[(port, slave, chip)] = rng.uniform(-2500, 2500)
        for k in FAMILY[a % 2]:
            channels.append(get_absolute_id(port, slave, chip, int(k)))
            delays.append(offsets[(port, slave, chip)] + profile[k] + rng.normal(0, noise))
    return np.array(channels), np.array(delays), profile, offsets


def test_asic_pattern_and_outliers():
    rng = np.random.default_rng(19)
    channels, delays, profile, offsets = synthetic(rng)
    injected = {}
    for position, jump in ((11, 3000.0), (95, -4000.0), (200, -1500.0)):
        delays[position] += jump
        injected[int(channels[position])] = jump
    delays[300] += 600.0  # below the threshold: not listed
    order = rng.permutation(len(channels))
    channels_in, delays_in = np.r_[channels[order], 999_999], np.r_[delays[order], np.nan]  # non-applied row ignored
    pattern = tpl.asic_pattern(channels_in, delays_in)
    assert len(pattern["asics"]) == 40 and len(pattern["delay_ps"]) == len(channels)
    # decoding matches PETsys port/slave/chip/channel
    k = int(np.flatnonzero(pattern["time_channel"] == channels[5])[0])
    assert pattern["channel"][k] == FAMILY[0][5] and pattern["port"][k] == 0 and pattern["chip"][k] == 0
    # profile and offsets recovered up to one shared constant
    shared = np.isfinite(pattern["profile_ps"])
    expected = profile - np.median(profile[shared])
    assert np.max(np.abs(pattern["profile_ps"][shared] - expected[shared])) < 40
    got = {tuple(map(int, (pattern["port"][pattern["asic_index"] == a][0], pattern["slave"][pattern["asic_index"] == a][0],
                            pattern["chip"][pattern["asic_index"] == a][0]))): pattern["offset_ps"][a]
           for a in range(40)}
    difference = np.array([got[key] - offsets[key] for key in offsets])
    assert np.ptp(difference) < 60
    residual = pattern["residual_ps"][np.isfinite(pattern["residual_ps"])]
    assert 20 < 1.4826 * np.median(np.abs(residual)) < 40
    outliers = tpl.pattern_outliers(pattern, 1000.0)
    assert {o["time_channel"] for o in outliers} == set(injected)
    for o in outliers:
        assert abs(o["residual_ps"] - injected[o["time_channel"]]) < 150
    assert len(tpl.pattern_outliers(pattern, 500.0)) == 4
    # an index seen on fewer than three ASICs has no profile and is never listed
    lonely = tpl.asic_pattern(np.r_[channels, get_absolute_id(0, 0, 0, 62)], np.r_[delays, 1e5])
    assert np.isnan(lonely["profile_ps"][62]) and get_absolute_id(0, 0, 0, 62) not in \
        {o["time_channel"] for o in tpl.pattern_outliers(lonely)}


def lut_file(path, channels, delays):
    rows = np.zeros(len(channels), ts.CHANNEL_DTYPE)
    rows["time_channel"], rows["delay_ps"], rows["sigma_ps"], rows["raw_sigma_ps"] = channels, delays, 5.0, 5.0
    rows["status"], rows["apply"] = ts.CALIBRATED, True
    rows["status"][-1], rows["apply"][-1] = ts.UNOBSERVED, False
    rows["delay_ps"][-1] = rows["sigma_ps"][-1] = rows["raw_sigma_ps"][-1] = np.nan
    write_lut(path, ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {"status_counts": {}}), {"test": True})
    return path


def test_plots_written_and_never_overwritten(tmp_path, capsys):
    rng = np.random.default_rng(23)
    channels, delays, _, _ = synthetic(rng, asics=12)
    delays[7] -= 3000.0
    (tmp_path / "out").mkdir()
    lut = lut_file(tmp_path / "out" / "v3.csv", channels, delays)
    before = {p.name: p.read_bytes() for p in lut.parent.iterdir()}
    assert tpl.main([str(lut)]) == 0
    printed = json.loads(capsys.readouterr().out)
    paths = tpl.plot_paths(lut)
    assert [p.name for p in paths] == ["v3_asic_within.png", "v3_asic_heatmap.png", "v3_asic_overview.png",
                                       "v3_asic_median_slope.png", "v3_asic_summary.csv"]
    assert printed["files"] == [str(p) for p in paths]
    for path in paths[:4]:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 10_000
    summary = paths[4].read_text().splitlines()
    assert summary[0].startswith("asic,port,slave,chip,channels,offset_ps") and len(summary) == 1 + 12
    assert sum(int(line.split(",")[-1]) for line in summary[1:]) == 1
    assert [o["time_channel"] for o in printed["outliers"]] == [int(channels[7])]
    assert printed["summary"]["channels"] == len(channels) - 1  # the unobserved row is not plotted
    # LUT and manifest untouched; nothing is overwritten on a second call
    assert all((lut.parent / name).read_bytes() == data for name, data in before.items())
    stamp = paths[0].stat().st_mtime_ns
    assert tpl.main([str(lut)]) == 2 and "exist" in capsys.readouterr().err
    assert paths[0].stat().st_mtime_ns == stamp
    with pytest.raises(tpl.PlotError, match="exist"):
        tpl.write_asic_plots(lut, channels, delays)
