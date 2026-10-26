"""Per-TIME-channel photopeak calibration from IMAS skew splits (spec 011 T5)."""
import json
import math

import numpy as np
import pytest

from imas_skew_helpers import CONFIG, T0, channels, side, write
from src.imas.skew import encal
from src.imas.skew.setup import ImasSetup
from src.utils import KevConverter

pytestmark = pytest.mark.fr("011-FR-4")


@pytest.fixture(scope="module")
def setup():
    return ImasSetup.from_files(CONFIG)


def fine(values):
    index = np.floor(np.asarray(values) / encal.FINE_BIN_AU).astype(int)
    return np.bincount(index[(index >= 0) & (index < encal.FINE_BINS)], minlength=encal.FINE_BINS)


def peak(rng, mu, n=4000, sigma_fraction=0.08, background=1500):
    # background from 2 a.u. (every ENERGY hit above en_min_ch) to the end of the 0-220 a.u. range
    return np.concatenate([rng.normal(mu, sigma_fraction * mu, n), rng.uniform(2.0, min(2.2 * mu, 219.0),
                                                                                  background)])


@pytest.mark.parametrize("mu", [35.0, 57.0, 110.0])
def test_peak_with_background_recovers_mu(mu):
    fitted, sigma, status, details = encal.fit_channel(fine(peak(np.random.default_rng(int(mu)), mu)))
    assert status == encal.FIT and abs(fitted / mu - 1) < 0.01
    assert sigma == pytest.approx(0.08 * mu, rel=0.15) and details["entries"] == 5500


def test_few_entries_and_no_peak():
    rng = np.random.default_rng(3)
    assert encal.fit_channel(fine(rng.normal(60, 4, 199)))[2] == encal.FEW
    assert encal.fit_channel(np.zeros(encal.FINE_BINS, np.uint64))[2] == encal.FEW
    mu, _, status, details = encal.fit_channel(fine(rng.uniform(0, 200, 5000)))
    assert status == encal.NO_PEAK and math.isnan(mu) and details["fit_status"] != "FIT"


def test_split_to_kev_converter(tmp_path, setup):
    """End to end: sides histogrammed by top TIME channel, fitted, written and read back."""
    rng = np.random.default_rng(7)
    a, b, c = channels(setup, 0, 5)[0][3], channels(setup, 60, 6)[0][1], channels(setup, 33, 2)[0][5]

    def sided(sm, mm, slab, total, stamp):
        return side(setup, sm, mm, time={slab: 9.0}, energy=[total / 5] * 5, stamp=stamp)
    records = [(sided(0, 5, 3, e1, T0), sided(60, 6, 1, e2, T0 - 200))
               for e1, e2 in zip(peak(rng, 60.0, 2400, background=600), peak(rng, 90.0, 2400, background=600))]
    records += [(sided(33, 2, 5, 70.0, T0), sided(60, 6, 1, 90.0, T0))] * 150
    paths = [write(tmp_path / "s0.ldat", records[:1500]), write(tmp_path / "s1.ldat", records[1500:])]
    path, manifest = encal.run(paths, tmp_path / "out", config_path=CONFIG, workers=1)
    factors = KevConverter(str(path), file_type="mu")
    assert factors.convert(a, 60.0) == pytest.approx(511.0, rel=0.01)
    assert factors.convert(b, 90.0) == pytest.approx(511.0, rel=0.01)
    # few entries (150) and never-seen channels are written as nan: NaN keV, never a number
    assert math.isnan(factors.convert(c, 70.0))
    assert math.isnan(factors.convert(channels(setup, 1, 1)[0][0], 70.0))
    assert len(factors.kev_factors) == 15_360
    assert manifest["status_counts"] == {encal.FIT: 2, encal.FEW: 15_358, encal.NO_PEAK: 0}
    assert manifest["record_counts"] == {"passed pre-geometry": 3150}
    assert [item["name"] for item in manifest["inputs"]] == ["s0.ldat", "s1.ldat"]
    stored = json.loads((tmp_path / "out" / "encal.json").read_text(encoding="utf-8"))
    assert stored["encal_sha256"] and stored["provenance"]["map_sha256"]
    rows = (tmp_path / "out" / "encal_status.csv").read_text(encoding="utf-8").splitlines()
    assert any(row.startswith(f"{c},33,2,5,few entries,150,") for row in rows)
    with pytest.raises(FileExistsError):
        encal.run(paths, tmp_path / "out", config_path=CONFIG)
