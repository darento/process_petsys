"""IMAS config/map lookup and ring-major SM numbering (spec 011 T3)."""
import math

import numpy as np
import pytest
import yaml

from imas_skew_helpers import CONFIG, config_copy, mu_file
from src.imas.skew.setup import MU_INVALID, MU_MISSING, MU_VALID, ImasSetup
from src.timing.shell import GeometryError

pytestmark = pytest.mark.fr("011-FR-2", "011-FR-6")


@pytest.fixture(scope="module")
def setup():
    return ImasSetup.from_files(CONFIG)


def test_tracked_config_and_map_pass(setup):
    assert setup.n_azimuth == 24 and len(setup.ring_z) == 5
    assert {sm for sm, _ in setup.sm_mM_map.values()} == set(range(120))
    assert len(setup.time_channels) == 15_360
    # ring_yx[0] = [Y, X] = [-53.5157, 406.4924]: 7.5 degrees below +X, 15 degrees per index
    assert math.degrees(setup.azimuth[0]) == pytest.approx(-7.5, abs=1e-4)
    assert np.allclose(np.diff(np.degrees(setup.azimuth))[:10], -15.0, atol=1e-3)
    assert setup.local_origin == (51.61, 51.61)
    assert setup.ring(0) == 0 and setup.ring(23) == 0 and setup.ring(24) == 1 and setup.ring(119) == 4


def test_slab_index_per_time_channel(setup):
    lookup = setup.lookup()
    time = lookup.kind == 1
    assert time.sum() == 15_360 and (lookup.kind == 2).sum() == 15_360
    assert set(np.unique(lookup.slab[time]).tolist()) == set(range(8))
    # eight distinct slabs per minimodule, ordered with the transaxial position
    keys = lookup.sm[time].astype(np.int64) * 16 + lookup.mm[time]
    for key in np.unique(keys)[:40]:
        chosen = np.flatnonzero(time)[keys == key]
        order = np.argsort(lookup.slab[chosen])
        assert lookup.slab[chosen][order].tolist() == list(range(8))
        assert (np.diff(lookup.coord_x[chosen][order]) > 0).all()


def test_swapped_ring_yx_entry_is_refused(tmp_path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ring_yx = dict(config["ring_yx"])
    ring_yx[3], ring_yx[4] = ring_yx[4], ring_yx[3]
    with pytest.raises(GeometryError, match="evenly spaced"):
        ImasSetup.from_files(config_copy(tmp_path / "c.yaml", ring_yx=ring_yx))


@pytest.mark.parametrize("changes, message", [
    ({"ring_z": [-152.7656, 0.0, 152.7656, 305.5312]}, "ring-major"),
    ({"ring_z": [0.0, -152.7656, 152.7656, 305.5312, 400.0]}, "increase"),
    ({"ring_r": 400.0}, "ring_r"),
])
def test_ring_table_mismatch_is_refused(tmp_path, changes, message):
    with pytest.raises(GeometryError, match=message):
        ImasSetup.from_files(config_copy(tmp_path / "c.yaml", **changes))


def test_ring_yx_with_missing_azimuth_is_refused(tmp_path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ring_yx = {k: v for k, v in config["ring_yx"].items() if k != 23}
    with pytest.raises(GeometryError, match="ring-major"):
        ImasSetup.from_files(config_copy(tmp_path / "c.yaml", ring_yx=ring_yx))


def test_mu_states(tmp_path, setup):
    a, b, c = setup.time_channels[:3].tolist()
    path = mu_file(tmp_path / "mu.tsv", {a: 120.5, b: float("nan"), c: -1.0})
    lookup = ImasSetup.from_files(CONFIG, path).lookup()
    assert lookup.mu_state[[a, b, c]].tolist() == [MU_VALID, MU_INVALID, MU_INVALID]
    assert lookup.mu[a] == 120.5 and np.isnan(lookup.mu[[b, c]]).all()
    assert lookup.mu_state[setup.time_channels[3]] == MU_MISSING
