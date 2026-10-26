"""IMAS nominal endpoints, frame signs and the bar shell (spec 011 T6)."""
import math

import numpy as np
import pytest
import yaml

from imas_skew_helpers import CONFIG
from src.imas.skew.geometry import IMAS_SKEW_BAR, ImasGeometry
from src.imas.skew.setup import ImasSetup
from src.timing.shell import C_MM_PER_PS, GEOMETRY_REJECTIONS, GeometryError, shell_intervals

pytestmark = pytest.mark.fr("011-FR-2", "011-FR-5")
R = 417.0                 # ring_r 410 + 7 mm depth


@pytest.fixture(scope="module")
def setup():
    return ImasSetup.from_files(CONFIG)


class Rows:
    def __init__(self, **values):
        self.values = {k: np.asarray(v) for k, v in values.items()}

    def column(self, name):
        return self.values[name]


def rows(pairs):
    names = ("sm_1", "x_1", "y_1", "sm_2", "x_2", "y_2")
    return Rows(**{name: [p[i] for p in pairs] for i, name in enumerate(names)})


@pytest.mark.parametrize("sm, x, y", [(0, 61.61, 40.0), (5, 51.61, 51.61), (23, 41.61, 70.0),
                                      (96, 61.61, 40.0), (101, 51.61, 51.61), (119, 41.61, 70.0)])
def test_known_endpoints(setup, sm, x, y):
    """Rings 0 and 4, azimuths 0, 5 and 23 against the formula written out by hand."""
    ring, k = divmod(sm, 24)
    y_k, x_k = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["ring_yx"][k]   # [Y, X]
    phi = math.atan2(y_k, x_k)
    z = (-305.5312, 305.5312)[ring // 4]
    u, v = x - 51.61, y - 51.61
    expected = [R * math.cos(phi) - u * math.sin(phi), R * math.sin(phi) + u * math.cos(phi), z - v]
    assert ImasGeometry.nominal(setup).endpoint(sm, x, y) == pytest.approx(expected, abs=1e-9)


def test_each_sign_flips_its_coordinate(setup):
    nominal = ImasGeometry.nominal(setup).endpoint(5, 61.61, 40.0)
    transverse = ImasGeometry.from_setup(setup, transverse_sign=-1).endpoint(5, 61.61, 40.0)
    axial = ImasGeometry.from_setup(setup, axial_sign=1).endpoint(5, 61.61, 40.0)
    centre = ImasGeometry.nominal(setup).endpoint(5, 51.61, 51.61)
    assert transverse[:2] == pytest.approx(2 * centre[:2] - nominal[:2]) and transverse[2] == nominal[2]
    assert axial[:2] == pytest.approx(nominal[:2]) and axial[2] - centre[2] == pytest.approx(-(nominal[2] - centre[2]))
    assert np.hypot(*(nominal[:2] - centre[:2])) == pytest.approx(10.0)
    with pytest.raises(GeometryError):
        ImasGeometry.from_setup(setup, axial_sign=0)


def test_bar_lor_crossings_and_rejections(setup):
    geometry = ImasGeometry.nominal(setup)
    through = (50, 51.61, 51.61, 62, 51.61, 51.61)           # ring 2, azimuth 2 against azimuth 14: diametric
    oblique = (50, 51.61, 51.61, 60, 51.61, 51.61)           # 150 degrees apart: crosses the bar, angle < 165
    beyond = (96, 51.61, 1.7, 108, 51.61, 1.7)               # ring 4 at z = +355.4 mm: past the bar's +350 mm
    unknown = (120, 51.61, 51.61, 62, 51.61, 51.61)
    result = geometry.apply(rows([through, oblique, beyond, unknown]))
    labels = ["accepted"] + list(GEOMETRY_REJECTIONS)
    assert [labels[c] for c in result.codes] == ["accepted", "opposition angle", "no shell crossing",
                                                 "invalid endpoint"]
    assert result.shell_segments[0] == 2
    tof = result.shell_tof_ps[0]
    # emission at radius r on the detector 1 side: t_1 - t_2 = -2 r / c (sign per the core convention)
    expected = sorted(abs(2 * r / C_MM_PER_PS) for r in (155.0, 165.0))
    assert sorted(np.abs(tof[:2])) == pytest.approx(expected, abs=1e-6)
    assert sorted(np.abs(tof[2:])) == pytest.approx(expected, abs=1e-6)
    p, q = result.endpoints_mm[0, :3], result.endpoints_mm[0, 3:]
    reference = shell_intervals(p, q, IMAS_SKEW_BAR)
    assert np.allclose(sorted(tof), sorted(v for s in reference for v in (s.tof_start_ps, s.tof_end_ps)))
