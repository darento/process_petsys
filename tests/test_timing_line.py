"""Spec 013 T2: line source from LOR endpoints (``src.timing.line``): recovery, steps, axis-parallel line, errors."""
import numpy as np
import pytest

from src.timing import line
from src.timing.ctr import CtrError
import line_helpers as h

HALF_MM = 350.0
GRID_Z = np.linspace(-350, 350, 15)


@pytest.fixture(scope="module")
def tilted():
    p, q, _ = h.line_lors(80_000)
    return h.true_line(), p, q, line.locate_line(p, q, radius_mm=h.RADIUS_MM, half_length_mm=HALF_MM)


def _at(origin, direction, z):
    return origin + (z / direction[2])[:, None] * direction


def _max_distance_mm(result, truth):
    fitted = _at(np.array(result["point_mm"]), np.array(result["direction"]), GRID_Z)
    return float(h.distance_to_line(fitted, *truth).max())


@pytest.mark.fr("013-FR-3")
def test_tilted_line_recovered_within_half_mm(tilted):
    truth, _, _, result = tilted
    assert _max_distance_mm(result, truth) < 0.5
    assert result["median_miss_mm"] < 2.5 and result["lors_used"] > 10_000


@pytest.mark.fr("013-FR-3")
def test_seed_and_slab_points_close_to_the_line(tilted):
    truth, p, q, result = tilted
    assert np.linalg.norm(np.array(result["seed_xy_mm"]) - truth[0][:2]) < 3.0
    points = np.array(result["slab_points"])
    xyz = np.column_stack((points[:, 1], points[:, 2], points[:, 0]))
    assert len(points) == 35 and result["empty_slabs"] == 0
    assert np.median(h.distance_to_line(xyz, *truth)) < 1.0
    assert result["spread_mm"]["median"] < 1.0
    steps = result["step_distance_mm"]  # the z-parallel seed is off a 0.011 rad tilt by a few mm at |z| = 350
    assert max(steps["principal"]) < 0.5 and max(steps["seed"]) < 6.0


@pytest.mark.fr("013-FR-3")
def test_axis_parallel_line_has_no_slope():
    truth = h.true_line(x0=25.0, y0=-60.0, ax=0.0, ay=0.0)
    p, q, _ = h.line_lors(60_000, line=truth, seed=3)
    result = line.locate_line(p, q, radius_mm=h.RADIUS_MM, half_length_mm=HALF_MM)
    assert np.abs(result["slopes"]).max() < 1e-3
    assert _max_distance_mm(result, truth) < 0.5


def _through_axis(z, count, rng, x=3.0, y=-4.0):
    """``count`` LORs through the z-parallel line (x, y) at height ``z``, in random directions."""
    v = rng.normal(size=(count, 3))
    v[:, :2] += 2.0 * np.sign(v[:, :2])
    s = np.array([x, y, z])
    return s - 300 * v, s + 300 * v


@pytest.mark.fr("013-FR-3")
def test_slab_with_fewer_than_three_lors_is_empty_and_counted():
    rng = np.random.default_rng(5)
    (p1, q1), (p2, q2) = _through_axis(0.0, 40, rng), _through_axis(100.0, 2, rng)
    points, empty = line.slab_points(np.vstack((p1, p2)), np.vstack((q1, q2)), (3.0, -4.0), half_length_mm=110.0)
    assert len(points) == 1 and empty == 10
    assert abs(points[0, 0]) < 1.0 and np.allclose(points[0, 1:3], (3.0, -4.0), atol=1e-6) and points[0, 3] == 40


@pytest.mark.fr("013-FR-3")
def test_gap_in_the_line_leaves_counted_empty_slabs():
    p, q, _ = h.line_lors(60_000, background=0.0, z_gaps=((80.0, 140.0),), seed=7)
    points, empty = line.slab_points(p, q, h.true_line()[0][:2], half_length_mm=HALF_MM)
    assert empty >= 2 and len(points) + empty == 35
    assert not np.any((points[:, 0] > 95) & (points[:, 0] < 125))


@pytest.mark.fr("013-FR-3")
def test_principal_line_and_distance_helper():
    z = np.linspace(-300, 300, 31)
    origin, direction = np.array([1.0, -2.0, 0.0]), np.array([0.02, 0.01, 1.0])
    direction /= np.linalg.norm(direction)
    xyz = _at(origin, direction, z)
    point, axis, spread = line.principal_line(np.column_stack((xyz[:, 2], xyz[:, 0], xyz[:, 1])), np.ones(31))
    assert np.allclose(point, origin, atol=1e-9) and np.allclose(axis, direction, atol=1e-9) and spread.max() < 1e-9
    p, q = np.array([[10.0, 0, -50], [0, 0, 100]]), np.array([[10.0, 0, 50], [0, 6, 100]])
    got = line.line_distance_mm(np.zeros(3), (0, 0, 1), p, q)  # parallel LOR 10 mm away; crossing LOR 0 mm
    assert np.allclose(got, (10.0, 0.0))


@pytest.mark.fr("013-FR-3")
def test_all_slabs_empty_raises():
    rng = np.random.default_rng(9)
    ends = [_through_axis(z, 1, rng) for z in (-250.0, -150.0, -50.0, 50.0, 150.0, 250.0)]
    p, q = np.vstack([e[0] for e in ends]), np.vstack([e[1] for e in ends])
    points, empty = line.slab_points(p, q, (3.0, -4.0), half_length_mm=HALF_MM)
    assert len(points) == 0 and empty == 35
    with pytest.raises(CtrError, match="slabs are empty"):
        line.locate_line(p, q, radius_mm=h.RADIUS_MM, half_length_mm=HALF_MM)


@pytest.mark.fr("013-FR-3")
def test_fewer_than_three_lors_raises():
    p, q = np.array([[400.0, 0, 0], [0, 400, 10]]), np.array([[-400.0, 0, 0], [0, -400, 10]])
    with pytest.raises(CtrError, match="fewer than three"):
        line.locate_line(p, q, radius_mm=h.RADIUS_MM, half_length_mm=HALF_MM)


@pytest.mark.fr("013-FR-3")
@pytest.mark.parametrize("kind", ["parallel to x", "parallel to z"])
def test_parallel_lors_raise(kind):
    rng = np.random.default_rng(11)
    a, b, z = rng.uniform(-8, 8, (3, 400))
    z = z * 40
    if kind == "parallel to x":
        p, q = np.column_stack((np.full(400, -400.0), a, z)), np.column_stack((np.full(400, 400.0), a, z))
    else:
        p, q = np.column_stack((a, b, np.full(400, -380.0))), np.column_stack((a, b, np.full(400, 380.0)))
    with pytest.raises(CtrError):
        line.locate_line(p, q, radius_mm=h.RADIUS_MM, half_length_mm=HALF_MM)
