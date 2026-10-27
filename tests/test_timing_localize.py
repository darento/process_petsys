"""Timing-blind source localization among scanner-wide background (bug-imas-ctr-localization).

The IMAS 22Na sources at z ~ 267 mm sit among coincidences spread over the whole scanner (natural LYSO
radioactivity, a weak source): only ~25 % of LORs pass the source, and the all-LOR least-squares start of
``converge_point`` landed 40-65 mm off. ``seeded_converge_point`` starts from a grid search instead.
"""
import numpy as np
import pytest

from src.timing import ctr

pytestmark = pytest.mark.fr("bug-imas-ctr-localization")

RING_R, HALF_LENGTH = 410.0, 356.0


def on_cylinder(rng, n):
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack((RING_R * np.cos(phi), RING_R * np.sin(phi), rng.uniform(-HALF_LENGTH, HALF_LENGTH, n)))


def lors_through(rng, source, n):
    """Endpoints of ``n`` LORs through ``source`` (within ~2 mm) that hit the scanner cylinder at both ends."""
    p, q = [], []
    while sum(len(x) for x in p) < n:
        u = rng.normal(size=(4 * n, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        b = source[0] * u[:, 0] + source[1] * u[:, 1]
        a = u[:, 0] ** 2 + u[:, 1] ** 2
        c = source[0] ** 2 + source[1] ** 2 - RING_R ** 2
        root = np.sqrt(b * b - a * c)
        t1, t2 = (-b - root) / a, (-b + root) / a
        e1, e2 = source + t1[:, None] * u, source + t2[:, None] * u
        inside = (np.abs(e1[:, 2]) < HALF_LENGTH) & (np.abs(e2[:, 2]) < HALF_LENGTH)
        p.append(e1[inside])
        q.append(e2[inside])
    p, q = np.concatenate(p)[:n], np.concatenate(q)[:n]
    return p + rng.normal(0, 1.5, p.shape), q + rng.normal(0, 1.5, q.shape)


@pytest.mark.parametrize("source", [(-1.4, 2.6, 267.3), (197.0, 3.3, 267.3), (198.5, 2.8, -0.4)])
def test_seeded_localization_among_background(source):
    rng = np.random.default_rng(7)
    source = np.array(source)
    p, q = lors_through(rng, source, 6_000)
    p = np.concatenate((p, on_cylinder(rng, 18_000)))       # 75 % background between random detector points
    q = np.concatenate((q, on_cylinder(rng, 18_000)))
    plain, _, _ = ctr.converge_point(p, q)
    assert np.linalg.norm(plain - source) > 10               # the defect: the plain start is pulled off
    point, miss, used, seed = ctr.seeded_converge_point(p, q, radius_mm=RING_R - 10, half_length_mm=HALF_LENGTH)
    assert np.linalg.norm(point - source) < 1.0 and miss < 2.5
    assert np.linalg.norm(seed - source) < 15 and used > 3_000


def test_seeded_matches_plain_without_background():
    rng = np.random.default_rng(8)
    source = np.array([0.0, 0.6, -0.2])
    p, q = lors_through(rng, source, 5_000)
    plain, _, _ = ctr.converge_point(p, q)
    point, _, _, _ = ctr.seeded_converge_point(p, q, radius_mm=RING_R - 10, half_length_mm=HALF_LENGTH)
    assert np.linalg.norm(point - plain) < 0.3
    with pytest.raises(ctr.CtrError, match="fewer than three"):
        ctr.seeded_converge_point(p[:2], q[:2], radius_mm=400.0, half_length_mm=300.0)
