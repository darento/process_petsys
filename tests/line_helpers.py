"""Synthetic LORs for a line source among scanner-wide background (spec 013 T2)."""
import numpy as np

RADIUS_MM = 410.0
ACCEPT_Z_MM = 380.0
EMIT_Z_MM = 350.0
BLUR_MM = 2.0


def true_line(x0=-37.0, y0=15.0, ax=0.01, ay=-0.005):
    direction = np.array([ax, ay, 1.0])
    return np.array([x0, y0, 0.0]), direction / np.linalg.norm(direction)


def _cylinder_hits(s, v):
    """Both intersections of the lines ``s + t v`` with the cylinder."""
    a = (v[:, :2] ** 2).sum(axis=1)
    b = (s[:, :2] * v[:, :2]).sum(axis=1)
    c = (s[:, :2] ** 2).sum(axis=1) - RADIUS_MM ** 2
    root = np.sqrt(b * b - a * c)
    return s + ((-b + root) / a)[:, None] * v, s + ((-b - root) / a)[:, None] * v


def line_lors(n_signal, *, line=None, background=0.3, z_gaps=(), seed=1):
    """LORs of a line source plus uniform background, as (p, q) endpoints in mm and the emission z (nan for background).

    Emission uniform along |z| <= 350 on the line (``z_gaps`` are excluded (lo, hi) ranges), isotropic
    back-to-back directions, endpoints on the cylinder within |z| <= 380, 2 mm blur. ``background`` is the
    fraction of all LORs that join two uniform random cylinder points.
    """
    rng = np.random.default_rng(seed)
    origin, direction = line if line is not None else true_line()
    ze = rng.uniform(-EMIT_Z_MM, EMIT_Z_MM, 3 * n_signal)
    for lo, hi in z_gaps:
        ze = ze[(ze < lo) | (ze > hi)]
    s = origin + (ze / direction[2])[:, None] * direction
    v = rng.normal(size=s.shape)
    p, q = _cylinder_hits(s, v / np.linalg.norm(v, axis=1)[:, None])
    ok = (np.abs(p[:, 2]) <= ACCEPT_Z_MM) & (np.abs(q[:, 2]) <= ACCEPT_Z_MM)
    p, q, ze = p[ok][:n_signal], q[ok][:n_signal], ze[ok][:n_signal]
    n_bg = round(len(p) * background / (1 - background))
    ends = [np.column_stack((RADIUS_MM * np.cos(phi), RADIUS_MM * np.sin(phi),
                             rng.uniform(-ACCEPT_Z_MM, ACCEPT_Z_MM, n_bg)))
            for phi in (rng.uniform(0, 2 * np.pi, n_bg), rng.uniform(0, 2 * np.pi, n_bg))]
    p, q = np.vstack((p, ends[0])), np.vstack((q, ends[1]))
    p, q = p + rng.normal(0, BLUR_MM, p.shape), q + rng.normal(0, BLUR_MM, q.shape)
    return p, q, np.concatenate((ze, np.full(n_bg, np.nan)))


def distance_to_line(points, origin, direction):
    """Distance (mm) of each point from the line."""
    w = points - origin
    return np.linalg.norm(w - np.outer(w @ direction, direction), axis=1)
