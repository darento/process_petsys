"""Scanner-neutral line-source location from LOR endpoints (spec 013 T2, FR-3).

Finds a straight line source (the NECR phantom's line, roughly parallel to the scanner z axis) without
timing: a seed over the bore, a robust convergence point per axial slab, a principal axis through the slab
points, and a robust refit of the four line parameters. Scanner front ends supply the endpoints.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from src.timing.ctr import CtrError

SEED_STEPS_MM = (10.0, 2.0)       # coarse grid over the bore, then a fine grid around the coarse best
SEED_NEAR_MM = 10.0               # LORs counted per grid point (half on the fine grid)
SEED_SAMPLE = 5_000
SLAB_MM = 20.0
SLAB_NEAR_MM = 15.0               # LORs of a slab: within this of the seed line
SLAB_FLOOR_MM = 3.0
SLAB_ITERATIONS = 6
MIN_SLAB_LORS = 3
MIN_XY_MM = 1.0                   # shorter xy projection: the LOR is parallel to z (a point in xy)
MIN_SIN = 0.1                     # LORs closer than this (sine) to the line direction are not refitted
WINDOWS_MM = (20.0, 10.0, 5.0)
F_SCALE_MM = 3.0
CHECK_Z_MM = (-350.0, 0.0, 350.0)


def _clean(p, q):
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    ok = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1) & (np.linalg.norm(q - p, axis=1) > 0)
    if np.count_nonzero(ok) < MIN_SLAB_LORS:
        raise CtrError("fewer than three LORs to locate the line")
    return p[ok], q[ok]


def line_distance_mm(origin, direction, p, q):
    """Unsigned distance (mm) of the line (``origin``, ``direction``) from each LOR (the infinite line through
    ``p`` and ``q``)."""
    a = np.asarray(direction, np.float64)
    a = a / np.linalg.norm(a)
    d = q - p
    d = d / np.linalg.norm(d, axis=1)[:, None]
    w = np.asarray(origin, np.float64) - p
    n = np.cross(a, d)
    s = np.linalg.norm(n, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = np.abs(np.einsum("ij,ij->i", w, n)) / s
    return np.where(s > 1e-9, skew, np.linalg.norm(w - np.outer(w @ a, a), axis=1))


def _xy(p, q):
    """xy projection of each LOR: point, direction, squared length and the mask of LORs with a direction."""
    dxy = (q - p)[:, :2]
    length2 = np.einsum("ij,ij->i", dxy, dxy)
    return p[:, :2], dxy, length2, length2 > MIN_XY_MM ** 2


def _xy_distance(c, p0, dxy, length2, ok):
    """Distance of the xy point ``c`` from each LOR's xy projection, or from its midpoint if it has no direction."""
    w = c - p0
    with np.errstate(invalid="ignore", divide="ignore"):
        line = np.abs(w[:, 0] * dxy[:, 1] - w[:, 1] * dxy[:, 0]) / np.sqrt(length2)
    return np.where(ok, line, np.linalg.norm(w - dxy / 2, axis=1))


def _near_counts_xy(grid, p0, dxy, length2, ok, radius_mm):
    """For each xy grid point, the number of LORs passing within ``radius_mm`` (3D distance of a z-parallel line)."""
    mid = p0 + dxy / 2
    u = dxy[ok] / np.sqrt(length2[ok])[:, None]
    c = p0[ok, 0] * u[:, 1] - p0[ok, 1] * u[:, 0]
    normal = np.column_stack((u[:, 1], -u[:, 0]))
    counts = np.empty(len(grid), np.int64)
    chunk = max(1, 4_000_000 // len(p0))
    for start in range(0, len(grid), chunk):
        s = grid[start:start + chunk]
        near = np.count_nonzero(np.abs(s @ normal.T - c) <= radius_mm, axis=1)
        gap = s[:, None, :] - mid[~ok][None]
        close = np.einsum("ijk,ijk->ij", gap, gap) <= radius_mm ** 2
        counts[start:start + chunk] = near + np.count_nonzero(close, axis=1)
    return counts


def seed_line(p, q, *, radius_mm, steps_mm=SEED_STEPS_MM, near_mm=SEED_NEAR_MM, sample=SEED_SAMPLE, seed=0):
    """The z-parallel line (x0, y0) that most LORs pass within ``near_mm`` of (bore disc ``radius_mm``).

    For a z-parallel line the 3D LOR distance is the 2D distance from (x0, y0) to the LOR's xy projection
    (to its midpoint for a LOR parallel to z). Coarse grid, then a fine grid around the best (half the
    radius), on up to ``sample`` LORs. Uses endpoints only. Returns the xy point (mm).
    """
    p, q = _clean(p, q)
    pick = np.random.default_rng(seed).permutation(len(p))[:sample]
    p0, dxy, length2, ok = _xy(p[pick], q[pick])
    coarse, fine = steps_mm
    axis = np.arange(-radius_mm, radius_mm + coarse / 2, coarse)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    grid = grid[np.einsum("ij,ij->i", grid, grid) <= radius_mm ** 2]
    best = grid[np.argmax(_near_counts_xy(grid, p0, dxy, length2, ok, near_mm))]
    offsets = np.arange(-coarse, coarse + fine / 2, fine)
    grid = best + np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)
    return grid[np.argmax(_near_counts_xy(grid, p0, dxy, length2, ok, near_mm / 2))]


def _slab_point(p0, dxy, length2, floor_mm, iterations=SLAB_ITERATIONS):
    """Point closest to the xy lines (least squares), refitted within max(median miss, floor); None if singular."""
    u = dxy / np.sqrt(length2)[:, None]
    used = np.ones(len(p0), bool)
    for _ in range(iterations):
        pu, uu = p0[used], u[used]
        a = len(pu) * np.eye(2) - uu.T @ uu
        if len(pu) < MIN_SLAB_LORS or abs(np.linalg.det(a)) < 1e-9 * len(pu) ** 2:
            return None
        point = np.linalg.solve(a, pu.sum(axis=0) - uu.T @ np.einsum("ij,ij->i", uu, pu))
        w = point - p0
        miss = np.abs(w[:, 0] * u[:, 1] - w[:, 1] * u[:, 0])
        used = miss <= max(float(np.median(miss)), floor_mm)
    return point, int(np.count_nonzero(used)), used


def slab_points(p, q, seed_xy, *, slab_mm=SLAB_MM, near_mm=SLAB_NEAR_MM, half_length_mm, floor_mm=SLAB_FLOOR_MM):
    """Per axial slab, the convergence point of the LORs near the seed line.

    LORs within ``near_mm`` of the z-parallel ``seed_xy`` line go to the slab of their z of closest approach
    to it (``|z| <= half_length_mm``). In a slab the least-squares point closest to the LORs' xy projections is
    refitted on the LORs within max(median miss, ``floor_mm``); its z is the median closest-approach z of the
    LORs used. A slab with fewer than three LORs (or parallel ones) is empty. Returns (rows of
    z, x, y, LORs used; empty slab count).
    """
    p, q = _clean(p, q)
    c = np.asarray(seed_xy, np.float64)
    p0, dxy, length2, ok = _xy(p, q)
    keep = ok & (_xy_distance(c, p0, dxy, length2, ok) <= near_mm)
    p0, dxy, length2 = p0[keep], dxy[keep], length2[keep]
    z = p[keep, 2] + np.einsum("ij,ij->i", c - p0, dxy) / length2 * (q - p)[keep, 2]
    count = int(np.ceil(2 * half_length_mm / slab_mm))
    slab = np.floor((z + half_length_mm) / slab_mm).astype(int)
    keep = (slab >= 0) & (slab < count)
    p0, dxy, length2, z, slab = p0[keep], dxy[keep], length2[keep], z[keep], slab[keep]
    order = np.argsort(slab, kind="stable")
    rows = []
    for index in np.split(order, np.searchsorted(slab[order], np.arange(1, count))):
        fit = _slab_point(p0[index], dxy[index], length2[index], floor_mm) if len(index) >= MIN_SLAB_LORS else None
        if fit is not None:
            point, used, mask = fit
            rows.append((float(np.median(z[index][mask])), point[0], point[1], used))
    return np.array(rows, np.float64).reshape(-1, 4), count - len(rows)


def principal_line(points, weights):
    """Weighted principal axis of the (z, x, y) ``points``: (point at z = 0, unit direction with dz > 0, spread).

    The spread is each point's distance (mm) from the line.
    """
    pts = np.column_stack((points[:, 1], points[:, 2], points[:, 0]))  # x, y, z
    w = np.asarray(weights, np.float64)
    if len(pts) < 2 or w.sum() <= 0:
        raise CtrError("fewer than two slab points to fit a line")
    centre = w @ pts / w.sum()
    cov = (pts - centre).T @ ((pts - centre) * w[:, None])
    direction = np.linalg.eigh(cov)[1][:, -1]
    if abs(direction[2]) < 0.1:
        raise CtrError("the slab points do not lie along z: degenerate line")
    direction = direction * np.sign(direction[2])
    origin = centre - centre[2] / direction[2] * direction
    along = (pts - centre) @ direction
    return origin, direction, np.linalg.norm(pts - centre - np.outer(along, direction), axis=1)


def _line(params):
    direction = np.array([params[2], params[3], 1.0])
    return np.array([params[0], params[1], 0.0]), direction / np.linalg.norm(direction)


def refine_line(p, q, line, *, windows_mm=WINDOWS_MM, f_scale_mm=F_SCALE_MM):
    """Robust refit of x = x0 + ax z, y = y0 + ay z from the LORs within each shrinking window of the line.

    ``line`` is (point, direction) with dz > 0. Minimises a ``soft_l1`` sum of LOR-to-line distances; LORs
    within ``MIN_SIN`` of the line direction are left out. Returns ((point at z = 0, unit direction), median
    miss (mm) and LORs used in the last window).
    """
    p, q = _clean(p, q)
    origin, direction = (np.asarray(v, np.float64) for v in line)
    if direction[2] <= 0:
        raise CtrError("line direction must have a positive z component")
    d = (q - p) / np.linalg.norm(q - p, axis=1)[:, None]
    params = np.array([*(origin - origin[2] / direction[2] * direction)[:2], *(direction[:2] / direction[2])])

    def residual(x, pw, dw):
        o, a = _line(x)
        n = np.cross(a, dw)
        return np.einsum("ij,ij->i", o - pw, n) / np.linalg.norm(n, axis=1)

    for window in windows_mm:
        o, a = _line(params)
        use = (line_distance_mm(o, a, p, q) <= window) & (np.linalg.norm(np.cross(a, d), axis=1) > MIN_SIN)
        if np.count_nonzero(use) < MIN_SLAB_LORS:
            raise CtrError(f"fewer than three LORs within {window:g} mm of the line")
        params = least_squares(residual, params, args=(p[use], d[use]), loss="soft_l1", f_scale=f_scale_mm,
                               x_scale="jac").x
    o, a = _line(params)
    miss = line_distance_mm(o, a, p, q)
    near = miss <= windows_mm[-1]
    return (o, a), float(np.median(miss[near])), int(np.count_nonzero(near))


def _at_z(origin, direction, z):
    return origin[:2] + (np.asarray(z) - origin[2])[..., None] * direction[:2] / direction[2]


def locate_line(p, q, *, radius_mm, half_length_mm, steps_mm=SEED_STEPS_MM, seed_near_mm=SEED_NEAR_MM,
                sample=SEED_SAMPLE, seed=0, slab_mm=SLAB_MM, near_mm=SLAB_NEAR_MM, floor_mm=SLAB_FLOOR_MM,
                windows_mm=WINDOWS_MM, check_z_mm=CHECK_Z_MM):
    """Seed -> slab points -> principal axis -> robust refit; the line the LORs converge to, from endpoints only.

    Returns a dict: the line (``point_mm`` at z = 0, unit ``direction``, ``slopes``), ``slab_points``
    (z, x, y, LORs), ``empty_slabs``, the principal-axis ``spread_mm``, ``median_miss_mm``, ``lors_used``,
    ``seed_xy_mm`` and, per step, the distance (mm) from the final line at ``check_z_mm``.
    Raises ``CtrError`` for too few LORs, no populated slab or degenerate geometry; never returns a fabricated line.
    """
    p, q = _clean(p, q)
    seed_xy = seed_line(p, q, radius_mm=radius_mm, steps_mm=steps_mm, near_mm=seed_near_mm, sample=sample, seed=seed)
    points, empty = slab_points(p, q, seed_xy, slab_mm=slab_mm, near_mm=near_mm, half_length_mm=half_length_mm,
                                floor_mm=floor_mm)
    if len(points) < 2:
        raise CtrError(f"{'all' if len(points) == 0 else 'all but one'} of {empty + len(points)} slabs are empty")
    start = principal_line(points, points[:, 3])
    (origin, direction), miss, used = refine_line(p, q, start[:2], windows_mm=windows_mm)
    z = np.asarray(check_z_mm, np.float64)
    final = _at_z(origin, direction, z)
    steps = {"seed": np.linalg.norm(seed_xy - final, axis=1),
             "principal": np.linalg.norm(_at_z(start[0], start[1], z) - final, axis=1)}
    return {"point_mm": origin.tolist(), "direction": direction.tolist(),
            "slopes": (direction[:2] / direction[2]).tolist(), "slab_points": points.tolist(),
            "empty_slabs": empty, "spread_mm": {"median": float(np.median(start[2])), "max": float(start[2].max())},
            "median_miss_mm": miss, "lors_used": used, "seed_xy_mm": seed_xy.tolist(),
            "check_z_mm": z.tolist(), "step_distance_mm": {k: v.tolist() for k, v in steps.items()}}
