"""Finite annular sources and the LOR stage on detector endpoints (spec 011 T1, FR-1).

Moved unchanged from ``src.cornell.skew.geometry`` and ``ingest`` (spec 008
T2a/T4c). A scanner adapter computes endpoints (mm, scanner frame); the LOR
stage then applies the finite shell and the 3D opposition angle. Shell
segments describe alternative source locations, not independent events.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numba
import numpy as np

from src.ldat_inspector.fastread import _pow
from src.timing.kernels import ACCEPTED

C_MM_PER_PS = 0.299792458
GEOMETRY_REJECTIONS = ("invalid endpoint", "no shell crossing", "opposition angle", "one shell segment")


class GeometryError(ValueError):
    """Required geometry is missing, inconsistent or nonfinite."""


def _vector(value, size, label):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{label}: expected {size} finite coordinates") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise GeometryError(f"{label}: expected {size} finite coordinates")
    return result


def _scalar(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise GeometryError(f"{label}: finite numeric value required")
    return float(value)


@dataclass(frozen=True)
class AnnularSource:
    inner_radius_mm: float
    outer_radius_mm: float
    axial_min_mm: float
    axial_max_mm: float
    center_mm: tuple[float, float, float]
    axis: tuple[float, float, float]

    def __post_init__(self):
        for name in ("inner_radius_mm", "outer_radius_mm", "axial_min_mm", "axial_max_mm"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if not 0 < self.inner_radius_mm < self.outer_radius_mm or self.axial_min_mm >= self.axial_max_mm:
            raise GeometryError("shell must have positive thickness and axial length")
        center = _vector(self.center_mm, 3, "source center")
        axis = _vector(self.axis, 3, "source axis")
        norm = float(np.linalg.norm(axis))
        if not math.isfinite(norm) or norm == 0:
            raise GeometryError("source axis must have finite nonzero norm")
        object.__setattr__(self, "center_mm", tuple(center))
        object.__setattr__(self, "axis", tuple(axis / norm))


@dataclass(frozen=True)
class ShellInterval:
    fraction_start: float
    fraction_end: float
    tof_start_ps: float
    tof_end_ps: float


def _line(endpoint_i_mm, endpoint_j_mm):
    start = _vector(endpoint_i_mm, 3, "detector i")
    end = _vector(endpoint_j_mm, 3, "detector j")
    delta = end - start
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length == 0:
        raise GeometryError("detector endpoints must form a finite nonzero LOR")
    return start, delta, length


def shell_intervals(endpoint_i_mm, endpoint_j_mm, source):
    """Positive-length source segments along i→j, clipped to detector endpoints.

    TOF is (distance to i - distance to j)/c; reversing endpoints negates
    and reverses intervals. Tangency alone has zero source length, hence no
    usable segment. A tangent to the inner cavity can give one merged segment.
    """
    if not isinstance(source, AnnularSource):
        raise GeometryError("explicit finite annular source required")
    start, direction, length = _line(endpoint_i_mm, endpoint_j_mm)
    axis = np.asarray(source.axis)
    relative = start - source.center_mm
    z0, dz = float(relative @ axis), float(direction @ axis)
    radial_start = relative - z0 * axis
    radial_direction = direction - dz * axis
    a = float(radial_direction @ radial_direction)
    b = 2.0 * float(radial_start @ radial_direction)
    radial2 = float(radial_start @ radial_start)
    if not all(math.isfinite(x) for x in (a, b, radial2, z0, dz)):
        raise GeometryError("LOR exceeds finite numerical geometry range")
    boundaries = [0.0, 1.0]
    if a > 0:
        for radius in (source.inner_radius_mm, source.outer_radius_mm):
            constant = radial2 - radius * radius
            discriminant = b * b - 4 * a * constant
            if discriminant >= 0:
                root = math.sqrt(discriminant)
                # Stable quadratic roots, including a double root at tangency.
                q = -0.5 * (b + math.copysign(root, b))
                roots = (-b / (2 * a),) if q == 0 else (q / a, constant / q)
                boundaries.extend(t for t in roots if 0 < t < 1)
    if dz != 0:
        boundaries.extend(t for z in (source.axial_min_mm, source.axial_max_mm)
                          if 0 < (t := (z - z0) / dz) < 1)
    boundaries = sorted(set(boundaries))
    segments = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        middle = (lo + hi) / 2
        radial = radial_start + middle * radial_direction
        r2 = float(radial @ radial)
        z = z0 + middle * dz
        # Radial boundaries themselves carry zero measure. An axial-parallel
        # line exactly on a radial surface is not assigned a fabricated volume.
        if (source.inner_radius_mm ** 2 < r2 < source.outer_radius_mm ** 2
                and source.axial_min_mm <= z <= source.axial_max_mm):
            if segments and segments[-1][1] == lo:
                segments[-1] = (segments[-1][0], hi)
            else:
                segments.append((lo, hi))
    return tuple(ShellInterval(lo, hi, length * (2 * lo - 1) / C_MM_PER_PS,
                               length * (2 * hi - 1) / C_MM_PER_PS) for lo, hi in segments)


def geometric_tof_ps(source_position_mm, endpoint_i_mm, endpoint_j_mm):
    """Known-position TOF, independent of measured timing or branch selection."""
    point = _vector(source_position_mm, 3, "source position")
    start, delta, _ = _line(endpoint_i_mm, endpoint_j_mm)
    result = (np.linalg.norm(point - start) - np.linalg.norm(point - start - delta)) / C_MM_PER_PS
    if not math.isfinite(result):
        raise GeometryError("nonfinite source flight distances")
    return float(result)


def legacy_opposition_accepts(endpoint_i_mm, endpoint_j_mm, *, tolerance_deg=15.0):
    """Legacy >=(180-tolerance) 3D scanner-origin angle; separate from shell."""
    tolerance_deg = _scalar(tolerance_deg, "opposition tolerance")
    if not 0 <= tolerance_deg <= 180:
        raise GeometryError("opposition tolerance must be in [0,180] degrees")
    start, delta, _ = _line(endpoint_i_mm, endpoint_j_mm)
    end = start + delta
    norms = np.linalg.norm(start), np.linalg.norm(end)
    if not all(math.isfinite(x) and x > 0 for x in norms):
        raise GeometryError("scanner-center endpoint has no opposition angle")
    cosine = float(np.clip((start / norms[0]) @ (end / norms[1]), -1.0, 1.0))
    # Avoid acos precision loss; equality accepts the declared boundary.
    limit = math.cos(math.radians(180 - tolerance_deg))
    return bool(cosine <= limit + 8 * np.finfo(np.float64).eps)


@numba.njit(cache=True)
def _add_bound(bounds, n, t):
    if 0 < t < 1:
        bounds[n] = t
        return n + 1
    return n


@numba.njit(cache=True)
def _shell(p, q, center, axis, r_in, r_out, z_min, z_max, fractions, tof):
    """geometry.shell_intervals port: segment count, -1 for a degenerate LOR."""
    d0, d1, d2 = q[0] - p[0], q[1] - p[1], q[2] - p[2]
    length = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    if not (np.isfinite(length) and length > 0):
        return -1
    v0, v1, v2 = p[0] - center[0], p[1] - center[1], p[2] - center[2]
    z0 = v0 * axis[0] + v1 * axis[1] + v2 * axis[2]
    dz = d0 * axis[0] + d1 * axis[1] + d2 * axis[2]
    s0, s1, s2 = v0 - z0 * axis[0], v1 - z0 * axis[1], v2 - z0 * axis[2]
    e0, e1, e2 = d0 - dz * axis[0], d1 - dz * axis[1], d2 - dz * axis[2]
    a = e0 * e0 + e1 * e1 + e2 * e2
    b = 2.0 * (s0 * e0 + s1 * e1 + s2 * e2)
    radial2 = s0 * s0 + s1 * s1 + s2 * s2
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(radial2) and np.isfinite(z0) and np.isfinite(dz)):
        return -1
    bounds = np.empty(8)
    bounds[0], bounds[1] = 0.0, 1.0
    n = 2
    if a > 0:
        for radius in (r_in, r_out):
            constant = radial2 - radius * radius
            discriminant = b * b - 4 * a * constant
            if discriminant >= 0:
                root = math.sqrt(discriminant)
                half = -0.5 * (b + math.copysign(root, b))
                if half == 0:
                    n = _add_bound(bounds, n, -b / (2 * a))
                else:
                    n = _add_bound(bounds, n, half / a)
                    n = _add_bound(bounds, n, constant / half)
    if dz != 0:
        n = _add_bound(bounds, n, (z_min - z0) / dz)
        n = _add_bound(bounds, n, (z_max - z0) / dz)
    bounds[:n].sort()
    inner2, outer2 = _pow(r_in, 2.0), _pow(r_out, 2.0)
    segments = 0
    for i in range(n - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if lo == hi:
            continue
        middle = (lo + hi) / 2
        w0, w1, w2 = s0 + middle * e0, s1 + middle * e1, s2 + middle * e2
        r2 = w0 * w0 + w1 * w1 + w2 * w2
        z = z0 + middle * dz
        if inner2 < r2 < outer2 and z_min <= z <= z_max:
            if segments and fractions[2 * segments - 1] == lo:
                fractions[2 * segments - 1] = hi
            elif segments == 2:
                return 3
            else:
                fractions[2 * segments] = lo
                fractions[2 * segments + 1] = hi
                segments += 1
    for i in range(2 * segments):
        tof[i] = length * (2 * fractions[i] - 1) / C_MM_PER_PS
    return segments


@numba.njit(cache=True)
def _lor_stage(endpoints, center, axis, r_in, r_out, z_min, z_max, cos_limit, codes, segments, fractions, tof):
    """Shell and angle codes for rows whose endpoints are defined (code 0 on entry)."""
    for k in range(codes.size):
        if codes[k] != 0:
            continue
        p, q = endpoints[k, :3], endpoints[k, 3:]
        n = _shell(p, q, center, axis, r_in, r_out, z_min, z_max, fractions[k], tof[k])
        segments[k] = n
        norm_p = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
        norm_q = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2])
        if n < 0 or not (norm_p > 0 and norm_q > 0 and np.isfinite(norm_p) and np.isfinite(norm_q)):
            codes[k] = 1
            continue
        cosine = ((p[0] / norm_p) * (q[0] / norm_q) + (p[1] / norm_p) * (q[1] / norm_q)
                  + (p[2] / norm_p) * (q[2] / norm_q))
        if n == 0:
            codes[k] = 2
        elif min(1.0, max(-1.0, cosine)) > cos_limit:
            codes[k] = 3
        elif n != 2:
            codes[k] = 4
        else:
            codes[k] = 0


@dataclass(frozen=True)
class GeometryResult:
    """Geometry stage over feature rows: code 0 accepted, k > 0 ``GEOMETRY_REJECTIONS[k - 1]``.

    Endpoints are detector 1 then detector 2 (mm); shell fractions/TOF (ps)
    hold up to two segments along detector 1 -> 2, NaN where absent.
    """
    hypothesis: str
    codes: np.ndarray
    endpoints_mm: np.ndarray
    shell_segments: np.ndarray
    shell_fractions: np.ndarray
    shell_tof_ps: np.ndarray

    def counts(self):
        found = np.bincount(self.codes, minlength=len(GEOMETRY_REJECTIONS) + 1)
        return {label: int(found[index]) for index, label in enumerate((ACCEPTED,) + GEOMETRY_REJECTIONS)}


def source_tables(source, opposition_tolerance_deg):
    """``_lor_stage`` source and angle arguments of a declared annular source."""
    if not isinstance(source, AnnularSource) or not 0 <= opposition_tolerance_deg <= 180:
        raise GeometryError("explicit finite annular source and opposition tolerance required")
    return (np.array(source.center_mm), np.array(source.axis), source.inner_radius_mm, source.outer_radius_mm,
            source.axial_min_mm, source.axial_max_mm,
            math.cos(math.radians(180 - opposition_tolerance_deg)) + 8 * np.finfo(np.float64).eps)


def lor_stage(name, codes, endpoints, source_args):
    """Run the LOR stage after an adapter's endpoint kernel; ``codes`` hold 1 for undefined endpoints."""
    count = len(codes)
    segments = np.zeros(count, np.int64)
    fractions, tof = np.full((count, 4), np.nan), np.full((count, 4), np.nan)
    _lor_stage(endpoints, *source_args, codes, segments, fractions, tof)
    if (segments > 2).any():
        raise GeometryError("annular shell produced more than two segments")
    return GeometryResult(name, codes, endpoints, segments, fractions, tof)
