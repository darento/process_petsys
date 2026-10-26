"""IMAS nominal detector endpoints and the rotating-bar source (spec 011 T6, FR-2, FR-5).

Each SM is a flat facet: centre azimuth ``phi`` from ``ring_yx`` (``[Y, X]``),
axial centre ``ring_z[ring]``, entrance face at ``ring_r`` (410 mm, bore 82 cm).
A side's endpoint is at the interaction depth (7 mm, owner) behind the face::

    p = (ring_r + depth)(cos phi, sin phi, 0) + s_t u (-sin phi, cos phi, 0) + (0, 0, z_ring + s_a v)

with ``u``, ``v`` the side's local transaxial x and axial y minus the SM centre
of the map frame (``ImasSetup.local_origin``). ``s_t``, ``s_a`` in {-1, +1} are
the frame signs; the legacy transform (``scripts_imas/
imas_skew_cal_bigdata_optimized.py:local_to_global``) is ``s_t = +1``,
``s_a = -1`` with a 48 mm origin and the endpoint on the face. The data
select the same pair (``SELECTED_*``, sign study of spec 011 T11). Nothing
here is physically qualified: no gaps, NN positions or DOI.

The source is the rotating PMMA bar's corona, an annulus coaxial with the
scanner (``IMAS_SKEW_BAR``, nominal). Endpoints then go through the core LOR
stage (``src.timing.shell``): shell segments, 3D opposition angle, codes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numba
import numpy as np

from src.timing.shell import AnnularSource, GeometryError, lor_stage, source_tables

# Bar 10 mm diameter, centre at 160 mm radius, ~700 mm long, axially centred, 120 s per turn (owner 2026-10-26).
IMAS_SKEW_BAR = AnnularSource(155.0, 165.0, -350.0, 350.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
INTERACTION_DEPTH_MM = 7.0
OPPOSITION_TOLERANCE_DEG = 15.0          # 3D LOR opening angle >= 165 degrees
NOMINAL_TRANSVERSE_SIGN, NOMINAL_AXIAL_SIGN = 1, -1
# Sign study (spec 011 T11, 3 splits, 947,155 common events, EM with free separation scale): the legacy pair
# wins; axial by 6,414 log-likelihood units (flip: sigma 279.9 -> 284.9 ps, delays move 5.9 ps robust sd),
# transverse by 565 units (flip moves delays 0.3 ps: weakly identified and immaterial to the LUT).
SELECTED_TRANSVERSE_SIGN, SELECTED_AXIAL_SIGN = 1, -1


@dataclass(frozen=True)
class ImasGeometry:
    """Declared hypothesis: frame signs, depth, source and opposition tolerance over one ``ImasSetup``."""
    name: str
    transverse_sign: int
    axial_sign: int
    ring_r: float
    ring_z: tuple
    azimuth: tuple
    local_origin: tuple
    source: AnnularSource = IMAS_SKEW_BAR
    depth_mm: float = INTERACTION_DEPTH_MM
    opposition_tolerance_deg: float = OPPOSITION_TOLERANCE_DEG

    def __post_init__(self):
        if self.transverse_sign not in (-1, 1) or self.axial_sign not in (-1, 1):
            raise GeometryError("frame signs must be -1 or +1")
        if not (math.isfinite(self.ring_r) and math.isfinite(self.depth_mm) and self.ring_r > 0):
            raise GeometryError("ring radius and depth must be finite")

    @classmethod
    def from_setup(cls, setup, *, transverse_sign=NOMINAL_TRANSVERSE_SIGN, axial_sign=NOMINAL_AXIAL_SIGN,
                   source=IMAS_SKEW_BAR, depth_mm=INTERACTION_DEPTH_MM,
                   opposition_tolerance_deg=OPPOSITION_TOLERANCE_DEG, name=None):
        name = name or f"imas facets s_t {transverse_sign:+d} s_a {axial_sign:+d}" + (
            "" if opposition_tolerance_deg == OPPOSITION_TOLERANCE_DEG else
            f" opening >= {180 - opposition_tolerance_deg:g} deg")
        return cls(name, int(transverse_sign), int(axial_sign), setup.ring_r, tuple(setup.ring_z),
                   tuple(float(a) for a in setup.azimuth), tuple(setup.local_origin), source, float(depth_mm),
                   float(opposition_tolerance_deg))

    @classmethod
    def nominal(cls, setup, *, source=IMAS_SKEW_BAR):
        """Legacy signs (s_t +1, s_a -1), 7 mm depth: the frame used until the T9 sign study."""
        return cls.from_setup(setup, source=source, name="imas nominal s_t +1 s_a -1")

    def describe(self):
        return {"name": self.name, "transverse_sign": self.transverse_sign, "axial_sign": self.axial_sign,
                "ring_r_mm": self.ring_r, "ring_z_mm": list(self.ring_z), "azimuth_rad": list(self.azimuth),
                "local_origin_mm": list(self.local_origin), "depth_mm": self.depth_mm,
                "source": repr(self.source), "opposition_tolerance_deg": self.opposition_tolerance_deg,
                "nominal": "flat SM facets from ring_yx/ring_z; not physically qualified"}

    def endpoint(self, sm, x, y):
        """One side's endpoint (mm), for checks and reports."""
        out = np.full(3, np.nan)
        ok = _endpoint(int(sm), float(x), float(y), *self._tables(), out)
        if not ok:
            raise GeometryError(f"SM {sm}: undefined endpoint")
        return out

    def _tables(self):
        return (np.array(self.azimuth), np.array(self.ring_z), self.ring_r + self.depth_mm, self.transverse_sign,
                self.axial_sign, self.local_origin[0], self.local_origin[1])

    def endpoints(self, features):
        """(codes, endpoints (n, 6) mm) over feature rows; code 1 where either side is undefined."""
        column = features.column
        count = len(column("sm_1"))
        codes = np.zeros(count, np.int64)
        endpoints = np.full((count, 6), np.nan)
        _endpoints(column("sm_1"), column("x_1"), column("y_1"), column("sm_2"), column("x_2"), column("y_2"),
                   *self._tables(), codes, endpoints)
        return codes, endpoints

    def apply(self, features):
        """Geometry stage over feature rows (``column(name)``); reads no LDAT."""
        codes, endpoints = self.endpoints(features)
        return lor_stage(self.name, codes, endpoints, source_tables(self.source, self.opposition_tolerance_deg))


@numba.njit(cache=True)
def _endpoint(sm, x, y, azimuth, ring_z, radius, s_t, s_a, x0, y0, out):
    n = azimuth.size
    if sm < 0 or sm >= n * ring_z.size or not (np.isfinite(x) and np.isfinite(y)):
        return False
    phi = azimuth[sm % n]
    c, s = math.cos(phi), math.sin(phi)
    u = s_t * (x - x0)
    out[0] = radius * c - u * s
    out[1] = radius * s + u * c
    out[2] = ring_z[sm // n] + s_a * (y - y0)
    return True


@numba.njit(cache=True)
def _endpoints(sm1, x1, y1, sm2, x2, y2, azimuth, ring_z, radius, s_t, s_a, x0, y0, codes, endpoints):
    """Both sides' endpoints; code 1 where either is undefined."""
    for k in range(sm1.size):
        if not (_endpoint(sm1[k], x1[k], y1[k], azimuth, ring_z, radius, s_t, s_a, x0, y0, endpoints[k, :3])
                and _endpoint(sm2[k], x2[k], y2[k], azimuth, ring_z, radius, s_t, s_a, x0, y0, endpoints[k, 3:])):
            codes[k] = 1
