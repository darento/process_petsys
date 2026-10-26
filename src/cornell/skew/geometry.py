"""Declared nominal detector frames and finite annular sources (spec 008 T2a).

No position/DOI calibration is inferred. Local origin, axis order and axial
sign are required choices; passing synthetic checks does not qualify a scanner.
Shell segments describe alternative source locations, not independent events.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.mapping_generator import ChannelType
from src.timing.shell import (C_MM_PER_PS, AnnularSource, GeometryError, ShellInterval,  # noqa: F401
                              _scalar, _vector, geometric_tof_ps, legacy_opposition_accepts,
                              shell_intervals)


@dataclass(frozen=True)
class DetectorFrame:
    sm: int
    center_mm: tuple[float, float, float]
    x_direction: tuple[float, float, float]
    y_direction: tuple[float, float, float]
    local_origin_mm: tuple[float, float]
    local_axes: str
    crystal_thickness_mm: float
    qualification: str = "nominal; physical origin/axes/interaction plane pending"

    def to_global(self, local_x_mm, local_y_mm):
        xy = _vector((local_x_mm, local_y_mm), 2, "local position")
        delta = xy - self.local_origin_mm
        return (np.asarray(self.center_mm) + delta[0] * np.asarray(self.x_direction)
                + delta[1] * np.asarray(self.y_direction))


def build_detector_frames(config, channel_modules, *, local_origin_mm, local_axes,
                          axial_sign, crystal_thickness_mm):
    """Cornell cassette-first IDs, verified against the selected channel map.

    Centers use actual ring_yx magnitudes ([Y,X]), not a rescaled ring_r.
    local_axes is explicitly time-axial (original) or time-transverse (optimized).
    Thickness is metadata: it never adds an assumed radial interaction shift.
    """
    origin = _vector(local_origin_mm, 2, "local origin")
    if local_axes not in ("time-axial", "time-transverse") or axial_sign not in (-1, 1):
        raise GeometryError("explicit local axes and axial sign (-1/+1) required")
    crystal_thickness_mm = _scalar(crystal_thickness_mm, "crystal thickness")
    if crystal_thickness_mm <= 0:
        raise GeometryError("positive finite crystal thickness required")
    frames = {}
    for cassette, x, y, radius, z in _cassette_placements(config, channel_modules):
        transverse = (-y / radius, x / radius, 0.0)
        axial = (0.0, 0.0, float(axial_sign))
        dx, dy = (axial, transverse) if local_axes == "time-axial" else (transverse, axial)
        for index, z_mm in enumerate(z):
            sm = cassette * len(z) + index
            frames[sm] = DetectorFrame(sm, (float(x), float(y), float(z_mm)), dx, dy,
                                       tuple(origin), local_axes, float(crystal_thickness_mm))
    return frames


def _cassette_placements(config, channel_modules):
    """Validated (cassette, x, y, radius, ring_z) from ring_yx/ring_z and the map."""
    yx = config.get("ring_yx")
    z = config.get("ring_z")
    if not isinstance(yx, dict) or not yx or not isinstance(z, (list, tuple)) or not z:
        raise GeometryError("ring_yx/ring_z required")
    if any(type(k) is not int for k in yx) or set(yx) != set(range(len(yx))):
        raise GeometryError("cassette indices must be consecutive integers from zero")
    z = _vector(z, len(z), "ring_z")
    if len(set(z)) != len(z):
        raise GeometryError("duplicate axial SM placement")
    sms = {value[0] for value in channel_modules.values()}
    if not sms or any(type(sm) is not int for sm in sms) or sms != set(range(len(yx) * len(z))):
        raise GeometryError("selected map does not match declared cassette-first placement")
    ring_r = config.get("ring_r")
    if ring_r is not None:
        ring_r = _scalar(ring_r, "ring_r")
        if ring_r <= 0:
            raise GeometryError("ring_r must be a positive consistency reference")
    placements = []
    for cassette in sorted(yx):
        y, x = _vector(yx[cassette], 2, "cassette center")
        radius = math.hypot(x, y)
        if radius == 0 or (ring_r is not None and abs(radius - ring_r) > 0.01):
            raise GeometryError("ring_yx/ring_r inconsistent or zero radius")
        placements.append((cassette, x, y, radius, z))
    return placements


@dataclass(frozen=True)
class TiledSketch:
    """Owner-supplied `Cornell Sketch r01 231010` (tolerances unspecified).

    Ø310 mm is between gamma entrance faces; SiPM faces are one crystal
    thickness farther out. Each SM spans two flat 18° tiles transversally and
    two 51.3 mm rows axially; each tile/row holds two minimodules.
    """
    entrance_apothem_mm: float = 155.0
    crystal_thickness_mm: float = 20.0
    tile_pitch_deg: float = 18.0
    minimodule_width_mm: float = 25.4
    minimodule_gap_mm: float = 0.5
    row_gap_mm: float = 1.2

    @property
    def minimodule_offset_mm(self):
        return (self.minimodule_width_mm + self.minimodule_gap_mm) / 2

    @property
    def row_pitch_mm(self):
        return 2 * self.minimodule_width_mm + self.minimodule_gap_mm + self.row_gap_mm


CORNELL_SKETCH_R01 = TiledSketch()
# Owner-declared 2026-10-13: fixed depth below the entrance face (LYSO 511 keV mean).
OWNER_INTERACTION_DEPTH_MM = 7.0


@dataclass(frozen=True)
class FacetedDetectorFrame:
    """Time-axial SM on two flat tiles; minimodule identity from map centers."""
    sm: int
    cassette_azimuth_rad: float
    axial_center_mm: float
    axial_sign: int
    transverse_sign: int
    interaction_apothem_mm: float
    map_centers_mm: tuple[tuple[float, ...], tuple[float, ...]]
    sketch: TiledSketch
    crystal_thickness_mm: float
    qualification: str = ("owner nominal; transverse handedness, spatial calibration "
                          "and source pose pending")

    def _minimodule(self, value, axis):
        centers = self.map_centers_mm[axis]
        distances = [abs(value - center) for center in centers]
        k = int(np.argmin(distances))
        # Edges are inclusive; 1 nm absorbs binary rounding of map coordinates.
        if not distances[k] <= self.sketch.minimodule_width_mm / 2 + 1e-6:
            raise GeometryError("local position outside every declared minimodule")
        return k, value - centers[k]

    def to_global(self, local_x_mm, local_y_mm):
        x, y = _vector((local_x_mm, local_y_mm), 2, "local position")
        kx, dx = self._minimodule(x, 0)
        ky, dy = self._minimodule(y, 1)
        offset = self.sketch.minimodule_offset_mm
        row = (kx // 2 - 0.5) * self.sketch.row_pitch_mm
        axial = row + (kx % 2 - 0.5) * 2 * offset + dx
        # transverse_sign +1: +ENERGY counter-clockwise; -1 mirrors each SM
        # across its cassette's radial plane (handedness unqualified).
        phi = self.cassette_azimuth_rad + self.transverse_sign * math.radians(
            (ky // 2 - 0.5) * self.sketch.tile_pitch_deg)
        along = self.transverse_sign * ((ky % 2 - 0.5) * 2 * offset + dy)
        c, s = math.cos(phi), math.sin(phi)
        r = self.interaction_apothem_mm
        return np.array((r * c - along * s, r * s + along * c,
                         self.axial_center_mm + self.axial_sign * axial))


def _minimodule_centers(local_map, channel_modules, channel_types):
    """Distinct map minimodule centers: TIME coordinate 0, ENERGY coordinate 1."""
    centers = []
    for kind, coordinate in ((ChannelType.TIME, 0), (ChannelType.ENERGY, 1)):
        ranges = {}
        for ch, key in channel_modules.items():
            if kind in channel_types.get(ch, ()):
                ranges.setdefault(key, []).append(float(local_map[ch][coordinate]))
        values = sorted({round((min(v) + max(v)) / 2, 6) for v in ranges.values()})
        if len(values) != 4:
            raise GeometryError("sketch needs four map minimodule centers per local axis")
        centers.append(tuple(values))
    return tuple(centers)


def build_faceted_detector_frames(config, local_map, channel_modules, channel_types, *,
                                  axial_sign, transverse_sign,
                                  interaction_depth_mm=OWNER_INTERACTION_DEPTH_MM,
                                  sketch=CORNELL_SKETCH_R01):
    """Owner-selected nominal Cornell geometry (T2c/T2d); planar is legacy only.

    Time-axial only (slab index axial). ring_yx gives cassette azimuths and
    ring_z only ascending SM order; the sketch sets 105 mm SM pitch with the
    middle SM at z=0. Interaction depth is a declared constant below the
    entrance face, never a per-event DOI conversion. Both handedness
    candidates stay explicit until data or a known-position source decides.
    """
    for sign, label in ((axial_sign, "axial"), (transverse_sign, "transverse")):
        if sign not in (-1, 1) or isinstance(sign, bool):
            raise GeometryError(f"explicit {label} sign (-1/+1) required")
    depth = _scalar(interaction_depth_mm, "interaction depth")
    if not 0 <= depth <= sketch.crystal_thickness_mm:
        raise GeometryError("interaction depth must lie within the crystal")
    placements = _cassette_placements(config, channel_modules)
    z = placements[0][4]
    if len(z) != 3 or not z[0] < z[1] < z[2] or z[1] != 0:
        raise GeometryError("sketch expects three ascending ring_z entries, middle at 0")
    centers = _minimodule_centers(local_map, channel_modules, channel_types)
    sm_pitch = 2 * sketch.row_pitch_mm
    frames = {}
    for cassette, x, y, _, _ in placements:
        for index in range(3):
            sm = cassette * 3 + index
            frames[sm] = FacetedDetectorFrame(
                sm, math.atan2(y, x), (index - 1) * sm_pitch, axial_sign, transverse_sign,
                sketch.entrance_apothem_mm + depth, centers, sketch, sketch.crystal_thickness_mm)
    return frames


# Normalization phantom (drawing Ph Norm D275L300 r01): fluid radii 137.5/142.5 mm,
# 300 mm long, centred at YAML z=0 by its fixture (owner, ~1 mm transverse).
NORMALIZATION_PHANTOM = AnnularSource(137.5, 142.5, -150.0, 150.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
# Slab index axial with sign -1 (T2b owner/drawing). Transverse handedness +1 is
# selected by run7 data (T6 hypothesis-exclusive events) and owner-accepted 2026-10-13.
SELECTED_AXIAL_SIGN = -1
SELECTED_TRANSVERSE_SIGN = 1
