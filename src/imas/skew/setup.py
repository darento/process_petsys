"""IMAS config/map lookup and SuperModule numbering (spec 011 T3, FR-2, FR-6).

Built once from ``configs/imas_2DAQ.yaml``, its map and, when given, a per-TIME-
channel ``mu`` energy calibration (``KevConverter(file_type="mu")``). SMs are
numbered ring-major, SM = azimuthal index + n_azimuth x ring, with SM centres
from ``ring_yx`` (``[Y, X]`` per azimuthal index) and ``ring_z`` (one per ring);
the map's SM set and the config's ring tables must agree.

Local frame: the map gives each TIME channel (one slab) a transaxial x and its
slab index 0-7 within the minimodule, and each ENERGY channel an axial y, in mm
over the SM. The SM centre in that frame is the midpoint of the map's extent,
the same for every SM. These are nominal positions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.mapping_generator import ChannelType, map_factory
from src.timing.kernels import _KIND_ENERGY, _KIND_TIME
from src.timing.shell import GeometryError

MU_MISSING, MU_VALID, MU_INVALID = 0, 1, 2
SLABS_PER_MINIMODULE = 8
ANGLE_TOLERANCE_RAD = 1e-3
RADIUS_TOLERANCE_MM = 0.5


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_mu(path):
    """``ID\\tmu`` table as {TIME channel: mu}; non-finite or non-positive values are kept (invalid)."""
    table = pd.read_csv(path, sep="\t")
    if list(table.columns[:2]) != ["ID", "mu"]:
        raise ValueError(f"{path}: expected columns ID, mu")
    return {int(ch): float(mu) for ch, mu in zip(table["ID"], table["mu"])}


def check_ring_tables(ring_r, ring_z, ring_yx, sms):
    """Ring-major numbering against the config: SM ids 0..n_rings*n_azimuth-1, evenly spaced azimuths.

    Returns per-azimuth centre angles (rad). A gap, swap or count mismatch is a ``GeometryError``.
    """
    n_azimuth, n_rings = len(ring_yx), len(ring_z)
    if sorted(ring_yx) != list(range(n_azimuth)) or n_azimuth < 3 or n_rings < 1:
        raise GeometryError("ring_yx keys must be 0..n-1 (n >= 3) and ring_z non-empty")
    if set(sms) != set(range(n_azimuth * n_rings)):
        raise GeometryError(f"map SMs are not 0..{n_azimuth * n_rings - 1} (ring-major, "
                            f"{n_rings} rings x {n_azimuth} azimuths)")
    if any(b <= a for a, b in zip(ring_z, ring_z[1:])):
        raise GeometryError("ring_z must increase with ring index")
    azimuth = np.array([math.atan2(float(ring_yx[k][0]), float(ring_yx[k][1])) for k in range(n_azimuth)])
    radius = np.array([math.hypot(float(ring_yx[k][0]), float(ring_yx[k][1])) for k in range(n_azimuth)])
    if np.abs(radius - ring_r).max() > RADIUS_TOLERANCE_MM:
        raise GeometryError("ring_yx centres are not at ring_r")
    step = np.angle(np.exp(1j * (np.roll(azimuth, -1) - azimuth)))
    if np.abs(np.abs(step) - 2 * math.pi / n_azimuth).max() > ANGLE_TOLERANCE_RAD or len(set(np.sign(step))) != 1:
        raise GeometryError("ring_yx azimuths are not evenly spaced in index order (swapped or missing entry)")
    return azimuth


@dataclass(frozen=True)
class ImasSetup:
    """Config, map and optional energy calibration; dense lookups come from ``lookup()``."""
    config_path: str
    map_path: str
    encal_path: str | None
    min_ch: int
    en_min_ch: float
    ring_r: float
    ring_z: tuple
    azimuth: np.ndarray          # per azimuthal index (rad)
    local_origin: tuple          # (transaxial x, axial y) of the SM centre in the map frame, mm
    local_map: dict
    sm_mM_map: dict
    chtype_map: dict
    mu: dict
    provenance: dict

    @classmethod
    def from_files(cls, config_path, encal_path=None, *, root="."):
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        map_path = Path(root) / config["map_file"]
        local_map, sm_mM_map, chtype_map, fem = map_factory(str(map_path))
        if not fem.sum_rows_cols:
            raise ValueError("IMAS selection requires sum-rows/cols readout")
        ring_yx = {int(k): v for k, v in config["ring_yx"].items()}
        ring_z = tuple(float(z) for z in config["ring_z"])
        azimuth = check_ring_tables(float(config["ring_r"]), ring_z, ring_yx, {sm for sm, _ in sm_mM_map.values()})
        origin = _local_origin(local_map, sm_mM_map, chtype_map)
        mu = read_mu(encal_path) if encal_path is not None else {}
        provenance = {"config": str(config_path), "config_sha256": _sha256(config_path), "map": str(map_path),
                      "map_sha256": _sha256(map_path)}
        if encal_path is not None:
            provenance |= {"encal": str(encal_path), "encal_sha256": _sha256(encal_path)}
        return cls(str(config_path), str(map_path), None if encal_path is None else str(encal_path),
                   int(config["min_ch"]), float(config["en_min_ch"]), float(config["ring_r"]), ring_z, azimuth,
                   origin, local_map, sm_mM_map, chtype_map, mu, provenance)

    @property
    def n_azimuth(self):
        return len(self.azimuth)

    @property
    def time_channels(self):
        return np.array(sorted(ch for ch, types in self.chtype_map.items() if ChannelType.TIME in types), np.int64)

    def ring(self, sm):
        return sm // self.n_azimuth

    def lookup(self):
        return ImasLookup.from_setup(self)


def _local_origin(local_map, sm_mM_map, chtype_map):
    """Midpoint of the TIME-channel x and ENERGY-channel y extents; equal for every SM."""
    extents = {}
    for ch, (sm, _) in sm_mM_map.items():
        axis = 0 if ChannelType.TIME in chtype_map[ch] else 1
        lo, hi = extents.get((sm, axis), (math.inf, -math.inf))
        value = local_map[ch][axis]
        extents[sm, axis] = (min(lo, value), max(hi, value))
    midpoints = {axis: {round((lo + hi) / 2, 6) for (_, a), (lo, hi) in extents.items() if a == axis}
                 for axis in (0, 1)}
    if any(len(values) != 1 for values in midpoints.values()):
        raise GeometryError("SM local frames differ between SMs")
    return midpoints[0].pop(), midpoints[1].pop()


@dataclass(frozen=True)
class ImasLookup:
    """Dense per-channel tables for the compiled selection (index = global channel ID)."""
    kind: np.ndarray        # 1 TIME, 2 ENERGY, 0 unmapped
    sm: np.ndarray
    mm: np.ndarray
    coord_x: np.ndarray     # transaxial local x (TIME channels), mm
    coord_y: np.ndarray     # axial local y (ENERGY channels), mm
    slab: np.ndarray        # slab index 0-7 within the minimodule (TIME channels)
    mu: np.ndarray          # 511 keV peak (a.u.) per TIME channel; NaN unless valid
    mu_state: np.ndarray    # MU_MISSING / MU_VALID / MU_INVALID
    min_ch: int
    en_min_ch: float

    @classmethod
    def from_setup(cls, setup):
        mapped = sorted(setup.chtype_map)
        if not mapped or mapped[0] < 0 or mapped[-1] >= 16_000_000:
            raise ValueError("channel IDs exceed bounded lookup capacity")
        size = mapped[-1] + 1
        kind = np.zeros(size, np.int8)
        sm, mm, slab = (np.full(size, -1, np.int32) for _ in range(3))
        coord_x, coord_y, mu = np.full(size, np.nan), np.full(size, np.nan), np.full(size, np.nan)
        mu_state = np.full(size, MU_MISSING, np.int8)
        for ch in mapped:
            types = tuple(setup.chtype_map[ch])
            if types not in ((ChannelType.TIME,), (ChannelType.ENERGY,)):
                raise ValueError(f"channel {ch}: exactly one TIME or ENERGY type required")
            sm[ch], mm[ch] = setup.sm_mM_map[ch]
            coord_x[ch], coord_y[ch] = setup.local_map[ch][0], setup.local_map[ch][1]
            if types[0] is ChannelType.TIME:
                kind[ch] = _KIND_TIME
                slab[ch] = setup.local_map[ch][2]
                if not 0 <= slab[ch] < SLABS_PER_MINIMODULE:
                    raise GeometryError(f"TIME channel {ch}: slab index {slab[ch]} outside 0-7")
                value = setup.mu.get(ch)
                if value is not None:
                    valid = math.isfinite(value) and value > 0
                    mu_state[ch] = MU_VALID if valid else MU_INVALID
                    mu[ch] = value if valid else np.nan
            else:
                kind[ch] = _KIND_ENERGY
        return cls(kind, sm, mm, coord_x, coord_y, slab, mu, mu_state, setup.min_ch, setup.en_min_ch)
