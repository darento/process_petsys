"""Cornell timing ingest (spec 008 T4).

T4a reference mode reproduces the original ``cornell_skew_cal_bigdata.py``
event selection by calling the same tracked helpers in the same order, on
records from the bounded ``src.ldat_stream`` reader. Inputs are compact
coincidence records (two detectors each), never singles. Energies are keV from
the selected ``(time_channel, slab)`` ``.encal``; timestamps and ``dt`` are
integer ps; positions are mm in the legacy planar frame.

Declared differences from the legacy script, all owner-approved (2026-10-13):
- no hidden 500,000-record stop: ``max_records`` is explicit, default full file;
- legacy crash paths become first-failure labels (unmapped channel, no positive
  minimodule energy, missing keV key); invalid keV factors are labelled rather
  than folded into the energy window;
- the one-TIME-channel slab coin flip is a per-record bit, so results do not
  depend on chunking, workers or global random state;
- ``en_min_ch`` is recorded as read but not applied, as in the legacy script.

T4c production mode keeps the compiled pre-geometry selection, labels
holdout-excluded records first (energy-calibration prefix, 1 µs file-edge
guards) and stores geometry-free features. A separate geometry stage applies a
declared faceted hypothesis, the finite phantom shell and the 3D opposition
angle to those features, so hypotheses never re-read LDAT.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path

import numba
import numpy as np
from scipy.constants import c as C_VAC
from scipy.spatial.transform import Rotation
import yaml

from src import utils
from src.detector_features import calculate_centroid
from src.filters import filter_coincidence, filter_min_ch, filter_total_energy
from src.ldat_inspector.fastread import _pow
from src.cornell.skew.geometry import (C_MM_PER_PS, NORMALIZATION_PHANTOM, OWNER_INTERACTION_DEPTH_MM,
                                       SELECTED_AXIAL_SIGN, SELECTED_TRANSVERSE_SIGN, AnnularSource,
                                       FacetedDetectorFrame, GeometryError, build_faceted_detector_frames)
from src.ldat_stream import LdatStream
from src.mapping_generator import ChannelType, map_factory

LEGACY_COINCIDENCE_WINDOW_PS = 15000
LEGACY_LOCAL_ORIGIN_MM = 48.0
LEGACY_ANGLE_ACCEPTANCE_DEG = 15
LEGACY_HIT_ENERGY_MIN = 0.0          # read_binary_file default: keeps energy >= 0
C_MM_PER_PS_LEGACY = C_VAC * 1000 / 1e12   # legacy expression, scipy constant

ACCEPTED = "accepted"
# First-failure labels in legacy evaluation order.
REJECTIONS = (
    "unmapped channel",
    "side min channels",
    "no positive minimodule energy",
    "minimodule min channels",
    "coincidence window",
    "unresolved slab",
    "missing energy calibration",
    "invalid energy calibration",
    "energy window",
    "no cylinder crossing",
    "opposition angle",
    "crossing outside segment",
)


def _splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def slab_bit(file_index, record_index, side, seed=0):
    """Deterministic coin flip for one detector side of one physical record."""
    if not (0 <= file_index < 2 ** 20 and 0 <= record_index < 2 ** 42 and side in (0, 1)):
        raise ValueError("slab seed key out of range")
    key = (file_index << 43) | (record_index << 1) | side
    return _splitmix64(key ^ _splitmix64(seed & 0xFFFFFFFFFFFFFFFF)) & 1


class _RecordCoin:
    """Stands in for ``random`` inside ``get_slab_cornell``: randint(0, 1) -> bit."""

    def __init__(self, bit):
        self.bit = bit

    def randint(self, a, b):
        if (a, b) != (0, 1):
            raise ValueError("unexpected legacy random draw")
        return self.bit


def _legacy_slab(det, setup, bit):
    original = utils.random
    utils.random = _RecordCoin(bit)
    try:
        return utils.get_slab_cornell(det, setup.chtype_map, setup.local_map)
    finally:
        utils.random = original


@dataclass(frozen=True)
class LegacySetup:
    chtype_map: dict
    sm_mM_map: dict
    local_map: dict
    sum_rows_cols: bool
    kev_factors: dict
    energy_range: tuple[float, float]
    min_ch: int
    en_min_ch: float
    pipe_radius: float
    sm_center_map: dict
    provenance: dict = field(default_factory=dict)
    unpopulated: frozenset = frozenset()     # configured (sm, mm) without SiPMs

    @classmethod
    def from_files(cls, config_path, encal_path, *, root="."):
        root = Path(root)
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        local_map, sm_mM_map, chtype_map, fem = map_factory(str(root / config["map_file"]))
        kev = utils.KevConverter(str(encal_path), file_type="cornell")
        centers = {}
        for cassette, yx in config["ring_yx"].items():
            for z_index, z in enumerate(config["ring_z"]):
                centers[z_index + int(cassette) * len(config["ring_z"])] = {
                    "sm_r": float(config["ring_r"]), "sm_ang": float(np.arctan2(*yx)), "sm_z": float(z)}
        return cls(chtype_map, sm_mM_map, local_map, bool(fem.sum_rows_cols), dict(kev.kev_factors),
                   (float(config["energy_range"][0]), float(config["energy_range"][1])),
                   int(config["min_ch"]), float(config["en_min_ch"]), float(config["pipe_radius"]), centers,
                   {"config": str(config_path), "encal": str(encal_path), "map": config["map_file"],
                    "en_min_ch_applied": False},
                   frozenset((int(sm), int(mm)) for sm, mms in (config.get("unpopulated_minimodules") or {}).items()
                             for mm in mms))


def legacy_local_to_global(x, y, sm):
    """Original script's planar transform (frozen-probe verified), unchanged."""
    x -= sm["sm_z"] + LEGACY_LOCAL_ORIGIN_MM
    y -= LEGACY_LOCAL_ORIGIN_MM
    rotation = Rotation.from_euler("z", sm["sm_ang"])
    return rotation.apply(np.array([sm["sm_r"], y, -x]))


def legacy_cylinder(line_start, line_end, radius):
    """Original ``cylinder_intersection_check`` with a reason instead of False.

    Returns (label or None, tof1, tof2); TOF values are int-truncated ps.
    """
    c_mm_per_ps = C_MM_PER_PS_LEGACY
    line_length = np.linalg.norm(line_end - line_start)
    line_vec = line_end - line_start
    a = np.dot(line_vec[:2], line_vec[:2])
    b = 2 * np.dot(line_start[:2], line_vec[:2])
    c = np.dot(line_start[:2], line_start[:2]) - radius ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return "no cylinder crossing", 0, 0
    start = line_start / np.linalg.norm(line_start)
    end = line_end / np.linalg.norm(line_end)
    angle_deg = np.degrees(np.arccos(np.clip(np.dot(start, end), -1.0, 1.0)))
    if angle_deg < 180 - LEGACY_ANGLE_ACCEPTANCE_DEG:
        return "opposition angle", 0, 0
    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)
    tof1 = int(line_length * (2 * t1 - 1) / c_mm_per_ps)
    tof2 = int(line_length * (2 * t2 - 1) / c_mm_per_ps)
    if not ((0 <= t1 <= 1) and (0 <= t2 <= 1)):
        return "crossing outside segment", tof1, tof2
    return None, tof1, tof2


@dataclass(frozen=True)
class LegacyFeatures:
    time_channels: tuple[int, int]
    modules: tuple[tuple[int, int], tuple[int, int]]
    slabs: tuple[int, int]
    energy_kev: tuple[float, float]
    local_mm: tuple[tuple[float, float], tuple[float, float]]
    global_mm: tuple[tuple[float, ...], tuple[float, ...]]
    dt_ps: int
    legacy_tof_ps: tuple[int, int]


def _kev(setup, key, energy):
    if key not in setup.kev_factors:
        return "missing energy calibration", None
    mu = setup.kev_factors[key]
    if not (math.isfinite(mu) and mu > 0):
        return "invalid energy calibration", None
    return None, 511.0 / mu * energy


def reference_record(det1, det2, setup, *, file_index, record_index, seed=0):
    """Legacy selection for one record: (label, LegacyFeatures or None).

    ``det1``/``det2`` are lists of (timestamp, energy, channel) after the legacy
    reader's ``energy >= 0`` hit filter.
    """
    chtype, modules, local = setup.chtype_map, setup.sm_mM_map, setup.local_map
    if any(hit[2] not in chtype or hit[2] not in modules or hit[2] not in local for hit in det1 + det2):
        return "unmapped channel", None
    if not (filter_min_ch(det1, setup.min_ch, chtype, setup.sum_rows_cols)
            and filter_min_ch(det2, setup.min_ch, chtype, setup.sum_rows_cols)):
        return "side min channels", None
    max1, energy1 = utils.get_maxEnergy_sm_mM(det1, modules, chtype)
    max2, energy2 = utils.get_maxEnergy_sm_mM(det2, modules, chtype)
    if not isinstance(max1, list) or not isinstance(max2, list):
        return "no positive minimodule energy", None
    if not (filter_min_ch(max1, setup.min_ch, chtype, setup.sum_rows_cols)
            and filter_min_ch(max2, setup.min_ch, chtype, setup.sum_rows_cols)):
        return "minimodule min channels", None
    in_window, tch1, tch2 = filter_coincidence(max1, max2, chtype, LEGACY_COINCIDENCE_WINDOW_PS)
    if not in_window:
        return "coincidence window", None
    slab1, _, _ = _legacy_slab(max1, setup, slab_bit(file_index, record_index, 0, seed))
    slab2, _, _ = _legacy_slab(max2, setup, slab_bit(file_index, record_index, 1, seed))
    if slab1 is None or slab2 is None:
        return "unresolved slab", None
    label, kev1 = _kev(setup, (tch1[2], slab1), energy1)
    if label:
        return label, None
    label, kev2 = _kev(setup, (tch2[2], slab2), energy2)
    if label:
        return label, None
    low, high = setup.energy_range
    if not (filter_total_energy(kev1, low, high) and filter_total_energy(kev2, low, high)):
        return "energy window", None
    x1, y1 = calculate_centroid(max1, local, x_rtp=1, y_rtp=2, chtype_map=chtype)
    x2, y2 = calculate_centroid(max2, local, x_rtp=1, y_rtp=2, chtype_map=chtype)
    sm1, sm2 = modules[tch1[2]][0], modules[tch2[2]][0]
    start = np.array(legacy_local_to_global(x1, y1, setup.sm_center_map[sm1]))
    end = np.array(legacy_local_to_global(x2, y2, setup.sm_center_map[sm2]))
    label, tof1, tof2 = legacy_cylinder(start, end, setup.pipe_radius)
    if label:
        return label, None
    return ACCEPTED, LegacyFeatures(
        (tch1[2], tch2[2]), (tuple(modules[tch1[2]]), tuple(modules[tch2[2]])), (slab1, slab2),
        (kev1, kev2), ((x1, y1), (x2, y2)), (tuple(map(float, start)), tuple(map(float, end))),
        int(tch1[0]) - int(tch2[0]), (tof1, tof2))


def _sides(batch, k):
    """Legacy reader view of record k: hit tuples with energy >= 0."""
    start, n1, n2 = int(batch.first_hit[k]), int(batch.n1[k]), int(batch.n2[k])
    hits = [(int(t), float(e), int(c)) for t, e, c in zip(batch.timestamp[start:start + n1 + n2].tolist(),
                                                         batch.energy[start:start + n1 + n2].tolist(),
                                                         batch.channel[start:start + n1 + n2].tolist())]
    keep = lambda side: [hit for hit in side if hit[1] >= LEGACY_HIT_ENERGY_MIN]
    return keep(hits[:n1]), keep(hits[n1:])


def reference_outcomes(path, setup, *, file_index, max_records=None, seed=0, block_bytes=None):
    """Yield (record_index, label, features) for every physical record read."""
    options = {} if block_bytes is None else {"block_bytes": block_bytes}
    for batch in LdatStream(path, max_records=max_records, **options):
        for k in range(batch.count):
            det1, det2 = _sides(batch, k)
            record = batch.first_record + k
            label, features = reference_record(det1, det2, setup, file_index=file_index,
                                               record_index=record, seed=seed)
            yield record, label, features


def reference_counts(path, setup, *, file_index, max_records=None, seed=0):
    """First-failure counters; every record read is counted exactly once."""
    counts = Counter(label for _, label, _ in reference_outcomes(
        path, setup, file_index=file_index, max_records=max_records, seed=seed))
    return {label: counts.get(label, 0) for label in (ACCEPTED,) + REJECTIONS}


# --- T4b: compiled selection up to the energy window and centroid ---------
# Codes: 0 passes every pre-geometry cut; k > 0 is REJECTIONS[k - 1].
PASSED = 0
_CODE = {label: index + 1 for index, label in enumerate(REJECTIONS)}
_KIND_TIME, _KIND_ENERGY = 1, 2
_MU_MISSING, _MU_VALID, _MU_INVALID = 0, 1, 2
FEATURE_INT = ("record", "time_channel_1", "time_channel_2", "sm_1", "mm_1", "sm_2", "mm_2",
               "slab_1", "slab_2", "dt_ps", "module_tie")
FEATURE_FLOAT = ("energy_kev_1", "energy_kev_2", "x_1", "y_1", "x_2", "y_2")
_PRE_GEOMETRY = REJECTIONS[:9]


@dataclass(frozen=True)
class CompiledLookup:
    """Dense per-channel tables built once from a ``LegacySetup``."""
    kind: np.ndarray
    sm: np.ndarray
    mm: np.ndarray
    coord_x: np.ndarray
    coord_y: np.ndarray
    position: np.ndarray
    time_index: np.ndarray
    mu: np.ndarray
    mu_state: np.ndarray
    min_ch: int
    energy_range: tuple[float, float]

    @classmethod
    def from_setup(cls, setup):
        if not setup.sum_rows_cols:
            raise ValueError("compiled Cornell selection requires sum-rows/cols readout")
        mapped = [ch for ch in setup.chtype_map if ch in setup.sm_mM_map and ch in setup.local_map]
        if not mapped or min(mapped) < 0 or max(mapped) >= 16_000_000:
            raise ValueError("channel IDs exceed bounded lookup capacity")
        size = max(mapped) + 1
        kind = np.zeros(size, np.int8)
        sm, mm, position, time_index = (np.full(size, -1, np.int32) for _ in range(4))
        coord_x, coord_y = np.full(size, np.nan), np.full(size, np.nan)
        times = sorted(ch for ch in mapped if ChannelType.TIME in setup.chtype_map[ch])
        mu = np.full((len(times), 16), np.nan)
        mu_state = np.full((len(times), 16), _MU_MISSING, np.int8)
        for ch in mapped:
            types = tuple(setup.chtype_map[ch])
            if types not in ((ChannelType.TIME,), (ChannelType.ENERGY,)):
                raise ValueError(f"channel {ch}: exactly one TIME or ENERGY type required")
            kind[ch] = _KIND_TIME if types[0] is ChannelType.TIME else _KIND_ENERGY
            sm[ch], mm[ch] = setup.sm_mM_map[ch]
            coord_x[ch], coord_y[ch] = setup.local_map[ch][0], setup.local_map[ch][1]
            position[ch] = setup.local_map[ch][2]
        for index, ch in enumerate(times):
            time_index[ch] = index
            for slab in range(16):
                value = setup.kev_factors.get((ch, slab))
                if value is None:
                    continue
                mu[index, slab] = value
                mu_state[index, slab] = _MU_VALID if math.isfinite(value) and value > 0 else _MU_INVALID
        return cls(kind, sm, mm, coord_x, coord_y, position, time_index, mu, mu_state,
                   setup.min_ch, setup.energy_range)


@numba.njit(cache=True)
def _mix(value):
    value = value + np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


@numba.njit(cache=True)
def _bit(file_index, record, side, seed_mix):
    key = (np.uint64(file_index) << np.uint64(43)) | (np.uint64(record) << np.uint64(1)) | np.uint64(side)
    return np.int64(_mix(key ^ seed_mix) & np.uint64(1))


@numba.njit(cache=True)
def _keep(start, n, energy, channel, kind, out):
    """Legacy reader hits (energy >= 0) into ``out``; second value flags unmapped."""
    m, unmapped = 0, False
    for i in range(start, start + n):
        if energy[i] >= 0:
            out[m] = i
            m += 1
            c = channel[i]
            if c < 0 or c >= kind.size or kind[c] == 0:
                unmapped = True
    return m, unmapped


@numba.njit(cache=True)
def _min_ch(hits, m, channel, kind, min_ch):
    n_energy = 0
    for j in range(m):
        if kind[channel[hits[j]]] == 2:
            n_energy += 1
    return min_ch <= n_energy < m


@numba.njit(cache=True)
def _max_module(hits, m, energy, channel, kind, sm, mm, out, keys, module_energy):
    """get_maxEnergy_sm_mM: (selected count, energy, ok, tie); first-seen order on ties."""
    first_key = sm[channel[hits[0]]] * 4096 + mm[channel[hits[0]]] if m else -1
    single = True
    for j in range(1, m):
        c = channel[hits[j]]
        if sm[c] * 4096 + mm[c] != first_key:
            single = False
    if single:
        total = 0.0
        for j in range(m):
            out[j] = hits[j]
            if kind[channel[hits[j]]] == 2:
                total += np.float64(energy[hits[j]])
        return m, total, True, False
    n_keys = 0
    for j in range(m):
        c = channel[hits[j]]
        key = sm[c] * 4096 + mm[c]
        seen = False
        for q in range(n_keys):
            if keys[q] == key:
                seen = True
        if not seen:
            keys[n_keys] = key
            n_keys += 1
    best, best_energy = -1, 0.0
    for q in range(n_keys):
        total = 0.0
        for j in range(m):
            c = channel[hits[j]]
            if sm[c] * 4096 + mm[c] == keys[q] and kind[c] == 2:
                total += np.float64(energy[hits[j]])
        module_energy[q] = total
        if total > best_energy:
            best_energy = total
            best = q
    if best < 0:
        return 0, 0.0, False, False
    tied = 0
    for q in range(n_keys):
        if module_energy[q] == best_energy:
            tied += 1
    count = 0
    for j in range(m):
        c = channel[hits[j]]
        if sm[c] * 4096 + mm[c] == keys[best]:
            out[count] = hits[j]
            count += 1
    return count, best_energy, True, tied > 1


@numba.njit(cache=True)
def _time_hits(hits, m, energy, channel, kind):
    """Top-two TIME hits by energy, earlier hit first on ties (stable sort)."""
    top, second, n_time = -1, -1, 0
    for j in range(m):
        i = hits[j]
        if kind[channel[i]] != 1:
            continue
        n_time += 1
        if top < 0 or energy[i] > energy[top]:
            second = top
            top = i
        elif second < 0 or energy[i] > energy[second]:
            second = i
    return top, second, n_time


@numba.njit(cache=True)
def _slab(top, second, n_time, channel, position, bit):
    p = position[channel[top]]
    count = 2 if n_time >= 2 else 1
    if p == 0:
        return count - 1
    if p == 7:
        return 14 + (1 - (count - 1))
    if count == 1:
        return 2 * p + bit
    diff = p - position[channel[second]]
    if diff > 1 or diff < -1:
        return -1
    return 2 * p if diff == 1 else 2 * p + 1


@numba.njit(cache=True)
def _centroid(hits, m, energy, channel, kind, coord_x, coord_y):
    sx = wx = sy = wy = 0.0
    for j in range(m):
        i = hits[j]
        c = channel[i]
        if kind[c] == 1:
            w = np.float64(energy[i]) + 0.00001
            sx += w * coord_x[c]
            wx += w
        else:
            # CPython float ** calls C pow(); numba's x*x differs by ulps (see fastread._c_pow).
            w = _pow(np.float64(energy[i]) + 0.00001, 2.0)
            sy += w * coord_y[c]
            wy += w
    return sx / wx, sy / wy


@numba.njit(cache=True)
def _select(n1, n2, first, timestamp, energy, channel, kind, sm, mm, coord_x, coord_y, position,
            time_index, mu, mu_state, min_ch, low, high, window, file_index, first_record, seed_mix,
            codes_map, codes, out_i, out_f):
    kept0, kept1 = np.empty(512, np.int64), np.empty(512, np.int64)
    sel0, sel1 = np.empty(512, np.int64), np.empty(512, np.int64)
    keys, module_energy = np.empty(512, np.int64), np.empty(512)
    n_out = 0
    for k in range(n1.size):
        m0, bad0 = _keep(first[k], n1[k], energy, channel, kind, kept0)
        m1, bad1 = _keep(first[k] + n1[k], n2[k], energy, channel, kind, kept1)
        if bad0 or bad1:
            codes[k] = codes_map[0]
            continue
        if not (_min_ch(kept0, m0, channel, kind, min_ch) and _min_ch(kept1, m1, channel, kind, min_ch)):
            codes[k] = codes_map[1]
            continue
        s0, e0, ok0, tie0 = _max_module(kept0, m0, energy, channel, kind, sm, mm, sel0, keys, module_energy)
        s1, e1, ok1, tie1 = _max_module(kept1, m1, energy, channel, kind, sm, mm, sel1, keys, module_energy)
        if not (ok0 and ok1):
            codes[k] = codes_map[2]
            continue
        if not (_min_ch(sel0, s0, channel, kind, min_ch) and _min_ch(sel1, s1, channel, kind, min_ch)):
            codes[k] = codes_map[3]
            continue
        top0, second0, nt0 = _time_hits(sel0, s0, energy, channel, kind)
        top1, second1, nt1 = _time_hits(sel1, s1, energy, channel, kind)
        dt = timestamp[top0] - timestamp[top1]
        if not abs(dt) < window:
            codes[k] = codes_map[4]
            continue
        record = first_record + k
        slab0 = _slab(top0, second0, nt0, channel, position, _bit(file_index, record, 0, seed_mix))
        slab1 = _slab(top1, second1, nt1, channel, position, _bit(file_index, record, 1, seed_mix))
        if slab0 < 0 or slab1 < 0:
            codes[k] = codes_map[5]
            continue
        row0, row1 = time_index[channel[top0]], time_index[channel[top1]]
        state0 = mu_state[row0, slab0] if row0 >= 0 else 0
        state1 = mu_state[row1, slab1] if row1 >= 0 else 0
        if state0 != 1:
            codes[k] = codes_map[6] if state0 == 0 else codes_map[7]
            continue
        if state1 != 1:
            codes[k] = codes_map[6] if state1 == 0 else codes_map[7]
            continue
        kev0 = 511.0 / mu[row0, slab0] * e0
        kev1 = 511.0 / mu[row1, slab1] * e1
        if not (kev0 > low and kev0 < high and kev1 > low and kev1 < high):
            codes[k] = codes_map[8]
            continue
        x0, y0 = _centroid(sel0, s0, energy, channel, kind, coord_x, coord_y)
        x1, y1 = _centroid(sel1, s1, energy, channel, kind, coord_x, coord_y)
        codes[k] = 0
        c0, c1 = channel[top0], channel[top1]
        out_i[n_out, 0] = record
        out_i[n_out, 1] = c0
        out_i[n_out, 2] = c1
        out_i[n_out, 3] = sm[c0]
        out_i[n_out, 4] = mm[c0]
        out_i[n_out, 5] = sm[c1]
        out_i[n_out, 6] = mm[c1]
        out_i[n_out, 7] = slab0
        out_i[n_out, 8] = slab1
        out_i[n_out, 9] = dt
        out_i[n_out, 10] = 1 if (tie0 or tie1) else 0
        out_f[n_out, 0] = kev0
        out_f[n_out, 1] = kev1
        out_f[n_out, 2] = x0
        out_f[n_out, 3] = y0
        out_f[n_out, 4] = x1
        out_f[n_out, 5] = y1
        n_out += 1
    return n_out


@dataclass(frozen=True)
class FeatureBatch:
    """Records passing every pre-geometry cut; ``codes`` cover every record read."""
    first_record: int
    codes: np.ndarray
    ints: np.ndarray
    floats: np.ndarray

    def column(self, name):
        if name in FEATURE_INT:
            return self.ints[:, FEATURE_INT.index(name)]
        return self.floats[:, FEATURE_FLOAT.index(name)]


def select_batch(batch, lookup, *, file_index, seed=0):
    """Compiled T4a selection up to the energy window/centroid for one ``LdatBatch``.

    Ties between equal-energy minimodules take the first seen in hit order
    (legacy uses Python set order) and are flagged in ``module_tie``.
    """
    if not 0 <= file_index < 2 ** 20 or batch.first_record + batch.count > 2 ** 42:
        raise ValueError("slab seed key out of range")
    count = batch.count
    codes = np.empty(count, np.int8)
    ints = np.empty((count, len(FEATURE_INT)), np.int64)
    floats = np.empty((count, len(FEATURE_FLOAT)))
    codes_map = np.array([_CODE[label] for label in _PRE_GEOMETRY], np.int8)
    low, high = lookup.energy_range
    seed_mix = np.uint64(_splitmix64(seed & 0xFFFFFFFFFFFFFFFF))
    n = _select(batch.n1, batch.n2, batch.first_hit, batch.timestamp, batch.energy, batch.channel,
                lookup.kind, lookup.sm, lookup.mm, lookup.coord_x, lookup.coord_y, lookup.position,
                lookup.time_index, lookup.mu, lookup.mu_state, lookup.min_ch, float(low), float(high),
                LEGACY_COINCIDENCE_WINDOW_PS, file_index, batch.first_record, seed_mix, codes_map,
                codes, ints, floats)
    return FeatureBatch(batch.first_record, codes, ints[:n].copy(), floats[:n].copy())


def compiled_batches(path, lookup, *, file_index, max_records=None, seed=0, block_bytes=None,
                     batch_records=None):
    """Yield one ``FeatureBatch`` per stream batch; memory bounded by the batch size."""
    options = {}
    if block_bytes is not None:
        options["block_bytes"] = block_bytes
    if batch_records is not None:
        options["batch_records"] = batch_records
    for batch in LdatStream(path, max_records=max_records, **options):
        yield select_batch(batch, lookup, file_index=file_index, seed=seed)


def compiled_legacy_outcomes(path, setup, lookup, *, file_index, max_records=None, seed=0, **options):
    """Compiled pre-geometry cuts plus the verified scalar legacy geometry: (record, label, features)."""
    for features in compiled_batches(path, lookup, file_index=file_index, max_records=max_records,
                                     seed=seed, **options):
        row = 0
        for k, code in enumerate(features.codes.tolist()):
            record = features.first_record + k
            if code:
                yield record, REJECTIONS[code - 1], None
                continue
            ints, floats = features.ints[row].tolist(), features.floats[row].tolist()
            row += 1
            start = np.array(legacy_local_to_global(floats[2], floats[3], setup.sm_center_map[ints[3]]))
            end = np.array(legacy_local_to_global(floats[4], floats[5], setup.sm_center_map[ints[5]]))
            label, tof1, tof2 = legacy_cylinder(start, end, setup.pipe_radius)
            if label:
                yield record, label, None
                continue
            yield record, ACCEPTED, LegacyFeatures(
                (ints[1], ints[2]), ((ints[3], ints[4]), (ints[5], ints[6])), (ints[7], ints[8]),
                (floats[0], floats[1]), ((floats[2], floats[3]), (floats[4], floats[5])),
                (tuple(map(float, start)), tuple(map(float, end))), ints[9], (tof1, tof2))


# --- T4c: production holdout, geometry stage and coverage -------------------
# Owner decision 2026-10-13: validation/test files skip their leading records up
# to ``records_validated`` of Manager calibrate run
# norm_10082026_r7_cal-P1-target_2026-10-11_1152, whose ``.encal`` photopeak
# factors used those prefixes. Train files keep every record.
RUN7_ENERGY_CALIBRATION_PREFIX = {3: 3_186_061, 5: 3_185_078, 8: 3_184_900, 11: 3_185_980, 14: 3_192_263}
HOLDOUT_EXCLUSIONS = ("energy calibration prefix", "file edge guard")
GEOMETRY_REJECTIONS = ("invalid endpoint", "no shell crossing", "opposition angle", "one shell segment")
PRODUCTION_LABELS = (ACCEPTED,) + HOLDOUT_EXCLUSIONS + _PRE_GEOMETRY + GEOMETRY_REJECTIONS
_HOLDOUT_CODE = {label: -1 - index for index, label in enumerate(HOLDOUT_EXCLUSIONS)}
PASSED_PRE_GEOMETRY = "passed pre-geometry"


def code_label(code):
    """Production ``FeatureBatch`` code: < 0 holdout exclusion, 0 passed, > 0 ``REJECTIONS``."""
    code = int(code)
    if code < 0:
        return HOLDOUT_EXCLUSIONS[-1 - code]
    return PASSED_PRE_GEOMETRY if code == 0 else REJECTIONS[code - 1]


@dataclass(frozen=True)
class Partition:
    """One file's frozen role; anchors (ps) are inclusive guarded bounds."""
    file_index: int
    role: str
    anchor_start_ps: int
    anchor_end_ps: int
    excluded_prefix_records: int

    @classmethod
    def from_holdout(cls, holdout, file_index, energy_calibration_prefix):
        """From ``dataset.freeze_holdout`` output; held-out files need a prefix bound."""
        if not holdout.get("qualified") or not holdout.get("roles_frozen_before_fitting"):
            raise ValueError("holdout is not qualified and frozen")
        matches = [p for p in holdout["partitions"] if p["index"] == file_index]
        if len(matches) != 1:
            raise ValueError(f"file {file_index}: no unique holdout partition")
        partition = matches[0]
        if partition["role"] not in ("train", "validation", "test") or None in (
                partition["anchor_start_ps"], partition["anchor_end_ps"]):
            raise ValueError(f"file {file_index}: invalid role or guarded interval")
        if partition["role"] == "train":
            prefix = 0
        elif int(energy_calibration_prefix.get(file_index, -1)) >= 0:
            prefix = int(energy_calibration_prefix[file_index])
        else:
            raise ValueError(f"held-out file {file_index}: energy-calibration prefix bound required")
        return cls(file_index, partition["role"], int(partition["anchor_start_ps"]),
                   int(partition["anchor_end_ps"]), prefix)


def holdout_codes(batch, partition):
    """Per-record holdout code: prefix first, then anchor outside the guarded interval.

    The anchor is the record's first hit (detector 1, else detector 2), as in
    ``dataset``; a record without hits has no anchor and is guarded out.
    """
    records = batch.first_record + np.arange(batch.count, dtype=np.int64)
    has_hits = (batch.n1.astype(np.int64) + batch.n2) > 0
    anchor = np.zeros(batch.count, np.int64)
    anchor[has_hits] = batch.timestamp[batch.first_hit[has_hits]]
    codes = np.zeros(batch.count, np.int8)
    codes[~has_hits | (anchor < partition.anchor_start_ps) | (anchor > partition.anchor_end_ps)] = \
        _HOLDOUT_CODE["file edge guard"]
    codes[records < partition.excluded_prefix_records] = _HOLDOUT_CODE["energy calibration prefix"]
    return codes


def production_batches(path, lookup, partition, *, max_records=None, seed=0, block_bytes=None,
                       batch_records=None):
    """Geometry-free features of one file's role partition; every record gets one code."""
    options = {}
    if block_bytes is not None:
        options["block_bytes"] = block_bytes
    if batch_records is not None:
        options["batch_records"] = batch_records
    for batch in LdatStream(path, max_records=max_records, **options):
        features = select_batch(batch, lookup, file_index=partition.file_index, seed=seed)
        excluded = holdout_codes(batch, partition)
        if excluded.any():
            keep = excluded[features.column("record") - batch.first_record] == 0
            features = FeatureBatch(features.first_record, np.where(excluded != 0, excluded, features.codes),
                                    features.ints[keep], features.floats[keep])
        yield features


@dataclass(frozen=True)
class GeometryHypothesis:
    """Declared production geometry: faceted frames, finite shell and 3D opposition angle."""
    name: str
    frames: dict
    source: AnnularSource
    opposition_tolerance_deg: float = float(LEGACY_ANGLE_ACCEPTANCE_DEG)

    @classmethod
    def faceted(cls, name, config_path, setup, *, axial_sign, transverse_sign, source,
                interaction_depth_mm=OWNER_INTERACTION_DEPTH_MM):
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        frames = build_faceted_detector_frames(config, setup.local_map, setup.sm_mM_map, setup.chtype_map,
                                               axial_sign=axial_sign, transverse_sign=transverse_sign,
                                               interaction_depth_mm=interaction_depth_mm)
        return cls(name, frames, source)

    @classmethod
    def selected(cls, config_path, setup, *, source=NORMALIZATION_PHANTOM):
        """Production nominal: faceted, 7 mm depth, owner/data-selected axial and transverse signs."""
        return cls.faceted(f"faceted transverse_sign {SELECTED_TRANSVERSE_SIGN:+d}", config_path, setup,
                           axial_sign=SELECTED_AXIAL_SIGN, transverse_sign=SELECTED_TRANSVERSE_SIGN, source=source)

    def tables(self):
        frames = list(self.frames.values())
        if not frames or not all(isinstance(frame, FacetedDetectorFrame) for frame in frames):
            raise GeometryError("geometry stage requires faceted detector frames")
        sketch, centers = frames[0].sketch, frames[0].map_centers_mm
        if any(frame.sketch != sketch or frame.map_centers_mm != centers for frame in frames):
            raise GeometryError("faceted frames must share one sketch and map minimodule centers")
        if not isinstance(self.source, AnnularSource) or not 0 <= self.opposition_tolerance_deg <= 180:
            raise GeometryError("explicit finite annular source and opposition tolerance required")
        size = max(self.frames) + 1
        valid = np.zeros(size, np.bool_)
        azimuth, axial_center, apothem = (np.full(size, np.nan) for _ in range(3))
        axial_sign, transverse_sign = np.zeros(size, np.int64), np.zeros(size, np.int64)
        for sm, frame in self.frames.items():
            valid[sm] = True
            azimuth[sm], axial_center[sm] = frame.cassette_azimuth_rad, frame.axial_center_mm
            apothem[sm] = frame.interaction_apothem_mm
            axial_sign[sm], transverse_sign[sm] = frame.axial_sign, frame.transverse_sign
        tile_rad = np.array([math.radians((q - 0.5) * sketch.tile_pitch_deg) for q in (0, 1)])
        source = self.source
        return (valid, azimuth, axial_center, axial_sign, transverse_sign, apothem,
                np.array(centers[0]), np.array(centers[1]), sketch.minimodule_width_mm / 2 + 1e-6,
                sketch.minimodule_offset_mm, sketch.row_pitch_mm, tile_rad,
                np.array(source.center_mm), np.array(source.axis), source.inner_radius_mm,
                source.outer_radius_mm, source.axial_min_mm, source.axial_max_mm,
                math.cos(math.radians(180 - self.opposition_tolerance_deg)) + 8 * np.finfo(np.float64).eps)


@numba.njit(cache=True)
def _nearest(value, centers, limit):
    """FacetedDetectorFrame._minimodule: first nearest center, inclusive edge."""
    k, best = 0, abs(value - centers[0])
    for q in range(1, centers.size):
        distance = abs(value - centers[q])
        if distance < best:
            k, best = q, distance
    return k, value - centers[k], best <= limit


@numba.njit(cache=True)
def _endpoint(sm, x, y, valid, azimuth, axial_center, axial_sign, transverse_sign, apothem,
              centers_x, centers_y, limit, offset, row_pitch, tile_rad, out):
    """FacetedDetectorFrame.to_global into ``out``; False when undefined."""
    if sm < 0 or sm >= valid.size or not valid[sm]:
        return False
    kx, dx, ok_x = _nearest(x, centers_x, limit)
    ky, dy, ok_y = _nearest(y, centers_y, limit)
    if not (ok_x and ok_y):
        return False
    row = (kx // 2 - 0.5) * row_pitch
    axial = row + (kx % 2 - 0.5) * 2 * offset + dx
    sign = transverse_sign[sm]
    phi = azimuth[sm] + sign * tile_rad[ky // 2]
    along = sign * ((ky % 2 - 0.5) * 2 * offset + dy)
    c, s = math.cos(phi), math.sin(phi)
    r = apothem[sm]
    out[0] = r * c - along * s
    out[1] = r * s + along * c
    out[2] = axial_center[sm] + axial_sign[sm] * axial
    return True


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
def _geometry(sm1, x1, y1, sm2, x2, y2, valid, azimuth, axial_center, axial_sign, transverse_sign,
              apothem, centers_x, centers_y, limit, offset, row_pitch, tile_rad, center, axis, r_in,
              r_out, z_min, z_max, cos_limit, codes, endpoints, segments, fractions, tof):
    for k in range(sm1.size):
        p, q = endpoints[k, :3], endpoints[k, 3:]
        if not (_endpoint(sm1[k], x1[k], y1[k], valid, azimuth, axial_center, axial_sign, transverse_sign,
                          apothem, centers_x, centers_y, limit, offset, row_pitch, tile_rad, p)
                and _endpoint(sm2[k], x2[k], y2[k], valid, azimuth, axial_center, axial_sign,
                              transverse_sign, apothem, centers_x, centers_y, limit, offset, row_pitch,
                              tile_rad, q)):
            codes[k] = 1
            continue
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


def apply_geometry(features, hypothesis):
    """Run one declared hypothesis on stored feature rows; reads no LDAT.

    ``features`` is any object with ``column(name)`` (``FeatureBatch`` or cached blocks).
    """
    column = features.column
    count = len(column("sm_1"))
    codes = np.zeros(count, np.int64)
    endpoints = np.full((count, 6), np.nan)
    segments = np.zeros(count, np.int64)
    fractions, tof = np.full((count, 4), np.nan), np.full((count, 4), np.nan)
    _geometry(column("sm_1"), column("x_1"), column("y_1"), column("sm_2"), column("x_2"), column("y_2"),
              *hypothesis.tables(), codes, endpoints, segments, fractions, tof)
    if (segments > 2).any():
        raise GeometryError("annular shell produced more than two segments")
    return GeometryResult(hypothesis.name, codes, endpoints, segments, fractions, tof)


def time_channel_classes(setup, down_regions):
    """Mapped TIME channels {channel: (sm, mm)}, operational and down channel lists.

    Operational excludes configured unpopulated minimodules and the declared
    acquisition-specific down regions.
    """
    down = {(int(sm), int(mm)) for sm, mm in down_regions}
    mapped = {tuple(key) for key in setup.sm_mM_map.values()}
    if not down <= mapped:
        raise ValueError(f"down regions not in the selected map: {sorted(down - mapped)}")
    time_channels = {ch: tuple(setup.sm_mM_map[ch]) for ch in setup.chtype_map
                     if ChannelType.TIME in setup.chtype_map[ch] and ch in setup.sm_mM_map}
    operational = sorted(ch for ch, key in time_channels.items() if key not in setup.unpopulated and key not in down)
    return time_channels, operational, sorted(ch for ch, key in time_channels.items() if key in down)


def time_channel_coverage(endpoint_counts, setup, down_regions):
    """Accepted endpoints per operational TIME channel (each accepted record adds two).

    Operational excludes configured unpopulated minimodules and the declared
    acquisition-specific down regions; endpoints in either are reported apart.
    """
    down = {(int(sm), int(mm)) for sm, mm in down_regions}
    time_channels, operational, _ = time_channel_classes(setup, down_regions)
    unknown = sorted(ch for ch, n in endpoint_counts.items() if n and ch not in time_channels)
    if unknown:
        raise ValueError(f"endpoint counts on non-TIME or unmapped channels: {unknown[:5]}")
    per_channel = {ch: int(endpoint_counts.get(ch, 0)) for ch in operational}

    def in_regions(regions):
        return sum(int(n) for ch, n in endpoint_counts.items() if n and time_channels[ch] in regions)

    return {
        "definition": "accepted coincidences add one endpoint to each selected TIME channel",
        "operational_time_channels": len(operational),
        "unpopulated_time_channels": sum(key in setup.unpopulated for key in time_channels.values()),
        "down_regions": sorted(down),
        "down_time_channels": sum(key in down for key in time_channels.values()),
        "endpoints_per_operational_channel": per_channel,
        "operational_without_endpoints": [ch for ch, n in per_channel.items() if n == 0],
        "down_region_endpoints": in_regions(down),
        "unpopulated_endpoints": in_regions(setup.unpopulated),
    }
