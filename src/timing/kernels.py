"""Scanner-neutral side-selection kernels, feature rows and file roles (spec 011 T1, FR-1).

Moved unchanged from ``src.cornell.skew.ingest`` (spec 008 T4b/T4c). Channel
``kind`` is 1 for TIME and 2 for ENERGY; scanner adapters build the dense
lookup tables and call these kernels in their own selection order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from src.ldat_inspector.fastread import _pow

LEGACY_COINCIDENCE_WINDOW_PS = 15000
ACCEPTED = "accepted"
_KIND_TIME, _KIND_ENERGY = 1, 2
FEATURE_INT = ("record", "time_channel_1", "time_channel_2", "sm_1", "mm_1", "sm_2", "mm_2",
               "slab_1", "slab_2", "dt_ps", "module_tie")
FEATURE_FLOAT = ("energy_kev_1", "energy_kev_2", "x_1", "y_1", "x_2", "y_2")
# spec 009 FR-1 raw walk ingredients per side: timestamped TIME-hit charge, summed TIME charge,
# summed ENERGY charge (the keV numerator) and listmode's DOI ratio (ENERGY sum / maximum), a.u.
WALK_FEATURES = ("q_top_1", "q_time_1", "e_raw_1", "doi_ratio_1", "q_top_2", "q_time_2", "e_raw_2", "doi_ratio_2")


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


@dataclass(frozen=True)
class FeatureBatch:
    """Records passing every pre-geometry cut; ``codes`` cover every record read."""
    first_record: int
    codes: np.ndarray
    ints: np.ndarray
    floats: np.ndarray
    walk: np.ndarray | None = None      # WALK_FEATURES columns (spec 009), None where not computed

    def column(self, name):
        if name in FEATURE_INT:
            return self.ints[:, FEATURE_INT.index(name)]
        if name in WALK_FEATURES:
            if self.walk is None:
                raise KeyError(f"{name}: walk ingredients were not computed for these features")
            return self.walk[:, WALK_FEATURES.index(name)]
        return self.floats[:, FEATURE_FLOAT.index(name)]


HOLDOUT_EXCLUSIONS = ("energy calibration prefix", "file edge guard")
_HOLDOUT_CODE = {label: -1 - index for index, label in enumerate(HOLDOUT_EXCLUSIONS)}


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
