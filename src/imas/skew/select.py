"""IMAS side selection for skew calibration (spec 011 T4, FR-3).

Per record, first failing cut counted (codes 1..8 = ``REJECTIONS``; 0 passes):

1. a kept hit on an unmapped channel;
2. either side fails ``min_ch`` (ENERGY hits >= ``min_ch`` and at least one TIME hit);
3. no minimodule with positive ENERGY sum (highest sum per side; first seen on ties, flagged);
4. the selected minimodule fails ``min_ch``;
5. ``|t_top1 - t_top2|`` not below the coincidence window (``t_top``: highest-charge TIME hit);
6./7. that TIME channel has no / an invalid ``mu``;
8. either side's minimodule ENERGY sum in keV outside the open keV window.

The slab is the top TIME channel itself (one slab per TIME channel); ``slab_*``
is its index 0-7 in the minimodule. Positions: transaxial ``x`` = the slab
centre (top TIME channel's map x), axial ``y`` = squared-charge-weighted
ENERGY-channel centroid, both local mm.

Production mode drops hits below ``en_min_ch`` before selection (as the LDAT
Inspector's compact reader does), uses the declared window (10 ns) and keV
window (408-613). Reference mode reproduces ``scripts_imas/
imas_skew_cal_bigdata_optimized.py``: hits with energy >= 0, 15 ns window and
``x`` from the TIME-channel centroid. ``calibration`` stops after cut 5 and
returns raw ENERGY sums for the energy calibration (FR-4).

Spec 014 FR-2: every passing record also carries, per side, the other TIME hits of the selected minimodule
(kept hits, so above the production floor) as ``timestamp.CANDIDATE_DTYPE`` slots sorted by raw lag to the
top (``FeatureBatch.candidates``, shape (records, 2, ``SLOTS``); channel -1 marks an empty slot). More
candidates than slots, or a lag outside int32, is refused, never truncated or wrapped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from src.ldat_stream import LdatStream
from src.timing.kernels import (ACCEPTED, FEATURE_FLOAT, FEATURE_INT, HOLDOUT_EXCLUSIONS,
                                LEGACY_COINCIDENCE_WINDOW_PS, WALK_FEATURES, FeatureBatch, _centroid,
                                _max_module, _min_ch, _time_hits, holdout_codes)
from src.imas.skew.timestamp import CANDIDATE_DTYPE, SLOTS
from src.timing.shell import GEOMETRY_REJECTIONS

REJECTIONS = (
    "unmapped channel",
    "side min channels",
    "no positive minimodule energy",
    "minimodule min channels",
    "coincidence window",
    "missing energy calibration",
    "invalid energy calibration",
    "energy window",
)
PASSED_PRE_GEOMETRY = "passed pre-geometry"
PRODUCTION_LABELS = (ACCEPTED,) + HOLDOUT_EXCLUSIONS + REJECTIONS + GEOMETRY_REJECTIONS
COINCIDENCE_WINDOW_PS = 10_000          # acquisition window (owner)
KEV_WINDOW = (408.0, 613.0)             # job and CTR (owner, spec 011 clarify 4)


@dataclass(frozen=True)
class Selection:
    """Declared cuts. ``energy_floor`` None keeps hits with energy >= 0 (legacy reader)."""
    window_ps: int = COINCIDENCE_WINDOW_PS
    kev_window: tuple = KEV_WINDOW
    energy_floor: float | None = None   # production: the config's en_min_ch
    x_from_time_centroid: bool = False

    @classmethod
    def production(cls, lookup, *, window_ps=COINCIDENCE_WINDOW_PS, kev_window=KEV_WINDOW):
        return cls(int(window_ps), tuple(map(float, kev_window)), float(lookup.en_min_ch), False)

    @classmethod
    def reference(cls, *, kev_window=KEV_WINDOW):
        """The legacy script's selection (its ``energy_range`` comes from the config it ran with)."""
        return cls(LEGACY_COINCIDENCE_WINDOW_PS, tuple(map(float, kev_window)), None, True)

    def describe(self):
        return {"window_ps": self.window_ps, "kev_window": list(self.kev_window),
                "hit_energy_floor": "energy >= 0 (legacy reader)" if self.energy_floor is None
                else f"energy >= {self.energy_floor} a.u. (en_min_ch)",
                "transaxial_position": "TIME-channel centroid (legacy)" if self.x_from_time_centroid
                else "top TIME channel (slab centre)"}


@numba.njit(cache=True)
def _keep(start, n, energy, channel, kind, floor, out):
    """Hits with energy >= ``floor`` into ``out``; second value flags an unmapped kept hit."""
    m, unmapped = 0, False
    for i in range(start, start + n):
        if energy[i] >= floor:
            out[m] = i
            m += 1
            c = channel[i]
            if c < 0 or c >= kind.size or kind[c] == 0:
                unmapped = True
    return m, unmapped


@numba.njit(cache=True)
def _walk_ingredients(hits, m, energy, channel, kind, top):
    """Timestamped TIME-hit charge, summed TIME charge and DOI ratio (ENERGY sum / maximum, 0 without)."""
    q_time = e_sum = e_max = 0.0
    for j in range(m):
        i = hits[j]
        if kind[channel[i]] == 1:
            q_time += energy[i]
        else:
            e_sum += energy[i]
            e_max = max(e_max, np.float64(energy[i]))
    return np.float64(energy[top]), q_time, e_sum / e_max if e_max != 0 else 0.0


@numba.njit(cache=True)
def _candidates(hits, m, timestamp, energy, channel, kind, top, row, side, out_c, out_l, out_q):
    """The other TIME hits of ``hits`` into slot row ``row``/``side`` by raw lag to ``top`` (stable); False when
    they exceed the slots or a lag leaves int32."""
    k = 0
    for j in range(m):
        i = hits[j]
        if i == top or kind[channel[i]] != 1:
            continue
        lag = timestamp[i] - timestamp[top]
        if k >= out_c.shape[2] or lag < -2147483648 or lag > 2147483647:
            return False
        p = k
        while p > 0 and out_l[row, side, p - 1] > lag:
            out_c[row, side, p], out_l[row, side, p], out_q[row, side, p] = (
                out_c[row, side, p - 1], out_l[row, side, p - 1], out_q[row, side, p - 1])
            p -= 1
        out_c[row, side, p], out_l[row, side, p], out_q[row, side, p] = channel[i], lag, energy[i]
        k += 1
    for p in range(k, out_c.shape[2]):
        out_c[row, side, p], out_l[row, side, p], out_q[row, side, p] = -1, 0, 0.0
    return True


@numba.njit(cache=True)
def _select(n1, n2, first, timestamp, energy, channel, kind, sm, mm, coord_x, coord_y, slab, mu, mu_state,
            min_ch, floor, low, high, window, x_centroid, calibration, first_record, codes, out_i, out_f, out_w,
            out_c, out_l, out_q, refused):
    kept0, kept1 = np.empty(512, np.int64), np.empty(512, np.int64)
    sel0, sel1 = np.empty(512, np.int64), np.empty(512, np.int64)
    keys, module_energy = np.empty(512, np.int64), np.empty(512)
    n_out = 0
    for k in range(n1.size):
        m0, bad0 = _keep(first[k], n1[k], energy, channel, kind, floor, kept0)
        m1, bad1 = _keep(first[k] + n1[k], n2[k], energy, channel, kind, floor, kept1)
        if bad0 or bad1:
            codes[k] = 1
            continue
        if not (_min_ch(kept0, m0, channel, kind, min_ch) and _min_ch(kept1, m1, channel, kind, min_ch)):
            codes[k] = 2
            continue
        s0, e0, ok0, tie0 = _max_module(kept0, m0, energy, channel, kind, sm, mm, sel0, keys, module_energy)
        s1, e1, ok1, tie1 = _max_module(kept1, m1, energy, channel, kind, sm, mm, sel1, keys, module_energy)
        if not (ok0 and ok1):
            codes[k] = 3
            continue
        if not (_min_ch(sel0, s0, channel, kind, min_ch) and _min_ch(sel1, s1, channel, kind, min_ch)):
            codes[k] = 4
            continue
        top0, _, _ = _time_hits(sel0, s0, energy, channel, kind)
        top1, _, _ = _time_hits(sel1, s1, energy, channel, kind)
        dt = timestamp[top0] - timestamp[top1]
        if not abs(dt) < window:
            codes[k] = 5
            continue
        c0, c1 = channel[top0], channel[top1]
        kev0 = kev1 = np.nan
        if not calibration:
            if mu_state[c0] != 1:
                codes[k] = 6 if mu_state[c0] == 0 else 7
                continue
            if mu_state[c1] != 1:
                codes[k] = 6 if mu_state[c1] == 0 else 7
                continue
            kev0 = 511.0 / mu[c0] * e0
            kev1 = 511.0 / mu[c1] * e1
            if not (kev0 > low and kev0 < high and kev1 > low and kev1 < high):
                codes[k] = 8
                continue
        xc0, y0 = _centroid(sel0, s0, energy, channel, kind, coord_x, coord_y)
        xc1, y1 = _centroid(sel1, s1, energy, channel, kind, coord_x, coord_y)
        codes[k] = 0
        out_i[n_out, 0] = first_record + k
        out_i[n_out, 1] = c0
        out_i[n_out, 2] = c1
        out_i[n_out, 3] = sm[c0]
        out_i[n_out, 4] = mm[c0]
        out_i[n_out, 5] = sm[c1]
        out_i[n_out, 6] = mm[c1]
        out_i[n_out, 7] = slab[c0]
        out_i[n_out, 8] = slab[c1]
        out_i[n_out, 9] = dt
        out_i[n_out, 10] = 1 if (tie0 or tie1) else 0
        out_f[n_out, 0] = kev0
        out_f[n_out, 1] = kev1
        out_f[n_out, 2] = xc0 if x_centroid else coord_x[c0]
        out_f[n_out, 3] = y0
        out_f[n_out, 4] = xc1 if x_centroid else coord_x[c1]
        out_f[n_out, 5] = y1
        q0, t0, r0 = _walk_ingredients(sel0, s0, energy, channel, kind, top0)
        q1, t1, r1 = _walk_ingredients(sel1, s1, energy, channel, kind, top1)
        out_w[n_out, 0], out_w[n_out, 1], out_w[n_out, 2], out_w[n_out, 3] = q0, t0, e0, r0
        out_w[n_out, 4], out_w[n_out, 5], out_w[n_out, 6], out_w[n_out, 7] = q1, t1, e1, r1
        if not (_candidates(sel0, s0, timestamp, energy, channel, kind, top0, n_out, 0, out_c, out_l, out_q)
                and _candidates(sel1, s1, timestamp, energy, channel, kind, top1, n_out, 1, out_c, out_l, out_q)):
            refused[0] = first_record + k
            return n_out
        n_out += 1
    return n_out


def select_batch(batch, lookup, selection, *, calibration=False):
    """Compiled selection of one ``LdatBatch``: a code per record and features of the passing ones.

    With ``calibration`` the cuts stop after the coincidence window and keV is NaN.
    """
    count = batch.count
    codes = np.empty(count, np.int8)
    ints = np.empty((count, len(FEATURE_INT)), np.int64)
    floats = np.empty((count, len(FEATURE_FLOAT)))
    walk = np.empty((count, len(WALK_FEATURES)))
    out_c = np.empty((count, 2, SLOTS), np.int32)
    out_l = np.empty((count, 2, SLOTS), np.int32)
    out_q = np.empty((count, 2, SLOTS), np.float32)
    refused = np.full(1, -1, np.int64)
    floor = 0.0 if selection.energy_floor is None else selection.energy_floor
    low, high = selection.kev_window
    n = _select(batch.n1, batch.n2, batch.first_hit, batch.timestamp, batch.energy, batch.channel,
                lookup.kind, lookup.sm, lookup.mm, lookup.coord_x, lookup.coord_y, lookup.slab, lookup.mu,
                lookup.mu_state, lookup.min_ch, float(floor), float(low), float(high), int(selection.window_ps),
                bool(selection.x_from_time_centroid), bool(calibration), batch.first_record, codes, ints, floats,
                walk, out_c, out_l, out_q, refused)
    if refused[0] >= 0:
        raise ValueError(f"record {int(refused[0])}: more than {SLOTS} other TIME hits in a selected minimodule, "
                         "or a TIME-hit lag outside int32 ps")
    candidates = np.empty((n, 2, SLOTS), CANDIDATE_DTYPE)
    candidates["channel"], candidates["lag_ps"], candidates["charge"] = out_c[:n], out_l[:n], out_q[:n]
    return FeatureBatch(batch.first_record, codes, ints[:n].copy(), floats[:n].copy(), walk[:n].copy(), candidates)


def code_label(code):
    """Feature code: < 0 holdout exclusion, 0 passed, > 0 ``REJECTIONS``."""
    code = int(code)
    if code < 0:
        return HOLDOUT_EXCLUSIONS[-1 - code]
    return PASSED_PRE_GEOMETRY if code == 0 else REJECTIONS[code - 1]


def feature_batches(path, lookup, selection, partition=None, *, max_records=None, calibration=False,
                    batch_records=None, on_block=None):
    """Features of one file, batch by batch; with ``partition`` its holdout codes replace the selection's.

    ``on_block(raw)`` sees every block read (e.g. to hash the input).
    """
    options = {} if batch_records is None else {"batch_records": batch_records}
    for batch in LdatStream(path, max_records=max_records, on_block=on_block, **options):
        features = select_batch(batch, lookup, selection, calibration=calibration)
        if partition is not None:
            excluded = holdout_codes(batch, partition)
            if excluded.any():
                keep = excluded[features.column("record") - batch.first_record] == 0
                features = FeatureBatch(features.first_record, np.where(excluded != 0, excluded, features.codes),
                                        features.ints[keep], features.floats[keep], features.walk[keep],
                                        features.candidates[keep])
        yield features


def count_codes(path, lookup, selection, *, max_records=None):
    """{label: records} by first failing cut over a file or its prefix."""
    counts = {}
    for features in feature_batches(path, lookup, selection, max_records=max_records):
        codes, found = np.unique(features.codes, return_counts=True)
        for code, n in zip(codes.tolist(), found.tolist()):
            counts[code_label(code)] = counts.get(code_label(code), 0) + n
    return counts
