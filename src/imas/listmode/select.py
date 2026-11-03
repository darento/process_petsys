"""IMAS listmode side selection (spec 015 T1, FR-2, FR-3, FR-5 inputs).

Per compact coincidence record, the first failing cut is counted (codes 1..7 = ``REJECTIONS``; 0 accepted):

1. a kept hit on an unmapped channel;
2. either side fails ``min_ch`` (sum rows/cols: ENERGY hits >= ``min_ch`` and fewer than all hits);
3. no minimodule with positive ENERGY sum (highest sum; first seen on ties, counted);
4. the selected minimodule fails ``min_ch``;
5./6. the top TIME channel (highest charge = the slab) of either side has no ``SLAB_EN_MAP`` mu / (both
   present) an invalid one;
7. either side's keV (511 / mu x minimodule ENERGY sum) outside the open ``energy_range``;
8. the SuperModule pair is not in the pair file in either order.

Hits below ``en_min_ch`` are dropped before every cut. Sides stay in LDAT order; ``swapped`` says the pair file
lists the pair as (side 2, side 1), and ``records.build`` applies it. Each accepted side carries its 8 ENERGY
charges by map row (zero where a row did not fire), the spec 012 walk ingredients and the spec 014 candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numba
import numpy as np
import pandas as pd

from src.imas.skew.select import _candidates, _keep, _walk_ingredients
from src.imas.skew.setup import MU_INVALID, MU_MISSING, MU_VALID, ImasLookup, read_mu
from src.imas.skew.timestamp import CANDIDATE_DTYPE, SLOTS
from src.ldat_stream import LdatStream
from src.mapping_generator import ChannelType
from src.timing.kernels import _max_module, _min_ch, _time_hits

REJECTIONS = (
    "unmapped channel",
    "side min channels",
    "no positive minimodule energy",
    "minimodule min channels",
    "missing energy calibration",
    "invalid energy calibration",
    "energy window",
    "pair",
)
ENERGY_ROWS = 8
# per accepted record, side s = 1, 2 in LDAT order
INTS = ("record", "time_channel_1", "time_channel_2", "sm_1", "mm_1", "sm_2", "mm_2", "pair", "swapped",
        "dt_ps", "t_min", "module_tie")
FLOATS = ("energy_kev_1", "energy_kev_2")
WALK = ("q_top_1", "q_time_1", "e_raw_1", "doi_ratio_1", "q_top_2", "q_time_2", "e_raw_2", "doi_ratio_2")


def read_pairs(path):
    """``p s0 s1`` tab-separated pair file as {(s0, s1): p} (``read_pair_map`` of the legacy script)."""
    table = pd.read_csv(path, sep="\t", names=["p", "s0", "s1"])
    if table.isna().any().any():
        raise ValueError(f"{path}: expected three tab-separated integer columns p s0 s1")
    pairs = {}
    for p, s0, s1 in zip(table["p"].astype(int), table["s0"].astype(int), table["s1"].astype(int)):
        pairs[s0, s1] = p          # a repeated key keeps the last row, as the legacy dict does
    return pairs


def _mu_arrays(size, mu, time_channels):
    values, state = np.full(size, np.nan), np.full(size, MU_MISSING, np.int8)
    for ch in time_channels:
        value = mu.get(int(ch))
        if value is not None:
            valid = math.isfinite(value) and value > 0
            state[ch] = MU_VALID if valid else MU_INVALID
            values[ch] = value if valid else np.nan
    return values, state


@dataclass(frozen=True)
class LmLookup:
    """Dense tables for the compiled listmode selection (index = global channel ID / SM id)."""
    base: ImasLookup            # kind, sm, mm, coord_x, coord_y, slab; its mu is unused
    energy_row: np.ndarray      # ENERGY channel row 0-7 (local_map[ch][2]); -1 otherwise
    time_y: np.ndarray          # TIME channel local y (mm), the slab's axial origin for the NN y
    mu: np.ndarray              # SLAB_EN_MAP mu per TIME channel (NaN unless valid)
    mu_state: np.ndarray
    walk_mu: np.ndarray         # walk-calibration mu (NaN unless valid or without walk)
    pair: np.ndarray            # (n_sm, n_sm) pair id, -1 if absent in that order
    kev_window: tuple
    min_ch: int
    en_min_ch: float

    @classmethod
    def build(cls, setup, slab_mu, pairs, energy_range, walk_mu=None):
        """``setup``: ``ImasSetup``; ``slab_mu``/``walk_mu``: {TIME channel: mu}; ``pairs``: {(s0, s1): p}."""
        base = setup.lookup()
        size = base.kind.size
        energy_row, time_y = np.full(size, -1, np.int8), np.full(size, np.nan)
        for ch, types in setup.chtype_map.items():
            if ChannelType.ENERGY in types:
                row = int(setup.local_map[ch][2])
                if not 0 <= row < ENERGY_ROWS:
                    raise ValueError(f"ENERGY channel {ch}: map row {row} outside 0-{ENERGY_ROWS - 1}")
                energy_row[ch] = row
            else:
                time_y[ch] = float(setup.local_map[ch][1])
        times = setup.time_channels
        mu, mu_state = _mu_arrays(size, slab_mu, times)
        wmu = _mu_arrays(size, walk_mu, times)[0] if walk_mu is not None else np.full(size, np.nan)
        n_sm = int(base.sm.max()) + 1
        pair = np.full((n_sm, n_sm), -1, np.int32)
        for (s0, s1), p in pairs.items():
            if not (0 <= s0 < n_sm and 0 <= s1 < n_sm) or not 0 <= p < 65536:
                raise ValueError(f"pair {p} ({s0}, {s1}): SM outside 0-{n_sm - 1} or id outside uint16")
            pair[s0, s1] = p
        low, high = map(float, energy_range)
        return cls(base, energy_row, time_y, mu, mu_state, wmu, pair, (low, high), base.min_ch, base.en_min_ch)


@numba.njit(cache=True)
def _rows(hits, m, energy, channel, kind, energy_row, out, row, side):
    for r in range(out.shape[2]):
        out[row, side, r] = 0.0
    for j in range(m):
        i = hits[j]
        if kind[channel[i]] == 2:
            out[row, side, energy_row[channel[i]]] = energy[i]


@numba.njit(cache=True)
def _lm_select(n1, n2, first, timestamp, energy, channel, kind, sm, mm, energy_row, mu, mu_state, pair, min_ch,
               floor, low, high, first_record, codes, out_i, out_f, out_w, out_e, out_c, out_l, out_q, refused):
    kept0, kept1 = np.empty(512, np.int64), np.empty(512, np.int64)
    sel0, sel1 = np.empty(512, np.int64), np.empty(512, np.int64)
    keys, module_energy = np.empty(512, np.int64), np.empty(512)
    n_sm = pair.shape[0]
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
        c0, c1 = channel[top0], channel[top1]
        if mu_state[c0] == 0 or mu_state[c1] == 0:
            codes[k] = 5
            continue
        if mu_state[c0] != 1 or mu_state[c1] != 1:
            codes[k] = 6
            continue
        kev0 = 511.0 / mu[c0] * e0
        kev1 = 511.0 / mu[c1] * e1
        if not (kev0 > low and kev0 < high and kev1 > low and kev1 < high):
            codes[k] = 7
            continue
        a, b = sm[c0], sm[c1]
        p, swapped = -1, 0
        if a < n_sm and b < n_sm:
            p = pair[a, b]
            if p < 0:
                p = pair[b, a]
                swapped = 1
        if p < 0:
            codes[k] = 8
            continue
        codes[k] = 0
        t0, t1 = timestamp[top0], timestamp[top1]
        out_i[n_out, 0] = first_record + k
        out_i[n_out, 1] = c0
        out_i[n_out, 2] = c1
        out_i[n_out, 3] = sm[c0]
        out_i[n_out, 4] = mm[c0]
        out_i[n_out, 5] = sm[c1]
        out_i[n_out, 6] = mm[c1]
        out_i[n_out, 7] = p
        out_i[n_out, 8] = swapped
        out_i[n_out, 9] = t0 - t1
        out_i[n_out, 10] = min(t0, t1)
        out_i[n_out, 11] = 1 if (tie0 or tie1) else 0
        out_f[n_out, 0] = kev0
        out_f[n_out, 1] = kev1
        q0, qt0, r0 = _walk_ingredients(sel0, s0, energy, channel, kind, top0)
        q1, qt1, r1 = _walk_ingredients(sel1, s1, energy, channel, kind, top1)
        out_w[n_out, 0], out_w[n_out, 1], out_w[n_out, 2], out_w[n_out, 3] = q0, qt0, e0, r0
        out_w[n_out, 4], out_w[n_out, 5], out_w[n_out, 6], out_w[n_out, 7] = q1, qt1, e1, r1
        _rows(sel0, s0, energy, channel, kind, energy_row, out_e, n_out, 0)
        _rows(sel1, s1, energy, channel, kind, energy_row, out_e, n_out, 1)
        if not (_candidates(sel0, s0, timestamp, energy, channel, kind, top0, n_out, 0, out_c, out_l, out_q)
                and _candidates(sel1, s1, timestamp, energy, channel, kind, top1, n_out, 1, out_c, out_l, out_q)):
            refused[0] = first_record + k
            return n_out
        n_out += 1
    return n_out


@dataclass(frozen=True)
class LmFeatures:
    """Accepted records of one ``LdatBatch``: a code per record and per accepted record its columns."""
    first_record: int
    codes: np.ndarray           # int8 per record (0 accepted, 1..8 ``REJECTIONS``)
    ints: np.ndarray            # (n, len(INTS)) int64
    floats: np.ndarray          # (n, len(FLOATS))
    walk: np.ndarray            # (n, len(WALK))
    energy_rows: np.ndarray     # (n, 2, 8) float32
    candidates: np.ndarray      # (n, 2, SLOTS) CANDIDATE_DTYPE

    def column(self, name):
        if name in INTS:
            return self.ints[:, INTS.index(name)]
        if name in FLOATS:
            return self.floats[:, FLOATS.index(name)]
        return self.walk[:, WALK.index(name)]

    def __len__(self):
        return len(self.ints)


def select_batch(batch, lookup):
    """Compiled listmode selection of one ``LdatBatch``."""
    count = batch.count
    codes = np.empty(count, np.int8)
    ints = np.empty((count, len(INTS)), np.int64)
    floats = np.empty((count, len(FLOATS)))
    walk = np.empty((count, len(WALK)))
    rows = np.empty((count, 2, ENERGY_ROWS), np.float32)
    out_c = np.empty((count, 2, SLOTS), np.int32)
    out_l = np.empty((count, 2, SLOTS), np.int32)
    out_q = np.empty((count, 2, SLOTS), np.float32)
    refused = np.full(1, -1, np.int64)
    base = lookup.base
    low, high = lookup.kev_window
    n = _lm_select(batch.n1, batch.n2, batch.first_hit, batch.timestamp, batch.energy, batch.channel, base.kind,
                   base.sm, base.mm, lookup.energy_row, lookup.mu, lookup.mu_state, lookup.pair, lookup.min_ch,
                   float(lookup.en_min_ch), low, high, batch.first_record, codes, ints, floats, walk, rows, out_c,
                   out_l, out_q, refused)
    if refused[0] >= 0:
        raise ValueError(f"record {int(refused[0])}: more than {SLOTS} other TIME hits in a selected minimodule, "
                         "or a TIME-hit lag outside int32 ps")
    candidates = np.empty((n, 2, SLOTS), CANDIDATE_DTYPE)
    candidates["channel"], candidates["lag_ps"], candidates["charge"] = out_c[:n], out_l[:n], out_q[:n]
    return LmFeatures(batch.first_record, codes, ints[:n].copy(), floats[:n].copy(), walk[:n].copy(),
                      rows[:n].copy(), candidates)


def count_codes(codes):
    """{label: records} of one batch's codes, accepted included."""
    out = {}
    values, found = np.unique(codes, return_counts=True)
    for code, n in zip(values.tolist(), found.tolist()):
        label = "accepted" if code == 0 else REJECTIONS[code - 1]
        out[label] = out.get(label, 0) + n
    return out


def feature_batches(path, lookup, *, max_records=None, batch_records=None):
    """``(LdatBatch, LmFeatures)`` of one file, batch by batch (bounded memory)."""
    options = {} if batch_records is None else {"batch_records": batch_records}
    for batch in LdatStream(path, max_records=max_records, **options):
        yield batch, select_batch(batch, lookup)


__all__ = ["REJECTIONS", "INTS", "FLOATS", "WALK", "ENERGY_ROWS", "LmLookup", "LmFeatures", "read_pairs",
           "read_mu", "select_batch", "count_codes", "feature_batches"]
