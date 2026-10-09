"""Fast LDAT coincidence reader for LDATInspector (spec 002, FR-1).

``process_file_fast`` returns the same ``FileResult`` as
``src.ldat_inspector.engine.process_file_reference``: identical accepted-pair and
rejection counts, channel counters and retained columns, compared by
``tests/test_ldat_scale.py``. It follows the reference's steps and their
order:

1. per-channel energy cut (``>= min_channel_energy``);
2. ``filter_min_ch`` on det1, then det2 (det2 is not checked if det1 fails);
3. per side ``get_maxEnergy_sm_mM`` (strict ``>`` against a running maximum
   starting at 0), minimum channels in the selected minimodule, the
   maximum-energy time channel, the Cornell slab, the centroid (TIME -> X
   power 1, ENERGY -> Y power 2, offset 1e-5), the DOI (sum / max of energy
   channels) and the finite / positive check.

Known differences: exact ties between minimodule energies are broken by hit
order instead of Python set order (counted in ``max_energy_ties``), and the
Cornell one-time-channel coin flip uses a per-file seeded generator, so only
the slab pair of those sides matches the reference. The same holds for the
FR-19 tie / no-neighbour coin flip under ``slab_rule="recover_non_adjacent"``.
Maps without sum-rows/cols readout fall back to the reference reader.
"""

from __future__ import annotations

from collections import Counter
import ctypes
import ctypes.util
from dataclasses import replace
from pathlib import Path
import sys

import llvmlite.binding
import numba
import numpy as np

from .engine import FileResult, Settings, SideTable, load_setup
from src.mapping_generator import ChannelType

CHUNK_PAIRS = 250_000


class Cancelled(Exception):
    """The operator cancelled processing; the file contributes nothing."""


# Set in each worker process by init_worker (ProcessPoolExecutor initializer).
_worker_cancel = None
_worker_progress = None


def init_worker(cancel, progress):
    """Share the GUI's cancel Event and per-file progress Array with this worker."""
    global _worker_cancel, _worker_progress
    _worker_cancel, _worker_progress = cancel, progress


def cancelled() -> bool:
    return _worker_cancel is not None and _worker_cancel.is_set()


def _report_progress(index, fraction):
    if _worker_progress is not None and 0 <= index < len(_worker_progress):
        _worker_progress[index] = fraction
_HIT = np.dtype([("timestamp", "<i8"), ("energy", "<f4"), ("channel", "<i4")])

# Per-pair outcome codes, in the reference's rejection labels.
OK, MIN_CHANNELS, KEY_ERROR, VALUE_ERROR, ZERO_DIVISION, UNRESOLVED_SLAB = range(6)
_LABELS = {MIN_CHANNELS: "min channels", KEY_ERROR: "KeyError", VALUE_ERROR: "ValueError",
           ZERO_DIVISION: "ZeroDivisionError", UNRESOLVED_SLAB: "unresolved Cornell slab"}
_TRUNCATED_HEADER, _TRUNCATED_HIT = 1, 2


def _c_pow():
    """The C library pow() that CPython's float ``**`` calls.

    numba lowers ``x ** 2`` to ``x * x``, which differs from CPython in the
    last bit for ~0.06 % of values (Cornell centroid Y differed by <= 3 ulp).
    Calling the same pow() keeps the retained Y identical to the reference.
    """
    library = ctypes.CDLL("ucrtbase" if sys.platform == "win32" else ctypes.util.find_library("m"))
    function = library.pow
    llvmlite.binding.add_symbol("ldat_c_pow", ctypes.cast(function, ctypes.c_void_p).value)
    return numba.types.ExternalFunction("ldat_c_pow", numba.types.float64(numba.types.float64, numba.types.float64))


_pow = _c_pow()


class _Lookup:
    """Dense per-channel arrays built from the selected map."""

    def __init__(self, setup):
        ids = set(setup.channel_types) | set(setup.channel_modules) | set(setup.coordinates)
        size = max(ids) + 1 if ids else 1
        self.in_types = np.zeros(size, np.bool_)
        self.is_time = np.zeros(size, np.bool_)
        self.is_energy = np.zeros(size, np.bool_)
        self.first_energy = np.zeros(size, np.bool_)  # centroid uses the first listed type
        self.in_modules = np.zeros(size, np.bool_)
        self.sm = np.zeros(size, np.int64)
        self.mm = np.zeros(size, np.int64)
        self.in_coords = np.zeros(size, np.bool_)
        self.x = np.zeros(size, np.float64)
        self.y = np.zeros(size, np.float64)
        self.pos = np.full(size, -1, np.int64)
        for ch, types in setup.channel_types.items():
            self.in_types[ch] = True
            self.is_time[ch] = ChannelType.TIME in types
            self.is_energy[ch] = ChannelType.ENERGY in types
            self.first_energy[ch] = types[0] is ChannelType.ENERGY
        for ch, (sm, mm) in setup.channel_modules.items():
            self.in_modules[ch], self.sm[ch], self.mm[ch] = True, sm, mm
        for ch, coords in setup.coordinates.items():
            self.in_coords[ch], self.x[ch], self.y[ch] = True, coords[0], coords[1]
            if len(coords) > 2:
                self.pos[ch] = coords[2]

    def arrays(self):
        return (self.in_types, self.is_time, self.is_energy, self.first_energy, self.in_modules,
                self.sm, self.mm, self.in_coords, self.x, self.y, self.pos)


@numba.njit(cache=True)
def _scan_headers(buf, start, limit):
    """Record offsets and side sizes from byte ``start``; stops at ``limit`` pairs.

    Returns (offsets, n1, n2, count, end, truncation code).
    """
    offsets = np.empty(limit, np.int64)
    n1 = np.empty(limit, np.int64)
    n2 = np.empty(limit, np.int64)
    pos, count, size = start, 0, buf.size
    while count < limit:
        if pos >= size:
            return offsets, n1, n2, count, pos, 0
        if pos + 2 > size:
            return offsets, n1, n2, count, pos, 1
        a, b = np.int64(buf[pos]), np.int64(buf[pos + 1])
        end = pos + 2 + 16 * (a + b)
        if end > size:
            return offsets, n1, n2, count, pos, 2
        offsets[count], n1[count], n2[count] = pos, a, b
        count += 1
        pos = end
    return offsets, n1, n2, count, pos, 0


@numba.njit(cache=True)
def _min_channels(idx, n, in_types, is_energy, channel, min_ch):
    """filter_min_ch with sum-rows/cols readout: 0 pass, 1 fail, 2 KeyError."""
    count = 0
    for k in range(n):
        ch = channel[idx[k]]
        if ch < 0 or ch >= in_types.size or not in_types[ch]:
            return 2
        if is_energy[ch]:
            count += 1
    return 0 if min_ch <= count < n else 1


@numba.njit(cache=True)
def _side(idx, n, timestamp, energy, channel, lut, min_ch, cornell, recover, sel, out_f, out_i, ties):
    """One detector side. Fills ``sel`` with the selected hit indices.

    ``out_f`` receives (raw energy, x, y, doi) and ``out_i`` (key, timestamp,
    sm, mm, random flag, recovered flag, selected count). Returns an outcome code.
    """
    (in_types, is_time, is_energy, first_energy, in_modules, sm_of, mm_of,
     in_coords, cx, cy, cpos) = lut
    # get_maxEnergy_sm_mM: every hit needs a (sm, mm).
    for k in range(n):
        ch = channel[idx[k]]
        if ch < 0 or ch >= in_modules.size or not in_modules[ch]:
            return KEY_ERROR
    first_group = sm_of[channel[idx[0]]] * 65536 + mm_of[channel[idx[0]]]
    single = True
    for k in range(1, n):
        ch = channel[idx[k]]
        if sm_of[ch] * 65536 + mm_of[ch] != first_group:
            single = False
            break
    count = 0
    raw = 0.0
    if single:
        for k in range(n):
            sel[k] = idx[k]
            if is_energy[channel[idx[k]]]:
                raw += np.float64(energy[idx[k]])
        count = n
    else:
        best = 0.0
        best_group = np.int64(-1)
        seen = np.empty(n, np.int64)
        n_seen = 0
        for k in range(n):
            ch = channel[idx[k]]
            group = sm_of[ch] * 65536 + mm_of[ch]
            new = True
            for j in range(n_seen):
                if seen[j] == group:
                    new = False
                    break
            if not new:
                continue
            seen[n_seen] = group
            n_seen += 1
            total = 0.0
            for j in range(n):
                cj = channel[idx[j]]
                if sm_of[cj] * 65536 + mm_of[cj] == group and is_energy[cj]:
                    total += np.float64(energy[idx[j]])
            if total > best:
                best = total
                best_group = group
            elif total == best and best > 0.0:
                ties[0] += 1
        if best_group < 0:
            return VALUE_ERROR  # no minimodule with positive energy
        raw = best
        for k in range(n):
            ch = channel[idx[k]]
            if sm_of[ch] * 65536 + mm_of[ch] == best_group:
                sel[count] = idx[k]
                count += 1
    # filter_min_ch on the selected minimodule
    n_energy = 0
    for k in range(count):
        if is_energy[channel[sel[k]]]:
            n_energy += 1
    if not (min_ch <= n_energy < count):
        return VALUE_ERROR
    # get_max_en_channel(TIME): first maximum in hit order
    t_hit = -1
    for k in range(count):
        h = sel[k]
        if is_time[channel[h]] and (t_hit < 0 or energy[h] > energy[t_hit]):
            t_hit = h
    if t_hit < 0:
        return VALUE_ERROR
    t_ch = channel[t_hit]
    slab = 0
    random_side = 0
    recovered = 0
    if cornell:
        # get_max_num_ch(TIME, 2): stable descending sort, so ties keep hit order
        second = -1
        n_time = 0
        for k in range(count):
            h = sel[k]
            if not is_time[channel[h]]:
                continue
            n_time += 1
            if h != t_hit and (second < 0 or energy[h] > energy[second]):
                second = h
        if not in_coords[t_ch]:
            return KEY_ERROR
        pos = cpos[t_ch]
        if pos == 0 or pos == 7:
            if pos == 0:
                slab = 0 if n_time == 1 else 1
            else:
                slab = 15 if n_time == 1 else 14
        elif n_time == 1:
            slab = 2 * pos + np.random.randint(0, 2)
            random_side = 1
        else:
            if not in_coords[channel[second]]:
                return KEY_ERROR
            diff = pos - cpos[channel[second]]
            if abs(diff) <= 1:
                slab = 2 * pos if diff == 1 else 2 * pos + 1
            elif not recover:
                return UNRESOLVED_SLAB
            else:
                # FR-19 (src.ldat_inspector.engine.cornell_slab): the strongest fired
                # adjacent time channel picks the side; a tie or none flips a coin.
                lower = -np.inf
                upper = -np.inf
                for k in range(count):
                    h = sel[k]
                    ch = channel[h]
                    if not is_time[ch]:
                        continue
                    if not in_coords[ch]:
                        return KEY_ERROR
                    e = np.float64(energy[h])
                    if cpos[ch] == pos - 1 and e > lower:
                        lower = e
                    elif cpos[ch] == pos + 1 and e > upper:
                        upper = e
                if lower > upper:
                    slab = 2 * pos
                elif upper > lower:
                    slab = 2 * pos + 1
                else:
                    slab = 2 * pos + np.random.randint(0, 2)
                    random_side = 1
                recovered = 1
    # calculate_centroid(..., 1, 2, chtype_map)
    sx = 0.0
    sy = 0.0
    wx = 0.0
    wy = 0.0
    for k in range(count):
        h = sel[k]
        ch = channel[h]
        if not in_coords[ch]:
            return KEY_ERROR
        e = np.float64(energy[h])
        if first_energy[ch]:
            w = _pow(e + 0.00001, 2.0)
            sy += w * cy[ch]
            wy += w
        else:
            w = (e + 0.00001) ** 1
            sx += w * cx[ch]
            wx += w
    if wx == 0.0 or wy == 0.0:
        return ZERO_DIVISION
    x = sx / wx
    y = sy / wy
    # calculate_DOI with sum-rows/cols: sum / max of energy channels
    top = 0.0
    found = False
    total = 0.0
    for k in range(count):
        h = sel[k]
        if is_energy[channel[h]]:
            e = np.float64(energy[h])
            total += e
            if not found or e > top:
                top = e
                found = True
    if top == 0.0:
        return ZERO_DIVISION
    doi = total / top
    if not (np.isfinite(raw) and np.isfinite(x) and np.isfinite(y) and np.isfinite(doi)) or raw <= 0.0:
        return VALUE_ERROR
    out_f[0], out_f[1], out_f[2], out_f[3] = raw, x, y, doi
    out_i[0], out_i[1] = (np.int64(t_ch) << 5) | slab, timestamp[t_hit]
    out_i[2], out_i[3], out_i[4], out_i[5], out_i[6] = sm_of[t_ch], mm_of[t_ch], random_side, recovered, count
    return OK


@numba.njit(cache=True)
def _pairs(first, n1, n2, timestamp, energy, channel, lut, cut, min_ch, cornell, recover,
           side_f, side_i, codes, t_counts, e_counts, ties):
    """Process a chunk of pairs; returns the number of accepted pairs.

    Accepted sides go to ``side_f`` (raw, x, y, doi) and ``side_i`` (key, ts,
    sm, mm, random, recovered) as 2 rows per accepted pair. ``codes[p]`` is each outcome.
    """
    in_types, is_time, is_energy = lut[0], lut[1], lut[2]
    idx1 = np.empty(256, np.int64)
    idx2 = np.empty(256, np.int64)
    sel1 = np.empty(256, np.int64)
    sel2 = np.empty(256, np.int64)
    f1 = np.empty(4, np.float64)
    f2 = np.empty(4, np.float64)
    i1 = np.empty(7, np.int64)
    i2 = np.empty(7, np.int64)
    accepted = 0
    for p in range(first.size):
        k1 = 0
        for h in range(first[p], first[p] + n1[p]):
            if np.float64(energy[h]) >= cut:
                idx1[k1] = h
                k1 += 1
        k2 = 0
        for h in range(first[p] + n1[p], first[p] + n1[p] + n2[p]):
            if np.float64(energy[h]) >= cut:
                idx2[k2] = h
                k2 += 1
        c = _min_channels(idx1, k1, in_types, is_energy, channel, min_ch)
        if c == 0:
            c = _min_channels(idx2, k2, in_types, is_energy, channel, min_ch)
        if c != 0:
            codes[p] = KEY_ERROR if c == 2 else MIN_CHANNELS
            continue
        code = _side(idx1, k1, timestamp, energy, channel, lut, min_ch, cornell, recover, sel1, f1, i1, ties)
        if code == OK:
            code = _side(idx2, k2, timestamp, energy, channel, lut, min_ch, cornell, recover, sel2, f2, i2, ties)
        codes[p] = code
        if code != OK:
            continue
        for s in range(2):
            of = f1 if s == 0 else f2
            oi = i1 if s == 0 else i2
            sel = sel1 if s == 0 else sel2
            row = 2 * accepted + s
            for j in range(4):
                side_f[row, j] = of[j]
            for j in range(6):
                side_i[row, j] = oi[j]
            for k in range(oi[6]):
                ch = channel[sel[k]]
                if is_time[ch]:
                    t_counts[ch] += 1
                if is_energy[ch]:
                    e_counts[ch] += 1
        accepted += 1
    return accepted


@numba.njit(cache=True)
def _seed(value):
    np.random.seed(value)


def _hits(buf, offsets, n1, n2, count):
    """Aligned hit arrays for ``count`` records, and each record's first hit index."""
    start, stop = offsets[0], offsets[count - 1] + 2 + 16 * (n1[count - 1] + n2[count - 1])
    keep = np.ones(stop - start, np.bool_)
    header = offsets[:count] - start
    keep[header] = False
    keep[header + 1] = False
    raw = np.ascontiguousarray(buf[start:stop][keep])
    hits = raw.view(_HIT)
    first = np.zeros(count, np.int64)
    np.cumsum(n1[:count - 1] + n2[:count - 1], out=first[1:])
    return hits["timestamp"].copy(), hits["energy"].copy(), hits["channel"].copy(), first


def process_file_fast(path: str, settings: Settings, index: int = 0, *, seed: int | None = None) -> FileResult:
    """Fast equivalent of ``process_file_reference`` (see module docstring)."""
    from .engine import process_file_reference

    result = FileResult(index=index, path=str(path))
    result.max_energy_ties = 0
    try:
        settings.validate()
        setup = load_setup(replace(settings, calibrated=False))
        if not setup.fem.sum_rows_cols:
            return process_file_reference(path, settings, index)
        if Path(path).stat().st_size == 0:
            raise ValueError("Empty LDAT file")
        lut = _Lookup(setup)
        buf = np.memmap(path, dtype=np.uint8, mode="r")
        limit = settings.max_pairs if settings.max_pairs is not None else np.iinfo(np.int64).max
        _seed(np.int64(seed if seed is not None else 2_000_003 + index))
        cornell = settings.system == "CORNELL"
        recover = cornell and settings.slab_rule == "recover_non_adjacent"
        t_counts = np.zeros(lut.in_types.size, np.int64)
        e_counts = np.zeros(lut.in_types.size, np.int64)
        ties = np.zeros(1, np.int64)
        errors = Counter()
        parts_f, parts_i = [], []
        position, read, truncation = 0, 0, 0
        while read < limit:
            if cancelled():
                raise Cancelled("Cancelled")
            want = int(min(CHUNK_PAIRS, limit - read))
            offsets, n1, n2, count, position, truncation = _scan_headers(buf, position, want)
            if count:
                timestamp, energy, channel, first = _hits(buf, offsets, n1, n2, count)
                side_f = np.empty((2 * count, 4), np.float64)
                side_i = np.empty((2 * count, 6), np.int64)
                codes = np.empty(count, np.int64)
                accepted = _pairs(first, n1[:count], n2[:count], timestamp, energy, channel, lut.arrays(),
                                  float(settings.min_channel_energy), settings.min_channels, cornell, recover,
                                  side_f, side_i, codes, t_counts, e_counts, ties)
                read += count
                for code, number in zip(*np.unique(codes, return_counts=True)):
                    if code != OK:
                        errors[_LABELS[int(code)]] += int(number)
                # Copy only accepted rows: the chunk buffers are sized for every pair.
                parts_f.append(side_f[:2 * accepted].copy())
                parts_i.append(side_i[:2 * accepted].copy())
                del side_f, side_i, codes, timestamp, energy, channel, first
            _report_progress(index, position / max(buf.size, 1))
            if truncation or count < want:
                break
        result.pairs_read = read
        result.errors = errors
        result.max_energy_ties = int(ties[0])
        if truncation:
            raise ValueError("Truncated LDAT pair header" if truncation == _TRUNCATED_HEADER
                             else "Truncated LDAT hit record")
        side_f = np.concatenate(parts_f) if parts_f else np.empty((0, 4))
        side_i = np.concatenate(parts_i) if parts_i else np.empty((0, 6), np.int64)
        del parts_f, parts_i
        result.pairs_accepted = side_f.shape[0] // 2
        result.prefix_limited = result.pairs_read == settings.max_pairs
        result.table = _table(side_f, side_i, index)
        result.time_counts = _counters(t_counts, lut)
        result.energy_counts = _counters(e_counts, lut)
    except Exception as exc:
        result.error = str(exc)
        result.table, result.time_counts, result.energy_counts = None, {}, {}
        result.pairs_accepted = 0
    return result


def _table(side_f, side_i, index):
    """The file's SideTable from pair-ordered sides (rows 2k and 2k + 1 are one pair)."""
    return SideTable.from_pairs(
        index, raw_energy=side_f[:, 0], x=side_f[:, 1], y=side_f[:, 2], doi=side_f[:, 3],
        calibration_key=side_i[:, 0], timestamp=side_i[:, 1], sm=side_i[:, 2], mm=side_i[:, 3],
        random_slab=side_i[:, 4].astype(bool), recovered_slab=side_i[:, 5].astype(bool))


def _counters(counts, lut):
    out = {}
    for ch in np.flatnonzero(counts):
        out.setdefault(int(lut.sm[ch]), Counter())[int(ch)] = int(counts[ch])
    return out
