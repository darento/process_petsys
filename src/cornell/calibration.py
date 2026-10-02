"""Cornell energy calibration per slab or per (slab, position region) (spec 003 FR-21, T20).

One algorithm for fixed coincidence, fixed group and compact coincidence input,
that of the owner-chosen reference ``scripts_cornell/cornell_slab_en_cal.py``
(owner decision 2026-10-01; it supersedes the T8 ``fit_gaussian`` port):

- hits below ``en_min_ch`` are dropped first, as the compact reader does;
- both sides pass ``filter_min_ch``; the highest-energy minimodule of each side
  passes it too; ``get_slab_cornell`` gives the slab; an event whose two slabs
  are both undetermined is skipped;
- only sides whose slab two time channels determined are kept (one-time-channel
  sides are mostly below the photopeak); the key is the side's highest-energy
  time channel and slab, plus the position region when ``positions`` >= 2;
- a file stops once more than ``event_limit`` (reference 10,000,000) events
  have passed;
- per key: at least 200 values, anchor at the highest 5-bin-smoothed bin of a
  150-bin 0-220 a.u. histogram, Gaussian plus constant or linear background
  (``fit_peak_background``; linear only when it improves the deviance by >= 8),
  a "higher-energy peak" check; outer slabs 0/15 take slab 1/14's factor; populated
  slabs without a fit are estimated from fitted neighbours or the minimodule median.

``positions`` = 1 writes the per-slab ``ID(t_ch, slab)`` calibration
(``KevConverter`` ``cornell``), byte-identical to the reference; >= 2 assigns
regions from the selected COG limits (normalized y COG, outside [0, 1]
excluded; T8 edge-multiplier boundaries) and writes ``ID(time_ch, slab, region)``
(``cornell_position``). Every mapped key is listed; keys without a factor are
``0\\t0`` as in the reference.

Storage is bounded by keys, never by events: the fit depends on the values
only through their count and two histograms, so pass 1 accumulates the count
and anchor histogram and pass 2 re-reads the same selection into each key's
anchor-dependent fit histogram. Fixed and compact batches are decoded into the
same padded arrays and every sum runs in hit order, so the two encodings of the
same events give identical results. Inputs are fully validated (T5 rules) while
pass 1 reads them; a changed input fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from numba import njit
import numpy as np
from scipy.signal import find_peaks

from src.ldat_inspector.engine import fit_peak_background
from src.mapping_generator import ChannelType
from src.utils_fixed import create_channel_type_mask, create_local_map_arrays, create_mm_map_array

from src.petsys_manager.contracts import DataFormat, InputDescriptor, Population
from .inputs import InputError, Limits, ProcessingConfig

ANCHOR_BINS, ANCHOR_RANGE = 150, (0.0, 220.0)   # a.u.
FIT_BINS = 100
MIN_EVENTS = 200
MIN_FITTED_FOR_ESTIMATE = 4
LINEAR_DEVIANCE_GAIN = 8
EVENT_LIMIT = 10_000_000          # reference: a file stops once more events than this have passed
DEFAULT_POSITIONS = 5
MAX_POSITIONS = 127
EDGE_MULTIPLIER = 1.8
DEFAULT_BATCH_RECORDS = 5000
MAX_SLABS = 16
MAX_KEYS = 200_000
READ_BLOCK_BYTES = 8 * 1024 * 1024
HIT = struct.Struct("<qfi")
HIT_DTYPE = np.dtype([("time", "<i8"), ("energy", "<f4"), ("channelID", "<i4")])
REJECTIONS = ("min_channels", "minimodule_channels", "no_minimodule_energy", "no_time_channel",
              "unresolved_slab", "one_time_channel", "missing_cog_limits", "cog_out_of_range")
NO_RESOLVED = "no fit: no resolved events"


class CalibrationCancelled(InputError):
    """Stopped before a calibration result existed."""


# Shared with listmode ------------------------------------------------------------------------------

def create_region_boundaries(num_regions, edge_multiplier=EDGE_MULTIPLIER):
    """Reference boundaries: edge regions ``edge_multiplier`` times a centre region."""
    if num_regions < 3:
        return np.linspace(0.0, 1.0, num_regions + 1)
    center = 1.0 / (num_regions - 2 + 2 * edge_multiplier)
    edge = edge_multiplier * center
    boundaries = [0.0, edge]
    for i in range(num_regions - 3):
        boundaries.append(edge + (i + 1) * center)
    boundaries += [1.0 - edge, 1.0]
    return np.array(boundaries, dtype=np.float64)


def limit_arrays(limits, max_ch):
    """Float32 (time channel, slab < 16) COG-limit lookup arrays."""
    if not isinstance(limits, Limits) or limits.kind != "cog":
        raise InputError("Position calibration requires typed COG limits")
    left = np.zeros((max_ch, MAX_SLABS), dtype=np.float32)
    right = np.zeros((max_ch, MAX_SLABS), dtype=np.float32)
    valid = np.zeros((max_ch, MAX_SLABS), dtype=bool)
    for (channel, slab), (low, high) in limits.values.items():
        if channel < max_ch and slab < MAX_SLABS:
            left[channel, slab], right[channel, slab], valid[channel, slab] = low, high, True
    return left, right, valid


def region_ids(y_cog, time_chs, slab_ids, left, right, boundaries):
    """Calibration region rule (T8): normalized COG outside [0, 1] has none; exactly 1 joins the last.

    Returns (regions, missing_limits); region -1 is "no region".
    """
    y_cog, time_chs, slab_ids = np.asarray(y_cog), np.asarray(time_chs), np.asarray(slab_ids)
    count = len(boundaries) - 1
    regions = np.full(len(y_cog), -1, dtype=np.int32)
    inside = (time_chs >= 0) & (slab_ids >= 0) & (slab_ids < MAX_SLABS) & (time_chs < left.shape[0])
    rows = np.flatnonzero(inside)
    low = left[time_chs[rows], slab_ids[rows]]
    high = right[time_chs[rows], slab_ids[rows]]
    usable = high > low
    missing = np.ones(len(y_cog), dtype=bool)
    missing[rows[usable]] = False
    rows, low, high = rows[usable], low[usable], high[usable]
    normalized = (y_cog[rows] - low) / (high - low)
    keep = (normalized >= 0.0) & (normalized <= 1.0)   # NaN fails both
    rows, normalized = rows[keep], normalized[keep]
    index = np.searchsorted(boundaries, normalized, side="right") - 1
    outside = (index < 0) | (index >= count)
    index[outside] = np.where(normalized[outside] >= 0.999, count - 1, -1)
    regions[rows] = index
    return regions, missing


@dataclass(frozen=True)
class CalibrationMaps:
    energy_mask: np.ndarray
    time_mask: np.ndarray
    minimodules: np.ndarray
    local: tuple
    sum_rows_cols: bool

    @classmethod
    def from_mapping(cls, mapping):
        types, modules, local = dict(mapping.types), dict(mapping.modules), dict(mapping.local)
        max_ch = max(max(types), max(modules), max(local)) + 1
        return cls(create_channel_type_mask(types, ChannelType.ENERGY, max_ch),
                   create_channel_type_mask(types, ChannelType.TIME, max_ch),
                   create_mm_map_array(modules, max_ch), create_local_map_arrays(local, max_ch),
                   bool(mapping.config["sum_rows_cols"]))

    @property
    def max_ch(self):
        return len(self.minimodules)


# Exact np.histogram binning ------------------------------------------------------------------------

def bin_index(values, low, high, edges, bins):
    """``np.histogram(values, bins, range=(low, high))`` bin of each value, -1 outside.

    ``low``/``high`` may be per value (one key's range each) with ``edges`` their
    ``np.linspace(low, high, bins + 1)`` rows. The result is the edge-consistent
    bin (edges[i] <= v < edges[i + 1], last bin closed), which numpy's uniform-bin
    path returns.
    """
    values = np.asarray(values, dtype=np.float64)
    low, high = np.broadcast_to(low, values.shape), np.broadcast_to(high, values.shape)
    edges = np.asarray(edges)
    per_value = edges.ndim == 2
    keep = (values >= low) & (values <= high)
    index = np.full(values.shape, -1, dtype=np.int64)
    rows = np.flatnonzero(keep)
    if not len(rows):
        return index
    v, lo, hi = values[rows], low[rows], high[rows]
    guess = ((v - lo) / (hi - lo) * bins).astype(np.int64)
    guess = np.clip(guess, 0, bins - 1)
    table = edges[rows] if per_value else None
    edge = (lambda i: np.take_along_axis(table, i[:, None], axis=1)[:, 0]) if per_value else (lambda i: edges[i])
    for _ in range(4):  # the arithmetic guess is off by at most one bin
        down = v < edge(guess)
        guess[down] -= 1
        up = (guess < bins - 1) & (v >= edge(np.minimum(guess + 1, bins)))
        guess[up] += 1
        if not (down.any() or up.any()):
            break
    index[rows] = guess
    return index


# Side kernel: the reference per-event rules, row by row in hit order ----------------------------

@njit(cache=True)
def _side_kernel(ch, energy, energy_mask, time_mask, mm_arr, pos_arr, y_arr, en_min, min_ch, sum_rows_cols,
                 forced_mm, want_cog):
    """Per side: full-side filter, max minimodule, its filter, slab rule, key channel and y COG.

    ``forced_mm[i]`` >= 0 imposes the maximum minimodule (reference tie fallback).
    status: 0 ok, 1 min_channels, 2 tie (resolve outside), 3 no_minimodule_energy,
    4 minimodule_channels, 5 no_time_channel, 6 unresolved (non-adjacent), 7 one time channel.
    """
    n, width = ch.shape
    status = np.zeros(n, np.int8)
    max_mm = np.full(n, -1, np.int64)
    side_energy = np.zeros(n, np.float64)
    slab = np.full(n, -1, np.int64)
    flag = np.full(n, -1, np.int8)
    time_ch = np.full(n, -1, np.int64)
    y_cog = np.zeros(n, np.float64)
    codes = np.empty(width, np.int64)
    sums = np.empty(width, np.float64)
    for i in range(n):
        valid = 0
        n_energy = 0
        for j in range(width):
            c = ch[i, j]
            if c >= 0 and np.float64(energy[i, j]) >= en_min:
                valid += 1
                if energy_mask[c]:
                    n_energy += 1
        passed = (min_ch <= n_energy and n_energy < valid) if sum_rows_cols else n_energy >= min_ch
        if not passed:
            status[i] = 1
            continue
        distinct = 0
        for j in range(width):
            c = ch[i, j]
            if c < 0 or np.float64(energy[i, j]) < en_min:
                continue
            m = mm_arr[c]
            k = 0
            while k < distinct and codes[k] != m:
                k += 1
            if k == distinct:
                codes[k] = m
                sums[k] = 0.0
                distinct += 1
            if energy_mask[c]:
                sums[k] += np.float64(energy[i, j])
        if forced_mm[i] >= 0:
            chosen = -1
            for k in range(distinct):
                if codes[k] == forced_mm[i]:
                    chosen = k
        elif distinct == 1:
            chosen = 0
        else:
            best = sums[0]
            for k in range(1, distinct):
                if sums[k] > best:
                    best = sums[k]
            ties = 0
            chosen = -1
            for k in range(distinct):
                if sums[k] == best:
                    ties += 1
                    if chosen < 0:
                        chosen = k
            if best <= 0.0:
                status[i] = 3   # the reference has no minimodule here and fails on it
                continue
            if ties > 1:
                status[i] = 2
                continue
        if chosen < 0:
            status[i] = 3
            continue
        m = codes[chosen]
        max_mm[i] = m
        side_energy[i] = sums[chosen]
        valid = 0
        n_energy = 0
        first = -1
        second = -1
        for j in range(width):
            c = ch[i, j]
            if c < 0 or np.float64(energy[i, j]) < en_min or mm_arr[c] != m:
                continue
            valid += 1
            if energy_mask[c]:
                n_energy += 1
            if time_mask[c]:   # stable descending order: the first of equal energies leads
                if first < 0 or energy[i, j] > energy[i, first]:
                    second = first
                    first = j
                elif second < 0 or energy[i, j] > energy[i, second]:
                    second = j
        passed = (min_ch <= n_energy and n_energy < valid) if sum_rows_cols else n_energy >= min_ch
        if not passed:
            status[i] = 4
            continue
        if first < 0:
            status[i] = 5
            continue
        n_time = 1 if second < 0 else 2
        c1 = ch[i, first]
        time_ch[i] = c1
        p1 = pos_arr[c1]
        if p1 == 0:
            slab[i] = 1 if n_time == 2 else 0
            flag[i] = 0
            if n_time == 1:
                status[i] = 7
        elif p1 == 7:
            slab[i] = 14 if n_time == 2 else 15
            flag[i] = 0
            if n_time == 1:
                status[i] = 7
        elif n_time == 1:
            flag[i] = 1   # reference coin flip; never calibrated
            status[i] = 7
        else:
            diff = p1 - pos_arr[ch[i, second]]
            if diff > 1 or diff < -1:
                flag[i] = 2
                status[i] = 6
            else:
                slab[i] = 2 * p1 if diff == 1 else 2 * p1 + 1
                flag[i] = 3
        if want_cog and status[i] == 0:
            weight = 0.0
            moment = 0.0
            for j in range(width):
                c = ch[i, j]
                if c < 0 or np.float64(energy[i, j]) < en_min or mm_arr[c] != m:
                    continue
                if sum_rows_cols and not energy_mask[c]:
                    continue
                w = (np.float64(energy[i, j]) + 1e-5) ** 2
                weight += w
                moment += w * np.float64(y_arr[c])
            y_cog[i] = moment / weight if weight != 0.0 else 0.0
    return status, max_mm, side_energy, slab, flag, time_ch, y_cog


# Readers: fixed and compact into the same padded arrays, validated while read -----------------------

@njit(cache=True)
def _compact_scan(buffer, sides, limit):
    """Record offsets and hit counts of the complete compact records in ``buffer`` (up to ``limit``)."""
    n = 0
    pos = 0
    size = len(buffer)
    offsets = np.empty(limit, np.int64)
    counts = np.zeros((limit, sides), np.int64)
    bad = -1
    while n < limit and pos + sides <= size:
        total = 0
        for s in range(sides):
            c = buffer[pos + s]
            if c == 0:
                bad = n
            counts[n, s] = c
            total += c
        if bad >= 0:
            break
        end = pos + sides + 16 * total
        if end > size:
            break
        offsets[n] = pos
        n += 1
        pos = end
    return offsets[:n], counts[:n], pos, bad


class _Reader:
    """Validated batches ``(records, [side hits (n, W) structured, ...])`` of one input."""

    def __init__(self, descriptor, mapped, batch_records, cancelled):
        self.descriptor = descriptor
        self.path = Path(descriptor.path)
        self.sides = 1 if descriptor.population == Population.GROUP else 2
        self.mapped = mapped
        self.batch_records = batch_records
        self.cancelled = cancelled
        self.records = 0
        self.hit_limit = None

    def _check(self, hits, active, first_record):
        channels = hits["channelID"]
        safe = np.where(active, channels, 0)
        unmapped = active & ((channels < 0) | (channels >= len(self.mapped)) |
                             ~self.mapped[np.clip(safe, 0, len(self.mapped) - 1)])
        if unmapped.any():
            row, column = np.argwhere(unmapped)[0]
            raise InputError(f"{self.path}: record {first_record + row}: unmapped channel {channels[row, column]}")
        if (active & ~np.isfinite(hits["energy"])).any():
            row = np.argwhere(active & ~np.isfinite(hits["energy"]))[0][0]
            raise InputError(f"{self.path}: record {first_record + row}: nonfinite energy")

    def _stop(self):
        if self.cancelled is not None and self.cancelled():
            raise CalibrationCancelled("Calibration cancelled")

    def __iter__(self):
        if self.descriptor.format == DataFormat.FIXED:
            return self._fixed()
        return self._compact()

    def _fixed(self):
        with self.path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            header = stream.read(4)
            if len(header) != 4:
                raise InputError(f"Truncated fixed hit-limit header: {self.path}")
            limit = struct.unpack("<i", header)[0]
            if not 1 <= limit <= 255:
                raise InputError(f"Invalid fixed hit limit {limit}: {self.path}")
            self.hit_limit = limit
            fields = [("header", "u1", (self.sides,))] + [(f"side{s}", HIT_DTYPE, (limit,))
                                                         for s in range(self.sides)]
            dtype = np.dtype(fields)
            if size <= 4 or (size - 4) % dtype.itemsize:
                raise InputError(f"Empty/truncated fixed records or inconsistent remainder/population: {self.path}")
            while True:
                self._stop()
                chunk = np.fromfile(stream, dtype=dtype, count=self.batch_records)
                if not len(chunk):
                    break
                counts = chunk["header"].astype(np.int64)
                if ((counts < 1) | (counts > limit)).any():
                    row = int(np.argwhere((counts < 1) | (counts > limit))[0][0])
                    raise InputError(f"{self.path}: invalid fixed hit count at record {self.records + row}")
                slots = np.arange(limit)
                batch = []
                for s in range(self.sides):
                    hits = chunk[f"side{s}"].copy()
                    active = slots[None, :] < counts[:, s][:, None]
                    self._check(hits, active, self.records)
                    hits["channelID"][~active] = -1   # inactive slots carry no population
                    batch.append(hits)
                self.records += len(chunk)
                yield len(chunk), batch

    def _compact(self):
        with self.path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size == 0:
                raise InputError(f"Empty LDAT: {self.path}")
            carry = b""
            while True:
                self._stop()
                block = stream.read(READ_BLOCK_BYTES)
                buffer = np.frombuffer(carry + block, dtype=np.uint8)
                if not len(buffer):
                    break
                start = 0
                while True:
                    offsets, counts, end, bad = _compact_scan(buffer[start:], self.sides, self.batch_records)
                    if bad >= 0:
                        raise InputError(f"{self.path}: invalid compact hit count at record {self.records + bad}")
                    if not len(offsets):
                        break
                    batch = self._gather(buffer, start, offsets, counts)
                    self.records += len(offsets)
                    yield len(offsets), batch
                    start += end
                    if len(offsets) < self.batch_records:
                        break
                carry = bytes(buffer[start:])
                if not block:
                    if carry:
                        raise InputError(f"{self.path}: truncated compact record at record {self.records}")
                    break

    def _gather(self, buffer, start, offsets, counts):
        width = int(counts.max()) if len(counts) else 1
        rows = len(offsets)
        side_start = offsets + start + self.sides
        batch = []
        for s in range(self.sides):
            n = counts[:, s]
            hits = np.zeros((rows, width), dtype=HIT_DTYPE)
            hits["channelID"] = -1
            row = np.repeat(np.arange(rows), n)
            slot = np.arange(len(row)) - np.repeat(np.cumsum(n) - n, n)
            first = side_start + 16 * (counts[:, :s].sum(axis=1) if s else 0)
            byte = np.repeat(first, n) + 16 * slot
            raw = buffer[byte[:, None] + np.arange(16)[None, :]]
            values = raw.reshape(-1).view(HIT_DTYPE)
            hits[row, slot] = values
            active = np.arange(width)[None, :] < n[:, None]
            self._check(hits, active, self.records)
            batch.append(hits)
        return batch


# Selection -------------------------------------------------------------------------------------------

class _Context:
    def __init__(self, config, limits, positions):
        mapping = config.mapping
        self.mapping = mapping
        self.maps = CalibrationMaps.from_mapping(mapping)
        self.positions = positions
        self.min_ch = int(config.values["min_ch"])
        if "en_min_ch" not in config.values:
            raise InputError("Calibration requires en_min_ch (a.u.) in the processing YAML")
        self.en_min_ch = float(config.values["en_min_ch"])
        self.mapped = np.zeros(self.maps.max_ch, dtype=bool)
        self.mapped[list(mapping.modules)] = True
        _, self.y_map, pos = self.maps.local
        self.pos_map = pos.astype(np.int64)
        self.mm_map = self.maps.minimodules.astype(np.int64)
        self.boundaries = create_region_boundaries(positions)
        if positions > 1:
            self.left, self.right, _ = limit_arrays(limits, self.maps.max_ch)
        self._reference = None

    def reference_max_mm(self, hits):
        """Exact energy tie between minimodules: the reference function decides (Python set order)."""
        from src.utils import get_maxEnergy_sm_mM
        if self._reference is None:
            self._reference = (dict(self.mapping.modules), dict(self.mapping.types))
        modules, types = self._reference
        det = [(int(t), float(e), int(c)) for t, e, c in zip(hits["time"], hits["energy"], hits["channelID"])
               if c >= 0 and float(e) >= self.en_min_ch]
        chosen, _ = get_maxEnergy_sm_mM(det, modules, types)
        sm, mm = modules[chosen[0][2]]
        return sm * 1000 + mm

    def side(self, hits):
        maps = self.maps
        forced = np.full(len(hits), -1, np.int64)
        args = (np.ascontiguousarray(hits["channelID"]), np.ascontiguousarray(hits["energy"]), maps.energy_mask,
                maps.time_mask, self.mm_map, self.pos_map, self.y_map, self.en_min_ch, self.min_ch,
                maps.sum_rows_cols)
        result = _side_kernel(*args, forced, self.positions > 1)
        ties = np.flatnonzero(result[0] == 2)
        if len(ties):
            for row in ties:
                forced[row] = self.reference_max_mm(hits[row])
            again = _side_kernel(*args, forced, self.positions > 1)
            for target, source in zip(result, again):
                target[ties] = source[ties]
        return result

    def select(self, batch, rejected):
        """Accepted (key code, energy, record row) and the per-record "passed" flags."""
        sides = [self.side(hits) for hits in batch]
        status = [s[0] for s in sides]
        after_mm = [np.isin(st, (0, 5, 6, 7)) for st in status]   # passed both side filters
        has_slab = [np.isin(st, (0, 7)) for st in status]          # flag 1 has a (random) slab here too
        if len(sides) == 2:
            full = (status[0] != 1) & (status[1] != 1)
            no_energy = full & ((status[0] == 3) | (status[1] == 3))
            mm_ok = full & after_mm[0] & after_mm[1]
            determined = has_slab[0] | has_slab[1]
            rejected["min_channels"] += int((~full).sum())
            rejected["no_minimodule_energy"] += int(no_energy.sum())
            rejected["minimodule_channels"] += int((full & ~no_energy & ~mm_ok).sum())
        else:
            mm_ok, determined = after_mm[0], has_slab[0]
            rejected["min_channels"] += int((status[0] == 1).sum())
            rejected["no_minimodule_energy"] += int((status[0] == 3).sum())
            rejected["minimodule_channels"] += int((status[0] == 4).sum())
        passed = mm_ok & determined
        rejected["unresolved_slab"] += int((mm_ok & ~determined).sum())
        keys, energies, records = [], [], []
        for st, _, side_energy, slab, _, time_ch, y_cog in sides:
            if len(sides) == 2:
                rejected["no_time_channel"] += int((passed & (st == 5)).sum())
                rejected["unresolved_slab"] += int((passed & (st == 6)).sum())
            rejected["one_time_channel"] += int((passed & (st == 7)).sum())
            rows = np.flatnonzero(passed & (st == 0))
            region = np.zeros(len(rows), np.int64)
            if self.positions > 1:
                found, missing = region_ids(y_cog[rows], time_ch[rows], slab[rows], self.left, self.right,
                                            self.boundaries)
                rejected["missing_cog_limits"] += int(missing.sum())
                rejected["cog_out_of_range"] += int((~missing & (found < 0)).sum())
                keep = found >= 0
                rows, region = rows[keep], found[keep].astype(np.int64)
            keys.append((time_ch[rows] * MAX_SLABS + slab[rows]) * self.positions + region)
            energies.append(side_energy[rows])
            records.append(rows)
        return np.concatenate(keys), np.concatenate(energies), np.concatenate(records), passed


# Accumulation ----------------------------------------------------------------------------------------

class _Accumulator:
    """Per key: value count, 150-bin anchor histogram; after pass 1, a 100-bin fit histogram."""

    def __init__(self, max_keys=MAX_KEYS):
        self.max_keys = max_keys
        self.rows = {}
        self.codes = []
        self.counts = np.zeros(0, np.int64)
        self.anchor = np.zeros((0, ANCHOR_BINS), np.int64)
        self.anchor_edges = np.linspace(*ANCHOR_RANGE, ANCHOR_BINS + 1)
        self.fit = None

    def _rows(self, codes, create):
        unique, inverse = np.unique(codes, return_inverse=True)
        rows = np.empty(len(unique), np.int64)
        for i, code in enumerate(unique.tolist()):
            row = self.rows.get(code)
            if row is None:
                if not create:
                    raise InputError("Calibration pass 2 found a key pass 1 did not (input changed)")
                if len(self.codes) >= self.max_keys:
                    raise InputError("Calibration keys exceed their bound")
                row = self.rows[code] = len(self.codes)
                self.codes.append(code)
            rows[i] = row
        extra = len(self.codes) - len(self.counts)
        if extra > 0:
            self.counts = np.concatenate((self.counts, np.zeros(extra, np.int64)))
            self.anchor = np.vstack((self.anchor, np.zeros((extra, ANCHOR_BINS), np.int64)))
        return rows[inverse.reshape(-1)]

    def first(self, codes, energies):
        if not len(codes):
            return
        rows = self._rows(codes, True)
        np.add.at(self.counts, rows, 1)
        index = bin_index(energies, *ANCHOR_RANGE, self.anchor_edges, ANCHOR_BINS)
        inside = index >= 0
        np.add.at(self.anchor, (rows[inside], index[inside]), 1)

    def prepare_fit(self):
        """Anchor and fit interval per key with enough values (reference ``fit_slab_photopeak``)."""
        self.fit_counts = np.zeros((len(self.codes), FIT_BINS), np.int64)
        self.interval = np.full((len(self.codes), 2), np.nan)
        self.fit_edges = np.zeros((len(self.codes), FIT_BINS + 1))
        self.smooth = {}
        centres = (self.anchor_edges[:-1] + self.anchor_edges[1:]) / 2
        for row in range(len(self.codes)):
            if self.counts[row] < MIN_EVENTS:
                continue
            smooth = np.convolve(self.anchor[row], np.ones(5) / 5, mode="same")
            anchor = float(centres[np.argmax(smooth)])
            self.smooth[row] = (smooth, anchor)
            self.interval[row] = (0.55 * anchor, 1.5 * anchor)
            self.fit_edges[row] = np.linspace(0.55 * anchor, 1.5 * anchor, FIT_BINS + 1)
        self.second_counts = np.zeros(len(self.codes), np.int64)

    def second(self, codes, energies):
        if not len(codes):
            return
        rows = self._rows(codes, False)
        np.add.at(self.second_counts, rows, 1)
        fitted = np.isfinite(self.interval[rows, 0])
        rows, energies = rows[fitted], np.asarray(energies)[fitted]
        index = bin_index(energies, self.interval[rows, 0], self.interval[rows, 1], self.fit_edges[rows], FIT_BINS)
        inside = index >= 0
        np.add.at(self.fit_counts, (rows[inside], index[inside]), 1)

    @property
    def nbytes(self):
        arrays = [self.counts, self.anchor] + [getattr(self, name) for name in
                                               ("fit_counts", "interval", "fit_edges", "second_counts")
                                               if hasattr(self, name)]
        return sum(a.nbytes for a in arrays)


def stand_in_values(counts, edges, events):
    """Values the reference fit cannot tell from the originals: each count at its bin centre, the rest
    beyond the interval. ``fit_peak_background`` uses only their number and
    ``np.histogram(values, len(counts), range=(edges[0], edges[-1]))``, which this reproduces exactly.
    Temporary, one key at a time; nothing per event is kept."""
    centres = (edges[:-1] + edges[1:]) / 2
    inside = np.repeat(centres, counts)
    outside = np.full(int(events) - len(inside), edges[-1] + 1.0)
    return np.concatenate((inside, outside))


def _fit_key(accumulator, row):
    """Reference ``fit_slab_photopeak`` from the key's histograms: (mu, sigma, status)."""
    events = int(accumulator.counts[row])
    if events < MIN_EVENTS:
        return 0.0, 0.0, f"no fit: {events} events (< {MIN_EVENTS})"
    smooth, anchor = accumulator.smooth[row]
    centres = (accumulator.anchor_edges[:-1] + accumulator.anchor_edges[1:]) / 2
    settings = dict(interval=(0.55 * anchor, 1.5 * anchor), search=(0.8 * anchor, 1.2 * anchor),
                    sigma0=0.07 * anchor, mu_halfwidth=0.15 * anchor)
    values = stand_in_values(accumulator.fit_counts[row], accumulator.fit_edges[row], events)
    linear = fit_peak_background(values, background_model="linear", bins=FIT_BINS, **settings)
    flat = fit_peak_background(values, background_model="constant", bins=FIT_BINS, **settings)
    best = (linear if linear["status"] == "FIT" and (flat["status"] != "FIT" or
                                                      flat["deviance"] - linear["deviance"] >= LINEAR_DEVIANCE_GAIN)
            else flat)
    if best["status"] != "FIT":
        return 0.0, 0.0, f"no fit: {best['status']}"
    mu, sigma = best["mu"], best["sigma"]
    peaks, _ = find_peaks(smooth, prominence=0.25 * smooth.max(), distance=8)
    height_at_mu = smooth[np.argmin(np.abs(centres - mu))]
    higher = [centres[p] for p in peaks if centres[p] > 1.2 * mu and smooth[p] >= 0.5 * height_at_mu]
    status = "fit"
    if higher:
        status = f"fit; check: higher-energy peak near {higher[0]:.1f}"
    return mu, sigma, status


# Result ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FileSample:
    path: Path
    format: DataFormat
    population: Population
    records_validated: int
    records_read: int
    events_passed: int
    accepted_sides: int
    stopped_at_limit: bool
    rejected: dict


@dataclass(frozen=True)
class CalibrationResult:
    positions: int
    boundaries: tuple
    data_format: DataFormat
    population: Population
    min_ch: int
    en_min_ch: float
    event_limit: int | None
    batch_records: int
    files: tuple
    keys: tuple                 # every mapped key, sorted
    factors: dict               # key -> (mu, sigma) for keys with a factor
    statuses: dict              # key -> status text for keys that received values or a factor
    edge_multiplier: float = EDGE_MULTIPLIER
    sources: dict = field(default_factory=dict)

    @property
    def layout(self):
        return "per_slab" if self.positions == 1 else "position"

    @property
    def num_regions(self):
        return self.positions

    def status(self, key):
        return self.statuses.get(key, NO_RESOLVED)

    def status_counts(self):
        counts = {}
        for key in self.keys:
            text = self.status(key)
            label = ("fitted" if text == "fit" else "fitted_check_higher_peak" if text.startswith("fit;")
                     else "borrowed" if text.startswith("borrowed") else "estimated" if text.startswith("estimated")
                     else "no_values" if text == NO_RESOLVED else "no_fit")
            counts[label] = counts.get(label, 0) + 1
        return counts


def _mapped_keys(mapping, positions):
    """(time channel, slab[, region]) for every mapped time channel, in the map's channel order."""
    keys = []
    for channel, types in mapping.types.items():
        if ChannelType.TIME in types:
            position = int(mapping.local[channel][2])
            for half in (0, 1):
                slab = position * 2 + half
                keys.extend([(int(channel), slab)] if positions == 1 else
                            [(int(channel), slab, region) for region in range(positions)])
    return keys


def _borrow_and_estimate(factors, statuses, mapping, positions):
    """Reference ``borrow_outer_slabs`` then ``estimate_missing_slabs``, per region when positions > 1."""
    regions = [None] if positions == 1 else list(range(positions))
    key = (lambda ch, slab, region: (ch, slab)) if positions == 1 else (lambda ch, slab, region: (ch, slab, region))
    counts = {"borrowed": 0, "estimated": 0}
    for region in regions:
        for channel, types in mapping.types.items():
            position = int(mapping.local[channel][2])
            if ChannelType.TIME not in types or position not in (0, 7):
                continue
            outer, inner = (0, 1) if position == 0 else (15, 14)
            if key(channel, inner, region) in factors:
                factors[key(channel, outer, region)] = factors[key(channel, inner, region)]
                statuses[key(channel, outer, region)] = f"borrowed from slab {inner}"
                counts["borrowed"] += 1
    fitted = {k for k, text in statuses.items() if text == "fit"}
    for region in regions:
        by_mm = {}
        for channel, types in mapping.types.items():
            if ChannelType.TIME in types:
                for half in (0, 1):
                    slab = int(mapping.local[channel][2]) * 2 + half
                    by_mm.setdefault(mapping.modules[channel], {})[slab] = key(int(channel), slab, region)
        for slabs in by_mm.values():
            sources = [k for k in slabs.values() if k in fitted]
            if len(sources) < MIN_FITTED_FOR_ESTIMATE:
                continue
            for slab, k in slabs.items():
                if k in factors:
                    continue
                near = [slabs[s] for s in (slab - 1, slab + 1) if slabs.get(s) in fitted]
                used = near or sources
                mu = (np.mean if near else np.median)([factors[u][0] for u in used])
                ratio = np.mean([factors[u][1] / factors[u][0] for u in used])
                factors[k] = (float(mu), float(mu * ratio))
                statuses[k] = (f"estimated from neighbour slabs {sorted(u[1] for u in near)}" if near
                               else f"estimated from minimodule median ({len(sources)} slabs)")
                counts["estimated"] += 1
    return counts


def calibrate(descriptors, config, limits=None, *, positions=DEFAULT_POSITIONS, event_limit=EVENT_LIMIT,
              batch_records=DEFAULT_BATCH_RECORDS, cancelled=None, progress=None):
    """Two validated passes over the ordered inputs, then the reference fits. No files are written."""
    descriptors = tuple(descriptors)
    if not isinstance(config, ProcessingConfig):
        raise InputError("Calibration requires a typed processing config")
    if not descriptors or any(not isinstance(d, InputDescriptor) or not Path(d.path).is_absolute()
                              for d in descriptors):
        raise InputError("Calibration requires absolute explicit input descriptors")
    shapes = {(d.format, d.population) for d in descriptors}
    if len(shapes) != 1:
        raise InputError("Do not mix formats or group/coincidence inputs in one calibration")
    data_format, population = shapes.pop()
    if data_format == DataFormat.COMPACT and population != Population.COINCIDENCE:
        raise InputError("Compact input must be coincidence")
    if type(positions) is not int or not 1 <= positions <= MAX_POSITIONS:
        raise InputError(f"positions must be an integer from 1 to {MAX_POSITIONS}")
    if positions > 1 and not isinstance(limits, Limits):
        raise InputError("Position calibration (positions >= 2) requires the selected COG limits")
    if event_limit is not None and (type(event_limit) is not int or event_limit < 1):
        raise InputError("The passing-event limit must be a positive integer or None")
    if type(batch_records) is not int or not 1 <= batch_records <= 1_000_000:
        raise InputError("batch_records must be an integer from 1 to 1,000,000")
    context = _Context(config, limits if positions > 1 else None, positions)
    accumulator = _Accumulator()
    files, fingerprints = [], []
    for index, descriptor in enumerate(descriptors):
        info = os.stat(descriptor.path)
        fingerprints.append((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
        sample = _sample(descriptor, context, accumulator.first, event_limit, batch_records, cancelled,
                         progress, validate=True)
        files.append(sample)
    accumulator.prepare_fit()
    for index, descriptor in enumerate(descriptors):
        again = _sample(descriptor, context, accumulator.second, event_limit, batch_records, cancelled, None,
                        validate=False)
        if (again.accepted_sides, again.events_passed) != (files[index].accepted_sides, files[index].events_passed):
            raise InputError(f"Input changed between calibration passes: {descriptor.path}")
    for descriptor, before in zip(descriptors, fingerprints):
        info = os.stat(descriptor.path)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != before:
            raise InputError(f"Input changed during calibration: {descriptor.path}")
    if not np.array_equal(accumulator.second_counts, accumulator.counts):
        raise InputError("Calibration passes disagree (input changed)")
    factors, statuses = {}, {}
    for row, code in enumerate(accumulator.codes):
        region = code % positions
        time_ch, slab = divmod(code // positions, MAX_SLABS)
        key = (int(time_ch), int(slab)) if positions == 1 else (int(time_ch), int(slab), int(region))
        mu, sigma, text = _fit_key(accumulator, row)
        statuses[key] = text
        if mu > 0:
            factors[key] = (mu, sigma)
    _borrow_and_estimate(factors, statuses, context.mapping, positions)
    mapping = config.mapping
    sources = {"processing_config": {"path": str(config.path), "sha256": config.sha256},
               "map": {"path": str(mapping.path), "sha256": mapping.sha256}}
    if positions > 1:
        sources["cog_limits"] = {"path": str(limits.path), "sha256": limits.sha256,
                                 "zero_width_keys": [list(k) for k in limits.zero_width]}
    return CalibrationResult(positions, tuple(float(b) for b in context.boundaries), data_format, population,
                             context.min_ch, context.en_min_ch, event_limit, batch_records, tuple(files),
                             tuple(sorted(_mapped_keys(mapping, positions))), factors, statuses, sources=sources)


def _sample(descriptor, context, add, event_limit, batch_records, cancelled, progress, *, validate):
    """One pass over one file, validating what it reads; both passes stop at the limit (FR-24).

    ``validate`` is kept for the call sites; records after the limit are neither read nor validated.
    """
    reader = _Reader(descriptor, context.mapped, batch_records, cancelled)
    rejected = dict.fromkeys(REJECTIONS, 0)
    passed_total = accepted = 0
    stopped = False
    read = 0
    for records, batch in reader:
        if stopped:
            break
        if progress is not None:
            progress(descriptor.path, reader.records)
        counted = dict.fromkeys(REJECTIONS, 0)
        codes, energies, _, passed = context.select(batch, counted)
        if event_limit is not None:
            before = passed_total + np.cumsum(passed) - passed    # events passed before each record
            included = before <= event_limit                      # reference: stop once count > limit
            if not included.all():
                stopped = True
                last = int(np.argmin(included))                   # first record not processed
                batch, records = [hits[:last] for hits in batch], last
                counted = dict.fromkeys(REJECTIONS, 0)
                codes, energies, _, passed = context.select(batch, counted)
        for reason, value in counted.items():
            rejected[reason] += value
        read += records
        passed_total += int(passed.sum())
        accepted += len(codes)
        add(codes, energies)
    return FileSample(Path(descriptor.path), descriptor.format, descriptor.population, reader.records, read,
                      passed_total, accepted, stopped, rejected)


# Outputs ---------------------------------------------------------------------------------------------

def _header(result):
    if result.positions == 1:
        return "ID(t_ch, slab)\tmu\tsigma\n"
    return (f"# Position-dependent energy calibration ({result.positions} regions per slab)\n"
            "ID(time_ch, slab, region)\tmu\tsigma\n")


def encal_text(result):
    """Reference ``write_slab_cal`` rows for every mapped key; no factor is ``0\\t0``."""
    lines = [_header(result)]
    for key in result.keys:
        if key in result.factors:
            mu, sigma = result.factors[key]
            lines.append(f"{key}\t{round(mu, 3)}\t{round(sigma, 3)}\n")
        else:
            lines.append(f"{key}\t0\t0\n")
    return "".join(lines)


def status_text(result):
    """Reference ``write_slab_status``: the fit status of every mapped key."""
    head = "ID(t_ch, slab)\tstatus\n" if result.positions == 1 else "ID(time_ch, slab, region)\tstatus\n"
    return head + "".join(f"{key}\t{result.status(key)}\n" for key in result.keys)


def sidecar(result, calibration_sha256):
    """JSON provenance accepted by ``inputs.load_calibration(metadata_path=...)``."""
    return {
        "schema_version": 1,
        "layout": result.layout,
        "num_regions": result.positions,
        "region_boundaries": list(result.boundaries),
        "calibration_sha256": calibration_sha256,
        "generator": "src.cornell.calibration (spec 003 T20; reference cornell_slab_en_cal.py)",
        "format": result.data_format.value,
        "population": result.population.value,
        "energy_units": "a.u.",
        "edge_multiplier": result.edge_multiplier,
        "cuts": {"min_ch": result.min_ch, "en_min_ch_au": result.en_min_ch,
                 "en_min_ch_note": "hits below en_min_ch are dropped before every filter",
                 "sides": "only sides whose slab two time channels determined",
                 "cog_policy": (None if result.positions == 1 else
                                "normalized y COG outside [0, 1] excluded; exactly 1 joins the last region")},
        "fit": {"anchor_histogram": {"bins": ANCHOR_BINS, "range_au": list(ANCHOR_RANGE), "smoothing_bins": 5},
                "fit_histogram_bins": FIT_BINS, "interval": "0.55-1.5 x anchor", "search": "0.8-1.2 x anchor",
                "model": "Gaussian + constant or linear background (src.ldat_inspector.engine)",
                "min_events": MIN_EVENTS, "outer_slabs": "0/15 take slab 1/14",
                "estimates": f"fitted neighbours, else minimodule median (>= {MIN_FITTED_FOR_ESTIMATE} fitted)"},
        "sampling": {"batch_records": result.batch_records, "passing_event_limit_per_file": result.event_limit,
                     "limit_semantics": "a file stops once more events than the limit have passed",
                     "validation": "records_validated = records read (whole reader batches); records after a limit "
                                   "stop are neither read nor validated (FR-24)"},
        "sources": result.sources,
        "inputs": [{"path": str(f.path), "format": f.format.value, "records_validated": f.records_validated,
                    "records_read": f.records_read, "events_passed": f.events_passed,
                    "accepted_sides": f.accepted_sides, "stopped_at_limit": f.stopped_at_limit,
                    "rejected": f.rejected} for f in result.files],
        "status_counts": result.status_counts(),
    }


def _write_exclusive(path, content):
    with open(path, "xb") as out:  # never replaces an existing calibration
        out.write(content)
        out.flush()
        os.fsync(out.fileno())


def write_calibration(result, encal_path, sidecar_path, status_path):
    """Exclusive .encal, status file and sidecar. Returns the .encal SHA-256."""
    if not result.factors:
        raise InputError("No key received a calibration factor (no fit, borrow or estimate)")
    content = encal_text(result).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    metadata = (json.dumps(sidecar(result, digest), indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_exclusive(encal_path, content)
    _write_exclusive(status_path, status_text(result).encode("utf-8"))
    _write_exclusive(sidecar_path, metadata)
    return digest


def plot_summary(result, path):
    """Reference ``plot_slab_dist``: histogram of every factor (fitted, borrowed, estimated)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    mus = [mu for mu, _ in result.factors.values()]
    figure = plt.figure()
    try:
        mean, std = np.mean(mus), np.std(mus)
        plt.hist(mus, bins=200, range=(0, 200), alpha=0.6)
        plt.axvline(mean, color="r", linestyle="dashed", linewidth=2)
        plt.fill_betweenx([0, plt.gca().get_ylim()[1]], mean - std, mean + std, color="r", alpha=0.2)
        plt.title(f"Histogram of Photopeak Distributions ({result.layout}, {result.positions} position(s))")
        plt.xlabel("Mu")
        plt.ylabel("Frequency")
        plt.text(0.6, 0.7, f"Mean: {mean}\nStd Dev: {std}", transform=plt.gca().transAxes)
        with open(path, "xb") as out:
            figure.savefig(out, format="png")
    finally:
        plt.close(figure)
