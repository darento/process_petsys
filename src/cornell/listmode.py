"""Fixed-position Cornell listmode generation (spec 003 T9).

Tracked port of the owner-confirmed reference
``scripts_cornell/cornell_listmode_cog_fixed_position.py``. Selection, the slab
rule and its ``np.random`` draws, COG/DOI decompression, region clipping,
calibration lookup, pair/swap logic and every ``CoincidenceV5`` field are the
reference expressions, applied per 1000-record reader batch as in the
reference. Changes:

- Header values come from supplied :class:`LMMetadata` (no fabricated 10 s,
  120 modules, 5 rings, 820 mm or 51.61 mm/100-pixel grid). ``identifier``,
  ``startTime`` 0 and version (9, 5) stay as in the reference; its zero fields
  stay zero and are listed in provenance.
- Every rejected pair is counted once, at its first failing stage. Cases where
  the reference would crash (partial decompression masks, an unmapped
  minimodule beyond its region array, a slab flag outside 0-3) or silently read
  pair-table row 99 (region -1) are counted rejections instead.
- Timestamps stay the raw LDAT values stored as float32 (no offset/scaling);
  the operator-declared ``timestamp_unit`` is provenance only. Wrapped int16
  ``dt`` and pixels outside the grid are written as the reference writes them,
  and counted.
- Debug keeps fixed histograms (energy, SuperModule hits, 64x64 per-region
  flood maps) instead of per-event lists.
- Output is streamed to exclusively created segments, merged in natural
  basename order (reference ``natsorted``). Resume reuses only segments whose
  completion record matches this job's settings and input identities; any
  other file in the job directory refuses the resume. Nothing is deleted.

Compact coincidence input (FR-22) is decoded into the fixed layout of the
conversion hit limit (empty slots channel -1) and regrouped into the same
batches, so both encodings of the same events give byte-identical output.

``en_min_ch`` is read by the reference but never applied; it is recorded as
not applied. DOI is a light-sharing ratio mapped linearly from its limits, not
an independently calibrated depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from uuid import uuid4

from numba import njit
import numpy as np

from src.detector_features_fixed import calculate_DOI_vectorized, calculate_centroid_vectorized
from src.filters_fixed import filter_min_ch_vectorized
from src.listmode import CoincidenceV5, LMHeader
from src.read_fixed import read_fixed_file_numpy
from src.utils_fixed import (create_dec_lookup_arrays, get_maxEnergy_sm_mM_vectorized,
                             get_slab_cornell_vectorized, get_timestamp_of_max_energy_hit_vectorized)

from src.petsys_manager.contracts import DataFormat, InputDescriptor, Population
from src.petsys_manager.settings import LMMetadata
from .calibration import HIT_DTYPE, CalibrationMaps, _Reader, create_region_boundaries
from .inputs import (Calibration, ChannelMap, InputError, Limits, NumericTable, ProcessingConfig,
                     _metadata_bytes, _rows, check_fixed_records, fixed_layout, mapped_channels)

GENERATOR = "src.cornell.listmode (spec 003 T9; reference cornell_listmode_cog_fixed_position.py)"
RTP_Y_VAL = 2
Y_OFFSET = 25.6             # mm, decompressed minimodule height
CRYSTAL_THICKNESS = 20      # mm, DOI mapping span
MAX_SLABS = 16
MAX_REGION = 100            # reference pair lookup width
DEFAULT_BATCH_RECORDS = 1000
LM_VERSION = (9, 5)
IDENTIFIER = "Cornell"
RECORD_DTYPE = np.dtype(CoincidenceV5)
HEADER_BYTES = 176
MAX_ENERGY_KEV = 65535      # uint16 energy field
JOB_FILE = "lm-job.json"
SEGMENTS = "segments"
REJECTIONS = ("min_channels", "minimodule_channels", "unresolved_slab", "no_position_region",
              "missing_calibration", "energy_window", "missing_doi_limits", "y_out_of_range",
              "doi_out_of_range", "unmapped_region", "no_pair", "missing_timestamp")
OBSERVATIONS = ("dt_wrapped_int16", "pixel_outside_grid")
SUPPLIED_HEADER_FIELDS = ("acqTime", "isotope", "detectorSizeX", "detectorSizeY", "measurementTime",
                          "moduleNumber", "ringNumber", "ringDistance", "detectorPixelSizeX",
                          "detectorPixelSizeY", "detectorPixelsX", "detectorPixelsY")
ZERO_HEADER_FIELDS = ("rawCounts", "activity", "detectorDistance", "isotopeHalfLife", "weight", "maxTemp",
                      "percentLoss", "reserved", "breast", "gatePeriod", "DOILayer", "method", "StudyId")
DEBUG_ENERGY_BINS, DEBUG_ENERGY_RANGE = 1500, (0, 1500)
DEBUG_FLOOD_BINS = 64


class ListmodeCancelled(InputError):
    """Stopped before the merged listmode existed."""


# Maps ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairMap:
    path: Path
    sha256: str
    values: NumericTable            # (region, region) -> (pair ID,)


@dataclass(frozen=True)
class RegionMap:
    path: Path
    sha256: str
    values: NumericTable            # (SuperModule, minimodule) -> (region, (offset x, offset y))


def _literal_tuple(text, length, label):
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, RecursionError) as exc:
        raise InputError(f"{label}: invalid tuple {text!r}") from exc
    if not isinstance(value, tuple) or len(value) != length:
        raise InputError(f"{label}: expected a {length}-value tuple, got {text!r}")
    return value


def load_pair_map(path):
    """Reference ``pairs_map_cornell.txt``: ``pair region region`` separated by one space."""
    if not Path(path).is_absolute():
        raise InputError("Pair map requires an absolute selected path")
    rows, digest = _rows(path)
    entries = []
    for number, line in rows:
        parts = line.split(" ")
        try:
            pair, first, second = (int(part) for part in parts) if len(parts) == 3 else (None,) * 3
        except ValueError:
            pair = None
        if pair is None:
            raise InputError(f"Pair map row {number}: expected three space-separated integers")
        if not (0 <= first < MAX_REGION and 0 <= second < MAX_REGION):
            raise InputError(f"Pair map row {number}: regions must be 0..{MAX_REGION - 1}")
        if not 0 <= pair <= 65535:
            raise InputError(f"Pair map row {number}: pair ID must fit the uint16 record field")
        entries.append(((first, second), (pair,)))
    if not entries:
        raise InputError("Pair map contains no entries")
    return PairMap(Path(path).resolve(), digest, NumericTable(tuple(entries)))


def load_region_map(path, mapping):
    """Reference ``region_sm_mm_map.tsv``: ``(sm, mm)<TAB>region<TAB>(offset x, offset y)``."""
    if not isinstance(mapping, ChannelMap) or not Path(path).is_absolute():
        raise InputError("Region map requires an absolute selected path and typed channel map")
    known = set(mapping.modules.values())
    rows, digest = _rows(path)
    entries = []
    for number, line in rows:
        parts = line.split("\t")
        if len(parts) != 3:
            raise InputError(f"Region map row {number}: expected key/region/offset")
        key = _literal_tuple(parts[0], 2, f"Region map row {number}")
        if any(type(v) is not int for v in key) or key not in known:
            raise InputError(f"Region map row {number}: {key} is not a minimodule of the selected map")
        try:
            region = int(parts[1])
        except ValueError:
            raise InputError(f"Region map row {number}: region must be an integer") from None
        if not 0 <= region < MAX_REGION:
            raise InputError(f"Region map row {number}: region must be 0..{MAX_REGION - 1}")
        offset = _literal_tuple(parts[2], 2, f"Region map row {number}")
        if any(type(v) not in (int, float) or not np.isfinite(v) for v in offset):
            raise InputError(f"Region map row {number}: offsets must be finite numbers (mm)")
        entries.append((key, (region, tuple(float(v) for v in offset))))
    if not entries:
        raise InputError("Region map contains no entries")
    return RegionMap(Path(path).resolve(), digest, NumericTable(tuple(entries)))


# Header -------------------------------------------------------------------------

def pixel_edges(size_mm, pixels):
    """Reference ``np.linspace(0, size, pixels + 1)`` grid (was 0-51.61 mm, 100 pixels)."""
    return np.linspace(0, size_mm, int(pixels) + 1)


def _complete_metadata(metadata):
    if not isinstance(metadata, LMMetadata):
        raise InputError("Listmode requires typed LM metadata")
    missing = metadata.missing()
    if missing:
        raise InputError(*(f"Required LM metadata unavailable: {name}" for name in missing))
    return metadata


def header_values(metadata):
    """Header fields written for ``metadata`` (supplied values plus reference constants)."""
    metadata = _complete_metadata(metadata)
    xpixels = pixel_edges(metadata.detector_size_x_mm, metadata.detector_pixels_x)
    ypixels = pixel_edges(metadata.detector_size_y_mm, metadata.detector_pixels_y)
    values = {"identifier": IDENTIFIER, "acqTime": metadata.acquisition_time_s, "isotope": metadata.isotope,
              "detectorSizeX": metadata.detector_size_x_mm, "detectorSizeY": metadata.detector_size_y_mm,
              "startTime": 0, "measurementTime": metadata.measurement_time_s,
              "moduleNumber": metadata.module_number, "ringNumber": metadata.ring_number,
              "ringDistance": metadata.ring_distance_mm, "detectorPixelSizeX": float(np.diff(xpixels)[0]),
              "detectorPixelSizeY": float(np.diff(ypixels)[0]), "version": list(LM_VERSION),
              "detectorPixelsX": xpixels.size - 1, "detectorPixelsY": ypixels.size - 1}
    for name in ("detectorPixelSizeX", "detectorPixelSizeY"):
        with np.errstate(over="ignore"):
            single = np.float32(values[name])
        if not np.isfinite(single) or single <= 0:
            raise InputError(f"LM header {name} does not fit its float32 field")
    return values, xpixels, ypixels


def header_bytes(metadata):
    """Reference ``write_header`` layout, filled from supplied metadata. Overflow raises."""
    values, _, _ = header_values(metadata)
    try:
        header = LMHeader(**{**values, "identifier": values["identifier"].encode("utf-8"),
                             "isotope": values["isotope"].encode("utf-8"),
                             "version": tuple(values["version"])})
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(f"LM metadata does not fit the header: {exc}") from exc
    content = bytes(header)
    if len(content) != HEADER_BYTES:
        raise InputError("Unexpected LM header size")
    # ctypes silently truncates integers; refuse anything that did not round-trip.
    decoded = LMHeader.from_buffer_copy(content)
    for name in ("moduleNumber", "ringNumber", "detectorPixelsX", "detectorPixelsY"):
        if getattr(decoded, name) != values[name]:
            raise InputError(f"LM header {name} overflows its field")
    return content


# Reference numerics ---------------------------------------------------------------

@njit
def compute_region_vectorized(y_cog, time_chs, slabs, cog_left, cog_right, region_boundaries, num_regions):
    """Reference region assignment: normalized COG clipped to [0, 0.999]."""
    n = len(y_cog)
    regions = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        tch = time_chs[i]
        sid = slabs[i]
        if tch < 0 or sid < 0 or sid >= 16 or tch >= cog_left.shape[0]:
            continue
        left = cog_left[tch, sid]
        right = cog_right[tch, sid]
        if np.isnan(left) or np.isnan(right) or right <= left:
            continue
        y_norm = (y_cog[i] - left) / (right - left)
        y_norm = max(0.0, min(y_norm, 0.999))
        for r in range(num_regions):
            if y_norm >= region_boundaries[r] and y_norm < region_boundaries[r + 1]:
                regions[i] = r
                break
    return regions


@njit
def convert_energy_numba(time_chs, slabs, regions, energies, cal_map, en_min, en_max):
    """Reference keV conversion 511 / mu * E and window test; 0/False without a factor."""
    n = len(energies)
    result = np.zeros(n, dtype=np.float32)
    valid_mask = np.zeros(n, dtype=np.bool_)
    max_tch = cal_map.shape[0]
    max_slab = cal_map.shape[1]
    max_reg = cal_map.shape[2]
    for i in range(n):
        t = time_chs[i]
        s = slabs[i]
        r = regions[i]
        e = energies[i]
        if t >= 0 and t < max_tch and s >= 0 and s < max_slab and r >= 0 and r < max_reg:
            mu = cal_map[t, s, r]
            if mu > 0:
                kev = 511.0 / mu * e
                result[i] = kev
                if kev >= en_min and kev <= en_max:
                    valid_mask[i] = True
    return result, valid_mask


def _limit_array(limits, max_ch):
    """Reference ``create_cog_limits_arrays``: float32, NaN where absent."""
    left = np.full((max_ch, MAX_SLABS), np.nan, dtype=np.float32)
    right = np.full((max_ch, MAX_SLABS), np.nan, dtype=np.float32)
    for (time_ch, slab), (low, high) in limits.values.items():
        if time_ch < max_ch and slab < MAX_SLABS:
            left[time_ch, slab], right[time_ch, slab] = low, high
    return left, right


def _has_factor(cal_map, time_chs, slabs, regions):
    inside = ((time_chs >= 0) & (time_chs < cal_map.shape[0]) & (slabs >= 0) & (slabs < cal_map.shape[1])
              & (regions >= 0) & (regions < cal_map.shape[2]))
    found = np.zeros(len(regions), dtype=bool)
    found[inside] = cal_map[time_chs[inside], slabs[inside], regions[inside]] > 0
    return found


@dataclass(frozen=True)
class ListmodeContext:
    maps: CalibrationMaps
    min_ch: int
    en_min: float
    en_max: float
    num_regions: int
    boundaries: np.ndarray
    cog_left: np.ndarray
    cog_right: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray
    doi_left: np.ndarray
    doi_right: np.ndarray
    cal_map: np.ndarray
    region_arr: np.ndarray
    offset_x: np.ndarray
    offset_y: np.ndarray
    pair_arr: np.ndarray
    xpixels: np.ndarray
    ypixels: np.ndarray

    @classmethod
    def build(cls, config, calibration, cog_limits, doi_limits, pairs, regions, metadata):
        maps = CalibrationMaps.from_mapping(config.mapping)
        max_ch = maps.max_ch
        num_regions = calibration.num_regions   # 1 for a per-slab calibration: one factor per slab
        cal_map = np.zeros((max_ch, MAX_SLABS, num_regions), dtype=np.float32)
        for key, (mu, _) in calibration.values.items():
            t, s, r = key if len(key) == 3 else (*key, 0)
            if t < max_ch and s < MAX_SLABS and r < num_regions:
                cal_map[t, s, r] = mu
        cog_left, cog_right = _limit_array(cog_limits, max_ch)
        dec = {key: value for key, value in cog_limits.values.items()}
        y_min, y_max = create_dec_lookup_arrays(dec, max_ch, max_slab=MAX_SLABS)
        doi_left, doi_right = _limit_array(doi_limits, max_ch)
        size = int(max(maps.minimodules.max(), max(sm * 1000 + mm for sm, mm in regions.values))) + 1
        region_arr = np.full(size, -1, dtype=np.int32)
        offset_x = np.zeros(size, dtype=np.float32)
        offset_y = np.zeros(size, dtype=np.float32)
        for (sm, mm), (region, (ox, oy)) in regions.values.items():
            region_arr[sm * 1000 + mm], offset_x[sm * 1000 + mm], offset_y[sm * 1000 + mm] = region, ox, oy
        pair_arr = np.full((MAX_REGION, MAX_REGION), -1, dtype=np.int32)
        for (first, second), (pair,) in pairs.values.items():
            pair_arr[first, second] = pair
        _, xpixels, ypixels = header_values(metadata)
        window = config.values["energy_range"]
        return cls(maps, int(config.values["min_ch"]), float(window[0]), float(window[1]), num_regions,
                   create_region_boundaries(num_regions), cog_left, cog_right, y_min, y_max, doi_left,
                   doi_right, cal_map, region_arr, offset_x, offset_y, pair_arr, xpixels, ypixels)


class DebugSummary:
    """Bounded replacement for the reference debug lists; histograms equal theirs."""

    def __init__(self):
        self.energy = np.zeros(DEBUG_ENERGY_BINS, dtype=np.int64)
        self.sm_hits = {}
        self.flood = np.zeros((MAX_REGION, DEBUG_FLOOD_BINS, DEBUG_FLOOD_BINS), dtype=np.int64)

    def add(self, energies1, energies2, mm1, mm2, regions1, regions2, x1, y1, x2, y2):
        for values in (energies1[::10], energies2[::10]):     # reference subsample, per batch
            self.energy += np.histogram(values, bins=DEBUG_ENERGY_BINS, range=DEBUG_ENERGY_RANGE)[0]
        for sm in np.concatenate((mm1 // 1000, mm2 // 1000)).tolist():
            self.sm_hits[sm] = self.sm_hits.get(sm, 0) + 1
        span = [[0, Y_OFFSET * 2], [0, Y_OFFSET * 2]]
        for regions, x, y in ((regions1, x1, y1), (regions2, x2, y2)):
            for region in np.unique(regions).tolist():
                chosen = regions == region
                self.flood[region] += np.histogram2d(x[chosen], y[chosen], bins=DEBUG_FLOOD_BINS,
                                                     range=span)[0].astype(np.int64)

    def merge(self, other):
        self.energy += other.energy
        for sm, count in other.sm_hits.items():
            self.sm_hits[sm] = self.sm_hits.get(sm, 0) + count
        self.flood += other.flood

    @property
    def nbytes(self):
        return self.energy.nbytes + self.flood.nbytes + 16 * len(self.sm_hits)

    def save(self, stream):
        sms = sorted(self.sm_hits)
        np.savez(stream, energy=self.energy, flood=self.flood, sm_ids=np.array(sms, dtype=np.int64),
                 sm_counts=np.array([self.sm_hits[sm] for sm in sms], dtype=np.int64))

    @classmethod
    def load(cls, path):
        summary = cls()
        with np.load(path, allow_pickle=False) as data:
            if data["energy"].shape != summary.energy.shape or data["flood"].shape != summary.flood.shape:
                raise InputError(f"Debug summary has an unexpected shape: {path}")
            summary.energy, summary.flood = data["energy"].astype(np.int64), data["flood"].astype(np.int64)
            summary.sm_hits = dict(zip(data["sm_ids"].tolist(), data["sm_counts"].tolist()))
        return summary


def _count(counter, name, mask):
    counter[name] += int(np.count_nonzero(mask))


def process_batch(chunk, ctx, rejected, observations, slab_flags, debug=None):
    """Reference ``cog_lm_loop_vectorized`` body for one reader batch. Returns record array."""
    maps = ctx.maps
    energy_mask, time_mask, sm_mM_map_arr, local_map_arrays = (maps.energy_mask, maps.time_mask,
                                                               maps.minimodules, maps.local)
    sum_rows_cols, min_ch = maps.sum_rows_cols, ctx.min_ch
    chtype_masks = (time_mask, energy_mask)
    empty = np.zeros(0, RECORD_DTYPE)
    side1_chunk = {"header": chunk["header"][:, 0], "hits": chunk["side1"]}
    side2_chunk = {"header": chunk["header"][:, 1], "hits": chunk["side2"]}

    mask1 = filter_min_ch_vectorized(side1_chunk, energy_mask, min_ch, sum_rows_cols)
    mask2 = filter_min_ch_vectorized(side2_chunk, energy_mask, min_ch, sum_rows_cols)
    both_pass_minch = mask1 & mask2
    energies1, max_mm1 = get_maxEnergy_sm_mM_vectorized(side1_chunk, sm_mM_map_arr, energy_mask)
    energies2, max_mm2 = get_maxEnergy_sm_mM_vectorized(side2_chunk, sm_mM_map_arr, energy_mask)
    channel_ids1 = side1_chunk["hits"]["channelID"]
    channel_ids2 = side2_chunk["hits"]["channelID"]
    valid_mask1 = channel_ids1 != -1
    valid_mask2 = channel_ids2 != -1
    safe_ids1 = np.where(valid_mask1, channel_ids1, 0)
    safe_ids2 = np.where(valid_mask2, channel_ids2, 0)
    mm_ids1 = sm_mM_map_arr[safe_ids1]
    mm_ids2 = sm_mM_map_arr[safe_ids2]
    is_max_mm1 = (mm_ids1 == max_mm1[:, None]) & valid_mask1
    is_max_mm2 = (mm_ids2 == max_mm2[:, None]) & valid_mask2
    is_energy1 = energy_mask[safe_ids1] & is_max_mm1
    is_energy2 = energy_mask[safe_ids2] & is_max_mm2
    num_eng_ch_mm1 = np.sum(is_energy1, axis=1)
    num_eng_ch_mm2 = np.sum(is_energy2, axis=1)
    if not sum_rows_cols:
        mask_mm_filter1 = num_eng_ch_mm1 >= min_ch
        mask_mm_filter2 = num_eng_ch_mm2 >= min_ch
    else:
        num_hits_mm1 = np.sum(is_max_mm1, axis=1)
        num_hits_mm2 = np.sum(is_max_mm2, axis=1)
        mask_mm_filter1 = (num_eng_ch_mm1 >= min_ch) & (num_eng_ch_mm1 < num_hits_mm1)
        mask_mm_filter2 = (num_eng_ch_mm2 >= min_ch) & (num_eng_ch_mm2 < num_hits_mm2)
    both_pass_mm = both_pass_minch & mask_mm_filter1 & mask_mm_filter2
    _count(rejected, "min_channels", ~both_pass_minch)
    _count(rejected, "minimodule_channels", both_pass_minch & ~both_pass_mm)
    if not np.any(both_pass_mm):        # reference early exit: no slab draws for this batch
        return empty

    filtered_side1 = {"header": side1_chunk["header"][both_pass_mm], "hits": side1_chunk["hits"][both_pass_mm]}
    filtered_side2 = {"header": side2_chunk["header"][both_pass_mm], "hits": side2_chunk["hits"][both_pass_mm]}
    filtered_max_mm1 = max_mm1[both_pass_mm]
    filtered_max_mm2 = max_mm2[both_pass_mm]
    filtered_energies1 = energies1[both_pass_mm]
    filtered_energies2 = energies2[both_pass_mm]
    # Same call order and array lengths as the reference: its np.random draws are preserved.
    slab1, flags1, x1, time_ch1 = get_slab_cornell_vectorized(filtered_side1, filtered_max_mm1, sm_mM_map_arr,
                                                              time_mask, local_map_arrays)
    slab2, flags2, x2, time_ch2 = get_slab_cornell_vectorized(filtered_side2, filtered_max_mm2, sm_mM_map_arr,
                                                              time_mask, local_map_arrays)
    valid_slabs = (slab1 != -1) & (slab2 != -1)
    _, y_centroid_early1 = calculate_centroid_vectorized(filtered_side1, filtered_max_mm1, sm_mM_map_arr,
                                                         local_map_arrays, chtype_masks, x_rtp=1, y_rtp=RTP_Y_VAL,
                                                         sum_rows_cols=sum_rows_cols)
    _, y_centroid_early2 = calculate_centroid_vectorized(filtered_side2, filtered_max_mm2, sm_mM_map_arr,
                                                         local_map_arrays, chtype_masks, x_rtp=1, y_rtp=RTP_Y_VAL,
                                                         sum_rows_cols=sum_rows_cols)
    pos_region1 = compute_region_vectorized(y_centroid_early1, time_ch1, slab1, ctx.cog_left, ctx.cog_right,
                                            ctx.boundaries, ctx.num_regions)
    pos_region2 = compute_region_vectorized(y_centroid_early2, time_ch2, slab2, ctx.cog_left, ctx.cog_right,
                                            ctx.boundaries, ctx.num_regions)
    energy_kev1, valid_en1 = convert_energy_numba(time_ch1, slab1, pos_region1, filtered_energies1, ctx.cal_map,
                                                  ctx.en_min, ctx.en_max)
    energy_kev2, valid_en2 = convert_energy_numba(time_ch2, slab2, pos_region2, filtered_energies2, ctx.cal_map,
                                                  ctx.en_min, ctx.en_max)
    en_filter = valid_en1 & valid_en2
    remaining_filters_pass = valid_slabs & en_filter
    no_region = valid_slabs & ((pos_region1 < 0) | (pos_region2 < 0))
    no_factor = (valid_slabs & ~no_region & ~(_has_factor(ctx.cal_map, time_ch1, slab1, pos_region1)
                                              & _has_factor(ctx.cal_map, time_ch2, slab2, pos_region2)))
    _count(rejected, "unresolved_slab", ~valid_slabs)
    _count(rejected, "no_position_region", no_region)
    _count(rejected, "missing_calibration", no_factor)
    _count(rejected, "energy_window", valid_slabs & ~no_region & ~no_factor & ~en_filter)
    if not np.any(remaining_filters_pass):
        return empty

    keep = remaining_filters_pass
    hits1 = filtered_side1["hits"][keep]
    hits2 = filtered_side2["hits"][keep]
    final_side1 = {"header": filtered_side1["header"][keep], "hits": hits1}
    final_side2 = {"header": filtered_side2["header"][keep], "hits": hits2}
    final_max_mm1, final_max_mm2 = filtered_max_mm1[keep], filtered_max_mm2[keep]
    final_slab1, final_slab2 = slab1[keep], slab2[keep]
    final_x1, final_x2 = x1[keep], x2[keep]
    final_time_ch1, final_time_ch2 = time_ch1[keep], time_ch2[keep]
    final_energy_kev1, final_energy_kev2 = energy_kev1[keep], energy_kev2[keep]
    for flags in (flags1[keep], flags2[keep]):
        for flag, count in zip(*np.unique(flags, return_counts=True)):
            slab_flags[int(flag)] = slab_flags.get(int(flag), 0) + int(count)
    y_centroid1, y_centroid2 = y_centroid_early1[keep], y_centroid_early2[keep]

    doi1 = calculate_DOI_vectorized(final_side1, final_max_mm1, sm_mM_map_arr, local_map_arrays, chtype_masks,
                                    sum_rows_cols=sum_rows_cols, slab_orientation="x")
    doi2 = calculate_DOI_vectorized(final_side2, final_max_mm2, sm_mM_map_arr, local_map_arrays, chtype_masks,
                                    sum_rows_cols=sum_rows_cols, slab_orientation="x")
    y1_min = ctx.y_min[final_time_ch1, final_slab1]
    y1_max = ctx.y_max[final_time_ch1, final_slab1]
    y2_min = ctx.y_min[final_time_ch2, final_slab2]
    y2_max = ctx.y_max[final_time_ch2, final_slab2]
    valid_decomp = ~(np.isnan(y1_min) | np.isnan(y1_max) | np.isnan(y2_min) | np.isnan(y2_max))
    doi_left1 = ctx.doi_left[final_time_ch1, final_slab1]
    doi_right1 = ctx.doi_right[final_time_ch1, final_slab1]
    doi_left2 = ctx.doi_left[final_time_ch2, final_slab2]
    doi_right2 = ctx.doi_right[final_time_ch2, final_slab2]
    valid_doi = ~(np.isnan(doi_left1) | np.isnan(doi_right1) | np.isnan(doi_left2) | np.isnan(doi_right2))
    # The reference indexes these with two differently sized masks (it raises unless every pair has
    # decompression limits). Region assignment already requires those limits, so apply one mask.
    _count(rejected, "no_position_region", ~valid_decomp)
    _count(rejected, "missing_doi_limits", valid_decomp & ~valid_doi)
    keep = valid_decomp & valid_doi
    if not np.any(keep):
        return empty
    y1_min, y1_max, y2_min, y2_max = y1_min[keep], y1_max[keep], y2_min[keep], y2_max[keep]
    doi_left1, doi_right1 = doi_left1[keep], doi_right1[keep]
    doi_left2, doi_right2 = doi_left2[keep], doi_right2[keep]
    y_centroid1, y_centroid2 = y_centroid1[keep], y_centroid2[keep]
    final_max_mm1, final_max_mm2 = final_max_mm1[keep], final_max_mm2[keep]
    final_x1, final_x2 = final_x1[keep], final_x2[keep]
    doi1, doi2 = doi1[keep], doi2[keep]
    final_energy_kev1, final_energy_kev2 = final_energy_kev1[keep], final_energy_kev2[keep]
    hits1, hits2 = hits1[keep], hits2[keep]

    mm1_row = (final_max_mm1 % 1000) // 4
    mm2_row = (final_max_mm2 % 1000) // 4
    y1_decompressed = (y_centroid1 - y1_min) * (Y_OFFSET / (y1_max - y1_min + 1e-9))
    y2_decompressed = (y_centroid2 - y2_min) * (Y_OFFSET / (y2_max - y2_min + 1e-9))
    doi1 = (doi1 - doi_right1) * (CRYSTAL_THICKNESS / (doi_left1 - doi_right1 + 1e-9))
    doi2 = (doi2 - doi_right2) * (CRYSTAL_THICKNESS / (doi_left2 - doi_right2 + 1e-9))
    valid_y = (y1_decompressed >= 0) & (y1_decompressed <= Y_OFFSET) & (y2_decompressed >= 0) & (y2_decompressed <= Y_OFFSET)
    valid_z = (doi1 >= 0) & (doi1 <= CRYSTAL_THICKNESS) & (doi2 >= 0) & (doi2 <= CRYSTAL_THICKNESS)
    _count(rejected, "y_out_of_range", ~valid_y)
    _count(rejected, "doi_out_of_range", valid_y & ~valid_z)
    keep = valid_y & valid_z
    if not np.any(keep):
        return empty
    y1_decompressed, y2_decompressed = y1_decompressed[keep], y2_decompressed[keep]
    doi1, doi2 = doi1[keep], doi2[keep]
    mm1_row, mm2_row = mm1_row[keep], mm2_row[keep]
    final_max_mm1, final_max_mm2 = final_max_mm1[keep], final_max_mm2[keep]
    final_x1, final_x2 = final_x1[keep], final_x2[keep]
    final_energy_kev1, final_energy_kev2 = final_energy_kev1[keep], final_energy_kev2[keep]
    hits1, hits2 = hits1[keep], hits2[keep]

    y1_final = y1_decompressed + (3 - mm1_row) * Y_OFFSET
    y2_final = y2_decompressed + (3 - mm2_row) * Y_OFFSET
    x1_final = final_x1 - ctx.offset_x[final_max_mm1]
    y1_final = y1_final - ctx.offset_y[final_max_mm1]
    x2_final = final_x2 - ctx.offset_x[final_max_mm2]
    y2_final = y2_final - ctx.offset_y[final_max_mm2]

    region1 = ctx.region_arr[final_max_mm1]
    region2 = ctx.region_arr[final_max_mm2]
    mapped = (region1 >= 0) & (region2 >= 0)       # the reference would read pair row 99 for -1
    safe1, safe2 = np.where(mapped, region1, 0), np.where(mapped, region2, 0)
    pairs = ctx.pair_arr[safe1, safe2]
    swap_mask = pairs == -1
    pairs = np.where(swap_mask, ctx.pair_arr[safe2, safe1], pairs)
    pairs = np.where(mapped, pairs, -1)
    _count(rejected, "unmapped_region", ~mapped)
    _count(rejected, "no_pair", mapped & (pairs == -1))
    keep = pairs != -1
    if not np.any(keep):
        return empty
    pairs, swap_mask = pairs[keep], swap_mask[keep]
    region1, region2 = region1[keep], region2[keep]
    x1_final, y1_final, x2_final, y2_final = x1_final[keep], y1_final[keep], x2_final[keep], y2_final[keep]
    doi1, doi2 = doi1[keep], doi2[keep]
    final_energy_kev1, final_energy_kev2 = final_energy_kev1[keep], final_energy_kev2[keep]
    hits1, hits2 = hits1[keep], hits2[keep]
    final_max_mm1, final_max_mm2 = final_max_mm1[keep], final_max_mm2[keep]

    timestamps1 = get_timestamp_of_max_energy_hit_vectorized(hits1, sm_mM_map_arr, time_mask, final_max_mm1)
    timestamps2 = get_timestamp_of_max_energy_hit_vectorized(hits2, sm_mM_map_arr, time_mask, final_max_mm2)
    valid_time_mask = (timestamps1 != -1) & (timestamps2 != -1)
    _count(rejected, "missing_timestamp", ~valid_time_mask)
    if not np.any(valid_time_mask):
        return empty
    timestamps = np.minimum(timestamps1, timestamps2)
    dt_values = (timestamps1 - timestamps2).astype(np.int32)
    keep = valid_time_mask
    timestamps, dt_values = timestamps[keep], dt_values[keep]
    pairs, swap_mask = pairs[keep], swap_mask[keep]
    region1, region2 = region1[keep], region2[keep]
    x1_final, y1_final, x2_final, y2_final = x1_final[keep], y1_final[keep], x2_final[keep], y2_final[keep]
    doi1, doi2 = doi1[keep], doi2[keep]
    final_energy_kev1, final_energy_kev2 = final_energy_kev1[keep], final_energy_kev2[keep]
    final_max_mm1, final_max_mm2 = final_max_mm1[keep], final_max_mm2[keep]

    n_write = len(timestamps)
    pixels_x1 = np.searchsorted(ctx.xpixels, x1_final, side="right") - 1
    pixels_y1 = np.searchsorted(ctx.ypixels, y1_final, side="right") - 1
    pixels_x2 = np.searchsorted(ctx.xpixels, x2_final, side="right") - 1
    pixels_y2 = np.searchsorted(ctx.ypixels, y2_final, side="right") - 1
    coincidences = np.zeros(n_write, CoincidenceV5)
    coincidences["time"] = timestamps
    coincidences["pair"] = pairs
    coincidences["amount"] = 1.0
    coincidences["energy1"] = np.where(swap_mask, np.round(final_energy_kev2), np.round(final_energy_kev1))
    coincidences["energy2"] = np.where(swap_mask, np.round(final_energy_kev1), np.round(final_energy_kev2))
    dt_written = np.where(swap_mask, -dt_values, dt_values)
    coincidences["dt"] = dt_written
    coincidences["xPosition1"] = np.where(swap_mask, pixels_x2, pixels_x1)
    coincidences["yPosition1"] = np.where(swap_mask, pixels_y2, pixels_y1)
    coincidences["zPosition1"] = np.where(swap_mask, doi2, doi1)
    coincidences["xPosition2"] = np.where(swap_mask, pixels_x1, pixels_x2)
    coincidences["yPosition2"] = np.where(swap_mask, pixels_y1, pixels_y2)
    coincidences["zPosition2"] = np.where(swap_mask, doi1, doi2)
    _count(observations, "dt_wrapped_int16", (dt_written < -32768) | (dt_written > 32767))
    nx, ny = len(ctx.xpixels) - 1, len(ctx.ypixels) - 1
    _count(observations, "pixel_outside_grid",
           (pixels_x1 < 0) | (pixels_x1 >= nx) | (pixels_x2 < 0) | (pixels_x2 >= nx)
           | (pixels_y1 < 0) | (pixels_y1 >= ny) | (pixels_y2 < 0) | (pixels_y2 >= ny))
    if debug is not None:
        debug.add(final_energy_kev1, final_energy_kev2, final_max_mm1, final_max_mm2, region1, region2,
                  x1_final, y1_final, x2_final, y2_final)
    return coincidences


# Files ----------------------------------------------------------------------------

def natural_key(name):
    """``natsort`` default order for names: digit runs compare as integers."""
    return tuple((1, int(part), "") if part.isdigit() else (0, 0, part)
                 for part in re.split(r"(\d+)", name)) + ((0, 0, name),)


def segment_name(input_path):
    name = Path(input_path).name.replace(".ldat", ".lm")     # reference per-file LM name
    return name if name.endswith(".lm") else name + ".lm"


def merged_name(first_segment):
    """Reference ``"_".join(name.split("_")[:-1]) + "_all.lm"``, on the basename only."""
    stem = "_".join(first_segment.split("_")[:-1])
    return (stem or first_segment[:-3]) + "_all.lm"


def default_job_name(first_input, num_regions, now):
    """Reference LM directory name ``lm-<YYYYmmdd-HHMMSS>-<experiment>_COG_POSITION_<n>REG``."""
    experiment = "_".join(Path(first_input).name.split("_")[1:-2])
    return f"lm-{now.strftime('%Y%m%d-%H%M%S')}-{experiment}_COG_POSITION_{num_regions}REG"


def _sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path, value):
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with open(path, "xb") as out:
        out.write(content)
        out.flush()
        os.fsync(out.fileno())


def _read_json(path):
    try:
        return json.loads(_metadata_bytes(path), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise InputError(f"Unreadable listmode record {path}: {exc}") from exc


def _plain_dir(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise InputError(f"Not a plain directory: {path}")


@dataclass(frozen=True)
class FileListmode:
    path: Path
    validated_records: int
    records_read: int
    records_written: int
    rejected: dict
    observations: dict
    slab_flags: dict
    segment: Path
    segment_sha256: str
    segment_bytes: int
    reused: bool = False


@dataclass(frozen=True)
class ListmodeResult:
    output: Path
    sidecar: Path
    sha256: str
    records: int
    files: tuple
    job_sha256: str
    resumed: bool
    ignored: tuple
    debug_outputs: tuple = ()
    debug: DebugSummary | None = field(default=None, compare=False, repr=False)

    def totals(self, name):
        total = {}
        for item in self.files:
            for key, value in getattr(item, name).items():
                total[key] = total.get(key, 0) + value
        return total


def _check_request(descriptors, config, calibration, cog_limits, doi_limits, pairs, regions, metadata,
                   batch_records, hit_limit=None):
    descriptors = tuple(descriptors)
    if not descriptors or any(not isinstance(d, InputDescriptor) or d.population != Population.COINCIDENCE
                              or not Path(d.path).is_absolute() for d in descriptors):
        raise InputError("Listmode requires absolute fixed or compact coincidence inputs with explicit descriptors")
    if len({d.format for d in descriptors}) > 1:
        raise InputError("Do not mix fixed and compact inputs in one listmode job")
    if descriptors[0].format == DataFormat.COMPACT:
        if type(hit_limit) is not int or not 1 <= hit_limit <= 255:
            raise InputError("Compact listmode requires the conversion hit limit (1-255 hits per side)")
    elif hit_limit is not None:
        raise InputError("A fixed input carries its own hit limit; give none")
    if not isinstance(config, ProcessingConfig) or "energy_range" not in config.values:
        raise InputError("Listmode requires a typed processing config with a keV energy_range")
    if float(config.values["energy_range"][1]) > MAX_ENERGY_KEV:
        raise InputError("energy_range upper exceeds the uint16 keV record field")
    if not isinstance(calibration, Calibration):
        raise InputError("Listmode requires a typed position calibration")
    expected = create_region_boundaries(calibration.num_regions)
    if calibration.region_boundaries is not None and not np.allclose(calibration.region_boundaries, expected,
                                                                     rtol=0, atol=1e-12):
        raise InputError("Calibration region boundaries differ from the listmode 1.8-edge layout")
    for limits, kind in ((cog_limits, "cog"), (doi_limits, "doi")):
        if not isinstance(limits, Limits) or limits.kind != kind:
            raise InputError(f"Listmode requires typed {kind.upper()} limits")
    if not isinstance(pairs, PairMap) or not isinstance(regions, RegionMap):
        raise InputError("Listmode requires typed pair and region maps")
    _complete_metadata(metadata)
    header_bytes(metadata)
    if type(batch_records) is not int or batch_records < 1:
        raise InputError("batch_records must be a positive integer")
    names = [Path(d.path).name for d in descriptors]
    if len({segment_name(name) for name in names}) != len(names):
        raise InputError("Listmode inputs need distinct basenames (one segment each)")
    if names != sorted(names, key=natural_key):
        raise InputError("List listmode inputs in natural file order (the reference merge order)")
    return descriptors


def job_record(descriptors, config, calibration, cog_limits, doi_limits, pairs, regions, metadata,
               batch_records, debug, hit_limit=None):
    """Everything that determines the LM bytes; resume requires an identical record."""
    inputs = []
    for d in descriptors:
        try:
            info = os.stat(d.path)
        except OSError as exc:
            raise InputError(f"Listmode input unavailable: {d.path}: {exc}") from exc
        inputs.append({"path": str(Path(d.path).resolve()), "size_bytes": info.st_size,
                       "mtime_ns": info.st_mtime_ns, "format": d.format.value, "population": d.population.value})
    source = lambda item: {"path": str(item.path), "sha256": item.sha256}
    job = {
        "generator": GENERATOR,
        "inputs": inputs,
        "processing_config": {**source(config), "min_ch": int(config.values["min_ch"]),
                              "energy_range_kev": [float(v) for v in config.values["energy_range"]],
                              "en_min_ch_au": config.values.get("en_min_ch")},
        "map": source(config.mapping),
        "calibration": {**source(calibration), "layout": calibration.layout, "num_regions": calibration.num_regions,
                        "keys_without_factor": calibration.unfitted,
                        "non_positive_mu_as_no_factor": {"count": len(calibration.non_positive),
                                                         "keys": [list(k) for k in calibration.non_positive]},
                        "region_provenance": calibration.region_provenance},
        "cog_limits": {**source(cog_limits), "zero_width_keys": [list(k) for k in cog_limits.zero_width]},
        "doi_limits": {**source(doi_limits), "zero_width_keys": [list(k) for k in doi_limits.zero_width]},
        "pair_map": source(pairs), "region_map": source(regions),
        "metadata": {item.name: getattr(metadata, item.name) for item in fields(metadata)},
        "header_sha256": hashlib.sha256(header_bytes(metadata)).hexdigest(),
        "batch_records": batch_records, "debug": bool(debug),
    }
    if hit_limit is not None:      # only compact jobs: fixed job records stay as before
        job["compact_decoding"] = {"hit_limit": hit_limit,
                                   "layout": "fixed coincidence of hit_limit slots, empty slots channel -1"}
    return job


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _differences(previous, current):
    keys = sorted(set(previous) | set(current))
    return [key for key in keys if previous.get(key) != current.get(key)]


def _prepare(destination, job, resume, in_place=False):
    """New job: create ``destination`` exclusively. Resume: accept only this job's own files.

    ``in_place`` (T29): ``destination`` is the caller's existing folder, created exclusively for this job; the
    job file, ``segments/`` and every output are still created exclusively. Entries not named like this job's
    files (job file, segments, the merged output's stem) belong to the caller and are ignored on resume.
    """
    destination = Path(destination)
    if not destination.is_absolute():
        raise InputError("Listmode destination must be absolute")
    segments = destination / SEGMENTS
    expected = {segment_name(item["path"]) for item in job["inputs"]}
    output = merged_name(sorted(expected, key=natural_key)[0])
    if not resume:
        if in_place:
            _plain_dir(destination)
            taken = [name for name in os.listdir(destination)
                     if name in (JOB_FILE, SEGMENTS) or name.startswith(output[:-3])]
            if taken:
                raise InputError("Listmode outputs already exist in the destination (never replaced): "
                                 + ", ".join(sorted(taken)))
        else:
            _plain_dir(destination.parent)
            destination.mkdir()            # exclusive; never adopt an existing directory
        _write_json_exclusive(destination / JOB_FILE, {"schema_version": 1, "job": job, "job_sha256": _digest(job)})
        segments.mkdir()
        return {}, ()
    _plain_dir(destination)
    _plain_dir(segments)
    record = _read_json(destination / JOB_FILE)
    if not isinstance(record, dict) or record.get("schema_version") != 1 or not isinstance(record.get("job"), dict):
        raise InputError("Resume refused: unsupported listmode job record")
    if record.get("job_sha256") != _digest(record["job"]):
        raise InputError("Resume refused: listmode job record is inconsistent")
    differing = _differences(record["job"], job)
    if differing:
        raise InputError("Resume refused: settings/inputs differ from the previous job: " + ", ".join(differing))
    completed, ignored, unexpected = {}, [], []
    for entry in sorted(os.listdir(segments)):
        if entry.endswith(".json") and entry[:-5] in expected:
            completed[entry[:-5]] = segments / entry
        elif any(re.fullmatch(re.escape(name) + r"-[0-9a-f]{32}(\.part|\.debug\.npz)", entry) for name in expected):
            ignored.append(segments / entry)
        else:
            unexpected.append(segments / entry)
    for entry in sorted(os.listdir(destination)):
        if entry in (JOB_FILE, SEGMENTS):
            continue
        if in_place and not entry.startswith(output[:-3]):
            continue                       # the caller's own files (request, result, run record)
        if entry == output:
            raise InputError(f"Resume refused: merged listmode already exists: {destination / entry}")
        if re.fullmatch(re.escape(output) + r"-[0-9a-f]{32}\.part", entry):
            ignored.append(destination / entry)
        else:
            unexpected.append(destination / entry)
    if unexpected:
        raise InputError("Resume refused: unrecorded files in the listmode job: "
                         + ", ".join(str(path) for path in unexpected))
    referenced = set()
    for path in completed.values():
        record = _read_json(path)
        for key in ("segment", "debug"):
            if isinstance(record, dict) and isinstance(record.get(key), dict):
                referenced.add(segments / str(record[key].get("name")))
    return completed, tuple(path for path in ignored if path not in referenced)


def _reuse(record_path, job_sha256, job_input, debug):
    """A completed segment is reused only if its record, input identity and bytes all match."""
    record = _read_json(record_path)
    segments = record_path.parent
    try:
        if record["schema_version"] != 1 or record["job_sha256"] != job_sha256 or record["input"] != job_input:
            raise InputError(f"Resume refused: segment record does not match this job: {record_path}")
        segment = segments / record["segment"]["name"]
        if segment.parent != segments or not re.fullmatch(r"[^/\\]+-[0-9a-f]{32}\.part", segment.name):
            raise InputError(f"Resume refused: invalid segment name in {record_path}")
        info = segment.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != record["segment"]["bytes"] or \
                info.st_size != record["records_written"] * RECORD_DTYPE.itemsize or \
                _sha256(segment) != record["segment"]["sha256"]:
            raise InputError(f"Resume refused: segment bytes changed: {segment}")
        summary = None
        if debug:
            path = segments / record["debug"]["name"]
            if path.parent != segments or _sha256(path) != record["debug"]["sha256"]:
                raise InputError(f"Resume refused: debug summary changed: {path}")
            summary = DebugSummary.load(path)
        item = FileListmode(Path(job_input["path"]), record["validated_records"], record["records_read"],
                            record["records_written"], dict(record["rejected"]), dict(record["observations"]),
                            {int(k): v for k, v in record["slab_flags"].items()}, segment,
                            record["segment"]["sha256"], record["segment"]["bytes"], reused=True)
    except (KeyError, TypeError, OSError) as exc:
        raise InputError(f"Resume refused: incomplete segment record {record_path}: {exc}") from exc
    return item, summary


def compact_chunks(descriptor, mapped, hit_limit, batch_records):
    """Compact coincidence records as ``read_fixed_file_numpy`` chunks of ``hit_limit`` slots (FR-22).

    Hits keep their order; empty slots are channel -1, time 0, energy 0. Chunks
    hold exactly ``batch_records`` records (the last may be shorter), as the
    fixed reader yields them. The slot count matters: the reference's float32
    sums group terms by row width.
    """
    dtype = np.dtype([("header", "u1", (2,)), ("side1", HIT_DTYPE, (hit_limit,)),
                      ("side2", HIT_DTYPE, (hit_limit,))])
    pending, count = [], 0
    for records, sides in _Reader(descriptor, mapped, batch_records, None):
        chunk = np.zeros(records, dtype)
        for s, (name, hits) in enumerate(zip(("side1", "side2"), sides)):
            if hits.shape[1] > hit_limit and (hits["channelID"][:, hit_limit:] != -1).any():
                raise InputError(f"{descriptor.path}: a side has more than {hit_limit} hits")
            width = min(hits.shape[1], hit_limit)
            chunk[name]["channelID"] = -1
            chunk[name][:, :width] = hits[:, :width]
            chunk["header"][:, s] = np.count_nonzero(hits["channelID"] != -1, axis=1)
        pending.append(chunk)
        count += records
        while count >= batch_records:
            joined = np.concatenate(pending) if len(pending) > 1 else pending[0]
            yield joined[:batch_records]
            pending, count = [joined[batch_records:]], count - batch_records
    if count:
        yield np.concatenate(pending) if len(pending) > 1 else pending[0]


def _chunks(descriptor, ctx, mapping, hit_limit, batch_records):
    if descriptor.format == DataFormat.FIXED:
        return read_fixed_file_numpy(str(descriptor.path), batch_size=batch_records, group_events=False)
    mapped = np.zeros(ctx.maps.max_ch, dtype=bool)
    mapped[[ch for ch in mapping.modules if ch < ctx.maps.max_ch]] = True
    return compact_chunks(descriptor, mapped, hit_limit, batch_records)


def file_seed(lm_seed, index):
    """The NumPy seed of one file's stream (FR-22, T25.5): fixed by the LM seed and the merge position."""
    return int(np.random.SeedSequence([lm_seed, index]).generate_state(1)[0])


def _process_file(descriptor, ctx, mapping, segments, job_sha256, job_input, *, batch_records, debug,
                  cancelled, progress, index, hit_limit=None, seed=None):
    # FR-24: one pass; each record is validated as it is read, before the merged LM exists.
    if seed is not None:
        np.random.seed(seed)      # this file's own stream; the slab rule keeps its reference call order inside
    before = os.stat(descriptor.path)
    fixed = descriptor.format == DataFormat.FIXED
    if fixed:
        limit, _, expected_records = fixed_layout(descriptor.path, 2)
        mapped = mapped_channels(mapping.modules)
    elif before.st_size == 0:
        raise InputError(f"Empty LDAT: {descriptor.path}")
    name = segment_name(descriptor.path)
    token = uuid4().hex
    segment = segments / f"{name}-{token}.part"
    rejected = dict.fromkeys(REJECTIONS, 0)
    observations = dict.fromkeys(OBSERVATIONS, 0)
    slab_flags = {}
    file_debug = DebugSummary() if debug else None
    digest = hashlib.sha256()
    records = written = 0
    with open(segment, "xb") as out:
        for chunk in _chunks(descriptor, ctx, mapping, hit_limit, batch_records):
            if cancelled is not None and cancelled():
                raise ListmodeCancelled("Listmode cancelled")
            if fixed:
                check_fixed_records(chunk, 2, limit, mapped, records, descriptor.path)
            records += len(chunk)
            payload = process_batch(chunk, ctx, rejected, observations, slab_flags, file_debug).tobytes()
            out.write(payload)
            digest.update(payload)
            written += len(payload) // RECORD_DTYPE.itemsize
            if progress is not None:
                progress(index, descriptor.path, records, written)
        out.flush()
        os.fsync(out.fileno())
    info = os.stat(descriptor.path)
    if (info.st_size, info.st_mtime_ns) != (before.st_size, before.st_mtime_ns) or (
            fixed and records != expected_records):
        raise InputError(f"Input changed while reading: {descriptor.path}")
    if records == 0:
        raise InputError(f"No coincidence records: {descriptor.path}")
    if records != written + sum(rejected.values()):
        raise InputError(f"Listmode pair accounting failed for {descriptor.path}")
    record = {"schema_version": 1, "job_sha256": job_sha256, "input": job_input,
              "validated_records": records, "records_read": records, "records_written": written,
              "rejected": rejected, "observations": observations,
              "slab_flags": {str(k): v for k, v in sorted(slab_flags.items())},
              "segment": {"name": segment.name, "bytes": segment.stat().st_size, "sha256": digest.hexdigest()},
              "debug": None}
    if file_debug is not None:
        path = segments / f"{name}-{token}.debug.npz"
        with open(path, "xb") as out:
            file_debug.save(out)
        record["debug"] = {"name": path.name, "sha256": _sha256(path)}
    _write_json_exclusive(segments / f"{name}.json", record)      # completion record, written last
    item = FileListmode(Path(descriptor.path), records, records, written, rejected, observations,
                        slab_flags, segment, digest.hexdigest(), record["segment"]["bytes"])
    return item, file_debug


_LM_WORKER = {}


def _lm_worker_init(event, ctx, mapping, segments, job_sha256, options):
    """Per LM worker process: the shared context (sent once), the job and the cancellation event."""
    _LM_WORKER.clear()
    _LM_WORKER.update(ctx=ctx, mapping=mapping, segments=segments, job_sha256=job_sha256, options=options,
                      cancelled=event.is_set)


def _lm_task(descriptor, job_input, index, seed):
    w = _LM_WORKER
    return _process_file(descriptor, w["ctx"], w["mapping"], w["segments"], w["job_sha256"], job_input,
                         cancelled=w["cancelled"], progress=None, index=index, seed=seed, **w["options"])


def _merge(destination, name, header, files):
    """Header plus segments in order, verified while copying; published without replacement."""
    temporary = destination / f"{name}-{uuid4().hex}.part"
    target = destination / name
    digest = hashlib.sha256(header)
    with open(temporary, "xb") as out:
        out.write(header)
        for item in files:
            check = hashlib.sha256()
            with open(item.segment, "rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    check.update(block)
                    digest.update(block)
                    out.write(block)
            if check.hexdigest() != item.segment_sha256:
                raise InputError(f"Segment changed before merging: {item.segment}")
        out.flush()
        os.fsync(out.fileno())
    os.link(temporary, target)              # fails if the target exists; never replaces
    os.unlink(temporary)                    # our own just-written name; data stays at the target
    return target, digest.hexdigest()


def sidecar(output, sha256, records, files, job, header_fields, *, resumed, ignored, debug_outputs):
    totals = lambda name: {key: sum(getattr(f, name).get(key, 0) for f in files)
                           for key in sorted({k for f in files for k in getattr(f, name)})}
    return {
        "schema_version": 1,
        "generator": GENERATOR,
        "job_sha256": _digest(job),
        "output": {"path": str(output), "sha256": sha256, "header_bytes": HEADER_BYTES,
                   "record_bytes": RECORD_DTYPE.itemsize, "records": records,
                   "record_type": "CoincidenceV5", "version": list(LM_VERSION)},
        "header": header_fields,
        "header_provenance": {"supplied": list(SUPPLIED_HEADER_FIELDS),
                              "reference_constants": {"identifier": IDENTIFIER, "startTime": 0,
                                                      "version": list(LM_VERSION)},
                              "zero_as_reference": list(ZERO_HEADER_FIELDS),
                              "pixel_grid": "np.linspace(0, detector size, pixels + 1) per axis"},
        "timestamp": {"field": "time (float32)",
                      "source": "raw LDAT timestamp of the max-energy time hit in each side's max minimodule; "
                                "minimum of the two sides",
                      "offset": None, "scaling": None,
                      "operator_declared_unit": job["metadata"]["timestamp_unit"],
                      "dt": "int16 of int32(side1 - side2), sign follows pair orientation; wraps outside int16"},
        "population": (f"{job['inputs'][0]['format']} coincidence pairs (two detector sides per record)"
                       + ("; compact decoded into the fixed layout of {} hits per side".format(
                           job["compact_decoding"]["hit_limit"]) if "compact_decoding" in job else "")),
        "cuts": {"min_ch": job["processing_config"]["min_ch"],
                 "energy_window_kev": job["processing_config"]["energy_range_kev"],
                 "en_min_ch_au": job["processing_config"]["en_min_ch_au"], "en_min_ch_applied": False,
                 "en_min_ch_note": "read by the reference listmode but not applied to its fixed loop",
                 "energy": "511 / mu(time channel, slab, region) * max-minimodule energy sum (a.u.)"},
        "position": {"region_policy": "normalized Y-COG clipped to [0, 0.999] (calibration excludes instead)",
                     "num_regions": job["calibration"]["num_regions"],
                     "region_boundaries": [float(b) for b in create_region_boundaries(job["calibration"]["num_regions"])],
                     "region_boundary_provenance": job["calibration"]["region_provenance"],
                     "y_cog_power": RTP_Y_VAL, "y_decompression_mm": Y_OFFSET,
                     "doi_mapping": f"linear (ratio - right) * {CRYSTAL_THICKNESS} / (left - right) mm-equivalent "
                                    "from DOI limits; light-sharing ratio, not calibrated depth",
                     "slab_rule": "src.utils_fixed.get_slab_cornell_vectorized (np.random for ambiguous sides)"},
        "sources": {key: job[key] for key in ("processing_config", "map", "calibration", "cog_limits",
                                              "doi_limits", "pair_map", "region_map")},
        "merge_order": "natural basename order (reference natsorted)",
        "random_streams": job.get("random_streams") or {
            "lm_seed": None, "rule": "reference: one unseeded NumPy stream continuing across files"},
        "inputs": [{"path": str(f.path), "validated_records": f.validated_records, "records_read": f.records_read,
                    "records_written": f.records_written, "rejected": f.rejected, "observations": f.observations,
                    "slab_flags": {str(k): v for k, v in sorted(f.slab_flags.items())},
                    "segment": str(f.segment), "segment_sha256": f.segment_sha256, "reused": f.reused}
                   for f in files],
        "totals": {"records_read": sum(f.records_read for f in files), "records_written": records,
                   "rejected": totals("rejected"), "observations": totals("observations")},
        "resume": {"resumed": resumed, "reused_segments": sum(f.reused for f in files),
                   "ignored_incomplete": [str(path) for path in ignored]},
        "segments_retained": True,
        "debug_outputs": [str(path) for path in debug_outputs],
    }


def generate_listmode(descriptors, config, calibration, cog_limits, doi_limits, pairs, regions, metadata,
                      destination, *, resume=False, debug=False, batch_records=DEFAULT_BATCH_RECORDS,
                      cancelled=None, progress=None, hit_limit=None, lm_seed=None, workers=1, in_place=False):
    """Validate, stream one segment per input, merge with the supplied header and write provenance.

    ``hit_limit``: the conversion hit limit, required for compact input only (FR-22).
    ``lm_seed`` (FR-22, T25.5): each file's ambiguous-slab draws come from its own NumPy stream,
    ``file_seed(lm_seed, index)``, so the .lm is the same on every run and for any ``workers`` count; None
    keeps the reference's one stream continuing across files (one worker only). ``workers`` > 1 processes
    files in spawned worker processes (``src.cornell.parallel``); segments are merged in the same order.
    ``in_place`` (T29): write into the caller's existing ``destination`` instead of creating it.
    """
    descriptors = _check_request(descriptors, config, calibration, cog_limits, doi_limits, pairs, regions,
                                 metadata, batch_records, hit_limit)
    if lm_seed is not None and (type(lm_seed) is not int or not 0 <= lm_seed < 2 ** 63):
        raise InputError("lm_seed must be a non-negative integer or None")
    if type(workers) is not int or workers < 1:
        raise InputError("workers must be a positive integer (resolve 0 = automatic before LM)")
    if workers > 1 and lm_seed is None:
        raise InputError("Parallel LM needs lm_seed: one random stream per file, independent of the workers")
    ctx = ListmodeContext.build(config, calibration, cog_limits, doi_limits, pairs, regions, metadata)
    job = job_record(descriptors, config, calibration, cog_limits, doi_limits, pairs, regions, metadata,
                     batch_records, debug, hit_limit)
    seeds = None if lm_seed is None else [file_seed(lm_seed, index) for index in range(len(descriptors))]
    if lm_seed is not None:            # part of the job digest: resume reuses only segments of the same seed
        job["random_streams"] = {"lm_seed": lm_seed, "file_seeds": seeds}
    job_sha256 = _digest(job)
    completed, ignored = _prepare(destination, job, resume, in_place)
    destination = Path(destination)
    segments = destination / SEGMENTS
    files, total_debug = [None] * len(descriptors), DebugSummary() if debug else None

    def keep(index, value):                  # debug summaries are sums: merged at once, never retained per file
        files[index], file_debug = value
        if total_debug is not None:
            total_debug.merge(file_debug)
    todo = []
    for index, (descriptor, job_input) in enumerate(zip(descriptors, job["inputs"])):
        name = segment_name(descriptor.path)
        if name in completed:
            keep(index, _reuse(completed[name], job_sha256, job_input, debug))
        else:
            todo.append((index, descriptor, job_input))
    options = dict(batch_records=batch_records, debug=debug, hit_limit=hit_limit)
    if workers == 1:
        for index, descriptor, job_input in todo:
            keep(index, _process_file(descriptor, ctx, config.mapping, segments, job_sha256, job_input,
                                      cancelled=cancelled, progress=progress, index=index,
                                      seed=None if seeds is None else seeds[index], **options))
    elif todo:
        from .parallel import OrderedPool, PoolCancelled

        def done(position, value):
            index, descriptor, _ = todo[position]
            keep(index, value)
            if progress is not None:
                progress(index, descriptor.path, value[0].records_read, value[0].records_written)
        try:
            with OrderedPool(min(workers, len(todo)), _lm_worker_init,
                             (ctx, config.mapping, segments, job_sha256, options), cancelled) as pool:
                pool.run(_lm_task, [(descriptor, job_input, index, seeds[index])
                                    for index, descriptor, job_input in todo], done)
        except PoolCancelled:
            raise ListmodeCancelled("Listmode cancelled") from None
    if cancelled is not None and cancelled():
        raise ListmodeCancelled("Listmode cancelled before merging")
    header = header_bytes(metadata)
    name = merged_name(segment_name(descriptors[0].path))
    output, sha256 = _merge(destination, name, header, files)
    records = sum(item.records_written for item in files)
    debug_outputs = debug_plots(total_debug, destination, name[:-3]) if debug else ()
    values, _, _ = header_values(metadata)
    side = output.with_name(output.name + ".json")
    _write_json_exclusive(side, sidecar(output, sha256, records, files, job, values, resumed=bool(resume),
                                        ignored=ignored, debug_outputs=debug_outputs))
    return ListmodeResult(output, side, sha256, records, tuple(files), job_sha256, bool(resume), ignored,
                          tuple(debug_outputs), total_debug)


def debug_plots(summary, directory, stem):
    """Reference debug figures from bounded histograms; exclusive creation."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from src.fits import fit_gaussian

    paths = []

    def save(figure, name, **options):
        path = Path(directory) / f"{stem}_{name}.png"
        with open(path, "xb") as out:
            figure.savefig(out, format="png", **options)
        plt.close(figure)
        paths.append(path)

    edges = np.histogram_bin_edges([], DEBUG_ENERGY_BINS, DEBUG_ENERGY_RANGE)
    figure = plt.figure()
    counts = summary.energy.astype(np.float64)
    plt.stairs(counts, edges, fill=True)
    try:
        x, y, pars, _, _ = fit_gaussian(counts, edges, cb=16)
        mu, sigma = pars[1], pars[2]
        plt.plot(x, y, "-r", label="fit")
        plt.legend([f"Energy res: {round(2.35 * sigma / mu * 100, 2)}%\nCentroid {round(mu, 2)}"])
    except (RuntimeError, ValueError, IndexError):
        plt.legend(["No fit (insufficient data)"])
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts (every 10th side per batch)")
    plt.title("Total energy")
    save(figure, "total_energy")

    figure = plt.figure(figsize=(15, 5))
    sms = sorted(summary.sm_hits)
    plt.bar([str(sm) for sm in sms], [summary.sm_hits[sm] for sm in sms])
    plt.xlabel("SuperModule (selected map)")
    plt.ylabel("Hits per SM (written pairs, both sides)")
    plt.title("Hits per SuperModule")
    save(figure, "sm_hits")

    figure, axes = plt.subplots(5, 20, figsize=(40, 10), constrained_layout=True)
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_under("white")
    vmax = max(int(summary.flood.max()), 1)
    image = None
    span = [0, Y_OFFSET * 2, 0, Y_OFFSET * 2]
    for region, ax in enumerate(axes.flatten()):
        if summary.flood[region].any():
            image = ax.imshow(summary.flood[region].T, origin="lower", extent=span, cmap=cmap, vmin=1.01,
                              vmax=vmax, aspect="auto")
        ax.set_title(f"R{region}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([0, 10, 20, 30, 40, 50])
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="Counts", pad=0.01)  # flat: 3.8 rejects nested lists
    figure.suptitle("Floodmaps for all regions (0-99)", fontsize=16)
    save(figure, "floodmap_all_regions", dpi=100, bbox_inches="tight")
    return tuple(paths)
