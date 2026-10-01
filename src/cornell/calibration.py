"""Fixed-position (time channel, slab, region) energy calibration (spec 003 T8).

Tracked port of the owner-confirmed reference
``scripts_cornell/cornell_slab_en_cal_fixed_position.py``; same selection,
region convention, histogram, fit call and ``.encal`` schema. Changes are
storage and provenance only:

- Per-key energy lists become a fixed 100-bin 0-200 a.u. histogram (binned
  exactly like ``np.histogram`` on the reference's float32 values) plus a count
  and float64 mean/M2 for the reference mean/std fallback. Storage grows with
  mapped keys, never with acquisition length.
- Fit status is kept: ``fitted``, ``fallback_mean_std`` (reference RuntimeError
  path, written but labelled), ``insufficient_samples`` (< 50, not written as
  in the reference), ``fit_error`` and ``invalid_result`` (not written; the
  reference would abort or write a non-positive/non-finite factor). A negative
  fitted sigma is written as its absolute value (the Gaussian is symmetric) and
  flagged.
- Side rejections are counted by reason; a JSON sidecar records boundaries,
  cuts, sampling, inputs and non-fitted keys for ``inputs.load_calibration``.

The reference applies no per-channel energy cut (``en_min_ch``) here; neither
does this port. Calibration excludes normalized COG outside [0, 1] (exactly 1
joins the last region); listmode clips instead. Do not unify them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from src.detector_features_fixed import calculate_centroid_vectorized
from src.filters_fixed import filter_min_ch_vectorized
from src.fits import fit_gaussian
from src.mapping_generator import ChannelType
from src.read_fixed import read_fixed_file_numpy
from src.utils_fixed import (create_channel_type_mask, create_local_map_arrays, create_mm_map_array,
                             get_maxEnergy_sm_mM_vectorized, get_slab_cornell_vectorized)

from src.petsys_manager.contracts import DataFormat, InputDescriptor, Population
from .inputs import InputError, Limits, ProcessingConfig, validate_ldat

HISTOGRAM_BINS = 100
HISTOGRAM_RANGE = (0, 200)          # a.u.
MIN_SAMPLES = 50
FIT_OPTIONS = {"cb": 6, "pk_finder": "peak"}
DEFAULT_REGIONS = 5
EDGE_MULTIPLIER = 1.8
DEFAULT_SIDE_LIMIT = 4_000_000      # accepted sides per file, checked between side batches
DEFAULT_BATCH_RECORDS = 5000
MAX_SLABS = 16                      # reference COG-limit array width
MAX_KEYS = 100_000
_ENERGY_DTYPE = np.dtype(np.float32)
REJECTIONS = ("min_channels", "minimodule_channels", "unresolved_slab", "missing_cog_limits",
              "cog_out_of_range")


class CalibrationCancelled(InputError):
    """Stopped before a calibration result existed."""


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
    """Reference float32 (time channel, slab < 16) COG-limit lookup arrays."""
    if not isinstance(limits, Limits) or limits.kind != "cog":
        raise InputError("Calibration requires typed COG limits")
    left = np.zeros((max_ch, MAX_SLABS), dtype=np.float32)
    right = np.zeros((max_ch, MAX_SLABS), dtype=np.float32)
    valid = np.zeros((max_ch, MAX_SLABS), dtype=bool)
    for (channel, slab), (low, high) in limits.values.items():
        if channel < max_ch and slab < MAX_SLABS:
            left[channel, slab], right[channel, slab], valid[channel, slab] = low, high, True
    return left, right, valid


def region_ids(y_cog, time_chs, slab_ids, left, right, boundaries):
    """Vectorized reference ``compute_region_id_numba``; also returns why a side has none.

    Returns (regions, missing_limits): region -1 is "no region"; missing_limits
    marks sides whose (time channel, slab) has no usable limits.
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
    keep = (normalized >= 0.0) & (normalized <= 1.0)   # NaN fails both, as in the reference
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


def select_sides(sides, maps, left, right, min_ch, boundaries):
    """Reference ``process_chunk_position`` returning arrays and rejection counts."""
    rejected = dict.fromkeys(REJECTIONS, 0)
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, _ENERGY_DTYPE))
    passed = filter_min_ch_vectorized(sides, maps.energy_mask, min_ch, maps.sum_rows_cols)
    rejected["min_channels"] = int(len(passed) - passed.sum())
    chunk = {key: value[passed] for key, value in sides.items()}
    if len(chunk.get("hits", [])) == 0:
        return (*empty, rejected)
    energies, max_mm = get_maxEnergy_sm_mM_vectorized(chunk, maps.minimodules, maps.energy_mask)
    channels = chunk["hits"]["channelID"]
    valid = channels != -1
    safe = np.where(valid, channels, 0)
    on_max = (maps.minimodules[safe] == max_mm[:, None]) & valid
    energy_hits = np.sum(maps.energy_mask[safe] & on_max, axis=1)
    if not maps.sum_rows_cols:
        keep = energy_hits >= min_ch
    else:
        keep = (energy_hits >= min_ch) & (energy_hits < np.sum(on_max, axis=1))
    rejected["minimodule_channels"] = int(len(keep) - keep.sum())
    chunk = {key: value[keep] for key, value in chunk.items()}
    max_mm, energies = max_mm[keep], energies[keep]
    if len(chunk.get("hits", [])) == 0:
        return (*empty, rejected)
    slabs, _, _, time_chs = get_slab_cornell_vectorized(chunk, max_mm, maps.minimodules, maps.time_mask,
                                                         maps.local)
    _, y_cog = calculate_centroid_vectorized(chunk, max_mm, maps.minimodules, maps.local,
                                             (maps.time_mask, maps.energy_mask), x_rtp=1.0, y_rtp=2.0,
                                             sum_rows_cols=maps.sum_rows_cols)
    regions, missing = region_ids(y_cog, time_chs, slabs, left, right, boundaries)
    resolved = slabs != -1
    accepted = resolved & (regions >= 0)
    rejected["unresolved_slab"] = int((~resolved).sum())
    rejected["missing_cog_limits"] = int((resolved & missing).sum())
    rejected["cog_out_of_range"] = int((resolved & ~missing & (regions < 0)).sum())
    return (np.asarray(time_chs)[accepted].astype(np.int64), np.asarray(slabs)[accepted].astype(np.int64),
            regions[accepted], energies[accepted], rejected)


class PositionAccumulator:
    """Per-key fixed histogram, count and float64 mean/M2; bounded by mapped keys."""

    def __init__(self, max_keys=MAX_KEYS):
        self.max_keys = max_keys
        _, self.edges = np.histogram(np.zeros(0, _ENERGY_DTYPE), bins=HISTOGRAM_BINS, range=HISTOGRAM_RANGE)
        self._rows = {}
        self.keys = []
        self.histograms = np.zeros((0, HISTOGRAM_BINS), dtype=np.int64)
        self.counts = np.zeros(0, dtype=np.int64)
        self.means = np.zeros(0, dtype=np.float64)
        self.m2 = np.zeros(0, dtype=np.float64)

    def _grow(self, size):
        extra = size - len(self.counts)
        if extra > 0:
            self.histograms = np.vstack((self.histograms, np.zeros((extra, HISTOGRAM_BINS), np.int64)))
            self.counts = np.concatenate((self.counts, np.zeros(extra, np.int64)))
            self.means = np.concatenate((self.means, np.zeros(extra)))
            self.m2 = np.concatenate((self.m2, np.zeros(extra)))

    def add(self, time_chs, slabs, regions, energies):
        energies = np.asarray(energies)
        if energies.dtype != _ENERGY_DTYPE:
            raise InputError(f"Calibration energies must be float32, got {energies.dtype}")
        if len(energies) == 0:
            return
        codes = np.stack((np.asarray(time_chs, np.int64), np.asarray(slabs, np.int64),
                          np.asarray(regions, np.int64)), axis=1)
        unique, inverse = np.unique(codes, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        rows = np.empty(len(unique), dtype=np.int64)
        for i, key in enumerate(map(tuple, unique.tolist())):
            row = self._rows.get(key)
            if row is None:
                if len(self.keys) >= self.max_keys:
                    raise InputError("Calibration keys exceed their bound")
                row = self._rows[key] = len(self.keys)
                self.keys.append(key)
            rows[i] = row
        self._grow(len(self.keys))
        target = rows[inverse]
        # Same inclusion/edges as np.histogram(values, 100, (0, 200)); last bin closed.
        keep = (energies >= HISTOGRAM_RANGE[0]) & (energies <= HISTOGRAM_RANGE[1])
        bins = np.searchsorted(self.edges, energies[keep], side="right") - 1
        bins[bins == HISTOGRAM_BINS] = HISTOGRAM_BINS - 1
        np.add.at(self.histograms, (target[keep], bins), 1)
        values = energies.astype(np.float64)
        count = np.bincount(inverse, minlength=len(unique)).astype(np.int64)
        mean = np.bincount(inverse, weights=values, minlength=len(unique)) / count
        m2 = np.bincount(inverse, weights=(values - mean[inverse]) ** 2, minlength=len(unique))
        total = self.counts[rows] + count
        delta = mean - self.means[rows]
        self.means[rows] += delta * count / total
        self.m2[rows] += m2 + delta ** 2 * self.counts[rows] * count / total
        self.counts[rows] = total

    @property
    def nbytes(self):
        return self.histograms.nbytes + self.counts.nbytes + self.means.nbytes + self.m2.nbytes

    def row(self, key):
        return self._rows[key]


@dataclass(frozen=True)
class FileSample:
    path: Path
    population: Population
    validated_records: int
    records_read: int
    sides_considered: int
    accepted_sides: int
    stopped_at_limit: bool
    rejected: dict


@dataclass(frozen=True)
class CalibrationEntry:
    key: tuple
    samples: int
    in_histogram: int
    status: str      # fitted | fallback_mean_std | insufficient_samples | fit_error | invalid_result
    mu: float | None = None
    sigma: float | None = None
    message: str = ""

    @property
    def written(self):
        return self.status in ("fitted", "fallback_mean_std")


@dataclass(frozen=True)
class CalibrationResult:
    num_regions: int
    boundaries: tuple
    population: Population
    min_ch: int
    side_limit: int | None
    batch_records: int
    files: tuple
    entries: tuple
    edge_multiplier: float = EDGE_MULTIPLIER
    sources: dict = field(default_factory=dict)

    @property
    def written(self):
        return tuple(entry for entry in self.entries if entry.written)

    def status_counts(self):
        counts = {}
        for entry in self.entries:
            counts[entry.status] = counts.get(entry.status, 0) + 1
        return counts


def sample_file(descriptor, maps, left, right, min_ch, boundaries, accumulator, *,
                side_limit=DEFAULT_SIDE_LIMIT, batch_records=DEFAULT_BATCH_RECORDS, cancelled=None,
                validated_records=0):
    """Reference ``extract_data_dict_position``: limit tested before each record batch and side batch."""
    coincidence = descriptor.population == Population.COINCIDENCE
    accepted = considered = records = 0
    rejected = dict.fromkeys(REJECTIONS, 0)
    stopped = False
    for chunk in read_fixed_file_numpy(str(descriptor.path), batch_records, group_events=not coincidence):
        if cancelled is not None and cancelled():
            raise CalibrationCancelled("Calibration cancelled")
        if side_limit and accepted >= side_limit:
            stopped = True
            break
        records += len(chunk)
        if coincidence:
            batches = [{"header": chunk["header"][:, 0], "hits": chunk["side1"]},
                       {"header": chunk["header"][:, 1], "hits": chunk["side2"]}]
        else:
            batches = [{"header": chunk["header"], "hits": chunk["hits"]}]
        for sides in batches:
            if side_limit and accepted >= side_limit:
                stopped = True
                break
            considered += len(sides["hits"])
            *selected, counts = select_sides(sides, maps, left, right, min_ch, boundaries)
            accumulator.add(*selected)
            accepted += len(selected[3])
            for reason, value in counts.items():
                rejected[reason] += value
        if stopped:
            break
    return FileSample(descriptor.path, descriptor.population, validated_records, records, considered,
                      accepted, stopped, rejected)


def fit_entries(accumulator):
    """Reference photopeak extraction per sorted key, with explicit status."""
    entries = []
    for key in sorted(accumulator.keys):
        row = accumulator.row(key)
        samples = int(accumulator.counts[row])
        histogram = accumulator.histograms[row].copy()
        in_histogram = int(histogram.sum())
        if samples < MIN_SAMPLES:
            entries.append(CalibrationEntry(key, samples, in_histogram, "insufficient_samples",
                                            message=f"{samples} < {MIN_SAMPLES} samples"))
            continue
        message = ""
        try:
            _, _, pars, _, _ = fit_gaussian(histogram, accumulator.edges, **FIT_OPTIONS)
            mu, sigma, status = float(pars[1]), float(pars[2]), "fitted"
        except RuntimeError as exc:
            mu = float(accumulator.means[row])
            sigma = math.sqrt(float(accumulator.m2[row]) / samples)
            status, message = "fallback_mean_std", f"Fit failed ({exc}); population mean/std written"
        except Exception as exc:  # The reference would abort the whole run here.
            entries.append(CalibrationEntry(key, samples, in_histogram, "fit_error",
                                            message=f"{type(exc).__name__}: {exc}"))
            continue
        if not (math.isfinite(mu) and mu > 0 and math.isfinite(sigma)):
            entries.append(CalibrationEntry(key, samples, in_histogram, "invalid_result",
                                            message=f"Non-usable factor mu={mu!r} sigma={sigma!r} ({status})"))
            continue
        if sigma < 0:
            sigma = -sigma
            message = (message + "; " if message else "") + "negative fitted sigma written as its absolute value"
        entries.append(CalibrationEntry(key, samples, in_histogram, status, mu, sigma, message))
    return tuple(entries)


def calibrate(descriptors, config, limits, *, num_regions=DEFAULT_REGIONS, side_limit=DEFAULT_SIDE_LIMIT,
              batch_records=DEFAULT_BATCH_RECORDS, cancelled=None):
    """Validate every input (T5), sample in input order, then fit. No files are written."""
    descriptors = tuple(descriptors)
    if not isinstance(config, ProcessingConfig):
        raise InputError("Calibration requires a typed processing config")
    if not descriptors or any(not isinstance(d, InputDescriptor) or d.format != DataFormat.FIXED
                              for d in descriptors):
        raise InputError("Calibration requires fixed LDAT inputs with explicit descriptors")
    populations = {d.population for d in descriptors}
    if len(populations) != 1:
        raise InputError("Do not mix group/coincidence calibration inputs")
    if type(num_regions) is not int or not 1 <= num_regions <= 127:
        raise InputError("num_regions must be an integer from 1 to 127")
    if side_limit is not None and (type(side_limit) is not int or side_limit < 1):
        raise InputError("Accepted-side limit must be a positive integer or None")
    mapping = config.mapping
    maps = CalibrationMaps.from_mapping(mapping)
    left, right, _ = limit_arrays(limits, maps.max_ch)
    boundaries = create_region_boundaries(num_regions)
    min_ch = int(config.values["min_ch"])
    accumulator = PositionAccumulator()
    files = []
    for descriptor in descriptors:
        summary = validate_ldat(descriptor, mapping.modules, batch_records=batch_records, cancelled=cancelled)
        sample = sample_file(descriptor, maps, left, right, min_ch, boundaries, accumulator,
                             side_limit=side_limit, batch_records=batch_records, cancelled=cancelled,
                             validated_records=summary.records)
        info = os.stat(descriptor.path)
        if (info.st_size, info.st_mtime_ns) != (summary.file_size, summary.mtime_ns):
            raise InputError(f"Input changed after validation: {descriptor.path}")
        files.append(sample)
    sources = {"processing_config": {"path": str(config.path), "sha256": config.sha256},
               "map": {"path": str(mapping.path), "sha256": mapping.sha256},
               "cog_limits": {"path": str(limits.path), "sha256": limits.sha256}}
    return CalibrationResult(num_regions, tuple(float(b) for b in boundaries), populations.pop(), min_ch,
                             side_limit, batch_records, tuple(files), fit_entries(accumulator),
                             sources=sources)


def encal_text(result):
    """Reference ``write_position_cal`` content for the written (fitted/fallback) entries."""
    lines = [f"# Position-dependent energy calibration ({result.num_regions} regions per slab)\n",
             "ID(time_ch, slab, region)\tmu\tsigma\n"]
    for entry in sorted(result.written, key=lambda e: e.key):
        lines.append(f"{tuple(int(v) for v in entry.key)}\t{entry.mu:.3f}\t{entry.sigma:.3f}\n")
    return "".join(lines)


def sidecar(result, calibration_sha256):
    """JSON provenance accepted by ``inputs.load_calibration(metadata_path=...)``. Bounded:
    fitted keys are implied by the .encal; only non-fitted keys are listed."""
    return {
        "schema_version": 1,
        "num_regions": result.num_regions,
        "region_boundaries": list(result.boundaries),
        "calibration_sha256": calibration_sha256,
        "generator": "src.cornell.calibration (spec 003 T8; reference cornell_slab_en_cal_fixed_position.py)",
        "population": result.population.value,
        "energy_units": "a.u.",
        "edge_multiplier": result.edge_multiplier,
        "cuts": {"min_ch": result.min_ch, "energy_channel_cut_au": None,
                 "energy_channel_cut_note": "reference calibration applies no per-channel energy cut",
                 "cog_policy": "normalized COG outside [0, 1] excluded; exactly 1 joins the last region"},
        "histogram": {"bins": HISTOGRAM_BINS, "range_au": list(HISTOGRAM_RANGE), "min_samples": MIN_SAMPLES,
                      "fit": "src.fits.fit_gaussian(cb=6, pk_finder='peak')",
                      "fallback": "population mean/std (ddof=0) after a fit RuntimeError"},
        "sampling": {"batch_records": result.batch_records, "accepted_side_limit_per_file": result.side_limit,
                     "limit_semantics": "tested before each record batch and each side batch; may overshoot",
                     "accepted_unit": "detector sides" if result.population == Population.COINCIDENCE else "groups"},
        "sources": result.sources,
        "inputs": [{"path": str(f.path), "validated_records": f.validated_records, "records_read": f.records_read,
                    "sides_considered": f.sides_considered, "accepted_sides": f.accepted_sides,
                    "stopped_at_limit": f.stopped_at_limit, "rejected": f.rejected} for f in result.files],
        "status_counts": result.status_counts(),
        "non_fitted": [{"key": list(e.key), "samples": e.samples, "status": e.status, "written": e.written,
                        "mu": e.mu, "sigma": e.sigma, "message": e.message}
                       for e in result.entries if e.status != "fitted" or e.message],
    }


def _write_exclusive(path, content):
    with open(path, "xb") as out:  # never replaces an existing calibration
        out.write(content)
        out.flush()
        os.fsync(out.fileno())


def write_calibration(result, encal_path, sidecar_path):
    """Write the .encal and its sidecar with exclusive creation. Returns the .encal SHA-256."""
    if not result.written:
        raise InputError("No calibration entry could be written (all keys below 50 samples or failed)")
    content = encal_text(result).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    metadata = (json.dumps(sidecar(result, digest), indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_exclusive(encal_path, content)
    _write_exclusive(sidecar_path, metadata)
    return digest


def plot_summary(result, path):
    """Reference summary plot (written entries), exclusive creation."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    written = result.written
    mus = [entry.mu for entry in written]
    fallback = sum(entry.status == "fallback_mean_std" for entry in written)
    figure = plt.figure(figsize=(12, 5))
    try:
        plt.subplot(1, 2, 1)
        plt.hist(mus, bins=200, range=(0, 200), alpha=0.6)
        plt.axvline(np.mean(mus), color="r", linestyle="dashed", linewidth=2)
        plt.title(f"Photopeak Distribution (All Regions; {fallback} mean/std fallbacks)")
        plt.xlabel("Mu (a.u.)")
        plt.ylabel("Frequency")
        plt.text(0.6, 0.8, f"Mean: {np.mean(mus):.2f}\nStd: {np.std(mus):.2f}", transform=plt.gca().transAxes)
        plt.subplot(1, 2, 2)
        by_region = {r: [e.mu for e in written if e.key[2] == r] for r in range(result.num_regions)}
        x = np.arange(result.num_regions)
        plt.bar(x, [np.mean(v) if v else 0 for v in by_region.values()],
                yerr=[np.std(v) if v else 0 for v in by_region.values()], capsize=5, alpha=0.7)
        plt.xlabel("Region")
        plt.ylabel("Mean Mu (a.u.)")
        plt.title("Photopeak by Region (Position Along Slab)")
        plt.xticks(x, [f"R{i}" for i in range(result.num_regions)])
        plt.tight_layout()
        with open(path, "xb") as out:
            figure.savefig(out, format="png")
    finally:
        plt.close(figure)
