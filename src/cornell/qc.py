"""Compact-coincidence legacy QC extraction, counting and fits (spec 003 T10).

Tracked port of the owner-confirmed reference
``scripts_cornell/cornell_system_validation.py``: same per-pair selection
(``en_min_ch`` reader cut, minimodule channel cuts, ``get_slab_cornell`` with
its ``random`` draws in the same call order), the same occupancy population and
the same photopeak histogram/fit call. Not a new QC model. Changes:

- Per-key energy lists become fixed 150-bin 0-250 a.u. histograms (binned
  exactly like ``np.histogram``) plus count and float64 mean/M2; floods become
  fixed 500 x 500 per-SuperModule histograms of the reference 40-200 a.u.
  window. Storage grows with mapped keys, never with acquisition length, and
  file results are merged as they finish instead of kept as lists.
- Populations are reported separately: records read from the iterator, pairs
  processed, pairs passing the channel cuts (the occupancy population, which
  still includes unresolved-slab pairs), accepted resolved pairs and their
  detector sides. Rejections are counted by reason.
- Fits keep a status; sparse/failed/invalid fits are unavailable rather than
  reported as a photopeak (the reference wrote mean/std as ``Mu``). Raw values
  stay in a.u.; QC applies no keV calibration.
- Expected channels come from the selected map minus the config's declared
  ``unpopulated_minimodules``; the legacy hardcoded rule (every third
  SuperModule populated only in minimodules 0, 1, 4, 5, 8, 9, 12, 13) is
  computed only to report the difference.

The reference x-COG profile (computed, never used) is not computed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import os
import struct

import numpy as np

from src.detector_features import calculate_centroid
from src.filters import filter_min_ch
from src.fits import fit_gaussian
from src.mapping_generator import ChannelType
from src.utils import get_max_num_ch, get_maxEnergy_sm_mM, get_slab_cornell

from src.petsys_manager.contracts import DataFormat, InputDescriptor, Population, SourceMode
from .inputs import InputError, ProcessingConfig, ValidationCancelled, validate_ldat

GENERATOR = "src.cornell.qc (spec 003 T10; reference cornell_system_validation.py)"
PAIR_LIMIT = 1_000_001              # reference breaks once accepted pairs exceed 1,000,000
RPY_Y_VAL = 2
PHOTOPEAK_BINS = 150
PHOTOPEAK_RANGE = (0, 250)          # a.u.
FIT_MIN_PEAK = 20
SLAB_CB = 10
MINIMODULE_CB = 12
PEAK_FINDER = "peak"
FLOOD_BINS = 500
FLOOD_RANGE = (0.0, 105.0)          # mm
FLOOD_ENERGY = (40, 200)            # a.u., inclusive
LEGACY_HALF_MINIMODULES = (0, 1, 4, 5, 8, 9, 12, 13)
REJECTIONS = ("min_channels", "minimodule_channels", "no_energy_minimodule", "unresolved_slab")
SLAB_FLAGS = ("edge", "single_time_random", "non_adjacent", "adjacent")   # get_slab_cornell flags 0-3
FIT_STATUSES = ("fitted", "sparse", "fit_failed", "fit_error", "invalid_result")
DEFAULT_FLUSH_SIDES = 16384
MAX_KEYS = 100_000
_HIT = struct.Struct("<qfi")


class QCCancelled(InputError):
    """Stopped before a QC result existed."""


class EnergyHistograms:
    """Per-key fixed photopeak histogram, count and float64 mean/M2."""

    def __init__(self, max_keys=MAX_KEYS):
        self.max_keys = max_keys
        _, self.edges = np.histogram(np.zeros(0), bins=PHOTOPEAK_BINS, range=PHOTOPEAK_RANGE)
        self._rows = {}
        self.keys = []
        self.histograms = np.zeros((0, PHOTOPEAK_BINS), dtype=np.int64)
        self.counts = np.zeros(0, dtype=np.int64)
        self.means = np.zeros(0, dtype=np.float64)
        self.m2 = np.zeros(0, dtype=np.float64)

    def _row(self, key):
        row = self._rows.get(key)
        if row is None:
            if len(self.keys) >= self.max_keys:
                raise InputError("QC keys exceed their bound")
            row = self._rows[key] = len(self.keys)
            self.keys.append(key)
        return row

    def add(self, keys, energies):
        if not keys:
            return
        energies = np.asarray(energies, dtype=np.float64)
        target = np.fromiter((self._row(key) for key in keys), dtype=np.int64, count=len(keys))
        extra = len(self.keys) - len(self.counts)
        if extra > 0:
            self.histograms = np.vstack((self.histograms, np.zeros((extra, PHOTOPEAK_BINS), np.int64)))
            self.counts = np.concatenate((self.counts, np.zeros(extra, np.int64)))
            self.means = np.concatenate((self.means, np.zeros(extra)))
            self.m2 = np.concatenate((self.m2, np.zeros(extra)))
        # Same inclusion/edges as np.histogram(values, 150, (0, 250)); last bin closed.
        keep = (energies >= PHOTOPEAK_RANGE[0]) & (energies <= PHOTOPEAK_RANGE[1])
        bins = np.searchsorted(self.edges, energies[keep], side="right") - 1
        bins[bins == PHOTOPEAK_BINS] = PHOTOPEAK_BINS - 1
        np.add.at(self.histograms, (target[keep], bins), 1)
        rows, inverse = np.unique(target, return_inverse=True)
        count = np.bincount(inverse).astype(np.int64)
        mean = np.bincount(inverse, weights=energies) / count
        m2 = np.bincount(inverse, weights=(energies - mean[inverse]) ** 2)
        total = self.counts[rows] + count
        delta = mean - self.means[rows]
        self.means[rows] += delta * count / total
        self.m2[rows] += m2 + delta ** 2 * self.counts[rows] * count / total
        self.counts[rows] = total

    def row(self, key):
        return self._rows[key]

    @property
    def nbytes(self):
        return self.histograms.nbytes + self.counts.nbytes + self.means.nbytes + self.m2.nbytes


class FloodHistograms:
    """Per-SuperModule 500 x 500 histogram over [0, 105] mm of 40-200 a.u. sides."""

    def __init__(self):
        self.histograms = {}
        self.edges = np.linspace(*FLOOD_RANGE, FLOOD_BINS + 1)

    def add(self, modules, xs, ys, energies):
        if not modules:
            return
        modules, xs, ys = np.asarray(modules), np.asarray(xs, np.float64), np.asarray(ys, np.float64)
        energies = np.asarray(energies, np.float64)
        window = (energies >= FLOOD_ENERGY[0]) & (energies <= FLOOD_ENERGY[1])
        for module in np.unique(modules).tolist():
            selected = window & (modules == module)
            counts, _, _ = np.histogram2d(xs[selected], ys[selected], bins=(FLOOD_BINS, FLOOD_BINS),
                                          range=[list(FLOOD_RANGE), list(FLOOD_RANGE)])
            if module not in self.histograms:
                self.histograms[module] = np.zeros((FLOOD_BINS, FLOOD_BINS), dtype=np.int64)
            self.histograms[module] += counts.astype(np.int64)

    @property
    def nbytes(self):
        return sum(h.nbytes for h in self.histograms.values())


@dataclass
class Accumulators:
    plots: bool
    slabs: bool
    minimodules: EnergyHistograms = field(default_factory=EnergyHistograms)
    slab_energy: EnergyHistograms = field(default_factory=EnergyHistograms)
    floods: FloodHistograms = field(default_factory=FloodHistograms)
    slab_counts: dict = field(default_factory=lambda: defaultdict(int))
    occupancy: dict = field(default_factory=dict)        # sm -> {channel: hits}

    @property
    def nbytes(self):
        return self.minimodules.nbytes + self.slab_energy.nbytes + self.floods.nbytes


def read_pairs(path, en_min_ch):
    """Reference ``read_compact.read_binary_file(path, en_min_ch)`` without tqdm.

    Yields (det1, det2) lists of (timestamp, energy, channel) tuples keeping hits
    with energy >= ``en_min_ch``. Truncation raises instead of yielding garbage.
    """
    with open(path, "rb") as stream:
        while True:
            header = stream.read(2)
            if not header:
                return
            if len(header) != 2:
                raise InputError(f"Truncated compact header: {path}")
            sides = []
            for count in header:
                payload = stream.read(count * _HIT.size)
                if len(payload) != count * _HIT.size:
                    raise InputError(f"Truncated compact hits: {path}")
                sides.append([hit for hit in _HIT.iter_unpack(payload) if hit[1] >= en_min_ch])
            yield sides[0], sides[1]


@dataclass(frozen=True)
class FileSample:
    path: str
    validated_records: int
    records_read: int            # iterator reads, including the one read before the limit break
    pairs_processed: int
    occupancy_pairs: int         # passed the channel cuts; unresolved-slab pairs included
    occupancy_hits: int
    accepted_pairs: int          # both slabs resolved
    stopped_at_limit: bool
    rejected: dict
    slab_flags: dict

    @property
    def accepted_sides(self):
        return 2 * self.accepted_pairs


def sample_file(path, config, accumulators, *, pair_limit=PAIR_LIMIT, validated_records=0,
                flush_sides=DEFAULT_FLUSH_SIDES, cancelled=None, progress=None):
    """Reference ``extract_data_dict`` for one file, merged into ``accumulators``."""
    mapping = config.mapping
    types, modules, local = mapping.types, mapping.modules, mapping.local
    min_ch = int(config.values["min_ch"])
    en_min_ch = float(config.values["en_min_ch"])
    sum_rows_cols = bool(mapping.config["sum_rows_cols"])
    acc = accumulators
    for sm in {sm for sm, _ in modules.values()}:
        acc.occupancy.setdefault(sm, defaultdict(int))
    rejected = dict.fromkeys(REJECTIONS, 0)
    flags = [0, 0, 0, 0]
    accepted = read = processed = occupancy_pairs = occupancy_hits = 0
    stopped = False
    mm_keys, slab_keys, energies, flood_sm, flood_x, flood_y = [], [], [], [], [], []

    def flush():
        if acc.plots:
            acc.minimodules.add(mm_keys, energies)
            if acc.slabs:
                acc.slab_energy.add(slab_keys, energies)
            acc.floods.add(flood_sm, flood_x, flood_y, energies)
        for values in (mm_keys, slab_keys, energies, flood_sm, flood_x, flood_y):
            values.clear()

    for det1, det2 in read_pairs(path, en_min_ch):
        read += 1
        if cancelled is not None and read % 1024 == 0 and cancelled():
            raise QCCancelled("QC cancelled")
        if progress is not None and read % 65536 == 0:
            progress(path, read)
        if accepted >= pair_limit:
            stopped = True
            break
        processed += 1
        if not (filter_min_ch(det1, min_ch, types, sum_rows_cols) and
                filter_min_ch(det2, min_ch, types, sum_rows_cols)):
            rejected["min_channels"] += 1
            continue
        max1, energy1 = get_maxEnergy_sm_mM(det1, modules, types)
        max2, energy2 = get_maxEnergy_sm_mM(det2, modules, types)
        if not isinstance(max1, list) or not isinstance(max2, list):
            rejected["no_energy_minimodule"] += 1     # reference raises TypeError here
            continue
        if not (filter_min_ch(max1, min_ch, types, sum_rows_cols) and
                filter_min_ch(max2, min_ch, types, sum_rows_cols)):
            rejected["minimodule_channels"] += 1
            continue
        sm_mm1, sm_mm2 = modules[max1[0][2]], modules[max2[0][2]]
        slab1, flag1, x1 = get_slab_cornell(max1, types, local)
        slab2, flag2, x2 = get_slab_cornell(max2, types, local)
        flags[flag1] += 1
        flags[flag2] += 1
        occupancy_pairs += 1
        for sm_mm, side in ((sm_mm1, max1), (sm_mm2, max2)):
            present = acc.occupancy[sm_mm[0]]
            for hit in side:
                present[hit[2]] += 1
            occupancy_hits += len(side)
        if slab1 is None or slab2 is None:
            rejected["unresolved_slab"] += 1
            continue
        for sm_mm, slab, x, energy, side in ((sm_mm1, slab1, x1, energy1, max1),
                                             (sm_mm2, slab2, x2, energy2, max2)):
            key = (sm_mm[0], sm_mm[1], slab)
            acc.slab_counts[key] += 1
            if acc.plots:
                mm_keys.append(sm_mm)
                slab_keys.append(key)
                energies.append(energy)
                hits = (get_max_num_ch(side, types, 8, ChannelType.ENERGY) +
                        get_max_num_ch(side, types, 2, ChannelType.TIME))
                _, y = calculate_centroid(hits, local, 1, RPY_Y_VAL, types)
                flood_sm.append(sm_mm[0])
                flood_x.append(x)
                flood_y.append(y)
        accepted += 1
        if len(energies) >= flush_sides:
            flush()
    flush()
    if progress is not None:
        progress(path, read)
    return FileSample(str(path), validated_records, read, processed, occupancy_pairs, occupancy_hits, accepted,
                      stopped, rejected, dict(zip(SLAB_FLAGS, flags)))


@dataclass(frozen=True)
class FitEntry:
    key: tuple
    samples: int
    in_histogram: int
    status: str
    mu: float | None = None
    sigma: float | None = None
    resolution_percent: float | None = None
    sample_mean: float | None = None    # population moments, never a photopeak
    sample_std: float | None = None
    message: str = ""
    histogram: np.ndarray | None = field(default=None, compare=False, repr=False)   # photopeak counts

    @property
    def available(self):
        return self.status == "fitted"


def fit_photopeaks(histograms, cb):
    """Reference ``fit_photopeak_worker`` per sorted key, with explicit status."""
    entries = []
    for key in sorted(histograms.keys):
        row = histograms.row(key)
        samples = int(histograms.counts[row])
        counts = histograms.histograms[row].copy()
        moments = dict(sample_mean=float(histograms.means[row]),
                       sample_std=math.sqrt(float(histograms.m2[row]) / samples))
        base = dict(key=key, samples=samples, in_histogram=int(counts.sum()), histogram=counts, **moments)
        if counts.max() < FIT_MIN_PEAK:      # fit_gaussian's first check, made explicit
            entries.append(FitEntry(**base, status="sparse",
                                    message=f"highest bin {int(counts.max())} < {FIT_MIN_PEAK} counts"))
            continue
        try:
            _, _, pars, _, _ = fit_gaussian(counts, histograms.edges, cb=cb, min_peak=FIT_MIN_PEAK,
                                            pk_finder=PEAK_FINDER)
        except RuntimeError as exc:
            entries.append(FitEntry(**base, status="fit_failed", message=str(exc)))
            continue
        except Exception as exc:              # the reference would abort the pool here
            entries.append(FitEntry(**base, status="fit_error", message=f"{type(exc).__name__}: {exc}"))
            continue
        mu, sigma = float(pars[1]), float(pars[2])
        message = ""
        if sigma < 0:
            sigma = -sigma
            message = "negative fitted sigma reported as its absolute value"
        if not (math.isfinite(mu) and math.isfinite(sigma) and mu > 0 and sigma > 0):
            entries.append(FitEntry(**base, status="invalid_result", message=f"mu={mu!r} sigma={sigma!r}"))
            continue
        entries.append(FitEntry(**base, status="fitted", mu=mu, sigma=sigma,
                                resolution_percent=2.35 * sigma / mu * 100, message=message))
    return tuple(entries)


def unpopulated(config):
    return frozenset((int(sm), int(mm)) for sm, mms in dict(config.values.get("unpopulated_minimodules") or {}).items()
                     for mm in mms)


def expected_channels(mapping, absent):
    """Selected-map Time/Energy channels per SuperModule, skipping declared unpopulated minimodules."""
    system = {sm: {"Time": [], "Energy": []} for sm in sorted({sm for sm, _ in mapping.modules.values()})}
    for channel, (sm, mm) in mapping.modules.items():
        if (sm, mm) in absent:
            continue
        if ChannelType.TIME in mapping.types[channel]:
            system[sm]["Time"].append(channel)
        elif ChannelType.ENERGY in mapping.types[channel]:
            system[sm]["Energy"].append(channel)
    return system


def legacy_unpopulated(mapping):
    """Minimodules the reference rule skips: SuperModule (sm + 1) % 3 == 0 outside its half."""
    return frozenset((sm, mm) for sm, mm in set(mapping.modules.values())
                     if (sm + 1) % 3 == 0 and mm not in LEGACY_HALF_MINIMODULES)


@dataclass(frozen=True)
class OccupancyFindings:
    missing: dict                 # sm -> {"Time": [...], "Energy": [...]} (expected, no hits in sample)
    expected_minimodules: tuple
    missing_minimodules: tuple    # expected, no hit in the sample
    unexpected_minimodules: tuple  # declared unpopulated, but hits observed

    @property
    def all_minimodules_observed(self):
        return not self.missing_minimodules

    def totals(self):
        return (sum(len(v["Time"]) for v in self.missing.values()),
                sum(len(v["Energy"]) for v in self.missing.values()))


def occupancy_findings(occupancy, expected, mapping):
    """Reference ``missing_channels_check`` computation (no printing/PDF)."""
    missing = {}
    for sm in sorted({sm for sm, _ in mapping.modules.values()}):
        present = set(occupancy.get(sm, {}))
        wanted = expected.get(sm, {"Time": [], "Energy": []})
        missing[sm] = {kind: [ch for ch in wanted[kind] if ch not in present] for kind in ("Time", "Energy")}
    expected_mm = {mapping.modules[ch] for channels in expected.values() for ch in channels["Time"] + channels["Energy"]}
    present_mm = {mapping.modules[ch] for channels in occupancy.values() for ch in channels if ch in mapping.modules}
    return OccupancyFindings(missing, tuple(sorted(expected_mm)), tuple(sorted(expected_mm - present_mm)),
                             tuple(sorted(present_mm - expected_mm)))


@dataclass(frozen=True)
class QCResult:
    mapping: object
    source_mode: SourceMode | None
    acquisition_time_s: float | None
    plots: bool
    slabs: bool
    min_ch: int
    en_min_ch: float
    pair_limit: int
    files: tuple
    occupancy: dict
    slab_counts: dict
    expected: dict
    absent: frozenset
    legacy_absent: frozenset
    findings: OccupancyFindings
    minimodule_fits: tuple
    slab_fits: tuple
    floods: dict
    flood_edges: np.ndarray
    sources: dict
    storage_bytes: int

    def totals(self):
        names = ("validated_records", "records_read", "pairs_processed", "occupancy_pairs", "occupancy_hits",
                 "accepted_pairs", "accepted_sides")
        totals = {name: sum(getattr(f, name) for f in self.files) for name in names}
        totals["rejected"] = {r: sum(f.rejected[r] for f in self.files) for r in REJECTIONS}
        totals["slab_flags"] = {s: sum(f.slab_flags[s] for f in self.files) for s in SLAB_FLAGS}
        return totals


def _check_options(descriptors, config, plots, slabs, source_mode, acquisition_time_s, pair_limit):
    if not isinstance(config, ProcessingConfig):
        raise InputError("QC requires a typed processing config")
    if not descriptors or any(not isinstance(d, InputDescriptor) or d.format != DataFormat.COMPACT
                              or d.population != Population.COINCIDENCE for d in descriptors):
        raise InputError("QC requires compact coincidence LDAT inputs with explicit descriptors")
    if len({os.path.normcase(os.path.abspath(d.path)) for d in descriptors}) != len(descriptors):
        raise InputError("Duplicate QC input")
    if type(plots) is not bool or type(slabs) is not bool:
        raise InputError("plots/slabs must be booleans")
    if slabs and not plots:
        raise InputError("Slab analysis requires plots")
    if source_mode is not None and not isinstance(source_mode, SourceMode):
        raise InputError("source_mode must be a SourceMode or None (not recorded)")
    if acquisition_time_s is not None and (isinstance(acquisition_time_s, bool) or
                                           not isinstance(acquisition_time_s, (int, float)) or
                                           not math.isfinite(acquisition_time_s) or acquisition_time_s <= 0):
        raise InputError("acquisition_time_s must be a positive finite number or None")
    if type(pair_limit) is not int or pair_limit < 1:
        raise InputError("pair_limit must be a positive integer")
    if "en_min_ch" not in config.values:
        raise InputError("QC requires en_min_ch (a.u.)")


def run_qc(descriptors, config, *, plots=False, slabs=False, source_mode=None, acquisition_time_s=None,
           pair_limit=PAIR_LIMIT, flush_sides=DEFAULT_FLUSH_SIDES, cancelled=None, progress=None):
    """Validate every input (T5), sample each file in the given order, then fit. Writes nothing."""
    descriptors = tuple(descriptors)
    _check_options(descriptors, config, plots, slabs, source_mode, acquisition_time_s, pair_limit)
    mapping = config.mapping
    acc = Accumulators(plots, slabs)
    files = []
    for descriptor in descriptors:
        try:
            summary = validate_ldat(descriptor, mapping.modules, cancelled=cancelled)
        except ValidationCancelled as exc:
            raise QCCancelled("QC cancelled during input validation") from exc
        files.append(sample_file(descriptor.path, config, acc, pair_limit=pair_limit,
                                 validated_records=summary.records, flush_sides=flush_sides,
                                 cancelled=cancelled, progress=progress))
        info = os.stat(descriptor.path)
        if (info.st_size, info.st_mtime_ns) != (summary.file_size, summary.mtime_ns):
            raise InputError(f"Input changed after validation: {descriptor.path}")
    absent = unpopulated(config)
    expected = expected_channels(mapping, absent)
    occupancy = {sm: dict(sorted(channels.items())) for sm, channels in sorted(acc.occupancy.items())}
    sources = {"processing_config": {"path": str(config.path), "sha256": config.sha256},
               "map": {"path": str(mapping.path), "sha256": mapping.sha256}}
    return QCResult(
        mapping, source_mode, None if acquisition_time_s is None else float(acquisition_time_s), plots, slabs,
        int(config.values["min_ch"]), float(config.values["en_min_ch"]), pair_limit, tuple(files), occupancy,
        dict(sorted(acc.slab_counts.items())), expected, absent, legacy_unpopulated(mapping),
        occupancy_findings(occupancy, expected, mapping),
        fit_photopeaks(acc.minimodules, MINIMODULE_CB) if plots else (),
        fit_photopeaks(acc.slab_energy, SLAB_CB) if slabs else (),
        dict(sorted(acc.floods.histograms.items())), acc.floods.edges, sources, acc.nbytes)
