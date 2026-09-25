"""Toolkit-independent LDAT ingestion and measurements for LDATInspector.

The event population is *detector sides of coincidence pairs*. Ingest cuts are
min-channel count and per-channel energy; the display's keV window is reversible.
Times in the PETsys LDAT examples are picoseconds (see imas_peak_fluctuation.py).
"""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
import struct

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.special import ndtr

from src.detector_features import calculate_DOI, calculate_centroid
from src.filters import filter_min_ch
from src.mapping_generator import ChannelType, map_factory
from src.utils import KevConverter, get_maxEnergy_sm_mM, get_max_en_channel, get_slab_cornell


MAX_PAIRS_PER_FILE = 1_000_000
TIMESTAMP_SECONDS = 1e-12
_HIT = struct.Struct("qfi")
_HEADER = struct.Struct("2B")
# ModuleEvents attributes shared with spec 001 (compared by scripts/ldat_scale_check.py).
_COLUMNS = {
    "energy": "d", "raw_energy": "d", "partner_energy": "d", "partner_raw_energy": "d",
    "calibration_key": "q", "partner_calibration_key": "q",
    "x": "d", "y": "d", "doi": "d", "timestamp": "q", "partner_timestamp": "q",
    "mm": "i", "file_index": "i",
}


def _path_from_config(config_path: str, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(__file__).resolve().parent.parent
    for candidate in (Path(config_path).resolve().parent / path, root / path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Map not found: {value} (relative to config or repo)")


@dataclass(frozen=True)
class Settings:
    config_path: str
    calibration_path: str
    system: str
    max_pairs: int | None = 10000  # None: whole files
    min_channels: int = 1
    min_channel_energy: float = 0.0
    calibrated: bool = True

    def validate(self):
        if self.system not in ("IMAS", "CORNELL"):
            raise ValueError("System must be IMAS or CORNELL")
        if self.max_pairs is not None and not 1 <= self.max_pairs <= MAX_PAIRS_PER_FILE:
            raise ValueError(f"Coincidence pairs per file must be 1–{MAX_PAIRS_PER_FILE:,} (or whole files)")
        if self.min_channels < 1 or not np.isfinite(self.min_channel_energy):
            raise ValueError("Invalid minimum channels or per-channel energy")
        if not Path(self.config_path).is_file():
            raise FileNotFoundError("Select an existing config")
        if self.calibrated and not Path(self.calibration_path).is_file():
            raise FileNotFoundError("Select an existing energy calibration or disable Calib")


@dataclass
class Setup:
    config: dict
    coordinates: dict
    channel_modules: dict
    channel_types: dict
    fem: object
    converter: object


def unpopulated_minimodules(config: dict) -> dict[int, frozenset[int]]:
    """Minimodules without sensors, from the config's `unpopulated_minimodules`.

    The key maps SuperModule -> list of minimodules (e.g. the half-populated
    Cornell SMs). Without the key every mapped minimodule is expected.
    """
    value = config.get("unpopulated_minimodules") or {}
    if not isinstance(value, dict):
        raise ValueError("unpopulated_minimodules must map SuperModule -> list of minimodules")
    try:
        return {int(sm): frozenset(int(mm) for mm in mms) for sm, mms in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid unpopulated_minimodules entry: {exc}") from exc


def load_setup(settings: Settings) -> Setup:
    settings.validate()
    with open(settings.config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not config.get("map_file"):
        raise ValueError("Configuration is missing map_file")
    unpopulated_minimodules(config)  # reject a malformed key before any file is read
    coords, channel_modules, channel_types, fem = map_factory(
        str(_path_from_config(settings.config_path, config["map_file"])))
    converter = (KevConverter(settings.calibration_path,
                              file_type="cornell" if settings.system == "CORNELL" else "mu")
                 if settings.calibrated else None)
    return Setup(config, coords, channel_modules, channel_types, fem, converter)


def iter_pairs(path: str, limit: int | None):
    """Read the same native-layout records as read_compact; reject truncation.

    A limit is a prefix of the file, never a random subsample or full-run rate;
    ``None`` reads the whole file.
    """
    if Path(path).stat().st_size == 0:
        raise ValueError("Empty LDAT file")
    with open(path, "rb") as handle:
        for _ in (range(limit) if limit is not None else iter(int, 1)):
            header = handle.read(_HEADER.size)
            if not header:
                return
            if len(header) != _HEADER.size:
                raise ValueError("Truncated LDAT pair header")
            sizes = _HEADER.unpack(header)
            sides = []
            for size in sizes:
                raw = handle.read(size * _HIT.size)
                if len(raw) != size * _HIT.size:
                    raise ValueError("Truncated LDAT hit record")
                sides.append(list(_HIT.iter_unpack(raw)))
            yield sides[0], sides[1]


# Stored once per detector side (~62 B/side with a calibrated energy column).
_TABLE_DTYPES = {
    "raw_energy": "f8", "x": "f8", "y": "f8", "doi": "f8", "timestamp": "i8",
    "calibration_key": "i4", "partner": "i4", "sm": "i2", "file_index": "i2",
    "mm": "i1", "random_slab": "?",
}
# Per-side values a reader produces, in pair order (rows 2k and 2k + 1 are one pair).
_SIDE_FIELDS = ("raw_energy", "calibration_key", "x", "y", "doi", "timestamp", "sm", "mm", "random_slab")


def _narrowest(values, dtype):
    """``dtype`` unless the values do not fit (calibration keys of very large channel IDs)."""
    values = np.asarray(values)
    info = np.iinfo(dtype) if np.dtype(dtype).kind == "i" else None
    if info is not None and values.size and (values.max() > info.max or values.min() < info.min):
        return values.astype(np.int64)
    return values.astype(dtype, copy=False)


class SideTable:
    """Detector sides of accepted coincidence pairs, sorted by (SM, file, record).

    Each column holds one value per side; ``partner[i]`` is the row of the other
    side of i's pair, so partner values are never stored twice. ``energy`` is the
    active energy view: the ``raw_energy`` array itself when uncalibrated.
    """

    def __init__(self, columns: dict, energy=None):
        self.columns = columns
        self.energy = columns["raw_energy"] if energy is None else energy

    @classmethod
    def from_pairs(cls, file_index: int, **sides):
        """Build from pair-ordered side arrays (see ``_SIDE_FIELDS``)."""
        n = len(sides["raw_energy"])
        columns = {name: np.asarray(sides[name]) for name in _SIDE_FIELDS}
        columns["file_index"] = np.full(n, file_index)
        columns["partner"] = np.arange(n) ^ 1
        return cls._sorted(columns)

    @classmethod
    def concatenate(cls, tables, *, consume=False):
        """One table from per-file tables, keeping file order within each SM.

        Each input is already SM-sorted, so every (SM, file) block is copied
        straight to its final rows, one column at a time. With ``consume`` each
        input column is released once copied, so peak memory stays close to one
        table plus one column rather than three copies.
        """
        tables = [t for t in tables if t is not None]
        total = sum(map(len, tables))
        segments = [dict(zip(*np.unique(t.columns["sm"], return_counts=True))) for t in tables]
        destination, start = [np.empty(len(t), np.int64) for t in tables], 0
        firsts = [0] * len(tables)
        for sm in sorted(set().union(*segments)) if segments else []:
            for i, counts in enumerate(segments):
                count = int(counts.get(sm, 0))
                destination[i][firsts[i]:firsts[i] + count] = np.arange(start, start + count)
                firsts[i] += count
                start += count
        columns = {}
        for name, dtype in _TABLE_DTYPES.items():
            if name == "partner":
                kind = np.int64  # global rows; narrowed below
            elif tables:
                kind = np.result_type(*(t.columns[name].dtype for t in tables))
            else:
                kind = np.dtype(dtype)
            out = np.empty(total, kind)
            for table, rows in zip(tables, destination):
                values = table.columns[name]
                out[rows] = rows[values] if name == "partner" else values
                if consume:
                    del table.columns[name]
                    if name == "raw_energy":
                        table.energy = None
            columns[name] = _narrowest(out, dtype)
        return cls(columns)

    @classmethod
    def _sorted(cls, columns):
        """Stable sort by SM (keeps file and record order) and remap partner rows."""
        order = np.argsort(columns["sm"], kind="stable")
        inverse = np.empty(order.size, np.int64)
        inverse[order] = np.arange(order.size)
        sorted_columns = {}
        for name, dtype in _TABLE_DTYPES.items():
            values = inverse[columns["partner"][order]] if name == "partner" else columns[name][order]
            sorted_columns[name] = np.ascontiguousarray(_narrowest(values, dtype))
        return cls(sorted_columns)

    def __len__(self):
        return len(self.columns["raw_energy"])

    def __getattr__(self, name):
        try:
            return self.__dict__["columns"][name]
        except KeyError:
            raise AttributeError(name) from None

    def with_energy(self, energy):
        """Same sides and columns with a different active energy view."""
        return SideTable(self.columns, energy)

    @property
    def nbytes(self):
        own = sum(values.nbytes for values in self.columns.values())
        return own + (0 if self.energy is self.columns["raw_energy"] else self.energy.nbytes)

    def by_sm(self):
        """``{sm: ModuleEvents}`` views onto this table's contiguous SM rows."""
        sm = self.columns["sm"]
        if not sm.size:
            return {}
        starts = np.flatnonzero(np.r_[True, sm[1:] != sm[:-1]])
        stops = np.r_[starts[1:], sm.size]
        return {int(sm[a]): ModuleEvents(self, int(a), int(b)) for a, b in zip(starts, stops)}


class ModuleEvents:
    """One SuperModule's rows of a ``SideTable``.

    Own columns are zero-copy slices; partner columns are gathered through
    ``partner`` on access and cannot be assigned.
    """

    _OWN = ("energy", "raw_energy", "calibration_key", "x", "y", "doi", "timestamp", "mm",
            "file_index", "random_slab")

    def __init__(self, table: SideTable, start: int, stop: int):
        self._table, self._start, self._stop = table, start, stop
        for name in self._OWN:
            setattr(self, name, getattr(table, name)[start:stop])

    def __len__(self):
        return self._stop - self._start

    def _partner(self, name):
        return getattr(self._table, name)[self._table.partner[self._start:self._stop]]

    partner_energy = property(lambda self: self._partner("energy"))
    partner_raw_energy = property(lambda self: self._partner("raw_energy"))
    partner_calibration_key = property(lambda self: self._partner("calibration_key"))
    partner_timestamp = property(lambda self: self._partner("timestamp"))
    partner_sm = property(lambda self: self._partner("sm"))
    partner_mm = property(lambda self: self._partner("mm"))

    def with_table(self, table: SideTable):
        """The same rows of ``table`` (a new energy view); other slices are shared."""
        view = copy.copy(self)
        view._table = table
        view.energy = table.energy[self._start:self._stop]
        return view


@dataclass
class FileResult:
    index: int
    path: str
    pairs_read: int = 0
    pairs_accepted: int = 0
    table: SideTable | None = None
    time_counts: dict[int, Counter] = field(default_factory=dict)
    energy_counts: dict[int, Counter] = field(default_factory=dict)
    errors: Counter = field(default_factory=Counter)
    error: str | None = None
    prefix_limited: bool = False

    @property
    def success(self):
        return self.error is None

    @property
    def modules(self) -> dict[int, ModuleEvents]:
        return self.table.by_sm() if self.table is not None else {}


class _UnresolvedCornellSlab(ValueError):
    """The two leading time hits are not adjacent; no slab can be calibrated."""


def process_file_reference(path: str, settings: Settings, index: int = 0) -> FileResult:
    """Decode raw coincidence sides once; a corrupt file contributes no prefix.

    Spec 001's per-pair Python reader. Spec 002 keeps it unchanged as the
    oracle that the fast reader is compared against (scripts/ldat_scale_check.py).
    """
    result = FileResult(index=index, path=str(path))
    try:
        settings.validate()
        setup = load_setup(replace(settings, calibrated=False))
        buffers = {name: array(kind) for name, kind in zip(_SIDE_FIELDS, "dqdddqqqb")}
        t_counts = defaultdict(Counter)
        e_counts = defaultdict(Counter)
        for det1, det2 in iter_pairs(path, settings.max_pairs):
            result.pairs_read += 1
            try:
                det1 = [hit for hit in det1 if hit[1] >= settings.min_channel_energy]
                det2 = [hit for hit in det2 if hit[1] >= settings.min_channel_energy]
                if not all(filter_min_ch(det, settings.min_channels,
                                         setup.channel_types, setup.fem.sum_rows_cols)
                           for det in (det1, det2)):
                    result.errors["min channels"] += 1
                    continue
                sides = []
                for det in (det1, det2):
                    selected, raw_energy = get_maxEnergy_sm_mM(
                        det, setup.channel_modules, setup.channel_types)
                    if not selected or not filter_min_ch(selected, settings.min_channels,
                                                          setup.channel_types,
                                                          setup.fem.sum_rows_cols):
                        raise ValueError("selected minimodule lacks channels")
                    t_hit = get_max_en_channel(selected, setup.channel_types, ChannelType.TIME)
                    if t_hit is None:
                        raise ValueError("selected minimodule lacks time channel")
                    t_ch = t_hit[2]
                    sm, mm = setup.channel_modules[t_ch]
                    slab, random_slab = 0, False
                    if settings.system == "CORNELL":
                        slab, slab_flag, _ = get_slab_cornell(selected, setup.channel_types,
                                                              setup.coordinates)
                        if slab is None:
                            raise _UnresolvedCornellSlab
                        random_slab = slab_flag == 1  # one time channel: coin flip
                    calibration_key = (int(t_ch) << 5) | int(slab)
                    energy = float(raw_energy)
                    x, y = calculate_centroid(selected, setup.coordinates, 1, 2,
                                              setup.channel_types)
                    doi = calculate_DOI(selected, setup.coordinates,
                                        setup.fem.sum_rows_cols, setup.channel_types)
                    if not np.all(np.isfinite((energy, x, y, doi))) or energy <= 0:
                        raise ValueError("non-finite or non-positive measurement")
                    times = (ch[2] for ch in selected
                             if ChannelType.TIME in setup.channel_types[ch[2]])
                    energies = (ch[2] for ch in selected
                                if ChannelType.ENERGY in setup.channel_types[ch[2]])
                    sides.append(((energy, calibration_key, float(x), float(y), float(doi),
                                   int(t_hit[0]), sm, mm, random_slab), tuple(times), tuple(energies)))
            except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
                result.errors["unresolved Cornell slab" if isinstance(exc, _UnresolvedCornellSlab)
                              else type(exc).__name__] += 1
                continue
            for values, times, energies in sides:
                for name, value in zip(_SIDE_FIELDS, values):
                    buffers[name].append(value)
                t_counts[values[6]].update(times)
                e_counts[values[6]].update(energies)
            result.pairs_accepted += 1
        # Reaching the prefix limit does not establish whether this was a full run.
        result.prefix_limited = result.pairs_read == settings.max_pairs
        result.table = SideTable.from_pairs(index, **{
            name: np.frombuffer(values, dtype=values.typecode) if values.typecode != "b"
            else np.frombuffer(values, dtype=np.int8).astype(bool)
            for name, values in buffers.items()})
        result.time_counts = dict(t_counts)
        result.energy_counts = dict(e_counts)
    except Exception as exc:
        result.error = str(exc)
        result.table = None
        result.time_counts = {}
        result.energy_counts = {}
        result.pairs_accepted = 0
    return result


def process_file(path: str, settings: Settings, index: int = 0) -> FileResult:
    """Read one LDAT file with the fast reader (same results as the reference)."""
    from src.ldat_fastread import process_file_fast
    return process_file_fast(path, settings, index)


@dataclass
class Dataset:
    settings: Settings
    files: list[FileResult]
    modules: dict[int, ModuleEvents]
    expected_time: dict[int, set[int]]
    expected_energy: dict[int, set[int]]
    time_counts: dict[int, Counter]
    energy_counts: dict[int, Counter]
    map_path: str
    config: dict
    file_spans: dict[int, tuple[int, int]]
    expected_mm: dict[int, set[int]] = field(default_factory=dict)
    table: SideTable | None = None
    # From the selected map: channel -> (x, y, index), (SM, minimodule) and types.
    coordinates: dict = field(default_factory=dict)
    channel_modules: dict = field(default_factory=dict)
    channel_types: dict = field(default_factory=dict)


def merge_results(settings: Settings, files: list[FileResult], setup: Setup | None = None,
                  *, consume: bool = False) -> Dataset:
    """Merged dataset; ``consume`` releases the per-file tables while merging."""
    setup = setup or load_setup(settings)
    expected_t = defaultdict(set)
    expected_e = defaultdict(set)
    expected_mm = defaultdict(set)
    unpopulated = unpopulated_minimodules(setup.config)
    for channel, (sm, mm) in setup.channel_modules.items():
        if mm in unpopulated.get(sm, ()):
            continue
        if channel not in setup.coordinates or channel not in setup.channel_types:
            continue
        expected_mm[sm].add(mm)
        if ChannelType.TIME in setup.channel_types[channel]:
            expected_t[sm].add(channel)
        if ChannelType.ENERGY in setup.channel_types[channel]:
            expected_e[sm].add(channel)
    t_counts, e_counts = defaultdict(Counter), defaultdict(Counter)
    spans = {}
    for result in files:
        if not result.success:
            continue
        for sm, counts in result.time_counts.items():
            t_counts[sm].update(counts)
        for sm, counts in result.energy_counts.items():
            e_counts[sm].update(counts)
        if result.table is not None and len(result.table):
            spans[result.index] = (int(result.table.timestamp.min()), int(result.table.timestamp.max()))
    table = SideTable.concatenate([r.table for r in files if r.success], consume=consume)
    # Keep per-file counts and provenance, not a second copy of every side.
    files = [replace(r, table=None) for r in files]
    dataset = Dataset(replace(settings, calibrated=False), files, table.by_sm(),
                      dict(expected_t), dict(expected_e), dict(t_counts), dict(e_counts),
                      str(_path_from_config(settings.config_path, setup.config["map_file"])),
                      setup.config.copy(), spans, dict(expected_mm), table,
                      setup.coordinates, setup.channel_modules, setup.channel_types)
    return apply_calibration(dataset, settings.calibration_path, True,
                             converter=setup.converter) if settings.calibrated else dataset


def apply_calibration(dataset: Dataset, path: str, enabled: bool,
                      *, converter=None) -> Dataset:
    """Derive an energy view without rereading or changing the accepted pairs.

    Missing Cornell slab/channel factors remain NaN (never plausible keV).
    Raw columns and file/channel counters are shared with the original view.
    """
    if enabled:
        if not Path(path).is_file():
            raise FileNotFoundError("Select an existing energy calibration")
        converter = converter or KevConverter(path, file_type=(
            "cornell" if dataset.settings.system == "CORNELL" else "mu"))

    def converted(raw, keys):
        if not enabled:
            return raw
        unique, inverse = np.unique(keys, return_inverse=True)
        factors = np.full(unique.shape, np.nan, dtype=float)
        for i, encoded in enumerate(unique):
            key = (int(encoded) >> 5, int(encoded) & 31) if dataset.settings.system == "CORNELL" else int(encoded) >> 5
            try:
                factor = converter.kev_factors[key]
            except KeyError:
                continue
            if np.isfinite(factor) and factor > 0:
                factors[i] = 511.0 / factor
        return raw * factors[inverse]

    # Only the energy column changes; partner energies follow through ``partner``.
    table = dataset.table.with_energy(converted(dataset.table.raw_energy, dataset.table.calibration_key))
    modules = {sm: data.with_table(table) for sm, data in dataset.modules.items()}
    return replace(dataset, table=table, modules=modules,
                   settings=replace(dataset.settings, calibrated=enabled, calibration_path=path))


@dataclass(frozen=True)
class Selection:
    energy_low: float = 400.0
    energy_high: float = 650.0
    doi_low: float = 0.0
    doi_high: float = float("inf")
    x_low: float = -float("inf")
    x_high: float = float("inf")
    y_low: float = -float("inf")
    y_high: float = float("inf")

    def mask(self, data: ModuleEvents, *, energy: bool = True):
        keep = ((data.doi >= self.doi_low) & (data.doi <= self.doi_high)
                & (data.x >= self.x_low) & (data.x <= self.x_high)
                & (data.y >= self.y_low) & (data.y <= self.y_high))
        if energy:
            keep &= ((data.energy >= self.energy_low) & (data.energy <= self.energy_high)
                     & (data.partner_energy >= self.energy_low)
                     & (data.partner_energy <= self.energy_high))
        return keep


def fit_peak(energies, *, interval=(350.0, 700.0), bins=140):
    """Constrained keV photopeak with the supported local continuum model.

    A narrow, unbounded Gaussian-only fit to the eight bins either side of a
    noisy argmax can move to a different peak and severely underestimate width.
    Use the linear model only when its Poisson deviance improves enough to pay
    for its extra parameter; explicit/manual linear settings remain opt-in.
    """
    flat = fit_peak_background(energies, interval=interval, bins=bins,
                               background_model="constant")
    slope = fit_peak_background(energies, interval=interval, bins=bins)
    if slope["status"] == "FIT" and (flat["status"] != "FIT" or
                                      flat["deviance"] - slope["deviance"] >= 8):
        return slope
    return flat


def fit_peak_background(energies, *, interval=(350.0, 700.0),
                         search=(425.0, 600.0), bins=140,
                         background_model="linear", sigma0=35.0, mu_halfwidth=45.0):
    """Poisson Gaussian plus nonnegative constant or linear continuum in keV.

    This is a *display estimate*, not an energy correction or clinical verdict.
    Endpoint background counts are constrained positive across the fit interval.
    ``sigma0`` (initial width) and ``mu_halfwidth`` (centroid freedom around the
    smoothed peak) default to keV values; callers fitting raw PETsys a.u. spectra
    scale both to their expected peak position.
    """
    unavailable = lambda why: {"status": why, "mu": None, "resolution": None}
    if not (np.isfinite([*interval, *search]).all()
            and interval[0] < search[0] < search[1] < interval[1]
            and 30 <= bins <= 1000
            and background_model in ("constant", "linear")):
        return unavailable("invalid keV fit/search interval")
    energies = np.asarray(energies, dtype=float)
    energies = energies[np.isfinite(energies)]
    if energies.size < 200:
        return unavailable("insufficient events")
    counts, edges = np.histogram(energies, bins=bins, range=interval)
    centres = (edges[:-1] + edges[1:]) / 2
    within = (centres >= search[0]) & (centres <= search[1])
    if not within.any():
        return unavailable("empty search interval")
    # A flat continuum alone should never acquire a manufactured photopeak.
    floor = float(np.percentile(counts, 25))
    peak = float(counts[within].max())
    if peak < max(10, floor + 5 * np.sqrt(max(floor, 1))):
        return unavailable("no supported peak above continuum")
    smooth = np.convolve(counts, np.ones(5) / 5, mode="same")
    peak_index = np.flatnonzero(within)[np.argmax(smooth[within])]
    mu0 = float(centres[peak_index])
    bin_width = edges[1] - edges[0]

    def components(params):
        area, mu, sigma, left = params[:4]
        right = params[4] if background_model == "linear" else left
        gaussian = area * (ndtr((edges[1:] - mu) / sigma)
                           - ndtr((edges[:-1] - mu) / sigma))
        background = left + (right - left) * (centres - centres[0]) / (centres[-1] - centres[0])
        return gaussian, background

    def deviance(params):
        gauss, background = components(params)
        prediction = np.maximum(gauss + background, 1e-10)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(counts > 0, counts * np.log(np.maximum(counts, 1) / prediction), 0)
        return np.sign(counts - prediction) * np.sqrt(np.maximum(
            2 * (term + prediction - counts), 0))

    # Keep a noisy secondary hump from pulling the fit away from the supported
    # mode. A 90-keV-wide centroid interval is still generous for a 511-keV peak.
    mu_low = max(search[0], mu0 - mu_halfwidth)
    mu_high = min(search[1], mu0 + mu_halfwidth)
    lower = [0.0, mu_low, bin_width, 0.0]
    upper = [max(energies.size * 2.0, 1.0), mu_high,
             (interval[1] - interval[0]) / 3, max(peak * 4, 1)]
    initial = [max((peak - floor) * 20, 1), mu0, sigma0, max(floor, 0.01)]
    if background_model == "linear":
        lower.append(0.0)
        upper.append(max(peak * 4, 1))
        initial.append(max(floor, 0.01))
    try:
        result = least_squares(
            deviance, initial,
            bounds=(lower, upper), max_nfev=300)
        area, mu, sigma, left = result.x[:4]
        right = result.x[4] if background_model == "linear" else left
        gauss, background = components(result.x)
        if (not result.success or not np.isfinite(result.x).all()
                or sigma <= lower[2] * 1.02 or sigma >= upper[2] * 0.98
                or mu <= mu_low + bin_width or mu >= mu_high - bin_width
                or gauss.sum() < max(100, 0.1 * (gauss + background).sum())):
            return unavailable("unresolved or background-dominated fit")
        return {"status": "FIT", "mu": float(mu), "sigma": float(sigma),
                "resolution": float(235.4820045 * sigma / mu),
                "area": float(area), "deviance": float(np.sum(deviance(result.x) ** 2)),
                "x": centres, "total": gauss + background, "gaussian": gauss,
                "background": background, "interval": interval, "search": search,
                "fit_bin_width": float(bin_width),
                "background_endpoints": (float(left), float(right)),
                "background_model": background_model}
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return unavailable(f"fit unavailable: {exc}")


def fit_on_display_bins(result, display_edges):
    """Return model components in *displayed counts per bin*, over the fit window.

    The fitter's histogram has its own bin width. Plotting those fitted counts
    directly over a differently binned chart can mis-scale every line by 3x.
    Integrate the Gaussian over each visible bin and convert the fitted local
    background from fit-bin counts to the visible bin width.
    """
    if result["status"] != "FIT":
        return None
    edges = np.asarray(display_edges, dtype=float)
    low, high = result["interval"]
    within = (edges[:-1] >= low) & (edges[1:] <= high)
    left_edges, right_edges = edges[:-1][within], edges[1:][within]
    centres = (left_edges + right_edges) / 2
    if not centres.size:
        return None
    mu, sigma, area = result["mu"], result["sigma"], result["area"]
    gauss = area * (ndtr((right_edges - mu) / sigma)
                    - ndtr((left_edges - mu) / sigma))
    left, right = result["background_endpoints"]
    a = low + result["fit_bin_width"] / 2
    b = high - result["fit_bin_width"] / 2
    per_fit_bin = left + (right - left) * (centres - a) / (b - a)
    background = per_fit_bin * (right_edges - left_edges) / result["fit_bin_width"]
    return {"x": centres, "total": gauss + background,
             "gaussian": gauss, "background": background}


def flood_counts(x, y, bins=100):
    """2D detector-side counts; empty bins are masked, not coloured as min counts."""
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 102], [0, 102]])
    return np.ma.masked_equal(counts.T, 0), xedges, yedges


def channel_status(dataset: Dataset, sm: int, *, min_events=100):
    data = dataset.modules.get(sm)
    count = len(data) if data is not None else 0
    expected_t, expected_e = dataset.expected_time.get(sm, set()), dataset.expected_energy.get(sm, set())
    active_t = set(dataset.time_counts.get(sm, Counter())) & expected_t
    active_e = set(dataset.energy_counts.get(sm, Counter())) & expected_e
    if count == 0:
        state = "NO DATA"
    elif count < min_events:
        state = "INSUFFICIENT EVENTS"
    else:
        state = "OBSERVED" if active_t == expected_t and active_e == expected_e else "NOT OBSERVED"
    return {"state": state, "events": count, "expected_time": expected_t,
            "expected_energy": expected_e, "active_time": active_t,
            "active_energy": active_e, "unobserved_time": expected_t - active_t,
            "unobserved_energy": expected_e - active_e}


# Channel findings (spec 002 FR-5-FR-7). Occupancy of the coincidence ingest
# population, never a dead/hot hardware verdict: each accepted side counts one
# hit per channel of its selected minimodule that passed the per-channel cut.
FINDINGS_POPULATION = "channel hit counts in the coincidence ingest population"
# Row state priority, highest first. A type that could not be assessed ranks
# below any finding but above OK, so a row is OK only when everything was.
FINDING_PRIORITY = ("NOT OBSERVED", "HIGH", "LOW", "INSUFFICIENT EVENTS", "OK")
# RAWInspector ADC Status colours. A mapped SM with no ingest sides is marked
# like a not-observed channel.
FINDING_COLOURS = {"OK": "#b3ffb3", "NOT OBSERVED": "#ffb3b3", "HIGH": "#ffc4e1",
                   "LOW": "#ffd9b3", "INSUFFICIENT EVENTS": "#dce0e4", "NO DATA": "#ffb3b3"}


@dataclass(frozen=True)
class FindingThresholds:
    """Provisional defaults (plan): RAWInspector's count-based dead test for low."""

    low_frac: float = 0.15    # LOW: hits < low_frac x median
    high_frac: float = 3.0    # HIGH: hits > high_frac x median
    min_median: float = 20.0  # per-type median below this: insufficient events
    min_events: int = 100     # SM ingest sides below this: insufficient events

    def validate(self):
        if not (np.isfinite(self.low_frac) and 0 <= self.low_frac < 1):
            raise ValueError("Low fraction must be at least 0 and below 1")
        if not (np.isfinite(self.high_frac) and self.high_frac > 1):
            raise ValueError("High factor must be above 1")
        if not (np.isfinite(self.min_median) and self.min_median >= 0):
            raise ValueError("Minimum median must be a nonnegative number of hits")
        if int(self.min_events) != self.min_events or self.min_events < 1:
            raise ValueError("Minimum ingest sides must be a positive integer")
        return self

    def text(self):
        return (f"low < {self.low_frac:g} × median, high > {self.high_frac:g} × median; "
                f"insufficient below a median of {self.min_median:g} hits "
                f"or {self.min_events:,} ingest sides")


def _type_findings(expected, counts, events, thresholds):
    channels = sorted(expected)
    hits = np.array([counts.get(ch, 0) for ch in channels], dtype=np.int64)
    median = float(np.median(hits)) if hits.size else None
    if events < thresholds.min_events:
        reason = f"{events:,} ingest sides < {thresholds.min_events:,}"
    elif median is None:
        reason = "no expected channels"
    elif median < thresholds.min_median:
        reason = f"median {median:g} hits < {thresholds.min_median:g}"
    else:
        reason = None
    if reason:
        states = ["INSUFFICIENT EVENTS"] * len(channels)
    else:
        states = np.select([hits == 0, hits > thresholds.high_frac * median,
                            hits < thresholds.low_frac * median],
                           ["NOT OBSERVED", "HIGH", "LOW"], "OK").tolist()
    return {"channels": channels, "hits": hits, "states": states, "median": median,
            "counts": Counter(states), "insufficient": reason,
            # hits on mapped channels the config does not expect (e.g. unpopulated minimodules)
            "unexpected": {ch: n for ch, n in sorted(counts.items()) if ch not in expected}}


def channel_findings(dataset: Dataset, sm: int, thresholds: FindingThresholds = FindingThresholds()):
    """Per-channel occupancy states for one SM, per channel type, and its row state.

    Each channel's hit count is compared with the median of its type (time or
    energy) over that SM's expected channels, zeros included.
    """
    thresholds.validate()
    data = dataset.modules.get(sm)
    events = len(data) if data is not None else 0
    findings = {"sm": sm, "events": events, "thresholds": thresholds}
    for name, expected, counts in (
            ("time", dataset.expected_time.get(sm, set()), dataset.time_counts.get(sm, Counter())),
            ("energy", dataset.expected_energy.get(sm, set()), dataset.energy_counts.get(sm, Counter()))):
        findings[name] = _type_findings(expected, counts, events, thresholds)
    present = set(findings["time"]["states"]) | set(findings["energy"]["states"])
    findings["state"] = ("NO DATA" if events == 0 or not present
                         else next(s for s in FINDING_PRIORITY if s in present))
    return findings


def system_channel_findings(dataset: Dataset, thresholds: FindingThresholds = FindingThresholds()):
    """``channel_findings`` for every mapped SM (and any SM with sides), in SM order."""
    sms = set(dataset.expected_time) | set(dataset.expected_energy) | set(dataset.modules)
    return [channel_findings(dataset, sm, thresholds) for sm in sorted(sms)]


def channel_geometry(dataset: Dataset, sm: int):
    """Channel positions of one SM from the selected map (sum rows/cols readout).

    Time channels carry their fine X, energy channels their fine Y. Each
    minimodule's box spans its time-channel X and energy-channel Y positions
    plus half a pitch; ``populated`` is False for minimodules the config lists
    as unpopulated.
    """
    by_mm = defaultdict(lambda: {"time": [], "energy": []})
    for ch, (s, mm) in dataset.channel_modules.items():
        if s != sm or ch not in dataset.coordinates or ch not in dataset.channel_types:
            continue
        x, y = dataset.coordinates[ch][:2]
        for kind, channel_type in (("time", ChannelType.TIME), ("energy", ChannelType.ENERGY)):
            if channel_type in dataset.channel_types[ch]:
                by_mm[mm][kind].append((ch, float(x), float(y)))
    minimodules, time, energy = {}, [], []
    for mm, channels in sorted(by_mm.items()):
        xs = sorted(x for _, x, _ in channels["time"]) or sorted(x for _, x, _ in channels["energy"])
        ys = sorted(y for _, _, y in channels["energy"]) or sorted(y for _, _, y in channels["time"])
        steps = np.diff(xs) if len(xs) > 1 else np.diff(ys)
        half = float(np.median(steps)) / 2 if len(steps) else 1.0
        box = (xs[0] - half, xs[-1] + half, ys[0] - half, ys[-1] + half)
        populated = mm in dataset.expected_mm.get(sm, set())
        minimodules[mm] = {"box": box, "populated": populated}
        if populated:
            time += [(ch, x, box[2], box[3], mm) for ch, x, _ in sorted(channels["time"], key=lambda c: c[1])]
            energy += [(ch, y, box[0], box[1], mm) for ch, _, y in sorted(channels["energy"], key=lambda c: c[2])]
    return {"minimodules": minimodules, "time": time, "energy": energy}


def _clusters(values, tolerance):
    """Index of each value's cluster, clusters ordered by value (gap > tolerance splits)."""
    order = np.argsort(values)
    labels = np.empty(len(values), dtype=int)
    label, previous = -1, None
    for i in order:
        if previous is None or values[i] - previous > tolerance:
            label += 1
        labels[i] = label
        previous = values[i]
    return labels, label + 1


def minimodule_layout(dataset: Dataset):
    """Minimodule grid of each SM, derived from the selected map's channel coordinates.

    A minimodule's centre is (mean fine X of its time channels, mean fine Y of
    its energy channels). Rows and columns are the distinct centres (within
    half a minimodule), so no shape is assumed and ``mM_disposition`` is not
    used. Row 0 is the largest Y (top, as the flood map is viewed); column 0 is
    the smallest X. Unpopulated minimodules keep their cell with
    ``populated`` False.

    Returns ``{sm: {"shape": (rows, cols), "cells": {mm: (row, col)},
    "centres": {mm: (x, y)}, "populated": {mm: bool}}}``.
    """
    points = defaultdict(lambda: defaultdict(lambda: {"x": [], "y": []}))
    for ch, (sm, mm) in dataset.channel_modules.items():
        if ch not in dataset.coordinates or ch not in dataset.channel_types:
            continue
        x, y = dataset.coordinates[ch][:2]
        if ChannelType.TIME in dataset.channel_types[ch]:
            points[sm][mm]["x"].append(float(x))
        if ChannelType.ENERGY in dataset.channel_types[ch]:
            points[sm][mm]["y"].append(float(y))
    layout = {}
    for sm, by_mm in sorted(points.items()):
        mms = sorted(mm for mm, p in by_mm.items() if p["x"] and p["y"])
        if not mms:
            continue
        centres = {mm: (float(np.mean(by_mm[mm]["x"])), float(np.mean(by_mm[mm]["y"]))) for mm in mms}
        xs = np.array([centres[mm][0] for mm in mms])
        ys = np.array([centres[mm][1] for mm in mms])
        # half the smallest spread of one minimodule's channels separates neighbouring centres
        width = min(max(np.ptp(by_mm[mm]["x"]), np.ptp(by_mm[mm]["y"])) for mm in mms)
        tolerance = max(width / 2, 1e-6)
        cols, ncols = _clusters(xs, tolerance)
        rows, nrows = _clusters(-ys, tolerance)
        cells = {mm: (int(r), int(c)) for mm, r, c in zip(mms, rows, cols)}
        if len(set(cells.values())) != len(cells):
            raise ValueError(f"SM {sm}: minimodules share a grid cell; the map's centres are not a grid")
        populated = {mm: mm in dataset.expected_mm.get(sm, set()) for mm in mms}
        layout[sm] = {"shape": (nrows, ncols), "cells": cells, "centres": centres, "populated": populated}
    return layout


def supermodule_layout(dataset: Dataset):
    """SuperModule placement from the config geometry: ``(rows, cols, {sm: (row, col)})``.

    Cornell: row = ring (SM within its cassette, ``sm % len(ring_z)``), column =
    cassette (``sm // len(ring_z)``). IMAS: ring-major, ``len(ring_yx)``
    SuperModules per ring. Moved from spec 001's GUI ``_layout``.
    """
    config = dataset.config
    sms = sorted(set(dataset.expected_time) | set(dataset.expected_energy) | set(dataset.modules))
    ncols = max(len(config.get("ring_yx") or {}), 1)
    if dataset.settings.system == "CORNELL":
        rings = len(config.get("ring_z") or [0, 1, 2])
        cells = {sm: (sm % rings, sm // rings) for sm in sms}
        ncols = max(ncols, max((c for _, c in cells.values()), default=0) + 1)
    else:
        rings = max(len(config.get("ring_z") or []), 1)
        cells = {sm: (sm // ncols, sm % ncols) for sm in sms}
        rings = max(rings, max((r for r, _ in cells.values()), default=0) + 1)
    if len(set(cells.values())) != len(cells):
        raise ValueError("SuperModules share a cell in the configured ring geometry")
    return rings, ncols, cells


RAW_FIT = {"status": "unavailable: raw a.u. (no keV calibration)", "mu": None, "resolution": None}


def minimodule_metrics(dataset: Dataset, selection: Selection, *, fits: bool = True, cancelled=None):
    """Per-(SM, minimodule) side counts and, when calibrated, the photopeak fit.

    - ``ingest``: accepted detector sides in that minimodule;
    - ``selected``: sides passing the current display cuts (paired energy,
      DOI and ROI, as ``Selection.mask``);
    - ``fit``: ``fit_peak`` on the ROI/DOI-selected energies with the energy
      window off, as in ``uniformity``; raw mode gives ``RAW_FIT``.

    Every expected minimodule gets a row (zeros when it has no sides).
    ``cancelled`` (a callable) is polled between SMs; when it returns True the
    function returns None. ``fits=False`` skips the fits (``fit`` is None).
    """
    calibrated = dataset.settings.calibrated
    metrics = {}
    for sm in sorted(set(dataset.expected_mm) | set(dataset.modules)):
        if cancelled is not None and cancelled():
            return None
        expected = dataset.expected_mm.get(sm, set())
        data = dataset.modules.get(sm)
        if data is None or len(data) == 0:
            for mm in sorted(expected):
                metrics[(sm, mm)] = {"ingest": 0, "selected": 0, "fit_sides": 0,
                                     "fit": (fit_peak([]) if calibrated else RAW_FIT) if fits else None}
            continue
        mm_col = data.mm.astype(np.int64)
        size = int(max(mm_col.max(initial=0), max(expected, default=0))) + 1
        ingest = np.bincount(mm_col, minlength=size)
        selected = np.bincount(mm_col[selection.mask(data)], minlength=size)
        spatial = selection.mask(data, energy=False)
        fit_counts = np.bincount(mm_col[spatial], minlength=size)
        order = np.argsort(mm_col[spatial], kind="stable")
        energies = data.energy[spatial][order]
        bounds = np.r_[0, np.cumsum(fit_counts)]
        for mm in sorted(expected | set(np.flatnonzero(ingest).tolist())):
            if not fits:
                fit = None
            elif not calibrated:
                fit = RAW_FIT
            else:
                fit = fit_peak(energies[bounds[mm]:bounds[mm + 1]])
            metrics[(sm, mm)] = {"ingest": int(ingest[mm]), "selected": int(selected[mm]),
                                 "fit_sides": int(fit_counts[mm]), "fit": fit}
    return metrics


def uniformity(dataset: Dataset, selection: Selection, target=511.0, tolerance_pct=10.0):
    if target <= 0 or tolerance_pct < 0:
        raise ValueError("Target and tolerance must be nonnegative with positive target")
    rows = []
    for sm in sorted(set(dataset.expected_time) | set(dataset.modules)):
        data = dataset.modules.get(sm)
        if data is None:
            fit = {"status": "no data", "mu": None, "resolution": None}
            count = 0
        elif not dataset.settings.calibrated:
            mask = selection.mask(data, energy=False)
            count = int(mask.sum())
            fit = {"status": "unavailable: raw a.u. (no keV calibration)",
                   "mu": None, "resolution": None}
        else:
            mask = selection.mask(data, energy=False)
            count = int(mask.sum())
            fit = fit_peak(data.energy[mask])
        mu = fit["mu"]
        rows.append({"sm": sm, "events": count, "fit": fit,
                     "deviation_pct": 100 * (mu - target) / target if mu is not None else None,
                     "result": ("UNAVAILABLE" if mu is None else
                                "IN TOLERANCE" if abs(mu - target) <= target * tolerance_pct / 100
                                else "OUT OF TOLERANCE")})
    return rows


def rate_series(dataset: Dataset, sm: int, file_index: int, bins: int = 60):
    """Within-file observed detector-side counts; never infer full-run live time."""
    if bins < 2:
        raise ValueError("At least two bins required")
    data = dataset.modules.get(sm)
    span = dataset.file_spans.get(file_index)
    if data is None or span is None or span[1] <= span[0]:
        return None
    times = data.timestamp[data.file_index == file_index]
    if times.size == 0:
        return None
    duration = (span[1] - span[0]) * TIMESTAMP_SECONDS
    seconds = (times.astype(np.float64) - float(span[0])) * TIMESTAMP_SECONDS
    counts, edges = np.histogram(seconds, bins=bins, range=(0, duration))
    duration = edges[1] - edges[0]
    return edges, counts / duration


def pair_offset_series(dataset: Dataset, sm: int, file_index: int, bins: int = 60):
    """Median signed local-minus-partner coincidence hit timestamp per time bin.

    This is a paired-event observation, *not* an absolute clock drift estimate.
    Event geometry and time-of-flight also affect it.
    """
    if bins < 2:
        raise ValueError("At least two bins required")
    span = dataset.file_spans.get(file_index)
    data = dataset.modules.get(sm)
    if span is None or span[1] <= span[0] or data is None:
        return None
    in_file = data.file_index == file_index
    if not in_file.any():
        return None
    seconds = (data.timestamp[in_file].astype(float) - float(span[0])) * TIMESTAMP_SECONDS
    difference_ns = (data.timestamp[in_file].astype(np.float64)
                     - data.partner_timestamp[in_file].astype(np.float64)) * 1e-3
    edges = np.linspace(0, (span[1] - span[0]) * TIMESTAMP_SECONDS, bins + 1)
    bin_ids = np.minimum(np.searchsorted(edges, seconds, side="right") - 1, bins - 1)
    medians = np.full(bins, np.nan)
    for position in np.unique(bin_ids):
        if 0 <= position < bins:
            medians[position] = np.median(difference_ns[bin_ids == position])
    return edges, medians
