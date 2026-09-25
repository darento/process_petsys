"""Toolkit-independent LDAT ingestion and measurements for LDATInspector.

The event population is *detector sides of coincidence pairs*. Ingest cuts are
min-channel count and per-channel energy; the display's keV window is reversible.
Times in the PETsys LDAT examples are picoseconds (see imas_peak_fluctuation.py).
"""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
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
    max_pairs: int = 10000
    min_channels: int = 1
    min_channel_energy: float = 0.0
    calibrated: bool = True

    def validate(self):
        if self.system not in ("IMAS", "CORNELL"):
            raise ValueError("System must be IMAS or CORNELL")
        if not 1 <= self.max_pairs <= MAX_PAIRS_PER_FILE:
            raise ValueError(f"Coincidence pairs per file must be 1–{MAX_PAIRS_PER_FILE:,}")
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


def iter_pairs(path: str, limit: int):
    """Read the same native-layout records as read_compact; reject truncation.

    A limit is a prefix of the file, never a random subsample or full-run rate.
    """
    if Path(path).stat().st_size == 0:
        raise ValueError("Empty LDAT file")
    with open(path, "rb") as handle:
        for _ in range(limit):
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


@dataclass
class ModuleEvents:
    energy: np.ndarray
    raw_energy: np.ndarray
    partner_energy: np.ndarray
    partner_raw_energy: np.ndarray
    calibration_key: np.ndarray
    partner_calibration_key: np.ndarray
    x: np.ndarray
    y: np.ndarray
    doi: np.ndarray
    timestamp: np.ndarray
    partner_timestamp: np.ndarray
    mm: np.ndarray
    file_index: np.ndarray

    def __len__(self):
        return len(self.energy)


def _empty_modules():
    return defaultdict(lambda: {key: array(kind) for key, kind in _COLUMNS.items()})


def _freeze_modules(buffers):
    return {
        sm: ModuleEvents(**{key: np.frombuffer(values, dtype=np.dtype(_COLUMNS[key])).copy()
                            for key, values in cols.items()})
        for sm, cols in buffers.items()
    }


@dataclass
class FileResult:
    index: int
    path: str
    pairs_read: int = 0
    pairs_accepted: int = 0
    modules: dict[int, ModuleEvents] = field(default_factory=dict)
    time_counts: dict[int, Counter] = field(default_factory=dict)
    energy_counts: dict[int, Counter] = field(default_factory=dict)
    errors: Counter = field(default_factory=Counter)
    error: str | None = None
    prefix_limited: bool = False

    @property
    def success(self):
        return self.error is None


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
        buffers = _empty_modules()
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
                    slab = 0
                    if settings.system == "CORNELL":
                        slab, _, _ = get_slab_cornell(selected, setup.channel_types,
                                                       setup.coordinates)
                        if slab is None:
                            raise _UnresolvedCornellSlab
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
                    sides.append((sm, mm, energy, calibration_key, float(x), float(y),
                                  float(doi), int(t_hit[0]), tuple(times), tuple(energies)))
            except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
                result.errors["unresolved Cornell slab" if isinstance(exc, _UnresolvedCornellSlab)
                              else type(exc).__name__] += 1
                continue
            for side, partner in ((sides[0], sides[1]), (sides[1], sides[0])):
                sm, mm, raw, key_id, x, y, doi, timestamp, times, energies = side
                columns = buffers[sm]
                for key, value in zip(_COLUMNS, (raw, raw, partner[2], partner[2],
                                                 key_id, partner[3], x, y, doi,
                                                 timestamp, partner[7], mm, index)):
                    columns[key].append(value)
                t_counts[sm].update(times)
                e_counts[sm].update(energies)
            result.pairs_accepted += 1
        # Reaching the prefix limit does not establish whether this was a full run.
        result.prefix_limited = result.pairs_read == settings.max_pairs
        result.modules = _freeze_modules(buffers)
        result.time_counts = dict(t_counts)
        result.energy_counts = dict(e_counts)
    except Exception as exc:
        result.error = str(exc)
        result.modules = {}
        result.time_counts = {}
        result.energy_counts = {}
        result.pairs_accepted = 0
    return result


# The GUI and checks call process_file; it switches to the fast reader in spec 002 T2.
process_file = process_file_reference


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


def merge_results(settings: Settings, files: list[FileResult], setup: Setup | None = None) -> Dataset:
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
    merged = {}
    by_sm = defaultdict(list)
    t_counts, e_counts = defaultdict(Counter), defaultdict(Counter)
    for result in files:
        if not result.success:
            continue
        for sm, data in result.modules.items():
            by_sm[sm].append(data)
        for sm, counts in result.time_counts.items():
            t_counts[sm].update(counts)
        for sm, counts in result.energy_counts.items():
            e_counts[sm].update(counts)
    for sm, parts in by_sm.items():
        merged[sm] = ModuleEvents(**{
            key: np.concatenate([getattr(part, key) for part in parts])
            for key in _COLUMNS
        })
    spans = {}
    for result in files:
        if not result.success:
            continue
        for data in result.modules.values():
            if len(data) == 0:
                continue
            low, high = int(data.timestamp.min()), int(data.timestamp.max())
            if result.index in spans:
                previous = spans[result.index]
                low, high = min(low, previous[0]), max(high, previous[1])
            spans[result.index] = low, high
    dataset = Dataset(replace(settings, calibrated=False), files, merged,
                      dict(expected_t), dict(expected_e), dict(t_counts), dict(e_counts),
                      str(_path_from_config(settings.config_path, setup.config["map_file"])),
                      setup.config.copy(), spans, dict(expected_mm))
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

    modules = {sm: replace(data,
                           energy=converted(data.raw_energy, data.calibration_key),
                           partner_energy=converted(data.partner_raw_energy, data.partner_calibration_key))
               for sm, data in dataset.modules.items()}
    return replace(dataset, modules=modules,
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
