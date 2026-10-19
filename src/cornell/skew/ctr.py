"""Independent CTR from known-position point sources (spec 008 T9b, FR-13).

Usage::

    python -u -m src.cornell.skew ctr CTR.json

The CTR file names the frozen canonical LUT (with its SHA-256), the
configuration and ``.encal`` of the job that solved it, and uncorrected
point-source LDATs, each with a declared source position and uncertainty
(mm, scanner frame of the selected geometry). The LUT's job file must still
exist: it proves that the configuration and ``.encal`` are the job's and that
no point-source LDAT was a job input.

A source position is either declared (mm) or ``"lor_convergence"``: the point
closest to the acquisition's own LORs (least squares, refitted on LORs within
max(median miss, 5 mm)), from endpoints only, never timestamps. Owner decision
2026-10-19 after the declared centre (0, 0, 0) proved 25 mm off axially; the
declared nominal position is recorded with it.

Owner decisions (2026-10-19), fixed before any point-source event was read:

* Cuts: the job's pre-geometry selection and seeded slab draw; endpoints from
  the selected faceted frames at 7 mm depth. An event is kept when its LOR
  passes within ``lor_distance_mm`` (15) of the source point, between its
  endpoints; the norm-run shell and scanner-origin opposition cuts do not apply.
* Residual: LUT-corrected ``t_1 - t_2`` minus the geometric TOF
  ``(|s - p_1| - |s - p_2|) / c``. Events with a channel lacking an applied
  delay are excluded and counted.
* Fit: binned Poisson likelihood of a Gaussian plus flat background in
  +-``window_ps`` with ``bin_ps`` bins. Reported: Gaussian FWHM, a model-free
  half-maximum FWHM above the fitted background, background fraction, and for
  tails the model-free full width at tenth maximum and FWTM/FWHM (1.82 for a
  Gaussian). (A tail excess beyond +-3 sigma was dropped: the likelihood fit
  balances it to about zero whatever the tails.)
* Pooling is event weighted; each position is also reported alone.
  Statistical uncertainty: seeded Poisson bootstrap of events. Systematics:
  every source shifted by +- its uncertainty along each axis (one common
  shift) and depth 3/11 mm, added in quadrature with components listed. LOR
  distances 10/25 mm are reported as sensitivity, not added.

The width includes residual offset errors and endpoint blur: it is the
system CTR at these cuts, not an intrinsic detector CTR. A shell
(normalization phantom) timing width depends on the assumed source model and
is never reported as CTR.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from src.cornell.skew.geometry import (C_MM_PER_PS, NORMALIZATION_PHANTOM, OWNER_INTERACTION_DEPTH_MM,
                                       SELECTED_AXIAL_SIGN, SELECTED_TRANSVERSE_SIGN)
from src.cornell.skew.ingest import (CompiledLookup, GeometryHypothesis, LegacySetup, apply_geometry,
                                     code_label, compiled_batches)
from src.cornell.skew.lut import UNCORRECTED, CanonicalLut

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "cornell-ctr-point/1"
REPORT_SCHEMA = "cornell-ctr-point-report/1"
_REQUIRED = {"schema", "config", "encal", "lut", "lut_sha256", "output", "acquisitions"}
_OPTIONAL = {"lor_distance_mm": 15.0, "sensitivity_lor_mm": [10.0, 25.0], "window_ps": 3000.0, "bin_ps": 10.0,
             "bootstrap": 200, "seed": 0, "depth_variants_mm": [3.0, 11.0], "max_records": None,
             "localization_events": 500_000}
_ACQUISITION_KEYS = {"name", "ldat", "source_mm", "source_sigma_mm", "input_correction"}
_ACQUISITION_OPTIONAL = {"provenance", "declared"}
LOCALIZE = "lor_convergence"
LOCALIZATION_ITERATIONS = 6
LOCALIZATION_FLOOR_MM = 5.0
FWHM_PER_SIGMA = 2 * math.sqrt(2 * math.log(2))
GAUSSIAN_FWTM_TO_FWHM = math.sqrt(math.log(10) / math.log(2))  # 1.823
MIN_FIT_EVENTS = 100
TOF_EDGES_PS = np.arange(-2000.0, 2000.0 + 1, 100.0)
MIN_TOF_BIN_EVENTS = 100
STAGES = ("records", "passed pre-geometry", "invalid endpoint", "no applied delay", "LOR outside distance",
          "outside window", "in window")
NOT_CTR = "shell (normalization phantom) timing width is model dependent and is not CTR"


class CtrError(ValueError):
    pass


@dataclass(frozen=True)
class Acquisition:
    name: str
    ldat: Path
    source_mm: tuple | None     # None: located from its own LOR endpoints (timing blind)
    source_sigma_mm: tuple
    provenance: Path | None
    declared: str | None = None  # operator's nominal position, recorded only


@dataclass(frozen=True)
class CtrSpec:
    path: Path
    sha256: str
    config: Path
    encal: Path
    lut: Path
    lut_sha256: str
    output: Path
    acquisitions: tuple
    lor_distance_mm: float
    sensitivity_lor_mm: tuple
    window_ps: float
    bin_ps: float
    bootstrap: int
    seed: int
    depth_variants_mm: tuple
    max_records: int | None
    localization_events: int = 500_000

    @property
    def edges_ps(self):
        bins = int(round(2 * self.window_ps / self.bin_ps))
        return np.linspace(-self.window_ps, self.window_ps, bins + 1)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(value, label, *, nonnegative=False):
    try:
        vector = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise CtrError(f"{label}: three numbers in mm required") from None
    if len(vector) != 3 or not all(math.isfinite(v) for v in vector) or (nonnegative and min(vector) < 0):
        raise CtrError(f"{label}: three finite{' nonnegative' if nonnegative else ''} numbers in mm required")
    return vector


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise CtrError(f"{label} must be a positive number")
    return float(value)


def load_ctr(path):
    """Validated CTR file; paths are relative to the file (config: absolute or repository)."""
    path = Path(path)
    raw = path.read_bytes()
    spec = json.loads(raw)
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise CtrError(f"CTR file schema must be {SCHEMA!r}")
    if not _REQUIRED <= set(spec) <= _REQUIRED | set(_OPTIONAL):
        raise CtrError(f"CTR file keys: required {sorted(_REQUIRED)}, optional {sorted(_OPTIONAL)}")
    options = {key: spec.get(key, default) for key, default in _OPTIONAL.items()}
    base = path.resolve().parent

    def local(value):
        value = Path(value)
        return value if value.is_absolute() else base / value

    acquisitions = []
    if not isinstance(spec["acquisitions"], list) or not spec["acquisitions"]:
        raise CtrError("at least one point-source acquisition required")
    for item in spec["acquisitions"]:
        if not isinstance(item, dict) or not _ACQUISITION_KEYS <= set(item) <= _ACQUISITION_KEYS | _ACQUISITION_OPTIONAL:
            raise CtrError(f"acquisitions need {sorted(_ACQUISITION_KEYS)} (optional 'provenance'); a shell or "
                           f"other extended source is not accepted ({NOT_CTR}): {item!r}")
        if item["input_correction"] != UNCORRECTED["state"]:
            raise CtrError(f"{item['name']}: only uncorrected LDAT is accepted; the LUT is applied offline and a "
                           "natively corrected conversion would be corrected twice")
        source = (None if item["source_mm"] == LOCALIZE
                  else _finite_vector(item["source_mm"], f"{item['name']} source_mm (or {LOCALIZE!r})"))
        acquisitions.append(Acquisition(
            str(item["name"]), local(item["ldat"]), source,
            _finite_vector(item["source_sigma_mm"], f"{item['name']} source_sigma_mm", nonnegative=True),
            local(item["provenance"]) if "provenance" in item else None,
            None if item.get("declared") is None else str(item["declared"])))
    if len({a.name for a in acquisitions}) != len(acquisitions):
        raise CtrError("acquisition names must be unique")
    window, width = _positive(options["window_ps"], "window_ps"), _positive(options["bin_ps"], "bin_ps")
    bins = 2 * window / width
    if abs(bins - round(bins)) > 1e-9 or round(bins) < 10:
        raise CtrError("2 * window_ps must be a whole number (>= 10) of bin_ps")
    depths = tuple(float(d) for d in options["depth_variants_mm"])
    if not all(0 <= d <= 20 for d in depths):
        raise CtrError("depth variants must lie within the 20 mm crystal")
    bootstrap = options["bootstrap"]
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0 or bootstrap == 1:
        raise CtrError("bootstrap: 0 (none) or at least 2 replicates")
    limit = options["max_records"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise CtrError("max_records: null or a positive integer")
    if isinstance(options["seed"], bool) or not isinstance(options["seed"], int) or options["seed"] < 0:
        raise CtrError("seed must be a nonnegative integer")
    events = options["localization_events"]
    if isinstance(events, bool) or not isinstance(events, int) or events < 100:
        raise CtrError("localization_events must be an integer >= 100")
    config = Path(spec["config"])
    return CtrSpec(path, hashlib.sha256(raw).hexdigest(), config if config.is_absolute() else ROOT / config,
                   local(spec["encal"]), local(spec["lut"]), str(spec["lut_sha256"]).lower(), local(spec["output"]),
                   tuple(acquisitions), _positive(options["lor_distance_mm"], "lor_distance_mm"),
                   tuple(_positive(d, "sensitivity_lor_mm") for d in options["sensitivity_lor_mm"]), window, width,
                   bootstrap, options["seed"], depths, limit, events)


def frozen_inputs(spec):
    """Check the LUT is the frozen, uncorrected job table and the point sources are disjoint from its job."""
    lut_sha = _sha256(spec.lut)
    if lut_sha != spec.lut_sha256:
        raise CtrError(f"LUT SHA-256 {lut_sha} is not the frozen {spec.lut_sha256}")
    lut = CanonicalLut.read(spec.lut)
    if lut.input_correction != UNCORRECTED:
        raise CtrError("LUT was solved from corrected data; offline application to uncorrected LDAT needs an "
                       "uncorrected-data LUT")
    provenance = lut.manifest.get("provenance") or {}
    job_path = Path(provenance.get("job", ""))
    if not provenance.get("job") or not job_path.is_file() or _sha256(job_path) != provenance.get("job_sha256"):
        raise CtrError(f"the LUT's job file {job_path} is missing or changed: cuts and disjointness cannot be verified")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_config = Path(job["config"])
    job_config = job_config if job_config.is_absolute() else ROOT / job_config
    config_sha = _sha256(spec.config)
    if config_sha != _sha256(job_config):
        raise CtrError(f"configuration {spec.config} differs from the LUT job's {job_config}")
    encal_sha = _sha256(spec.encal)
    if encal_sha not in set((provenance.get("encal_sha256") or {}).values()):
        raise CtrError(f".encal {spec.encal} is not one the LUT job used")
    job_base = job_path.resolve().parent
    for run in job["runs"]:
        directory = Path(run["directory"])
        directory = (directory if directory.is_absolute() else job_base / directory).resolve()
        for acquisition in spec.acquisitions:
            if acquisition.ldat.resolve().parent == directory and acquisition.ldat.name.startswith(run["prefix"]):
                raise CtrError(f"{acquisition.name}: {acquisition.ldat} is an input of the LUT job run {run['name']}")
    return {"lut": str(spec.lut), "lut_sha256": lut_sha, "lut_table_sha256": lut.manifest["table"]["sha256"],
            "lut_input_correction": lut.input_correction, "lut_job": str(job_path),
            "lut_job_sha256": provenance["job_sha256"], "lut_runs": provenance.get("runs"),
            "config": str(spec.config), "config_sha256": config_sha, "encal": str(spec.encal),
            "encal_sha256": encal_sha}, lut


class Delays:
    """Applied LUT delays by TIME channel; channels without an applied delay map to -1."""

    def __init__(self, lut):
        rows = lut.rows[lut.rows["apply"]]
        order = np.argsort(rows["time_channel"])
        self.channels = rows["time_channel"][order].astype(np.int64)
        self.delays_ps = rows["delay_ps"][order].astype(np.float64)

    def index(self, channels):
        channels = np.asarray(channels, np.int64)
        if not len(self.channels):
            return np.full(channels.shape, -1, np.int64)
        k = np.clip(np.searchsorted(self.channels, channels), 0, len(self.channels) - 1)
        return np.where(self.channels[k] == channels, k, -1)


def geometric_tof_ps(source_mm, endpoint_1_mm, endpoint_2_mm):
    """Expected ``t_1 - t_2`` (ps) for annihilation at the source; rows of endpoints in mm."""
    source = np.asarray(source_mm, np.float64)
    return (np.linalg.norm(source - endpoint_1_mm, axis=-1)
            - np.linalg.norm(source - endpoint_2_mm, axis=-1)) / C_MM_PER_PS


def lor_distance_mm(source_mm, endpoint_1_mm, endpoint_2_mm):
    """Distance (mm) of the source from each LOR and its position along it (0 at endpoint 1, 1 at endpoint 2)."""
    source = np.asarray(source_mm, np.float64)
    direction = endpoint_2_mm - endpoint_1_mm
    with np.errstate(invalid="ignore", divide="ignore"):
        along = np.einsum("ij,ij->i", source - endpoint_1_mm, direction) / np.einsum("ij,ij->i", direction, direction)
        distance = np.linalg.norm(source - (endpoint_1_mm + along[:, None] * direction), axis=-1)
    return distance, along


def residual_ps(dt_corrected_ps, source_mm, endpoint_1_mm, endpoint_2_mm):
    """Corrected ``t_1 - t_2`` minus the known-position geometric TOF."""
    return np.asarray(dt_corrected_ps, np.float64) - geometric_tof_ps(source_mm, endpoint_1_mm, endpoint_2_mm)


def converge_point(endpoint_1_mm, endpoint_2_mm, *, iterations=LOCALIZATION_ITERATIONS,
                   floor_mm=LOCALIZATION_FLOOR_MM):
    """Point closest to the LORs (least squares), refitted on LORs within max(median miss, floor) of it.

    Uses endpoints only, never timestamps. Returns (point, median miss of the last fit's LORs, LORs used).
    """
    direction = endpoint_2_mm - endpoint_1_mm
    length = np.linalg.norm(direction, axis=1)
    ok = np.isfinite(length) & (length > 0)
    if np.count_nonzero(ok) < 3:
        raise CtrError("fewer than three LORs to locate the source")
    p, d = endpoint_1_mm[ok], direction[ok] / length[ok, None]
    used = np.ones(len(p), bool)
    for _ in range(iterations):
        pu, du = p[used], d[used]
        # sum over LORs of (I - d d^T), and of (I - d d^T) p
        a = len(pu) * np.eye(3) - du.T @ du
        b = pu.sum(axis=0) - du.T @ np.einsum("ij,ij->i", du, pu)
        if abs(np.linalg.det(a)) < 1e-9 * len(pu) ** 3:
            raise CtrError("LORs are parallel: the source cannot be located")
        point = np.linalg.solve(a, b)
        along = np.einsum("ij,ij->i", point - p, d)
        miss = np.linalg.norm(point - (p + along[:, None] * d), axis=1)
        used = miss <= max(float(np.median(miss)), floor_mm)
    return point, float(np.median(miss[used])), int(np.count_nonzero(used))


def localize(acquisition, index, spec, lookup, hypothesis):
    """Timing-blind source position from the acquisition's first ``localization_events`` pre-geometry events."""
    p, q, events = [], [], 0
    for features in compiled_batches(acquisition.ldat, lookup, file_index=index, max_records=spec.max_records,
                                     seed=spec.seed):
        if not len(features.ints):
            continue
        geometry = apply_geometry(features, hypothesis)
        valid = geometry.codes != 1
        take = min(int(np.count_nonzero(valid)), spec.localization_events - events)
        rows = geometry.endpoints_mm[valid][:take]
        p.append(rows[:, :3])
        q.append(rows[:, 3:])
        events += take
        if events >= spec.localization_events:
            break
    if not events:
        raise CtrError(f"{acquisition.name}: no LOR with valid endpoints to locate the source")
    point, miss, used = converge_point(np.concatenate(p), np.concatenate(q))
    return {"method": LOCALIZE, "position_mm": [float(v) for v in point], "lors": events, "lors_used": used,
            "median_miss_mm": miss, "geometry": hypothesis.name}


def _model(params, edges):
    mean, sigma, signal, background = params
    return signal * np.diff(ndtr((edges - mean) / sigma)) + background / (len(edges) - 1)


def _width_at(counts, centers, background_per_bin, fraction):
    """Model-free full width at ``fraction`` of the peak height above the background (3-bin smoothing)."""
    smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    k = int(np.argmax(smooth))
    height = smooth[k] - background_per_bin
    if not height > 0:
        return None
    level = background_per_bin + height * fraction
    width = centers[1] - centers[0]
    left = k
    while left > 0 and smooth[left - 1] >= level:
        left -= 1
    right = k
    while right < len(smooth) - 1 and smooth[right + 1] >= level:
        right += 1
    if left == 0 or right == len(smooth) - 1:
        return None
    x_left = centers[left - 1] + width * (level - smooth[left - 1]) / (smooth[left] - smooth[left - 1])
    x_right = centers[right] + width * (smooth[right] - level) / (smooth[right] - smooth[right + 1])
    return float(x_right - x_left)


def fit_histogram(counts, edges, *, start=None):
    """Gaussian + flat background, binned Poisson likelihood; ``status`` says why numbers are absent."""
    counts = np.asarray(counts, np.float64)
    edges = np.asarray(edges, np.float64)
    total = float(counts.sum())
    if total < MIN_FIT_EVENTS:
        return {"status": "insufficient events", "events": total}
    centers, width, bins = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0], len(counts)
    if start is None:
        outer = max(bins // 10, 1)
        level = float(np.concatenate((counts[:outer], counts[-outer:])).mean())
        smooth = np.convolve(counts, np.ones(5) / 5, mode="same")
        k = int(np.argmax(smooth))
        above = smooth - level
        sigma = max(np.count_nonzero(above > above[k] / 2) * width / FWHM_PER_SIGMA, width)
        background = max(level * bins, 1.0)
        start = (centers[k], sigma, max(total - background, 1.0), background)
    mean, sigma, signal, background = start
    x0 = np.array([mean, math.log(sigma), math.log(max(signal, 1.0)), math.log(max(background, 1e-3))])

    def nll(x):
        mu = np.maximum(_model((x[0], math.exp(x[1]), math.exp(x[2]), math.exp(x[3])), edges), 1e-300)
        return float(np.sum(mu - counts * np.log(mu)))

    result = minimize(nll, x0, method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 6000, "maxfev": 12000})
    mean, sigma, signal, background = result.x[0], *np.exp(result.x[1:])
    mu = _model((mean, sigma, signal, background), edges)
    inside = (sigma > width / 2) and (sigma < edges[-1] - edges[0]) and (edges[0] < mean < edges[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        deviance = 2 * np.sum(mu - counts + np.where(counts > 0, counts * np.log(counts / mu), 0.0))
    half = _width_at(counts, centers, background / bins, 0.5)
    tenth = _width_at(counts, centers, background / bins, 0.1)
    return {"status": "fitted" if result.success and inside else "fit failed",
            "events": total, "mean_ps": float(mean), "sigma_ps": float(sigma),
            "fwhm_ps": float(FWHM_PER_SIGMA * sigma), "model_free_fwhm_ps": half, "model_free_fwtm_ps": tenth,
            "fwtm_to_fwhm": None if half is None or tenth is None else tenth / half,
            "signal_events": float(signal), "background_events": float(background),
            "background_fraction": float(background / (signal + background)),
            "deviance_per_dof": float(deviance / max(bins - 4, 1)), "_start": (mean, sigma, signal, background)}


def _public(fit):
    return {k: v for k, v in fit.items() if not k.startswith("_")}


def combine_uncertainty(nominal, variants, statistical):
    """Systematic components (largest |change| per group), quadrature total with the statistical part."""
    components = {group: max(abs(v - nominal) for v in values) for group, values in variants.items() if values}
    systematic = math.sqrt(sum(c * c for c in components.values()))
    return {"statistical": statistical, "systematic_components": components, "systematic": systematic,
            "total": math.hypot(statistical, systematic) if statistical is not None else None}


@dataclass(frozen=True)
class Variant:
    name: str
    depth_index: int      # 0 nominal depth, k > 0 depth_variants_mm[k - 1]
    shift: tuple          # in units of each acquisition's source_sigma_mm
    lor_distance_mm: float
    group: str | None     # systematic group; None for nominal and sensitivity


def variants(spec):
    out = [Variant("nominal", 0, (0.0, 0.0, 0.0), spec.lor_distance_mm, None)]
    out += [Variant(f"lor_{d:g}mm", 0, (0.0, 0.0, 0.0), d, None) for d in spec.sensitivity_lor_mm]
    for axis, label in enumerate("xyz"):
        for sign in (-1.0, 1.0):
            shift = [0.0, 0.0, 0.0]
            shift[axis] = sign
            out.append(Variant(f"source_{label}{'+' if sign > 0 else '-'}", 0, tuple(shift), spec.lor_distance_mm,
                               f"source {label}"))
    out += [Variant(f"depth_{d:g}mm", k + 1, (0.0, 0.0, 0.0), spec.lor_distance_mm, "depth")
            for k, d in enumerate(spec.depth_variants_mm)]
    return out


class Accumulator:
    """Bounded per-variant, per-acquisition residual histograms, bootstrap replicates and coverage."""

    def __init__(self, spec, delays, sources=None):
        self.spec, self.delays = spec, delays
        self.sources = [np.array(a.source_mm, np.float64) for a in spec.acquisitions] if sources is None \
            else [np.asarray(s, np.float64) for s in sources]
        self.variants = variants(spec)
        self.edges = spec.edges_ps
        bins, n = len(self.edges) - 1, len(spec.acquisitions)
        self.hist = np.zeros((len(self.variants), n, bins))
        self.boot = np.zeros((spec.bootstrap, n, bins))
        self.tof = np.zeros((n, len(TOF_EDGES_PS) - 1, bins))
        self.stages = [dict.fromkeys(STAGES, 0) for _ in range(n)]
        self.pre_geometry = [{} for _ in range(n)]
        m = len(delays.channels)
        self.channel_seen = np.zeros(m, bool)
        self.pair_seen = np.zeros((m, m), bool)

    def add(self, index, features, geometries):
        spec, acquisition = self.spec, self.spec.acquisitions[index]
        stages = self.stages[index]
        codes, found = np.unique(features.codes, return_counts=True)
        for code, count in zip(codes.tolist(), found.tolist()):
            label = code_label(code)
            self.pre_geometry[index][label] = self.pre_geometry[index].get(label, 0) + count
        stages["records"] += len(features.codes)
        stages["passed pre-geometry"] += len(features.ints)
        if not len(features.ints):
            return
        k1, k2 = self.delays.index(features.column("time_channel_1")), self.delays.index(features.column("time_channel_2"))
        applied = (k1 >= 0) & (k2 >= 0)
        delays = self.delays.delays_ps
        # integers first: (t_1 - t_2) is exact, then - b_1 + b_2
        corrected = np.where(applied, features.column("dt_ps").astype(np.float64)
                             - delays[np.maximum(k1, 0)] + delays[np.maximum(k2, 0)], np.nan)
        bins = len(self.edges) - 1
        sigma = np.array(acquisition.source_sigma_mm)
        for v, variant in enumerate(self.variants):
            geometry = geometries[variant.depth_index]
            valid = geometry.codes != 1  # endpoints defined; shell codes are irrelevant here
            p, q = geometry.endpoints_mm[:, :3], geometry.endpoints_mm[:, 3:]
            source = self.sources[index] + np.array(variant.shift) * sigma
            distance, along = lor_distance_mm(source, p, q)
            with np.errstate(invalid="ignore"):
                near = valid & applied & (distance <= variant.lor_distance_mm) & (along > 0) & (along < 1)
            residual = residual_ps(corrected[near], source, p[near], q[near])
            b = np.floor((residual + spec.window_ps) / spec.bin_ps).astype(np.int64)
            inside = (b >= 0) & (b < bins)
            self.hist[v, index] += np.bincount(b[inside], minlength=bins)
            if v:
                continue
            stages["invalid endpoint"] += int(np.count_nonzero(~valid))
            stages["no applied delay"] += int(np.count_nonzero(valid & ~applied))
            stages["LOR outside distance"] += int(np.count_nonzero(valid & applied & ~near))
            stages["outside window"] += int(np.count_nonzero(~inside))
            stages["in window"] += int(np.count_nonzero(inside))
            b = b[inside]
            i1, i2 = k1[near][inside], k2[near][inside]
            self.channel_seen[i1] = self.channel_seen[i2] = True
            self.pair_seen[np.minimum(i1, i2), np.maximum(i1, i2)] = True
            tof = geometric_tof_ps(source, p[near][inside], q[near][inside])
            t = np.searchsorted(TOF_EDGES_PS, tof, side="right") - 1
            ok = (t >= 0) & (t < len(TOF_EDGES_PS) - 1)
            self.tof[index] += np.bincount(t[ok] * bins + b[ok], minlength=self.tof[index].size).reshape(
                self.tof[index].shape)
            if spec.bootstrap:
                rng = np.random.default_rng([spec.seed, index, features.first_record])
                for r in range(spec.bootstrap):
                    self.boot[r, index] += np.bincount(b, weights=rng.poisson(1.0, len(b)), minlength=bins)


def tof_dependence(tof_hist, edges, sigma_ps):
    """Median residual per geometric-TOF bin and its weighted linear slope (ps per ps of TOF)."""
    rows = []
    for k in range(tof_hist.shape[0]):
        counts = tof_hist[k]
        n = counts.sum()
        if n < MIN_TOF_BIN_EVENTS:
            continue
        cumulative = np.cumsum(counts)
        j = int(np.searchsorted(cumulative, n / 2))
        below = cumulative[j - 1] if j else 0.0
        median = edges[j] + (n / 2 - below) / counts[j] * (edges[1] - edges[0])
        rows.append({"tof_ps": float((TOF_EDGES_PS[k] + TOF_EDGES_PS[k + 1]) / 2), "events": float(n),
                     "median_residual_ps": float(median), "median_sigma_ps": float(1.2533 * sigma_ps / math.sqrt(n))})
    out = {"bins": rows, "slope": None, "slope_sigma": None, "intercept_ps": None}
    if len(rows) >= 3 and sigma_ps:
        x = np.array([r["tof_ps"] for r in rows])
        y = np.array([r["median_residual_ps"] for r in rows])
        w = 1 / np.array([r["median_sigma_ps"] for r in rows]) ** 2
        design = np.column_stack((np.ones_like(x), x))
        cov = np.linalg.inv(design.T @ (design * w[:, None]))
        coef = cov @ design.T @ (w * y)
        out.update(slope=float(coef[1]), slope_sigma=float(math.sqrt(cov[1, 1])), intercept_ps=float(coef[0]))
    return out


def summarize(accumulator):
    """Fits, bootstrap and systematics per acquisition and pooled (event weighted)."""
    spec, edges = accumulator.spec, accumulator.edges
    names = [a.name for a in spec.acquisitions]
    scopes = [("pooled", slice(None))] + [(name, k) for k, name in enumerate(names)]
    results = {}
    for scope, selector in scopes:
        def pick(array):
            return array[..., selector, :].sum(axis=-2) if scope == "pooled" else array[..., selector, :]

        hist = pick(accumulator.hist)
        nominal = fit_histogram(hist[0], edges)
        entry = {"fit": _public(nominal)}
        if nominal["status"] != "fitted":
            entry["uncertainty"] = None
            results[scope] = entry
            continue
        boot = pick(accumulator.boot) if spec.bootstrap else np.zeros((0, len(edges) - 1))
        replicate = [fit_histogram(h, edges, start=nominal["_start"]) for h in boot]
        good = [f for f in replicate if f["status"] == "fitted"]
        entry["bootstrap"] = {"replicates": len(replicate), "fitted": len(good)}
        variant_fits = {v.name: fit_histogram(hist[i], edges, start=nominal["_start"])
                        for i, v in enumerate(accumulator.variants) if i}
        entry["variants"] = {name: _public(f) for name, f in variant_fits.items()}
        for key in ("fwhm_ps", "model_free_fwhm_ps", "model_free_fwtm_ps"):
            values = [f[key] for f in good if f[key] is not None]
            statistical = float(np.std(values, ddof=1)) if len(values) >= 2 else None
            groups = {}
            for v in accumulator.variants:
                fit = variant_fits.get(v.name)
                if v.group and fit and fit["status"] == "fitted" and fit[key] is not None:
                    groups.setdefault(v.group, []).append(fit[key])
            entry.setdefault("uncertainty", {})[key] = (
                combine_uncertainty(nominal[key], groups, statistical) if nominal[key] is not None else None)
        entry["lor_sensitivity_fwhm_ps"] = {v.name: variant_fits[v.name].get("fwhm_ps")
                                            for v in accumulator.variants[1:] if v.group is None}
        tof = accumulator.tof.sum(axis=0) if scope == "pooled" else accumulator.tof[selector]
        entry["tof_dependence"] = tof_dependence(tof, edges, nominal["sigma_ps"])
        results[scope] = entry
    return results


def _hypotheses(spec, setup):
    depths = (OWNER_INTERACTION_DEPTH_MM,) + spec.depth_variants_mm
    return [GeometryHypothesis.faceted(f"faceted transverse_sign {SELECTED_TRANSVERSE_SIGN:+d} depth {d:g} mm",
                                       spec.config, setup, axial_sign=SELECTED_AXIAL_SIGN,
                                       transverse_sign=SELECTED_TRANSVERSE_SIGN, source=NORMALIZATION_PHANTOM,
                                       interaction_depth_mm=d) for d in depths]


def run_ctr(spec, *, log=print):
    """Stream every acquisition once, fit, write the report JSON, histogram CSV and plot; return the report."""
    started = time.perf_counter()
    inputs, lut = frozen_inputs(spec)
    acquisitions = []
    for acquisition in spec.acquisitions:
        if not acquisition.ldat.is_file():
            raise CtrError(f"{acquisition.name}: missing LDAT {acquisition.ldat}")
        acquisitions.append({"name": acquisition.name, "ldat": str(acquisition.ldat),
                             "size_bytes": acquisition.ldat.stat().st_size, "sha256": _sha256(acquisition.ldat),
                             "declared": acquisition.declared,
                             "source_sigma_mm": list(acquisition.source_sigma_mm),
                             "input_correction": UNCORRECTED["state"],
                             "provenance": None if acquisition.provenance is None else str(acquisition.provenance),
                             "provenance_sha256": None if acquisition.provenance is None
                             else _sha256(acquisition.provenance)})
    setup = LegacySetup.from_files(spec.config, spec.encal, root=ROOT)
    lookup = CompiledLookup.from_setup(setup)
    hypotheses = _hypotheses(spec, setup)
    delays = Delays(lut)
    sources = []
    for index, (acquisition, entry) in enumerate(zip(spec.acquisitions, acquisitions)):
        if acquisition.source_mm is None:
            entry["source"] = localize(acquisition, index, spec, lookup, hypotheses[0])
            log(f"{acquisition.name}: source located at {np.round(entry['source']['position_mm'], 2).tolist()} mm "
                f"(median LOR miss {entry['source']['median_miss_mm']:.2f} mm)")
        else:
            entry["source"] = {"method": "declared", "position_mm": list(acquisition.source_mm)}
        sources.append(entry["source"]["position_mm"])
    accumulator = Accumulator(spec, delays, sources)
    for index, acquisition in enumerate(spec.acquisitions):
        t0 = time.perf_counter()
        for features in compiled_batches(acquisition.ldat, lookup, file_index=index, max_records=spec.max_records,
                                         seed=spec.seed):
            geometries = [apply_geometry(features, h) for h in hypotheses]
            accumulator.add(index, features, geometries)
        stages = accumulator.stages[index]
        log(f"{acquisition.name}: {stages['records']:,} records, {stages['in window']:,} in window "
            f"({time.perf_counter() - t0:.1f} s)")
    results = summarize(accumulator)
    applied = len(delays.channels)
    report = {
        "schema": REPORT_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ctr_file": str(spec.path), "ctr_file_sha256": spec.sha256, "frozen_inputs": inputs,
        "acquisitions": acquisitions,
        "cuts": {"pre_geometry": "LUT job selection (energy window, minimodule, slab draw seed "
                                 f"{spec.seed}, coincidence window)",
                 "geometry": hypotheses[0].name, "lor_distance_mm": spec.lor_distance_mm,
                 "lor_between_endpoints": True, "window_ps": spec.window_ps, "bin_ps": spec.bin_ps,
                 "max_records": spec.max_records},
        "model": "Gaussian + flat background, binned Poisson likelihood; model-free FWHM at half height above "
                 "the fitted background (3-bin smoothing); tails: model-free FWTM and FWTM/FWHM "
                 f"(Gaussian {GAUSSIAN_FWTM_TO_FWHM:.3f})",
        "weighting": "event weighted: pooled histogram is the sum over acquisitions",
        "uncertainty": f"statistical: {spec.bootstrap} seeded Poisson bootstrap replicates of events; systematic: "
                       "source +- sigma per axis (common shift), depth "
                       f"{'/'.join(f'{d:g}' for d in spec.depth_variants_mm)} mm; LOR "
                       f"{'/'.join(f'{d:g}' for d in spec.sensitivity_lor_mm)} mm sensitivity not added",
        "counts": {a.name: {"stages": accumulator.stages[k], "pre_geometry": accumulator.pre_geometry[k]}
                   for k, a in enumerate(spec.acquisitions)},
        "coverage": {"applied_channels": applied, "applied_channels_seen": int(accumulator.channel_seen.sum()),
                     "applied_channel_pairs_seen": int(accumulator.pair_seen.sum())},
        "results": results,
        "interpretation": "system CTR at these cuts (includes residual offset errors and endpoint blur); "
                          f"not an intrinsic detector CTR; {NOT_CTR}",
        "seconds": round(time.perf_counter() - started, 1),
    }
    spec.output.parent.mkdir(parents=True, exist_ok=True)
    spec.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    _write_histograms(spec, accumulator)
    _plot(spec, accumulator, results)
    return report


def _write_histograms(spec, accumulator):
    edges = accumulator.edges
    names = [a.name for a in spec.acquisitions]
    with open(spec.output.with_name(spec.output.stem + "_histograms.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["residual_low_ps", "residual_high_ps", *names, "pooled"])
        nominal = accumulator.hist[0]
        for k in range(len(edges) - 1):
            writer.writerow([f"{edges[k]:g}", f"{edges[k + 1]:g}", *(int(nominal[a, k]) for a in range(len(names))),
                             int(nominal[:, k].sum())])


def _plot(spec, accumulator, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = accumulator.edges
    centers = (edges[:-1] + edges[1:]) / 2
    pooled = accumulator.hist[0].sum(axis=0)
    fit = results["pooled"]["fit"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, scale in zip(axes, ("linear", "log")):
        ax.step(centers, pooled, where="mid", lw=0.8, label="pooled residual")
        if fit["status"] == "fitted":
            model = _model((fit["mean_ps"], fit["sigma_ps"], fit["signal_events"], fit["background_events"]), edges)
            ax.plot(centers, model, lw=1.2, label=f"Gaussian + flat, FWHM {fit['fwhm_ps']:.0f} ps")
        ax.set_yscale(scale)
        ax.set_xlabel("corrected t1 - t2 minus geometric TOF (ps)")
        ax.set_ylabel(f"events / {spec.bin_ps:g} ps")
        ax.legend(fontsize=8)
    figure.suptitle(f"Point-source CTR, LUT {spec.lut.name}, LOR <= {spec.lor_distance_mm:g} mm")
    figure.tight_layout()
    figure.savefig(spec.output.with_suffix(".png"), dpi=120)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.cornell.skew ctr",
                                     description="Independent CTR from known-position point sources (FR-13).")
    parser.add_argument("ctr_file", help=f"CTR file (schema {SCHEMA})")
    args = parser.parse_args(argv)
    try:
        report = run_ctr(load_ctr(args.ctr_file))
    except CtrError as error:
        print(f"ctr refused: {error}", file=sys.stderr)
        return 2
    pooled = report["results"]["pooled"]
    fit = pooled["fit"]
    if fit["status"] != "fitted":
        print(f"pooled fit unavailable: {fit['status']}")
        return 1
    total = (pooled["uncertainty"]["fwhm_ps"] or {}).get("total")
    print(f"pooled CTR FWHM {fit['fwhm_ps']:.1f} ps" + (f" +- {total:.1f} ps" if total is not None else "")
          + f" (model-free {fit['model_free_fwhm_ps']}, {fit['events']:.0f} events)")
    return 0
