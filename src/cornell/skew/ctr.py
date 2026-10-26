"""Independent CTR from known-position point sources (spec 008 T9b, FR-13).

Usage::

    python -u -m src.cornell.skew ctr CTR.json

The CTR file names the frozen canonical LUT (with its SHA-256), the
configuration and ``.encal`` of the job that solved it, and uncorrected
point-source LDATs, each with a declared source position and uncertainty
(mm, scanner frame of the selected geometry). The LUT's job file must still
exist: it proves that the configuration and ``.encal`` are the job's and that
no point-source LDAT was a job input. The configuration may differ from the
job's only in ``energy_range`` (owner 2026-10-19): such a run is reported as a
study at other cuts (``cuts_identical_to_lut_job: false``), not the FR-13 result.

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
import yaml

from src.cornell.skew.geometry import (NORMALIZATION_PHANTOM, OWNER_INTERACTION_DEPTH_MM, SELECTED_AXIAL_SIGN,
                                       SELECTED_TRANSVERSE_SIGN)
from src.cornell.skew.ingest import (CompiledLookup, GeometryHypothesis, LegacySetup, apply_geometry,
                                     code_label, compiled_batches)
from src.cornell.skew.lut import UNCORRECTED, CanonicalLut
from src.timing.ctr import (  # noqa: F401  (moved core, re-exported)
    FWHM_PER_SIGMA, GAUSSIAN_FWTM_TO_FWHM, LOCALIZATION_FLOOR_MM, LOCALIZATION_ITERATIONS, MIN_FIT_EVENTS,
    MIN_TOF_BIN_EVENTS, TOF_EDGES_PS, CtrError, _model, _public, _width_at, combine_uncertainty,
    converge_point, fit_histogram, geometric_tof_ps, level_crossings, lor_distance_mm, residual_ps,
    Variant, summarize, tof_dependence, variants)

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
ANALYSIS_KEY = "energy_range"   # the one configuration key a CTR run may change against the LUT job (owner 2026-10-19)
STAGES = ("records", "passed pre-geometry", "invalid endpoint", "no applied delay", "walk input missing",
          "LOR outside distance", "outside window", "in window")
# spec 009 FR-5: optional walk LUT (with the listmode files its inputs come from) and earlier reports to tabulate
WALK_OPTIONAL = {"walk_lut": None, "walk_lut_sha256": None, "listmode": None, "previous_reports": []}
NOT_CTR = "shell (normalization phantom) timing width is model dependent and is not CTR"


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
    extra: dict | None = None       # keys of a file built on this one (spec 009 walk fit), validated there

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


def load_ctr(path, *, schema=SCHEMA, extra=None):
    """Validated CTR file; paths are relative to the file (config: absolute or repository).

    ``schema``/``extra`` let a file built on the CTR file (spec 009 walk fit) reuse it: ``extra`` maps its
    own keys to defaults (``...`` = required); their raw values land in ``CtrSpec.extra``.
    """
    path = Path(path)
    raw = path.read_bytes()
    spec = json.loads(raw)
    extra = (WALK_OPTIONAL if schema == SCHEMA else {}) if extra is None else extra
    if not isinstance(spec, dict) or spec.get("schema") != schema:
        raise CtrError(f"file schema must be {schema!r}")
    required = _REQUIRED | {key for key, default in extra.items() if default is ...}
    optional = set(_OPTIONAL) | set(extra)
    if not required <= set(spec) <= required | optional:
        raise CtrError(f"file keys: required {sorted(required)}, optional {sorted(optional - required)}")
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
                   bootstrap, options["seed"], depths, limit, events,
                   {key: spec.get(key, default) for key, default in extra.items()})


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
    ours = yaml.safe_load(spec.config.read_text(encoding="utf-8"))
    theirs = yaml.safe_load(job_config.read_text(encoding="utf-8"))
    differ = sorted(key for key in set(ours) | set(theirs) if ours.get(key) != theirs.get(key))
    if set(differ) - {ANALYSIS_KEY}:
        raise CtrError(f"configuration {spec.config} differs from the LUT job's {job_config} in "
                       f"{sorted(set(differ) - {ANALYSIS_KEY})}; only {ANALYSIS_KEY!r} may differ")
    window = ours.get(ANALYSIS_KEY)
    if not (isinstance(window, list) and len(window) == 2 and all(isinstance(v, (int, float)) for v in window)
            and all(math.isfinite(v) for v in window) and 0 <= window[0] < window[1]):
        raise CtrError(f"{ANALYSIS_KEY} must be [low, high] keV with 0 <= low < high")
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
            "config": str(spec.config), "config_sha256": config_sha, "lut_job_config": str(job_config),
            "lut_job_config_sha256": _sha256(job_config), "energy_range_kev": [float(v) for v in window],
            "lut_job_energy_range_kev": theirs.get(ANALYSIS_KEY), "cuts_identical_to_lut_job": not differ,
            "encal": str(spec.encal), "encal_sha256": encal_sha}, lut


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

    def add(self, index, features, geometries, walk_dt=None, keep=None):
        """``keep``: events with walk inputs (both accumulators of a walk run hold the same events);
        ``walk_dt``: W(side 1) - W(side 2) subtracted from the corrected ``t_1 - t_2`` (spec 009)."""
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
        has_delay = applied
        if keep is not None:
            applied = applied & keep
        delays = self.delays.delays_ps
        # integers first: (t_1 - t_2) is exact, then - b_1 + b_2
        corrected = np.where(applied, features.column("dt_ps").astype(np.float64)
                             - delays[np.maximum(k1, 0)] + delays[np.maximum(k2, 0)], np.nan)
        if walk_dt is not None:
            corrected = corrected - np.where(applied, walk_dt, 0.0)
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
            stages["no applied delay"] += int(np.count_nonzero(valid & ~has_delay))
            stages["walk input missing"] += int(np.count_nonzero(valid & has_delay & ~applied))
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
    if not inputs["cuts_identical_to_lut_job"]:
        log(f"energy window {inputs['energy_range_kev']} keV differs from the LUT job's "
            f"{inputs['lut_job_energy_range_kev']} keV: a study at other cuts, not the FR-13 result")
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
    walk_info = _walk(spec, lut)
    accumulator = Accumulator(spec, delays, sources)
    plain = Accumulator(spec, delays, sources) if walk_info else None
    for index, acquisition in enumerate(spec.acquisitions):
        t0 = time.perf_counter()
        for features in compiled_batches(acquisition.ldat, lookup, file_index=index, max_records=spec.max_records,
                                         seed=spec.seed):
            geometries = [apply_geometry(features, h) for h in hypotheses]
            if walk_info is None:
                accumulator.add(index, features, geometries)
                continue
            walk_dt, keep = _walk_dt(walk_info, features, index) if len(features.ints) else (None, None)
            accumulator.add(index, features, geometries, walk_dt=walk_dt, keep=keep)
            plain.add(index, features, geometries, keep=keep)
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
                 "energy_window_kev": inputs["energy_range_kev"],
                 "identical_to_lut_job": inputs["cuts_identical_to_lut_job"],
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
        **({} if walk_info is None else {
            "walk": {"walk_lut": str(walk_info["path"]), "walk_lut_sha256": walk_info["lut"].sha256,
                     "fit_kind": walk_info["lut"].manifest.get("fit_kind"),
                     "fitted_with_skew_lut_sha256": walk_info["lut"].manifest.get("skew_lut_sha256"),
                     "listmode_files": walk_info["context"].files, "clamped": walk_info["clamped"],
                     **walk_info["calibration"],
                     "note": "results are with walk; results_without_walk hold the same events without it"},
            "results_without_walk": summarize(plain)}),
        "interpretation": "system CTR at these cuts (includes residual offset errors and endpoint blur); "
                          f"not an intrinsic detector CTR; {NOT_CTR}",
        "seconds": round(time.perf_counter() - started, 1),
    }
    if spec.extra and spec.extra.get("previous_reports"):
        report["round_trip"] = _round_trip(spec, report)
    spec.output.parent.mkdir(parents=True, exist_ok=True)
    spec.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    _write_histograms(spec, accumulator)
    _plot(spec, accumulator, results, inputs["energy_range_kev"])
    return report


def _walk(spec, lut):
    """Spec 009 FR-5: the walk LUT, its pairing with the skew LUT and the listmode files, or None."""
    extra = spec.extra or {}
    named = [extra.get(k) is not None for k in ("walk_lut", "walk_lut_sha256", "listmode")]
    skew_walk = (lut.manifest.get("provenance") or {}).get("walk_lut_sha256")
    if not any(named):
        if skew_walk:
            raise CtrError(f"the skew LUT was solved with walk LUT {skew_walk[:12]}...: name a walk LUT paired with it")
        return None
    if not all(named):
        raise CtrError("walk_lut, walk_lut_sha256 and listmode {calibration, cog_limits, doi_limits} go together")
    from src.cornell.skew import walk
    base = spec.path.resolve().parent

    def local(value):
        value = Path(value)
        return value if value.is_absolute() else base / value

    path = local(extra["walk_lut"])
    files = extra["listmode"]
    if not isinstance(files, dict) or set(files) != {"calibration", "cog_limits", "doi_limits"}:
        raise CtrError("listmode: {calibration, cog_limits, doi_limits} paths required")
    try:
        table = walk.WalkLut.read(path)
        context = walk.WalkContext.from_files(spec.config, *(local(files[k]) for k in ("calibration", "cog_limits",
                                                                                        "doi_limits")))
        walk.check_pairing(lut, _sha256(spec.lut), table)
        calibration = walk.check_files(table, context.files)
    except walk.WalkError as error:
        raise CtrError(str(error)) from None
    if table.sha256 != str(extra["walk_lut_sha256"]).lower():
        raise CtrError(f"walk LUT table SHA-256 {table.sha256} is not the frozen {extra['walk_lut_sha256']}")
    clamped = [dict.fromkeys(walk.INPUTS, 0) | {"walk input missing": 0} for _ in spec.acquisitions]
    return {"path": path, "lut": table, "context": context, "clamped": clamped, "module": walk,
            "calibration": calibration}


def _walk_dt(info, features, index):
    """W(side 1) - W(side 2) and the events with every walk input on both sides; clamps counted."""
    walk = info["module"]
    values, status1, status2 = walk.feature_inputs(info["context"], features)
    keep = (status1 == 0) & (status2 == 0)
    side1 = {n: values[f"{n}_1"] for n in walk.INPUTS}
    side2 = {n: values[f"{n}_2"] for n in walk.INPUTS}
    difference, out1, out2 = walk.walk_difference(info["lut"], side1, side2)
    counts = info["clamped"][index]
    counts["walk input missing"] += int(np.count_nonzero(~keep))
    for name in walk.INPUTS:
        counts[name] += int(np.count_nonzero(keep & out1[name])) + int(np.count_nonzero(keep & out2[name]))
    return np.where(keep, difference, 0.0), keep


def _round_trip(spec, report):
    """Pooled and per-position numbers of the listed earlier CTR reports and of this one (spec 009 FR-5)."""
    def numbers(results):
        fit = results["fit"]
        unc = results.get("uncertainty") or {}
        return {"model_free_fwhm_ps": fit.get("model_free_fwhm_ps"), "fwhm_ps": fit.get("fwhm_ps"),
                "model_free_fwtm_ps": fit.get("model_free_fwtm_ps"), "events": fit.get("events"),
                "model_free_fwhm_total_ps": (unc.get("model_free_fwhm_ps") or {}).get("total"),
                "fwhm_total_ps": (unc.get("fwhm_ps") or {}).get("total")}

    def row(label, data, sha):
        walk_info = data.get("walk") or {}
        return {"report": label, "report_sha256": sha, "skew_lut_sha256": data["frozen_inputs"]["lut_sha256"],
                "walk_lut_sha256": walk_info.get("walk_lut_sha256"),
                "energy_window_kev": data["cuts"]["energy_window_kev"],
                "results": {scope: numbers(entry) for scope, entry in data["results"].items()
                            if entry.get("fit", {}).get("status") == "fitted"}}

    base = spec.path.resolve().parent
    rows = []
    for name in spec.extra["previous_reports"]:
        path = Path(name) if Path(name).is_absolute() else base / name
        rows.append(row(str(path), json.loads(path.read_text(encoding="utf-8")), _sha256(path)))
    rows.append(row("this report", report, None))
    return rows


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


def _plot(spec, accumulator, results, energy_window_kev):
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
            background = fit["background_events"] / len(centers)
            for fraction, label, colour in ((0.5, "model-free FWHM", "tab:red"), (0.1, "model-free FWTM", "tab:green")):
                crossing = level_crossings(pooled, centers, background, fraction)
                if crossing is not None:
                    left, right, level = crossing
                    ax.plot([left, right], [level, level], "|-", color=colour, lw=1.4, ms=10,
                            label=f"{label} {right - left:.0f} ps")
        ax.set_yscale(scale if pooled.any() else "linear")
        if scale == "linear":
            ax.set_xlim(-1500, 1500)
        ax.set_xlabel("corrected t1 - t2 minus geometric TOF (ps)")
        ax.set_ylabel(f"events / {spec.bin_ps:g} ps")
        ax.legend(fontsize=8)
    walk_name = (spec.extra or {}).get("walk_lut")
    figure.suptitle(f"Point-source CTR, LUT {spec.lut.name}" + (f" + walk {Path(walk_name).name}" if walk_name else "")
                    + f", energy {energy_window_kev[0]:g}-{energy_window_kev[1]:g} keV, LOR <= {spec.lor_distance_mm:g} mm")
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
    free = (pooled["uncertainty"]["model_free_fwhm_ps"] or {}).get("total")
    print(f"pooled CTR: model-free FWHM "
          + (f"{fit['model_free_fwhm_ps']:.1f}" if fit["model_free_fwhm_ps"] is not None else "n/a")
          + (f" +- {free:.1f}" if free is not None else "") + f" ps, Gaussian FWHM {fit['fwhm_ps']:.1f}"
          + (f" +- {total:.1f}" if total is not None else "") + f" ps; {fit['events']:.0f} events, energy window "
          f"{report['cuts']['energy_window_kev']} keV"
          + ("" if report["cuts"]["identical_to_lut_job"] else " (differs from the LUT job)"))
    if "results_without_walk" in report:
        without = report["results_without_walk"]["pooled"]["fit"]
        print(f"with walk LUT {report['walk']['walk_lut']}; same events without walk: model-free FWHM "
              + (f"{without['model_free_fwhm_ps']:.1f}" if without.get("model_free_fwhm_ps") is not None else "n/a")
              + f" ps, Gaussian FWHM {without['fwhm_ps']:.1f} ps")
    return 0
