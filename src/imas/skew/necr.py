"""IMAS NEMA NU 2-2018 CTR on the NECR phantom (spec 013).

``python -m src.imas.skew necr FILE``: per frame of a NECR acquisition, the NEMA
line-source timing resolution (``src.timing.nema``) for the uncorrected data,
the legacy skew table and the frozen spec 011/012 tables, on the same events.
The headline width and the gates use spec 011's model-free FWHM (owner, T9);
the NEMA widths are reported beside it. The line is fitted from LOR convergence (``src.timing.line``) or declared.
Inputs are frozen as in the point-source CTR file (``src.imas.skew.ctr``).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from src.imas.skew import ctr, select as sel
from src.imas.skew import walk as iw
from src.imas.skew.geometry import ImasGeometry
from src.imas.skew.setup import ImasSetup
from src.timing import ctr as core, line as tl, nema
from src.timing.cache import _sha256
from src.timing.ctr import CtrError

SCHEMA = "imas-necr-ctr/1"
REPORT_SCHEMA = "imas-necr-ctr-report/2"
FITTED = "lor_convergence"
_REQUIRED = {"schema", "job_work", "encal", "luts", "legacy", "output", "frames", "lines", "headline_line"}
_OPTIONAL = {"walks": {}, "gates": None, "legacy_results": None, "bootstrap": 200, "seed": 0, "max_records": None,
             "localization_events": 1_000_000, "model_free_window_ps": 3000.0, "model_free_bin_ps": 10.0,
             "energy_check_records": 2_000_000}
_FRAME_KEYS = {"name", "frame", "ldat", "activity_kbq_ml", "activity_source"}
_LINE_OPTIONS = {"slab_mm", "near_mm", "seed_near_mm", "floor_mm", "windows_mm"}
_GATE_KEYS = {"frame", "transfer", "acceptance"}
_LEGACY_KEYS = {"fwhm_raw_ps", "fwtm_raw_ps", "fwhm_bg_ps", "fwtm_bg_ps"}
STAGES = ("records", "passed pre-geometry", "invalid endpoint", "no applied delay", "walk input missing",
          "same events")
LINE_HALF_LENGTH_MM = 356.109009      # legacy P1/P2 z; fitted lines are written with the same end points
FRAME_LINE_FLAG_MM = 1.0              # plan: a frame line farther than this from the pooled line is flagged
GATE_KIND, GATE_KEY = "model_free", "model_free_fwhm_ps"     # spec FR-6 (T9): the headline width


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise CtrError(f"{label}: a positive number required")
    return float(value)


def _line(name, entry):
    if not isinstance(entry, dict):
        raise CtrError(f"lines.{name}: an object required")
    if entry.get("method") == FITTED:
        if not set(entry) - {"method"} <= _LINE_OPTIONS:
            raise CtrError(f"lines.{name}: options {sorted(_LINE_OPTIONS)}")
        return SimpleNamespace(name=name, method=FITTED, options={k: v for k, v in entry.items() if k != "method"})
    if set(entry) != {"p1", "p2", "source"}:
        raise CtrError(f"lines.{name}: {{method: {FITTED!r}, ...options}} or {{p1, p2, source}}")
    p1, p2 = ctr._vector(entry["p1"], f"lines.{name}.p1"), ctr._vector(entry["p2"], f"lines.{name}.p2")
    if math.dist(p1, p2) < 1.0:
        raise CtrError(f"lines.{name}: P1 and P2 must be at least 1 mm apart")
    return SimpleNamespace(name=name, method="declared", p1=p1, p2=p2, source=str(entry["source"]))


def _pair(value, label, names):
    if not isinstance(value, list) or len(value) != 2 or value[0] == value[1] or not set(value) <= names:
        raise CtrError(f"{label}: [reference, candidate] of table names {sorted(names)}")
    return tuple(value)


def load_necr(path):
    """Validated NECR CTR file (paths relative to it); the options as ``SimpleNamespace`` fields.

    The frozen keys (``job_work``, ``encal``, ``luts``, ``legacy``, ``walks``) keep the point-source CTR
    file's names and rules, so ``ctr.frozen_inputs`` checks them unchanged (with the splits as its
    ``acquisitions`` and the gate pairs as its ``walk_acceptance``).
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise CtrError(f"file schema must be {SCHEMA!r}")
    if not _REQUIRED <= set(raw) <= _REQUIRED | set(_OPTIONAL):
        raise CtrError(f"file keys: required {sorted(_REQUIRED)}, optional {sorted(_OPTIONAL)}")
    base = path.resolve().parent

    def local(value):
        value = Path(value)
        return value if value.is_absolute() else base / value

    if not isinstance(raw["luts"], dict) or not raw["luts"]:
        raise CtrError("luts: at least one {name: {path, sha256}}")
    if {ctr.UNCORRECTED, ctr.LEGACY} & set(raw["luts"]):
        raise CtrError(f"LUT names {ctr.UNCORRECTED!r} and {ctr.LEGACY!r} are reserved")
    walks = {}
    if not isinstance(raw.get("walks", {}), dict):
        raise CtrError("walks: {name: {skew, path, sha256}}")
    for name, entry in raw.get("walks", {}).items():
        if name in (ctr.UNCORRECTED, ctr.LEGACY) or name in raw["luts"]:
            raise CtrError(f"walk table name {name!r} is reserved or a LUT name")
        if not isinstance(entry, dict) or entry.get("skew") not in raw["luts"]:
            raise CtrError(f"walks.{name}: 'skew' must name one of the luts")
        walk_path, sha = ctr._frozen({k: v for k, v in entry.items() if k != "skew"}, f"walks.{name}", base)
        walks[name] = SimpleNamespace(skew=entry["skew"], path=walk_path, sha256=sha)
    frames = []
    for item in raw["frames"] if isinstance(raw["frames"], list) else []:
        if not isinstance(item, dict) or set(item) != _FRAME_KEYS:
            raise CtrError(f"frames need exactly {sorted(_FRAME_KEYS)}: {item!r}")
        if not isinstance(item["ldat"], list) or not item["ldat"]:
            raise CtrError(f"frame {item['name']}: ldat must list at least one split")
        if isinstance(item["frame"], bool) or not isinstance(item["frame"], int):
            raise CtrError(f"frame {item['name']}: frame must be an integer")
        frames.append(SimpleNamespace(
            name=str(item["name"]), frame=item["frame"], ldat=tuple(local(p) for p in item["ldat"]),
            activity_kbq_ml=_positive(item["activity_kbq_ml"], f"frame {item['name']} activity_kbq_ml"),
            activity_source=str(item["activity_source"])))
    if not frames:
        raise CtrError("at least one frame required")
    if len({f.name for f in frames}) != len(frames):
        raise CtrError("frame names must be unique")
    if not isinstance(raw["lines"], dict) or not raw["lines"]:
        raise CtrError("lines: at least one {name: line}")
    lines = {name: _line(name, entry) for name, entry in raw["lines"].items()}
    if raw["headline_line"] not in lines:
        raise CtrError(f"headline_line must name one of the lines {sorted(lines)}")
    options = {key: raw.get(key, default) for key, default in _OPTIONAL.items()}
    names = {ctr.UNCORRECTED, ctr.LEGACY} | set(raw["luts"]) | set(walks)
    gates = None
    if options["gates"] is not None:
        entry = options["gates"]
        if not isinstance(entry, dict) or "frame" not in entry or not set(entry) <= _GATE_KEYS:
            raise CtrError("gates: {frame, transfer?, acceptance?}")
        if entry["frame"] not in {f.name for f in frames}:
            raise CtrError("gates.frame must name one of the frames")
        gates = SimpleNamespace(frame=entry["frame"], **{key: _pair(entry[key], f"gates.{key}", names)
                                                         if key in entry else None
                                                         for key in ("transfer", "acceptance")})
    legacy_results = None
    if options["legacy_results"] is not None:
        entry = options["legacy_results"]
        if not isinstance(entry, dict) or set(entry) != {"source", "frames"} or not isinstance(entry["frames"], dict):
            raise CtrError("legacy_results: {source, frames: {frame name: {fwhm/fwtm raw/bg ps}}}")
        for name, values in entry["frames"].items():
            if name not in {f.name for f in frames} or not isinstance(values, dict) or set(values) != _LEGACY_KEYS:
                raise CtrError(f"legacy_results.frames.{name}: a frame name with {sorted(_LEGACY_KEYS)}")
            for key, value in values.items():
                _positive(value, f"legacy_results.frames.{name}.{key}")
        legacy_results = entry
    bootstrap = options["bootstrap"]
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0 or bootstrap == 1:
        raise CtrError("bootstrap: 0 (none) or at least 2 replicates")
    window, width = (_positive(options[k], k) for k in ("model_free_window_ps", "model_free_bin_ps"))
    bins = 2 * window / width
    if abs(bins - round(bins)) > 1e-9 or round(bins) < 10:
        raise CtrError("2 * model_free_window_ps must be a whole number (>= 10) of model_free_bin_ps")
    for key in ("localization_events", "energy_check_records"):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 1000:
            raise CtrError(f"{key}: an integer >= 1000 required")
    return SimpleNamespace(
        path=path, sha256=_sha256(path), job_work=local(raw["job_work"]), encal=local(raw["encal"]),
        luts={name: ctr._frozen(entry, f"luts.{name}", base) for name, entry in raw["luts"].items()},
        legacy=ctr._frozen(raw["legacy"], "legacy", base), walks=walks, output=local(raw["output"]),
        frames=tuple(frames), lines=lines, headline_line=raw["headline_line"], gates=gates,
        legacy_results=legacy_results, bootstrap=bootstrap, seed=int(options["seed"]),
        max_records=options["max_records"], localization_events=options["localization_events"],
        energy_check_records=options["energy_check_records"], model_free_window_ps=window,
        model_free_bin_ps=width, table_names=names,
        # read by ctr.frozen_inputs: every split must exist and not be a job input; gate tables must be evaluated
        acquisitions=tuple(SimpleNamespace(name=f"{f.name}/{p.name}", ldat=p) for f in frames for p in f.ldat),
        walk_acceptance=[pair for pair in (gates.transfer, gates.acceptance) if pair] if gates else [])


def prepare_output(spec):
    """Create the output directory; an existing one is refused (outputs are never overwritten)."""
    if spec.output.exists():
        raise CtrError(f"{spec.output}: output directory already exists; use a new name")
    spec.output.mkdir(parents=True)
    return spec.output


# --- localization (FR-3) --------------------------------------------------------------------------------------

def _bore(geometry):
    return geometry.ring_r - 10.0, max(abs(z) for z in geometry.ring_z) + geometry.local_origin[1]


def localization_sample(frame, spec, lookup, selection, geometry):
    """Endpoints of the first ``localization_events`` LORs with valid endpoints over the frame's splits."""
    p, q, events = [], [], 0
    for path in frame.ldat:
        for features in sel.feature_batches(path, lookup, selection, max_records=spec.max_records):
            if not len(features.ints):
                continue
            codes, rows = geometry.endpoints(features)
            rows = rows[codes != 1][:spec.localization_events - events]
            p.append(rows[:, :3])
            q.append(rows[:, 3:])
            events += len(rows)
            if events >= spec.localization_events:
                return np.concatenate(p), np.concatenate(q)
    if not events:
        raise CtrError(f"{frame.name}: no LOR with valid endpoints to locate the line")
    return np.concatenate(p), np.concatenate(q)


def _ends(point, direction):
    """Two points on the line, at z = -/+ ``LINE_HALF_LENGTH_MM``."""
    point, direction = np.asarray(point, np.float64), np.asarray(direction, np.float64)
    return tuple((point + (z - point[2]) / direction[2] * direction).tolist()
                 for z in (-LINE_HALF_LENGTH_MM, LINE_HALF_LENGTH_MM))


def xy_at(p1, p2, z):
    """(x, y) of the line P1-P2 at each ``z``."""
    p1, p2, z = np.asarray(p1, np.float64), np.asarray(p2, np.float64), np.asarray(z, np.float64)
    return p1[:2] + ((z - p1[2]) / (p2[2] - p1[2]))[:, None] * (p2 - p1)[:2]


def locate_lines(spec, samples, geometry, *, log=print):
    """Each line's end points P1/P2 in the map frame; for a fitted line also the pooled fit and the per-frame
    fits with their distance from the pooled line (flagged beyond ``FRAME_LINE_FLAG_MM``)."""
    radius, half = _bore(geometry)
    check = np.array(tl.CHECK_Z_MM)
    out = {}
    for name, line in spec.lines.items():
        if line.method != FITTED:
            out[name] = {"method": "declared", "p1_mm": list(line.p1), "p2_mm": list(line.p2), "source": line.source}
            continue
        options = {"radius_mm": radius, "half_length_mm": half, "seed": spec.seed,
                   **{k: tuple(v) if isinstance(v, list) else v for k, v in line.options.items()}}
        p = np.concatenate([samples[f.name][0] for f in spec.frames])
        q = np.concatenate([samples[f.name][1] for f in spec.frames])
        pooled = tl.locate_line(p, q, **options)
        p1, p2 = _ends(pooled["point_mm"], pooled["direction"])
        frames = {}
        for frame in spec.frames:
            try:
                fit = tl.locate_line(*samples[frame.name], **options)
            except CtrError as error:
                frames[frame.name] = {"status": str(error)}
                continue
            distance = np.linalg.norm(xy_at(*_ends(fit["point_mm"], fit["direction"]), check) - xy_at(p1, p2, check),
                                      axis=1)
            frames[frame.name] = {"status": "fitted", "point_mm": fit["point_mm"], "direction": fit["direction"],
                                  "median_miss_mm": fit["median_miss_mm"], "lors_used": fit["lors_used"],
                                  "distance_to_pooled_mm": distance.tolist(),
                                  "flagged": bool(distance.max() > FRAME_LINE_FLAG_MM)}
        log(f"line {name}: {np.round(pooled['point_mm'], 2).tolist()} mm at z = 0, slopes "
            f"{np.round(pooled['slopes'], 5).tolist()}, median LOR miss {pooled['median_miss_mm']:.2f} mm")
        out[name] = {"method": FITTED, "p1_mm": list(p1), "p2_mm": list(p2), "lors": int(len(p)),
                     "options": {k: list(v) if isinstance(v, tuple) else v for k, v in options.items()},
                     "fit": pooled, "per_frame": frames,
                     "axis_distance_mm": np.linalg.norm(xy_at(p1, p2, check), axis=1).tolist()}
    names = list(out)
    distances = {f"{a} - {b}": np.linalg.norm(xy_at(out[a]["p1_mm"], out[a]["p2_mm"], check)
                                              - xy_at(out[b]["p1_mm"], out[b]["p2_mm"], check), axis=1).tolist()
                 for i, a in enumerate(names) for b in names[i + 1:]}
    return out, {"z_mm": check.tolist(), "distance_mm": distances}


# --- accumulation and widths (FR-2, FR-4, FR-5) ---------------------------------------------------------------

class Accumulator:
    """NEMA r x Δt histograms and model-free Δt histograms (|r| <= 20 mm) per line x table x frame, on the same
    events.

    Bootstrap replicates (one Poisson weight per event, shared by lines and tables) are held for the current
    frame and summed over frames; ``finish_frame`` reduces a frame to its results.
    """

    def __init__(self, spec, tables, lines):
        self.spec, self.names, self.tables = spec, list(tables), list(tables.values())
        self.lines = list(lines)
        self.ends = [(np.array(lines[n]["p1_mm"]), np.array(lines[n]["p2_mm"])) for n in self.lines]
        n_l, n_t, n_f, n_b = len(self.lines), len(self.tables), len(spec.frames), spec.bootstrap
        bins = int(round(2 * spec.model_free_window_ps / spec.model_free_bin_ps))
        self.mf_edges = np.linspace(-spec.model_free_window_ps, spec.model_free_window_ps, bins + 1)
        self.hist = nema.NemaHistograms((n_l, n_t, n_f))
        self.mf = np.zeros((n_l, n_t, n_f, bins))
        self.boot = nema.NemaHistograms((n_l, n_t, n_b), np.float32)
        self.mf_boot = np.zeros((n_l, n_t, n_b, bins), np.float32)
        self.pooled_boot = nema.NemaHistograms((n_l, n_t, n_b), np.float32)
        self.pooled_mf_boot = np.zeros_like(self.mf_boot)
        self.stages = [dict.fromkeys(STAGES, 0) | {
            "parallel to line": dict.fromkeys(self.lines, 0),
            "kept (NEMA)": {line: dict.fromkeys(self.names, 0) for line in self.lines},
            "model-free band": {line: dict.fromkeys(self.names, 0) for line in self.lines}} for _ in spec.frames]
        self.pre_geometry = [{} for _ in spec.frames]
        self.clamped = {}
        self.results = {}

    def corrected(self, features):
        """Corrected ``dt`` per table, the mask of events with both delays in every table, and the mask of
        events missing a walk input."""
        ch1, ch2 = features.column("time_channel_1"), features.column("time_channel_2")
        dt = features.column("dt_ps").astype(np.float64)
        applied, walk_missing, out = np.ones(len(dt), bool), np.zeros(len(dt), bool), []
        for name, table in zip(self.names, self.tables):
            if isinstance(table, ctr.WalkTable):
                values, status, counts = iw.correct_features(features, table.skew, table.walk, policy="drop")
                clamped = self.clamped.setdefault(name, dict.fromkeys(iw.INPUTS, 0))
                for key in iw.INPUTS:
                    clamped[key] += counts[f"clamped {key}"]
                applied &= status != 1
                walk_missing |= status == 2
                out.append(values)
                continue
            b1, f1 = table.lookup(ch1)
            b2, f2 = table.lookup(ch2)
            applied &= f1 & f2
            out.append(dt - b1 + b2)       # t_1 - t_2 exact in integers, then - b_1 + b_2
        return out, applied, walk_missing

    def add(self, index, features, endpoints, split=0):
        """``endpoints``: (codes, rows) of the nominal geometry; ``split`` seeds the bootstrap with the batch."""
        spec, stages = self.spec, self.stages[index]
        codes, found = np.unique(features.codes, return_counts=True)
        for code, count in zip(codes.tolist(), found.tolist()):
            label = sel.code_label(code)
            self.pre_geometry[index][label] = self.pre_geometry[index].get(label, 0) + count
        stages["records"] += len(features.codes)
        stages["passed pre-geometry"] += len(features.ints)
        if not len(features.ints):
            return
        corrected, applied, walk_missing = self.corrected(features)
        codes, rows = endpoints
        valid = codes != 1
        same = valid & applied & ~walk_missing
        stages["invalid endpoint"] += int(np.count_nonzero(~valid))
        stages["no applied delay"] += int(np.count_nonzero(valid & ~applied))
        stages["walk input missing"] += int(np.count_nonzero(valid & applied & walk_missing))
        stages["same events"] += int(np.count_nonzero(same))
        p, q = rows[same, :3], rows[same, 3:]
        size, bins = nema.R_BINS * nema.T_BINS, self.mf.shape[-1]
        binned = []
        for l, (line, (p1, p2)) in enumerate(zip(self.lines, self.ends)):
            r, _, tof = nema.line_frame(p, q, p1, p2)
            stages["parallel to line"][line] += int(np.count_nonzero(np.isnan(r)))
            with np.errstate(invalid="ignore"):
                band = np.abs(r) <= nema.R_SIGNAL_MM
            for t, name in enumerate(self.names):
                error = corrected[t][same] - tof
                ok, flat = nema.bin_index(error, r)
                self.hist.band[l, t, index] += np.bincount(flat, minlength=size).reshape(nema.R_BINS, nema.T_BINS)
                with np.errstate(invalid="ignore"):
                    b = np.floor((error + spec.model_free_window_ps) / spec.model_free_bin_ps)
                inside = band & (b >= 0) & (b < bins)
                b = b[inside].astype(np.int64)
                self.mf[l, t, index] += np.bincount(b, minlength=bins)
                stages["kept (NEMA)"][line][name] += int(np.count_nonzero(ok))
                stages["model-free band"][line][name] += int(np.count_nonzero(inside))
                binned.append((l, t, ok, flat, inside, b))
        if spec.bootstrap:
            rng = np.random.default_rng([spec.seed, index, split, features.first_record])
            for k in range(spec.bootstrap):
                weights = rng.poisson(1.0, len(p))
                for l, t, ok, flat, inside, b in binned:
                    self.boot.band[l, t, k] += np.bincount(flat, weights=weights[ok], minlength=size).reshape(
                        nema.R_BINS, nema.T_BINS)
                    self.mf_boot[l, t, k] += np.bincount(b, weights=weights[inside], minlength=bins)

    def finish_frame(self, index):
        frame = self.spec.frames[index].name
        self.results[frame] = self.summarize(self.hist.band[:, :, index], self.mf[:, :, index], self.boot.band,
                                             self.mf_boot)
        self.pooled_boot.band += self.boot.band
        self.pooled_mf_boot += self.mf_boot
        self.boot.band[:] = 0
        self.mf_boot[:] = 0

    def finish(self):
        """Results per frame and pooled over frames (counts summed)."""
        self.results["pooled"] = self.summarize(self.hist.band.sum(axis=2), self.mf.sum(axis=2),
                                                self.pooled_boot.band, self.pooled_mf_boot)
        return self.results

    def summarize(self, band, mf, boot, mf_boot):
        """Per line and table: NEMA widths (raw, background-subtracted) and the model-free fit with bootstrap
        standard deviations; ``_draws`` keeps each replicate's values for the paired differences."""
        out = {}
        for l, line in enumerate(self.lines):
            out[line] = {}
            for t, name in enumerate(self.names):
                fit = core.fit_histogram(mf[l, t], self.mf_edges, crossings=ctr.MODEL_FREE)
                draws = {}
                for k in range(boot.shape[2]):
                    replicate = nema.summarize(boot[l, t, k].sum(axis=0), boot[l, t, k])
                    for kind in ("raw", "background_subtracted"):
                        for key in ("fwhm_ps", "fwtm_ps"):
                            draws.setdefault((kind, key), []).append(replicate[kind][key])
                    if fit["status"] == "fitted":
                        value = core.fit_histogram(mf_boot[l, t, k], self.mf_edges, start=fit["_start"],
                                                   crossings=ctr.MODEL_FREE)
                        draws.setdefault(("model_free", "model_free_fwhm_ps"), []).append(
                            value["model_free_fwhm_ps"] if value["status"] == "fitted" else None)
                out[line][name] = {"nema": nema.summarize(band[l, t].sum(axis=0), band[l, t]),
                                   "model_free": core._public(fit),
                                   "uncertainty": {f"{kind} {key}": _sd(values)
                                                   for (kind, key), values in draws.items()},
                                   "_draws": draws}
        return out


def _sd(values):
    good = [v for v in values if v is not None]
    return {"sd_ps": float(np.std(good, ddof=1)) if len(good) >= 2 else None, "replicates": len(values),
            "fitted": len(good)}


# --- energy check (FR-7) ----------------------------------------------------------------------------------------

KEV_BIN, KEV_RANGE = 2.0, (0.0, 1200.0)
ANCHOR_KEV = (350.0, 800.0)          # the photopeak is searched here, away from the low-energy scatter


def photopeak(histogram, setup):
    """The 511 keV photopeak (keV) of one frame: each TIME channel's ENERGY-sum histogram (``encal.accumulate``,
    a.u.) moved to keV with its calibration ``511 / mu``, then ``fit_peak_background`` (Gaussian + linear
    background) around the highest smoothed ``KEV_BIN`` bin in ``ANCHOR_KEV``. Channels without a valid ``mu``
    are excluded and counted."""
    from src.imas.skew import encal
    from src.ldat_inspector.engine import fit_peak_background
    lookup = setup.lookup()
    mu = lookup.mu[np.asarray(setup.time_channels)]
    rows, cols = np.nonzero(histogram)
    valid = np.isfinite(mu[rows]) & (mu[rows] > 0)
    counts = histogram[rows, cols].astype(np.float64)
    centres = encal.RANGE_AU[0] + encal.FINE_BIN_AU * (cols + 0.5)
    kev = 511.0 / mu[rows[valid]] * centres[valid]
    bins = int(round((KEV_RANGE[1] - KEV_RANGE[0]) / KEV_BIN))
    hist, edges = np.histogram(kev, bins, range=KEV_RANGE, weights=counts[valid])
    out = {"entries": float(counts[valid].sum()), "excluded_channels": int(len(np.unique(rows[~valid]))),
           "excluded_entries": float(counts[~valid].sum()), "kev_bin": KEV_BIN}
    centre = (edges[:-1] + edges[1:]) / 2
    window = (centre >= ANCHOR_KEV[0]) & (centre <= ANCHOR_KEV[1])
    if hist[window].sum() < encal.MIN_ENTRIES:
        return out | {"status": "few entries"}
    smooth = np.convolve(hist, np.ones(5) / 5, mode="same")
    anchor = float(centre[window][np.argmax(smooth[window])])
    # fitted on the per-channel keV values (no re-binning), in KEV_BIN fit bins: fit_peak_background's start
    # needs bins several times narrower than the peak sigma
    values = np.repeat(kev, counts[valid].astype(np.int64))
    fit = fit_peak_background(values, background_model="linear", interval=(0.55 * anchor, 1.5 * anchor),
                              search=(0.8 * anchor, 1.2 * anchor), sigma0=0.07 * anchor,
                              mu_halfwidth=0.15 * anchor, bins=int(round(0.95 * anchor / KEV_BIN)))
    if fit["status"] != "FIT":
        return out | {"status": "no peak", "anchor_kev": anchor}
    return out | {"status": "fitted", "anchor_kev": anchor, "peak_kev": float(fit["mu"]),
                  "sigma_kev": float(fit["sigma"]), "fwhm_percent": float(235.48 * fit["sigma"] / fit["mu"])}


def energy_check(spec, setup, inputs, *, log=print):
    """FR-7: the calibration used and, per frame, the photopeak of its first ``energy_check_records`` records
    per split (rate check)."""
    from src.imas.skew import encal
    frames = {}
    for frame in spec.frames:
        histogram = None
        for path in frame.ldat:
            h, _, _, _ = encal.accumulate(path, setup, max_records=spec.energy_check_records)
            histogram = h.astype(np.uint64) if histogram is None else histogram + h
        frames[frame.name] = photopeak(histogram, setup) | {"activity_kbq_ml": frame.activity_kbq_ml}
        peak = frames[frame.name].get("peak_kev")
        log(f"{frame.name}: photopeak " + (f"{peak:.1f} keV" if peak else frames[frame.name]["status"]))
    return {"calibration": inputs["calibration"], "records_per_split": spec.energy_check_records,
            "frames": frames}


# --- comparisons and gates (FR-5, FR-6) -----------------------------------------------------------------------

DRAW_KEYS = (("background_subtracted", "fwhm_ps"), ("raw", "fwhm_ps"), ("model_free", "model_free_fwhm_ps"))


def _value(entry, kind, key):
    return entry["model_free"].get(key) if kind == "model_free" else entry["nema"][kind][key]


def paired(entry, reference, kind, key):
    """Reference minus table on the same events: nominal and bootstrap SD of the per-replicate differences."""
    nominal = (None if None in (_value(reference, kind, key), _value(entry, kind, key))
               else _value(reference, kind, key) - _value(entry, kind, key))
    draws = [a - b for a, b in zip(reference["_draws"].get((kind, key), ()), entry["_draws"].get((kind, key), ()))
             if a is not None and b is not None]
    return {"improvement_ps": nominal, "sd_ps": float(np.std(draws, ddof=1)) if len(draws) >= 2 else None,
            "replicates": len(draws)}


def paired_differences(results, reference=ctr.LEGACY):
    """Per scope (frame or pooled), line and table: ``reference`` minus the table for each ``DRAW_KEYS``."""
    return {scope: {line: {name: {f"{kind} {key}": paired(entry, tables[reference], kind, key)
                                  for kind, key in DRAW_KEYS}
                           for name, entry in tables.items() if name not in (ctr.UNCORRECTED, reference)}
                    for line, tables in lines.items()}
            for scope, lines in results.items()}


def judge(candidate, reference, kind=GATE_KIND, key=GATE_KEY):
    """Spec 011 FR-10 rule: the candidate is better when reference - candidate exceeds the quadrature sum of
    their bootstrap SDs; the paired same-event SD is reported beside it."""
    value, old = _value(candidate, kind, key), _value(reference, kind, key)
    u_new = candidate["uncertainty"].get(f"{kind} {key}", {}).get("sd_ps")
    u_old = reference["uncertainty"].get(f"{kind} {key}", {}).get("sd_ps")
    if None in (value, old, u_new, u_old):
        return {"status": "unavailable", "judged_on": f"{kind} {key}"}
    combined = math.hypot(u_new, u_old)
    return {"judged_on": f"{kind} {key}", "table_ps": value, "reference_ps": old, "improvement_ps": old - value,
            "combined_uncertainty_ps": combined, "accepted": bool(old - value > combined),
            "paired": paired(candidate, reference, kind, key)}


def evaluate_gates(spec, results):
    """FR-6 on the headline line: the transfer check and the acceptance on the gate frame (per frame and pooled
    reported), and the headline (model-free) value per table with the NEMA widths beside it (withheld for the
    frozen tables when the transfer check fails)."""
    line = spec.headline_line
    frame = spec.gates.frame if spec.gates else None
    out = {"line": line, "frame": frame, "judged_on": f"{GATE_KIND} {GATE_KEY}"}
    transfer_ok = True
    for gate in ("transfer", "acceptance"):
        pair = getattr(spec.gates, gate) if spec.gates else None
        if pair is None:
            continue
        reference, candidate = pair
        verdicts = {scope: judge(lines[line][candidate], lines[line][reference]) for scope, lines in results.items()}
        out[gate] = {"reference": reference, "candidate": candidate, "verdict": verdicts[frame],
                     "accepted": bool(verdicts[frame].get("accepted")), "per_scope": verdicts}
        if gate == "transfer":
            transfer_ok = out[gate]["accepted"]
    out["transfer_passed"] = transfer_ok if spec.gates and spec.gates.transfer else None
    scope = frame or "pooled"
    values = {}
    for name, entry in results[scope][line].items():
        if not transfer_ok and name not in (ctr.UNCORRECTED, ctr.LEGACY):
            values[name] = {"status": "withheld: the transfer check failed; the frozen tables do not transfer to "
                                      "this acquisition's hardware state"}
            continue
        fit, background = entry["model_free"], entry["nema"]["background_subtracted"]
        status = fit["status"] if fit["status"] != "fitted" or fit.get(GATE_KEY) else "no crossing"
        values[name] = {"status": status, "fwhm_ps": fit.get(GATE_KEY), "fwtm_ps": fit.get("model_free_fwtm_ps"),
                        "sd_ps": entry["uncertainty"].get(f"{GATE_KIND} {GATE_KEY}", {}).get("sd_ps"),
                        "nema_background_subtracted": {
                            "status": background["status"], "fwhm_ps": background["fwhm_ps"],
                            "sd_ps": entry["uncertainty"]["background_subtracted fwhm_ps"]["sd_ps"]},
                        "nema_raw_fwhm_ps": entry["nema"]["raw"]["fwhm_ps"]}
    out["headline_value"] = {"frame": scope, "line": line, "width": f"{GATE_KIND} {GATE_KEY}", "values": values}
    return out


def _public(results):
    return {scope: {line: {name: {k: v for k, v in entry.items() if not k.startswith("_")}
                           for name, entry in tables.items()} for line, tables in lines.items()}
            for scope, lines in results.items()}


# --- run --------------------------------------------------------------------------------------------------------

def run_necr(spec, *, log=print):
    from src.timing.report import peak_memory
    started = time.perf_counter()
    inputs, setup, tables = ctr.frozen_inputs(spec)
    unavailable = {name: "no applied delay" for name, table in tables.items() if not table.delays}
    tables = {name: table for name, table in tables.items() if name not in unavailable}
    out = prepare_output(spec)
    setup = ImasSetup.from_files(inputs["config"], spec.encal)
    lookup = setup.lookup()
    selection = sel.Selection.production(lookup)
    geometry = ctr.geometries(setup, inputs["signs"], (ImasGeometry.from_setup(setup).depth_mm,))[0]
    frames = [{"name": f.name, "frame": f.frame, "activity_kbq_ml": f.activity_kbq_ml,
               "activity_source": f.activity_source,
               "splits": [{"path": str(p), "size_bytes": p.stat().st_size, "sha256": _sha256(p)} for p in f.ldat]}
              for f in spec.frames]
    clock = time.perf_counter()
    samples = {f.name: localization_sample(f, spec, lookup, selection, geometry) for f in spec.frames}
    lines, distances = locate_lines(spec, samples, geometry, log=log)
    del samples
    seconds = {"localization": round(time.perf_counter() - clock, 1)}
    accumulator = Accumulator(spec, tables, lines)
    for index, frame in enumerate(spec.frames):
        clock = time.perf_counter()
        for split, path in enumerate(frame.ldat):
            for features in sel.feature_batches(path, lookup, selection, max_records=spec.max_records):
                accumulator.add(index, features, geometry.endpoints(features) if len(features.ints) else None, split)
        read = time.perf_counter() - clock
        accumulator.finish_frame(index)
        seconds[frame.name] = {"read_and_accumulate": round(read, 1),
                               "summarize": round(time.perf_counter() - clock - read, 1)}
        stages = accumulator.stages[index]
        log(f"{frame.name}: {stages['records']:,} records, {stages['same events']:,} same events "
            f"({time.perf_counter() - clock:.1f} s)")
    clock = time.perf_counter()
    results = accumulator.finish()
    seconds["pooled_summarize"] = round(time.perf_counter() - clock, 1)
    gates = evaluate_gates(spec, results)
    clock = time.perf_counter()
    energy = energy_check(spec, setup, inputs, log=log)
    seconds["energy_check"] = round(time.perf_counter() - clock, 1)
    report = {
        "schema": REPORT_SCHEMA, "requirement": "013-FR-1..FR-8",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "necr_file": str(spec.path), "necr_file_sha256": spec.sha256, "frozen_inputs": inputs,
        "frames": frames, "unavailable_tables": unavailable, "lines": lines, "line_distances": distances,
        "headline_line": spec.headline_line,
        "cuts": {"pre_geometry": selection.describe(), "geometry": geometry.name, "opening_angle": "none",
                 "same_events": "valid endpoints, both TIME channels with an applied delay in every table"
                 + (", and every walk input on both sides" if any(isinstance(t, ctr.WalkTable) for t in
                                                                   tables.values()) else ""),
                 "max_records_per_split": spec.max_records},
        "method": {"nema": "legacy imas_NEMA_CTR_multiprocess port (src.timing.nema): kept |r| <= "
                           f"{nema.R_KEEP_MM:g} mm and |dt error| <= {nema.T_RANGE_PS:g} ps; raw histogram of every "
                           f"kept event and r x dt {nema.R_BINS} x {nema.T_BINS} bins over +-{nema.R_RANGE_MM:g} mm, "
                           f"+-{nema.T_RANGE_PS:g} ps; background per dt bin interpolated between the r bins "
                           f"nearest +-{nema.R_SIGNAL_MM:g} mm; widths from a parabola through the peak bin and "
                           "linear interpolation; c = 0.299792 mm/ps",
                   "model_free": f"|r| <= {nema.R_SIGNAL_MM:g} mm, {spec.model_free_bin_ps:g} ps bins over "
                                 f"+-{spec.model_free_window_ps:g} ps, flat background, local quadratic crossings "
                                 "(spec 011 FR-10)",
                   "uncertainty": f"{spec.bootstrap} seeded Poisson bootstrap replicates, one weight per event "
                                  "shared by lines and tables; gates add the two SDs in quadrature (spec 011 FR-10), "
                                  "the paired same-event SD is reported beside"},
        "counts": {f.name: {"stages": accumulator.stages[k], "pre_geometry": accumulator.pre_geometry[k]}
                   for k, f in enumerate(spec.frames)},
        "walk_clamped": accumulator.clamped,
        "results": _public(results), "paired_against_legacy": paired_differences(results), "gates": gates,
        "energy": energy,
        "legacy_results": spec.legacy_results,
        "interpretation": "system CTR of this acquisition with the NEMA NU 2-2018 line-source method as the legacy "
                          "script implements it (includes residual offset errors, endpoint blur and the 7 mm "
                          "fixed depth); the legacy .lm results used another table, geometry and position "
                          "estimator, so they are not expected to equal ours",
        "seconds": {**seconds, "total": round(time.perf_counter() - started, 1)},
        "peak_memory": peak_memory(),
    }
    np.savez_compressed(out / "necr_histograms.npz", band=accumulator.hist.band, model_free=accumulator.mf,
                        model_free_edges=accumulator.mf_edges, r_edges=nema.R_EDGES, t_edges=nema.T_EDGES,
                        lines=np.array(accumulator.lines), tables=np.array(accumulator.names),
                        frames=np.array([f.name for f in spec.frames]))
    from src.imas.skew import necrplots
    report["plots"] = [path.relative_to(out).as_posix() for path in necrplots.write_plots(out, report)]
    (out / "necr_ctr.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew necr", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("necr_file", type=Path, help=f"NECR CTR file (schema {SCHEMA})")
    args = parser.parse_args(argv)
    try:
        spec = load_necr(args.necr_file)
        report = run_necr(spec)
    except CtrError as error:
        print(f"necr refused: {error}", file=sys.stderr)
        return 2
    headline = report["gates"]["headline_value"]
    print(f"Headline value ({headline['frame']}, line {headline['line']}, model-free FWHM; NEMA "
          "background-subtracted FWHM beside):")
    for name, entry in headline["values"].items():
        value = (f"{entry['fwhm_ps']:.0f} +- {entry['sd_ps']:.0f} ps" if entry.get("fwhm_ps")
                 and entry.get("sd_ps") is not None else "")
        background = entry.get("nema_background_subtracted", {})
        beside = (f" (NEMA {background['fwhm_ps']:.0f} +- {background['sd_ps']:.0f} ps)"
                  if background.get("status") == nema.FITTED and background.get("sd_ps") is not None
                  else f" (NEMA: {background.get('status', 'unavailable')})")
        print(f"  {name}: {value}" + ("" if entry["status"] == nema.FITTED else f" ({entry['status']})") + beside)
    for gate in ("transfer", "acceptance"):
        if gate in report["gates"]:
            g = report["gates"][gate]
            print(f"{gate}: {g['candidate']} vs {g['reference']}: {'accepted' if g['accepted'] else 'not accepted'}")
    print(f"-> {spec.output}")
    return 0
