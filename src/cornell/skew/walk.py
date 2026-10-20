"""Cornell time-walk correction (spec 009): walk inputs, walk LUT and its fits.

Per side, the corrected timestamp is ``t - b[TIME channel] - W[charge, share] - D[DOI] - K[keV]`` (ps,
subtractive). The four walk inputs (FR-1) are listmode's own quantities, so a table fitted here and
applied by ``src/cornell/listmode.py`` sees the same values for the same event:

* charge: raw QDC charge (a.u.) of the timestamped TIME hit, the highest-charge TIME hit of the selected
  minimodule (first maximum);
* share: that charge over the minimodule's summed TIME charge;
* DOI: listmode's light-sharing ratio (ENERGY sum / ENERGY maximum) mapped linearly through the DOI limits
  of its (TIME channel, slab) onto 0-20 mm (``listmode.map_doi_mm``); not a calibrated depth;
* keV: listmode's ``511 / mu * E`` with its calibration's (TIME channel, slab, region) factor, the region
  from the ENERGY centroid and the COG limits (``listmode.compute_region_vectorized``).

Raw ingredients (charges, ENERGY sum, DOI ratio, centroid, TIME channel, slab) come from the selection;
``WalkContext`` maps them with listmode's files, so cached ingredients stay valid when those files change.
Missing inputs are status codes, never zero corrections.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from src.cornell import listmode as lm
from src.cornell.calibration import CalibrationMaps, create_region_boundaries
from src.cornell.inputs import load_calibration, load_limits, load_processing_config

ROOT = Path(__file__).resolve().parents[3]
INPUTS = ("charge", "share", "doi_mm", "kev")
# status per side: 0 ok, else the first missing input
STATUS = ("ok", "no TIME charge", "no ENERGY charge", "no position region", "no calibration factor",
          "no DOI limits")
_STATUS = {label: code for code, label in enumerate(STATUS)}


@dataclass(frozen=True)
class WalkContext:
    """Listmode's calibration, COG and DOI limits as lookup arrays, with their identities."""
    cal_map: np.ndarray
    num_regions: int
    boundaries: np.ndarray
    cog_left: np.ndarray
    cog_right: np.ndarray
    doi_left: np.ndarray
    doi_right: np.ndarray
    files: dict          # {"calibration"|"cog_limits"|"doi_limits": {"path", "sha256"}}

    @classmethod
    def build(cls, mapping, calibration, cog_limits, doi_limits):
        max_ch = CalibrationMaps.from_mapping(mapping).max_ch
        cog_left, cog_right = lm._limit_array(cog_limits, max_ch)
        doi_left, doi_right = lm._limit_array(doi_limits, max_ch)
        files = {name: {"path": str(item.path), "sha256": item.sha256}
                 for name, item in (("calibration", calibration), ("cog_limits", cog_limits),
                                    ("doi_limits", doi_limits))}
        files["calibration"].update(layout=calibration.layout, num_regions=calibration.num_regions)
        return cls(lm.calibration_array(calibration, max_ch), calibration.num_regions,
                   create_region_boundaries(calibration.num_regions), cog_left, cog_right, doi_left,
                   doi_right, files)

    @classmethod
    def from_files(cls, config, calibration, cog_limits, doi_limits, *, root=ROOT):
        """Load as the listmode command does (absolute paths; the calibration's own region count)."""
        mapping = load_processing_config(config, processing_root=root, action="listmode").mapping
        paths = [Path(p) if Path(p).is_absolute() else Path(root) / p for p in (calibration, cog_limits, doi_limits)]
        return cls.build(mapping, load_calibration(paths[0], mapping),
                         load_limits(paths[1], mapping, kind="cog"), load_limits(paths[2], mapping, kind="doi"))


def inputs(context, time_channel, slab, q_top, q_time, e_raw, doi_ratio, y_cog):
    """Walk inputs of one side for arrays of events: ({name: float64 array}, status int8 array)."""
    time_channel = np.asarray(time_channel, np.int64)
    slab = np.asarray(slab, np.int64)
    q_top, q_time = np.asarray(q_top, np.float64), np.asarray(q_time, np.float64)
    n = len(time_channel)
    inside = (time_channel >= 0) & (time_channel < context.cal_map.shape[0]) & (slab >= 0) & (slab < lm.MAX_SLABS)
    tch, sl = np.where(inside, time_channel, 0), np.where(inside, slab, 0)
    region = lm.compute_region_vectorized(np.asarray(y_cog, np.float64), np.where(inside, time_channel, -1), sl,
                                          context.cog_left, context.cog_right, context.boundaries,
                                          context.num_regions)
    kev, _ = lm.convert_energy_numba(np.where(inside, time_channel, -1), sl, region,
                                     np.asarray(e_raw, np.float32), context.cal_map, -np.inf, np.inf)
    has_factor = lm._has_factor(context.cal_map, np.where(inside, time_channel, -1), sl, region)
    left, right = context.doi_left[tch, sl], context.doi_right[tch, sl]
    doi = lm.map_doi_mm(np.asarray(doi_ratio, np.float32), left, right)
    status = np.zeros(n, np.int8)
    for label, bad in (("no DOI limits", ~inside | np.isnan(left) | np.isnan(right)),
                       ("no calibration factor", ~has_factor),
                       ("no position region", region < 0),
                       ("no ENERGY charge", np.asarray(doi_ratio) <= 0),
                       ("no TIME charge", (q_top < 0) | ~(q_time > 0))):
        status[bad] = _STATUS[label]       # applied in reverse order: the first missing input wins
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(q_time > 0, q_top / q_time, np.nan)
    values = {"charge": q_top, "share": share, "doi_mm": doi.astype(np.float64), "kev": kev.astype(np.float64)}
    for name in values:
        values[name] = np.where(status == 0, values[name], np.nan)
    return values, status


def feature_inputs(context, features):
    """Walk inputs of both sides of ``ingest.FeatureBatch`` rows: ({name_1/_2: array}, status_1, status_2)."""
    out, statuses = {}, []
    for s in (1, 2):
        values, status = inputs(context, features.column(f"time_channel_{s}"), features.column(f"slab_{s}"),
                                features.column(f"q_top_{s}"), features.column(f"q_time_{s}"),
                                features.column(f"e_raw_{s}"), features.column(f"doi_ratio_{s}"),
                                features.column(f"y_{s}"))
        out.update({f"{name}_{s}": value for name, value in values.items()})
        statuses.append(status)
    return out, statuses[0], statuses[1]


# --- Walk LUT (FR-2) ----------------------------------------------------------
LUT_SCHEMA = "cornell-walk-lut/1"
CONVENTION = "per side t_corrected = t - b[TIME channel] - walk2d[charge, share] - doi[doi_mm] - kev[kev] (ps)"
BLOCKS = {"walk2d": ("charge", "share"), "doi": ("doi_mm",), "kev": ("kev",)}
DEFAULT_BINS = {"charge": 20, "share": 10, "doi_mm": 20, "kev": 20}
MIN_EVENTS = 200


class WalkError(ValueError):
    pass


def manifest_path(path):
    """``walk_lut.json`` -> ``walk_lut.manifest.json``."""
    path = Path(path)
    return path.with_name(path.stem + ".manifest.json")


def _fill_sparse(offset, events, minimum):
    """Bins below ``minimum`` events take the nearest populated bin (Manhattan in bin indices, first in
    C order on ties); returns the filled offsets and the listed bins."""
    offset = np.array(offset, np.float64)
    populated = np.argwhere(events >= minimum)
    if not len(populated):
        raise WalkError(f"no bin reaches {minimum} events")
    filled = []
    for index in np.argwhere(events < minimum):
        nearest = populated[np.argmin(np.abs(populated - index).sum(axis=1))]
        offset[tuple(index)] = offset[tuple(nearest)]
        filled.append([int(v) for v in index])
    return offset, filled


def _bin(edges, values):
    """Bin index (edges clamped) and whether each value lay outside the edges; NaN -> index -1."""
    values = np.asarray(values, np.float64)
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    outside = (values < edges[0]) | (values > edges[-1])
    return np.where(np.isnan(values), -1, index), outside & ~np.isnan(values)


class WalkLut:
    """Frozen walk table: per block bin edges (one array per axis), offsets (ps) and fit events per bin."""

    def __init__(self, edges, offset_ps, events, filled, manifest):
        self.edges = {b: [np.asarray(e, np.float64) for e in edges[b]] for b in BLOCKS}
        self.offset_ps = {b: np.asarray(offset_ps[b], np.float64) for b in BLOCKS}
        self.events = {b: np.asarray(events[b], np.int64) for b in BLOCKS}
        self.filled = filled
        self.manifest = manifest
        for b, axes in BLOCKS.items():
            shape = tuple(len(e) - 1 for e in self.edges[b])
            if len(self.edges[b]) != len(axes) or self.offset_ps[b].shape != shape or self.events[b].shape != shape:
                raise WalkError(f"block {b}: edges/offsets/events shapes disagree")
            if any(np.any(np.diff(e) <= 0) for e in self.edges[b]) or not np.all(np.isfinite(self.offset_ps[b])):
                raise WalkError(f"block {b}: edges must strictly increase and offsets be finite")

    @classmethod
    def from_fit(cls, edges, offset_ps, events, manifest, *, minimum=MIN_EVENTS):
        """Fill sparse bins, then fix the gauge: each block zero-mean over the fit events (only side
        differences are observable, so a constant per block is free)."""
        offsets, filled = {}, {}
        for b in BLOCKS:
            value, filled[b] = _fill_sparse(offset_ps[b], np.asarray(events[b]), minimum)
            weights = np.asarray(events[b], np.float64)
            offsets[b] = value - (weights * value).sum() / weights.sum()
        return cls(edges, offsets, events, filled, dict(manifest, min_events=minimum))

    def table(self):
        return {"schema": LUT_SCHEMA, "units": "ps", "convention": CONVENTION, "edges": "clamp",
                "gauge": "each block zero-mean over the fit events (events-weighted)",
                "blocks": {b: {"inputs": list(BLOCKS[b]), "edges": [e.tolist() for e in self.edges[b]],
                               "offset_ps": self.offset_ps[b].tolist(), "events": self.events[b].tolist(),
                               "filled_from_nearest": self.filled[b]} for b in BLOCKS}}

    def write(self, path):
        """Write the table and its manifest (with the table SHA-256); never overwrite."""
        path = Path(path)
        text = (json.dumps(self.table(), indent=1) + "\n").encode("utf-8")
        manifest = dict(self.manifest, schema=LUT_SCHEMA, table={"name": path.name,
                                                                 "sha256": hashlib.sha256(text).hexdigest()})
        for target in (path, manifest_path(path)):
            if target.exists():
                raise WalkError(f"output exists: {target}; choose a new name")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as handle:
            handle.write(text)
        with open(manifest_path(path), "x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=1) + "\n")
        self.manifest = manifest
        return manifest

    @classmethod
    def read(cls, path):
        path = Path(path)
        text = path.read_bytes()
        manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
        table = json.loads(text)
        if table.get("schema") != LUT_SCHEMA or table.get("units") != "ps" or table.get("convention") != CONVENTION:
            raise WalkError("unknown walk LUT schema, units or sign convention")
        if manifest.get("table", {}).get("sha256") != hashlib.sha256(text).hexdigest():
            raise WalkError("walk LUT table does not match its manifest")
        blocks = table["blocks"]
        if set(blocks) != set(BLOCKS) or any(blocks[b]["inputs"] != list(BLOCKS[b]) for b in BLOCKS):
            raise WalkError(f"walk LUT blocks must be {BLOCKS}")
        return cls({b: blocks[b]["edges"] for b in BLOCKS}, {b: blocks[b]["offset_ps"] for b in BLOCKS},
                   {b: blocks[b]["events"] for b in BLOCKS}, {b: blocks[b]["filled_from_nearest"] for b in BLOCKS},
                   manifest)

    @property
    def sha256(self):
        return self.manifest["table"]["sha256"]

    def bins(self, values):
        """Per block a flat bin index (-1 where an input is missing) and per input an outside-edges flag."""
        index, outside = {}, {}
        for b, axes in BLOCKS.items():
            flat = None
            for axis, (name, edges) in enumerate(zip(axes, self.edges[b])):
                i, outside[name] = _bin(edges, values[name])
                flat = i if flat is None else np.where((flat < 0) | (i < 0), -1, flat * (len(edges) - 1) + i)
            index[b] = flat
        return index, outside

    def correction(self, values):
        """Per-side correction (ps; NaN where an input is missing) and per-input clamp flags for
        ``values`` = {charge, share, doi_mm, kev: arrays}."""
        index, outside = self.bins(values)
        total = 0.0
        for b in BLOCKS:
            flat = self.offset_ps[b].ravel()
            total = total + np.where(index[b] >= 0, flat[np.maximum(index[b], 0)], np.nan)
        return total, outside


# --- Point-source fit (FR-3, FR-8) --------------------------------------------
FIT_SCHEMA = "cornell-walk-fit/1"
REPORT_SCHEMA = "cornell-walk-fit-report/1"
FIT_EXTRA = {"listmode": ..., "fit_window_kev": [300.0, 700.0], "bins": dict(DEFAULT_BINS),
             "min_events": MIN_EVENTS, "core_ps": 600.0, "iterations": 4}
FIT_STAGES = ("records", "passed pre-geometry", "invalid endpoint", "no applied delay", "LOR outside distance",
              "walk input missing", "outside window", "kept")


def quantile_edges(values, bins):
    """Strictly increasing quantile edges of ``values`` (repeated quantiles merge, so fewer bins)."""
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2 * bins:
        raise WalkError(f"too few fit events ({len(values)}) for {bins} bins")
    edges = np.unique(np.percentile(values, np.linspace(0, 100, bins + 1)))
    if len(edges) < 2:      # one value: a single bin, i.e. no correction from this input after the gauge
        edges = np.array([edges[0], edges[0] + 1e-6 * max(1.0, abs(edges[0]))])
    return edges


def _empty(edges):
    shapes = {b: tuple(len(e) - 1 for e in edges[b]) for b in BLOCKS}
    return WalkLut(edges, {b: np.zeros(shapes[b]) for b in BLOCKS},
                   {b: np.zeros(shapes[b], np.int64) for b in BLOCKS}, {b: [] for b in BLOCKS}, {})


def design(edges, side1, side2):
    """Sparse +1 (side 1) / -1 (side 2) bin columns per block (walk2d flat bins | doi | kev), and the
    events per column counted over both sides."""
    import scipy.sparse as sp
    template = _empty(edges)
    (i1, _), (i2, _) = template.bins(side1), template.bins(side2)
    sizes = [template.offset_ps[b].size for b in BLOCKS]
    starts = np.cumsum([0] + sizes[:-1])
    n = len(side1["charge"])
    columns = [i1[b] + s for b, s in zip(BLOCKS, starts)] + [i2[b] + s for b, s in zip(BLOCKS, starts)]
    cols = np.concatenate(columns)
    rows = np.tile(np.arange(n), 2 * len(BLOCKS))
    vals = np.repeat([1.0, -1.0], len(BLOCKS) * n)
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n, sum(sizes)))
    matrix.sum_duplicates()
    return matrix, np.bincount(cols, minlength=sum(sizes)), starts, sizes


def fit_tables(side1, side2, residual, train, bins, *, core_ps, iterations, minimum, manifest):
    """Joint least squares of residual = W(side 1) - W(side 2) over the blocks on ``train`` events.

    Edges are quantiles of the train events (both sides pooled); iterated core window
    |residual - model| < ``core_ps`` (first pass |residual| < core_ps); a 1e-6 relative ridge fixes the
    per-block constants, then ``WalkLut.from_fit`` fills sparse bins and sets the gauge.
    """
    edges = {b: [quantile_edges(np.concatenate([side1[name][train], side2[name][train]]), bins[name])
                 for name in BLOCKS[b]] for b in BLOCKS}
    matrix, _, starts, sizes = design(edges, side1, side2)
    keep = train & (np.abs(residual) < core_ps)
    history = []
    for _ in range(iterations):
        rows = matrix[keep]
        normal = (rows.T @ rows).toarray()
        ridge = 1e-6 * max(np.trace(normal), 1.0) / len(normal)
        solution = np.linalg.solve(normal + ridge * np.eye(len(normal)), rows.T @ residual[keep])
        model = matrix @ solution
        history.append({"events": int(keep.sum()), "rms_ps": float(np.sqrt(np.mean((residual - model)[keep] ** 2)))})
        keep = train & (np.abs(residual - model) < core_ps)
    _, counts, _, _ = design(edges, {k: v[train] for k, v in side1.items()}, {k: v[train] for k, v in side2.items()})
    offsets, events = {}, {}
    for b, start, size in zip(BLOCKS, starts, sizes):
        shape = tuple(len(e) - 1 for e in edges[b])
        offsets[b] = solution[start:start + size].reshape(shape)
        events[b] = counts[start:start + size].reshape(shape)
    lut = WalkLut.from_fit(edges, offsets, events, dict(manifest, iterations=history, core_ps=core_ps),
                           minimum=minimum)
    return lut, history


def walk_difference(lut, side1, side2):
    """W(side 1) - W(side 2) per event (ps) and per side the clamp flags."""
    w1, out1 = lut.correction(side1)
    w2, out2 = lut.correction(side2)
    return w1 - w2, out1, out2


def check_pairing(skew_lut, skew_sha256, walk_lut):
    """FR-4/FR-5 pairing: a walk LUT is applied with the skew LUT it was fitted with, or with a skew LUT solved
    with it (round trip: skew re-solved with walk 1, walk 2 refitted with that skew); a skew LUT solved with a walk
    LUT is never applied without walk."""
    skew_walk = (skew_lut.manifest.get("provenance") or {}).get("walk_lut_sha256")
    if walk_lut is None:
        if skew_walk:
            raise WalkError(f"the skew LUT was solved with walk LUT {skew_walk[:12]}...: apply a walk LUT paired with it")
        return
    if walk_lut.manifest.get("skew_lut_sha256") != skew_sha256 and walk_lut.sha256 != skew_walk:
        raise WalkError(f"walk LUT {walk_lut.sha256[:12]}... was fitted with skew LUT "
                        f"{str(walk_lut.manifest.get('skew_lut_sha256'))[:12]}..., not with {skew_sha256[:12]}..., "
                        "and that skew LUT was not solved with it")


def check_files(walk_lut, context):
    """The COG and DOI limits must be the ones the walk LUT was fitted with. The calibration may be another one
    of the same layout and region count (owner 2026-10-20: each run converts keV with its own calibration).
    Returns {"calibration_differs_from_fit": bool, "fitted_calibration_sha256": ...}."""
    fitted = walk_lut.manifest.get("listmode_files") or {}
    differ = sorted(k for k in ("cog_limits", "doi_limits")
                    if (fitted.get(k) or {}).get("sha256") != context.files[k]["sha256"])
    if differ:
        raise WalkError(f"listmode files {differ} differ from those the walk LUT was fitted with")
    ours, theirs = context.files["calibration"], fitted.get("calibration") or {}
    same = theirs.get("sha256") == ours["sha256"]
    if not same and (theirs.get("layout"), theirs.get("num_regions")) != (ours["layout"], ours["num_regions"]):
        raise WalkError(f"listmode calibration {ours['layout']} with {ours['num_regions']} region(s) differs in "
                        f"layout from the walk LUT's ({theirs.get('layout')}, {theirs.get('num_regions')} region(s))")
    return {"calibration_differs_from_fit": not same, "fitted_calibration_sha256": theirs.get("sha256")}


def load_fit(path):
    """Walk fit file: the CTR file's keys (schema ``cornell-walk-fit/1``) plus ``listmode`` (calibration,
    cog_limits, doi_limits), ``fit_window_kev``, ``bins``, ``min_events``, ``core_ps``, ``iterations``."""
    from dataclasses import replace
    from src.cornell.skew import ctr
    spec = ctr.load_ctr(path, schema=FIT_SCHEMA, extra=FIT_EXTRA)
    extra = dict(spec.extra)
    files = extra["listmode"]
    if not isinstance(files, dict) or set(files) != {"calibration", "cog_limits", "doi_limits"}:
        raise WalkError("listmode: {calibration, cog_limits, doi_limits} paths required")
    base = spec.path.resolve().parent
    extra["listmode"] = {k: (Path(v) if Path(v).is_absolute() else base / v) for k, v in files.items()}
    window = extra["fit_window_kev"]
    if not (isinstance(window, list) and len(window) == 2 and 0 <= float(window[0]) < float(window[1])):
        raise WalkError("fit_window_kev must be [low, high] keV with 0 <= low < high")
    extra["fit_window_kev"] = [float(window[0]), float(window[1])]
    bins = extra["bins"]
    if not isinstance(bins, dict) or set(bins) != set(DEFAULT_BINS) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in bins.values()):
        raise WalkError(f"bins: integers >= 2 for {sorted(DEFAULT_BINS)}")
    for key in ("min_events", "iterations"):
        if isinstance(extra[key], bool) or not isinstance(extra[key], int) or extra[key] < 1:
            raise WalkError(f"{key} must be a positive integer")
    if not float(extra["core_ps"]) > 0:
        raise WalkError("core_ps must be positive")
    return replace(spec, extra=extra)


def collect(spec, *, log=print):
    """Stream every acquisition once with the CTR cuts at the fit window; keep each kept event's walk
    inputs (float32), residual, record parity and acquisition (~40 bytes per event)."""
    from dataclasses import replace
    from src.cornell.skew import ctr
    from src.cornell.skew.ingest import CompiledLookup, LegacySetup, apply_geometry, code_label, compiled_batches
    inputs_info, lut = ctr.frozen_inputs(spec)
    setup = LegacySetup.from_files(spec.config, spec.encal, root=ROOT)
    lookup = replace(CompiledLookup.from_setup(setup), energy_range=tuple(spec.extra["fit_window_kev"]))
    files = spec.extra["listmode"]
    context = WalkContext.from_files(spec.config, files["calibration"], files["cog_limits"], files["doi_limits"])
    hypothesis = ctr._hypotheses(spec, setup)[0]
    delays = ctr.Delays(lut)
    parts, acquisitions = [], []
    for index, acquisition in enumerate(spec.acquisitions):
        if not acquisition.ldat.is_file():
            raise WalkError(f"{acquisition.name}: missing LDAT {acquisition.ldat}")
        entry = {"name": acquisition.name, "ldat": str(acquisition.ldat), "sha256": ctr._sha256(acquisition.ldat),
                 "declared": acquisition.declared}
        if acquisition.source_mm is None:
            entry["source"] = ctr.localize(acquisition, index, spec, lookup, hypothesis)
        else:
            entry["source"] = {"method": "declared", "position_mm": list(acquisition.source_mm)}
        source = np.asarray(entry["source"]["position_mm"], np.float64)
        stages, status_counts, pre = dict.fromkeys(FIT_STAGES, 0), {}, {}
        for features in compiled_batches(acquisition.ldat, lookup, file_index=index, max_records=spec.max_records,
                                         seed=spec.seed):
            stages["records"] += len(features.codes)
            for code, count in zip(*np.unique(features.codes, return_counts=True)):
                pre[code_label(code)] = pre.get(code_label(code), 0) + int(count)
            stages["passed pre-geometry"] += len(features.ints)
            if not len(features.ints):
                continue
            geometry = apply_geometry(features, hypothesis)
            k1 = delays.index(features.column("time_channel_1"))
            k2 = delays.index(features.column("time_channel_2"))
            p, q = geometry.endpoints_mm[:, :3], geometry.endpoints_mm[:, 3:]
            valid, applied = geometry.codes != 1, (k1 >= 0) & (k2 >= 0)
            distance, along = ctr.lor_distance_mm(source, p, q)
            with np.errstate(invalid="ignore"):
                near = valid & applied & (distance <= spec.lor_distance_mm) & (along > 0) & (along < 1)
            values, status1, status2 = feature_inputs(context, features)
            ok = (status1 == 0) & (status2 == 0)
            for s, status in ((1, status1), (2, status2)):
                for code in np.unique(status[near & (status != 0)]):
                    label = f"side {s}: {STATUS[code]}"
                    status_counts[label] = status_counts.get(label, 0) + int(np.count_nonzero(near & (status == code)))
            b = delays.delays_ps
            dt = features.column("dt_ps").astype(np.float64) - b[np.maximum(k1, 0)] + b[np.maximum(k2, 0)]
            with np.errstate(invalid="ignore"):
                residual = np.where(near, dt - ctr.geometric_tof_ps(source, p, q), np.nan)
                inside = near & ok & (np.abs(residual) < spec.window_ps)
            stages["invalid endpoint"] += int(np.count_nonzero(~valid))
            stages["no applied delay"] += int(np.count_nonzero(valid & ~applied))
            stages["LOR outside distance"] += int(np.count_nonzero(valid & applied & ~near))
            stages["walk input missing"] += int(np.count_nonzero(near & ~ok))
            stages["outside window"] += int(np.count_nonzero(near & ok & ~inside))
            stages["kept"] += int(np.count_nonzero(inside))
            part = {"residual_ps": residual[inside].astype(np.float32),
                    "fit": features.column("record")[inside] % 2 == 0,
                    "acquisition": np.full(int(inside.sum()), index, np.int8)}
            for name in INPUTS:
                for s in (1, 2):
                    part[f"{name}_{s}"] = values[f"{name}_{s}"][inside].astype(np.float32)
            parts.append(part)
        entry.update(stages=stages, pre_geometry=pre, walk_status=status_counts)
        acquisitions.append(entry)
        log(f"{acquisition.name}: {stages['records']:,} records, {stages['kept']:,} kept "
            f"({stages['walk input missing']:,} without walk inputs)")
    if not parts:
        raise WalkError("no event passed the cuts")
    events = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return events, acquisitions, inputs_info, context


def _sides(events, mask=None):
    pick = (lambda v: v) if mask is None else (lambda v: v[mask])
    return ({n: pick(events[f"{n}_1"]).astype(np.float64) for n in INPUTS},
            {n: pick(events[f"{n}_2"]).astype(np.float64) for n in INPUTS})


def ctr_summary(residual, spec, bootstrap, seed):
    """Spec 008 CTR numbers (Gaussian + flat, model-free FWHM/FWTM) of residuals, with bootstrap spreads."""
    from src.cornell.skew import ctr
    edges = spec.edges_ps
    counts = np.histogram(residual, edges)[0].astype(float)
    fit = ctr.fit_histogram(counts, edges)
    out = ctr._public(fit)
    if fit["status"] == "fitted" and bootstrap:
        rng = np.random.default_rng(seed)
        reps = [ctr.fit_histogram(rng.poisson(counts), edges, start=fit["_start"]) for _ in range(bootstrap)]
        for key in ("fwhm_ps", "model_free_fwhm_ps", "model_free_fwtm_ps"):
            values = [r[key] for r in reps if r["status"] == "fitted" and r[key] is not None]
            out[key + "_stat"] = float(np.std(values, ddof=1)) if len(values) > 2 else None
    return out


def _report_path(output):
    output = Path(output)
    return output.with_name(output.stem + "_report.json")


def run_fit(spec, *, log=print):
    started = time.perf_counter()
    output = Path(spec.output)
    for target in (output, manifest_path(output), _report_path(output)):
        if target.exists():
            raise WalkError(f"output exists: {target}; choose a new name")
    events, acquisitions, inputs_info, context = collect(spec, log=log)
    extra = spec.extra
    residual = events["residual_ps"].astype(np.float64)
    train = events["fit"]
    side1, side2 = _sides(events)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fit_kind": "point_source",
        "fit_file": str(spec.path), "fit_file_sha256": spec.sha256, "skew_lut": inputs_info["lut"],
        "skew_lut_sha256": inputs_info["lut_sha256"], "config": inputs_info["config"],
        "config_sha256": inputs_info["config_sha256"], "encal": inputs_info["encal"],
        "encal_sha256": inputs_info["encal_sha256"], "listmode_files": context.files,
        "fit_window_kev": extra["fit_window_kev"],
        "cuts": {"lor_distance_mm": spec.lor_distance_mm, "window_ps": spec.window_ps, "seed": spec.seed},
        "split": "fit on even record indices, held out odd",
        "acquisitions": [{k: a[k] for k in ("name", "ldat", "sha256", "source")} for a in acquisitions],
        "events": {"fit": int(train.sum()), "held_out": int((~train).sum())}, "bins": extra["bins"]}
    lut, history = fit_tables(side1, side2, residual, train, extra["bins"], core_ps=float(extra["core_ps"]),
                              iterations=extra["iterations"], minimum=extra["min_events"], manifest=manifest)
    difference, out1, out2 = walk_difference(lut, side1, side2)
    corrected = residual - difference
    held = ~train
    results = {"pooled": {"before": ctr_summary(residual[held], spec, spec.bootstrap, spec.seed),
                          "after": ctr_summary(corrected[held], spec, spec.bootstrap, spec.seed)}}
    for index, acquisition in enumerate(spec.acquisitions):
        mask = held & (events["acquisition"] == index)
        results[acquisition.name] = {"before": ctr_summary(residual[mask], spec, 0, spec.seed),
                                     "after": ctr_summary(corrected[mask], spec, 0, spec.seed)}
    clamped = {f"side {s} {name}": int(np.count_nonzero(flags[name][held]))
               for s, flags in ((1, out1), (2, out2)) for name in INPUTS}
    lut.write(output)
    report = {"schema": REPORT_SCHEMA, "walk_lut": str(output), "walk_lut_sha256": lut.sha256,
              "frozen_inputs": inputs_info, "acquisitions": acquisitions, "manifest": lut.manifest,
              "iterations": history, "held_out_clamped": clamped, "results": results,
              "seconds": round(time.perf_counter() - started, 1)}
    _report_path(output).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    plot_fit(lut, events, residual, corrected, output)
    b, a = results["pooled"]["before"], results["pooled"]["after"]
    if b.get("model_free_fwhm_ps") is not None and a.get("model_free_fwhm_ps") is not None:
        log(f"walk LUT {output}: held-out model-free FWHM {b['model_free_fwhm_ps']:.1f} -> "
            f"{a['model_free_fwhm_ps']:.1f} ps, Gaussian {b['fwhm_ps']:.1f} -> {a['fwhm_ps']:.1f} ps "
            f"({int(held.sum()):,} events)")
    return report


def plot_fit(lut, events, residual, corrected, output):
    """FR-8 diagnostics: 2D walk table with counts, DOI/keV curves, held-out residual profiles."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt
    output = Path(output)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    ec, es = lut.edges["walk2d"]
    mesh = axes[0].pcolormesh(ec, es, lut.offset_ps["walk2d"].T, cmap="coolwarm", shading="flat")
    fig.colorbar(mesh, ax=axes[0], label="offset (ps)")
    axes[0].set(xlabel="timing-hit charge (a.u.)", ylabel="light-sharing share", title="walk2d offset")
    counts = axes[1].pcolormesh(ec, es, np.maximum(lut.events["walk2d"].T, 1), shading="flat", norm=LogNorm())
    fig.colorbar(counts, ax=axes[1], label="fit events (both sides)")
    axes[1].set(xlabel="timing-hit charge (a.u.)", ylabel="light-sharing share", title="walk2d fit events")
    for ax, block, label in ((axes[2], "doi", "DOI (mm, listmode)"), (axes[3], "kev", "energy (keV, listmode)")):
        e = lut.edges[block][0]
        ax.step(e, np.append(lut.offset_ps[block], lut.offset_ps[block][-1]), where="post")
        ax.set(xlabel=label, ylabel="offset (ps)", title=f"{block} offset")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output.with_name(output.stem + "_tables.png"), dpi=110)
    plt.close(fig)
    held = ~events["fit"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, name in zip(axes, INPUTS):
        values = np.concatenate([events[f"{name}_1"][held], events[f"{name}_2"][held]]).astype(np.float64)
        edges = np.unique(np.nanpercentile(values, np.linspace(0, 100, 21)))
        centre = (edges[:-1] + edges[1:]) / 2
        index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
        for label, r in (("before", residual), ("after", corrected)):
            signed = np.concatenate([r[held], -r[held]])
            core = np.abs(signed) < 1000
            medians = [np.median(signed[core & (index == i)]) if np.any(core & (index == i)) else np.nan
                       for i in range(len(centre))]
            ax.plot(centre, medians, "o-", ms=3, label=label)
        ax.axhline(0, color="k", lw=0.6)
        ax.set(xlabel=name, ylabel="median side-signed residual (ps)", title=f"held out: residual vs {name}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output.with_name(output.stem + "_profiles.png"), dpi=110)
    plt.close(fig)


# --- Normalization-phantom fit (FR-7) -------------------------------------------
PHANTOM_SCHEMA = "cornell-walk-phantom-fit/1"
_PHANTOM_REQUIRED = {"schema", "job", "lut", "lut_sha256", "listmode", "output"}
_PHANTOM_OPTIONAL = {"bins": dict(DEFAULT_BINS), "min_events": MIN_EVENTS, "core_ps": 600.0, "iterations": 4,
                     "branch_iterations": 3, "max_events": 5_000_000}


def load_phantom(path):
    """Phantom fit file: the skew LUT's job (its feature caches hold the walk ingredients, schema 3), the frozen
    skew LUT, listmode files, output and fit options; paths relative to the file."""
    path = Path(path)
    raw = path.read_bytes()
    spec = json.loads(raw)
    if not isinstance(spec, dict) or spec.get("schema") != PHANTOM_SCHEMA:
        raise WalkError(f"file schema must be {PHANTOM_SCHEMA!r}")
    if not _PHANTOM_REQUIRED <= set(spec) <= _PHANTOM_REQUIRED | set(_PHANTOM_OPTIONAL):
        raise WalkError(f"file keys: required {sorted(_PHANTOM_REQUIRED)}, optional {sorted(_PHANTOM_OPTIONAL)}")
    base = path.resolve().parent

    def local(value):
        value = Path(value)
        return value if value.is_absolute() else base / value

    options = _PHANTOM_OPTIONAL | {k: spec[k] for k in _PHANTOM_OPTIONAL if k in spec}
    files = spec["listmode"]
    if not isinstance(files, dict) or set(files) != {"calibration", "cog_limits", "doi_limits"}:
        raise WalkError("listmode: {calibration, cog_limits, doi_limits} paths required")
    for key in ("min_events", "iterations", "branch_iterations", "max_events"):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 1:
            raise WalkError(f"{key} must be a positive integer")
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "job": local(spec["job"]),
            "lut": local(spec["lut"]), "lut_sha256": str(spec["lut_sha256"]).lower(), "output": local(spec["output"]),
            "listmode": {k: local(v) for k, v in files.items()}, **options}


def collect_phantom(spec, *, log=print):
    """Train-role feature rows of every job run passing the selected geometry, with LUT-corrected ``t_1 - t_2``,
    the near/far shell-branch centre TOFs and the walk inputs; an evenly strided subsample of at most
    ``max_events`` (no LDAT is read)."""
    from src.cornell.skew import cache as tc, ctr, job as tj
    from src.cornell.skew.ingest import GeometryHypothesis, LegacySetup, apply_geometry
    from src.cornell.skew.lut import CanonicalLut
    job = tj.load_job(spec["job"])
    if ctr._sha256(spec["lut"]) != spec["lut_sha256"]:
        raise WalkError(f"LUT SHA-256 of {spec['lut']} is not the frozen {spec['lut_sha256']}")
    lut = CanonicalLut.read(spec["lut"])
    if (lut.manifest.get("provenance") or {}).get("job_sha256") != job.sha256:
        raise WalkError("the skew LUT was not solved by this job file: its caches are not the LUT's events")
    if (lut.input_correction or {}).get("state") != "uncorrected":
        raise WalkError("phantom walk fits need an uncorrected-data skew LUT")
    files = spec["listmode"]
    context = WalkContext.from_files(job.config, files["calibration"], files["cog_limits"], files["doi_limits"])
    setup = LegacySetup.from_files(job.config, job.encal, root=ROOT)
    hypothesis = GeometryHypothesis.selected(job.config, setup)
    delays = ctr.Delays(lut)
    caches = []
    for run in job.runs:
        tag = "" if run.encal is None else f"-{hashlib.sha256(run.encal.read_bytes()).hexdigest()[:12]}"
        cache = tc.FeatureCache(job.work / run.name / f"features{tag}").require_complete()
        if cache.manifest["fingerprint"]["schema_version"] != tc.SCHEMA_VERSION:
            raise WalkError(f"{run.name}: feature cache schema {cache.manifest['fingerprint']['schema_version']}, "
                            f"walk ingredients need {tc.SCHEMA_VERSION}")
        caches.append((run.name, cache))
    total = sum(block["rows"] for _, cache in caches for done in cache.files.values() if done["role"] == "train"
                for block in done["blocks"])
    stride = max(1, -(-total // spec["max_events"]))
    parts, counts = [], {"train rows": int(total), "stride": int(stride), "geometry rejected": 0,
                         "no applied delay": 0, "walk input missing": 0}
    for name, cache in caches:
        for _, role, block in cache.blocks():
            if role != "train":
                continue
            rows = np.array(block[::stride])
            result = apply_geometry(tc._Columns(rows), hypothesis)
            accepted = result.codes == 0
            k1, k2 = delays.index(rows["time_channel_1"]), delays.index(rows["time_channel_2"])
            applied = (k1 >= 0) & (k2 >= 0)
            values, statuses = {}, []
            for s in (1, 2):
                values[s], status = inputs(context, rows[f"time_channel_{s}"], rows[f"slab_{s}"], rows[f"q_top_{s}"],
                                           rows[f"q_time_{s}"], rows[f"e_raw_{s}"], rows[f"doi_ratio_{s}"],
                                           rows[f"y_{s}"])
                statuses.append(status)
            ok = (statuses[0] == 0) & (statuses[1] == 0)
            counts["geometry rejected"] += int(np.count_nonzero(~accepted))
            counts["no applied delay"] += int(np.count_nonzero(accepted & ~applied))
            counts["walk input missing"] += int(np.count_nonzero(accepted & applied & ~ok))
            keep = accepted & applied & ok
            b = delays.delays_ps
            dt = rows["dt_ps"].astype(np.float64) - b[np.maximum(k1, 0)] + b[np.maximum(k2, 0)]
            tof = result.shell_tof_ps
            part = {"dt_ps": dt[keep].astype(np.float32),
                    "near_ps": ((tof[:, 0] + tof[:, 1]) / 2)[keep].astype(np.float32),
                    "far_ps": ((tof[:, 2] + tof[:, 3]) / 2)[keep].astype(np.float32),
                    "fit": rows["record"][keep] % 2 == 0}
            for n in INPUTS:
                for s in (1, 2):
                    part[f"{n}_{s}"] = values[s][n][keep].astype(np.float32)
            parts.append(part)
        log(f"{name}: train rows read with stride {stride}")
    if not parts:
        raise WalkError("no train event in the job caches")
    events = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    counts["kept"] = int(len(events["dt_ps"]))
    return events, counts, job, lut, context


def run_phantom(spec, *, log=print):
    """FR-7: fit the walk LUT from the phantom; the residual is taken against the nearer shell branch, chosen
    again after each walk update (branches ~2 x 930 ps apart, walk < ~200 ps)."""
    started = time.perf_counter()
    output = Path(spec["output"])
    for target in (output, manifest_path(output), _report_path(output)):
        if target.exists():
            raise WalkError(f"output exists: {target}; choose a new name")
    events, counts, job, lut, context = collect_phantom(spec, log=log)
    dt = events["dt_ps"].astype(np.float64)
    near, far = events["near_ps"].astype(np.float64), events["far_ps"].astype(np.float64)
    side1, side2 = _sides(events)
    train = events["fit"]
    model = np.zeros(len(dt))
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fit_kind": "phantom",
                "fit_file": str(spec["path"]), "fit_file_sha256": spec["sha256"], "job": str(job.path),
                "job_sha256": job.sha256, "skew_lut": str(spec["lut"]), "skew_lut_sha256": spec["lut_sha256"],
                "config": str(job.config), "listmode_files": context.files, "events": counts,
                "split": "fit on even record indices of train files", "bins": spec["bins"]}
    history = []
    for _ in range(spec["branch_iterations"]):
        branch = np.where(np.abs(dt - model - near) <= np.abs(dt - model - far), near, far)
        residual = dt - branch
        walk_lut, fit_history = fit_tables(side1, side2, residual, train, spec["bins"], core_ps=float(spec["core_ps"]),
                                           iterations=spec["iterations"], minimum=spec["min_events"],
                                           manifest=manifest)
        model, _, _ = walk_difference(walk_lut, side1, side2)
        history.append({"near_fraction": float(np.mean(branch == near)), "fit": fit_history})
    walk_lut.manifest["branch_iterations"] = history
    walk_lut.write(output)
    report = {"schema": "cornell-walk-phantom-report/1", "walk_lut": str(output), "walk_lut_sha256": walk_lut.sha256,
              "manifest": walk_lut.manifest, "counts": counts, "seconds": round(time.perf_counter() - started, 1)}
    _report_path(output).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    log(f"phantom walk LUT {output}: {counts['kept']:,} train events (stride {counts['stride']})")
    return report


def compare_tables(first, second, side1, side2):
    """Per-event difference of two walk LUTs' corrections W(1) - W(2) on the same events (ps): mean and rms."""
    a, _, _ = walk_difference(first, side1, side2)
    b, _, _ = walk_difference(second, side1, side2)
    d = (a - b)[np.isfinite(a) & np.isfinite(b)]
    return {"events": int(len(d)), "mean_ps": float(d.mean()), "rms_ps": float(np.sqrt(np.mean(d ** 2))),
            "p95_abs_ps": float(np.percentile(np.abs(d), 95))}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.cornell.skew walk",
                                     description="Fit a walk LUT with a frozen skew LUT (spec 009): from point "
                                                 f"sources ({FIT_SCHEMA}) or a normalization phantom job's "
                                                 f"caches ({PHANTOM_SCHEMA}).")
    parser.add_argument("fit_file", help="walk fit file")
    args = parser.parse_args(argv)
    try:
        schema = json.loads(Path(args.fit_file).read_bytes()).get("schema")
        if schema == PHANTOM_SCHEMA:
            run_phantom(load_phantom(args.fit_file), log=lambda message: print(message, flush=True))
        else:
            run_fit(load_fit(args.fit_file), log=lambda message: print(message, flush=True))
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
