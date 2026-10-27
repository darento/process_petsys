"""Scanner-neutral time-walk table and fits (spec 012 T1, FR-2, FR-3; moved from spec 009's Cornell walk).

Per side, ``t_corrected = t - b[TIME channel] - sum over blocks of offset[block][inputs]`` (ps). A
``WalkModel`` names the table's schema, sign convention and blocks (inputs per block); a scanner binds
it to a ``WalkLut`` subclass (``model = ...``). Cornell: ``src.cornell.skew.walk``; IMAS:
``src.imas.skew.walk``.

* ``WalkLut``: frozen bin edges per block axis (clamped), offsets and fit events per bin; sparse bins
  take the nearest populated bin; each block is zero-mean over the fit events (only side differences
  are observable). Written with a manifest holding the table's SHA-256; never overwritten.
* ``fit_tables``: joint least squares of ``residual = W(side 1) - W(side 2)``.
* ``fit_two_branch``: the same fit when each event's expected ``dt`` is one of two branches (an
  annular source's two crossings), the nearer branch taken again after each walk update.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

MIN_EVENTS = 200


class WalkError(ValueError):
    pass


@dataclass(frozen=True)
class WalkModel:
    """A walk table's identity: ``blocks`` = {block: (input, ...)} in table order."""
    schema: str
    convention: str
    blocks: dict
    default_bins: dict

    @property
    def inputs(self):
        return tuple(name for axes in self.blocks.values() for name in axes)


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
    """Frozen walk table: per block bin edges (one array per axis), offsets (ps) and fit events per bin.
    Subclasses set ``model``."""
    model: WalkModel = None

    def __init__(self, edges, offset_ps, events, filled, manifest):
        blocks = self.model.blocks
        self.edges = {b: [np.asarray(e, np.float64) for e in edges[b]] for b in blocks}
        self.offset_ps = {b: np.asarray(offset_ps[b], np.float64) for b in blocks}
        self.events = {b: np.asarray(events[b], np.int64) for b in blocks}
        self.filled = filled
        self.manifest = manifest
        for b, axes in blocks.items():
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
        for b in cls.model.blocks:
            value, filled[b] = _fill_sparse(offset_ps[b], np.asarray(events[b]), minimum)
            weights = np.asarray(events[b], np.float64)
            offsets[b] = value - (weights * value).sum() / weights.sum()
        return cls(edges, offsets, events, filled, dict(manifest, min_events=minimum))

    @classmethod
    def empty(cls, edges):
        shapes = {b: tuple(len(e) - 1 for e in edges[b]) for b in cls.model.blocks}
        return cls(edges, {b: np.zeros(shapes[b]) for b in shapes},
                   {b: np.zeros(shapes[b], np.int64) for b in shapes}, {b: [] for b in shapes}, {})

    def table(self):
        blocks = self.model.blocks
        return {"schema": self.model.schema, "units": "ps", "convention": self.model.convention, "edges": "clamp",
                "gauge": "each block zero-mean over the fit events (events-weighted)",
                "blocks": {b: {"inputs": list(blocks[b]), "edges": [e.tolist() for e in self.edges[b]],
                               "offset_ps": self.offset_ps[b].tolist(), "events": self.events[b].tolist(),
                               "filled_from_nearest": self.filled[b]} for b in blocks}}

    def write(self, path):
        """Write the table and its manifest (with the table SHA-256); never overwrite."""
        path = Path(path)
        text = (json.dumps(self.table(), indent=1) + "\n").encode("utf-8")
        manifest = dict(self.manifest, schema=self.model.schema, table={"name": path.name,
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
        model = cls.model
        if table.get("schema") != model.schema or table.get("units") != "ps" or table.get("convention") != model.convention:
            raise WalkError("unknown walk LUT schema, units or sign convention")
        if manifest.get("table", {}).get("sha256") != hashlib.sha256(text).hexdigest():
            raise WalkError("walk LUT table does not match its manifest")
        blocks = table["blocks"]
        if set(blocks) != set(model.blocks) or any(blocks[b]["inputs"] != list(model.blocks[b]) for b in model.blocks):
            raise WalkError(f"walk LUT blocks must be {model.blocks}")
        names = model.blocks
        return cls({b: blocks[b]["edges"] for b in names}, {b: blocks[b]["offset_ps"] for b in names},
                   {b: blocks[b]["events"] for b in names}, {b: blocks[b]["filled_from_nearest"] for b in names},
                   manifest)

    @property
    def sha256(self):
        return self.manifest["table"]["sha256"]

    def bins(self, values):
        """Per block a flat bin index (-1 where an input is missing) and per input an outside-edges flag."""
        index, outside = {}, {}
        for b, axes in self.model.blocks.items():
            flat = None
            for axis, (name, edges) in enumerate(zip(axes, self.edges[b])):
                i, outside[name] = _bin(edges, values[name])
                flat = i if flat is None else np.where((flat < 0) | (i < 0), -1, flat * (len(edges) - 1) + i)
            index[b] = flat
        return index, outside

    def correction(self, values):
        """Per-side correction (ps; NaN where an input is missing) and per-input clamp flags for
        ``values`` = {input: array}."""
        index, outside = self.bins(values)
        total = 0.0
        for b in self.model.blocks:
            flat = self.offset_ps[b].ravel()
            total = total + np.where(index[b] >= 0, flat[np.maximum(index[b], 0)], np.nan)
        return total, outside


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


def design(edges, side1, side2, *, table):
    """Sparse +1 (side 1) / -1 (side 2) bin columns per block of ``table`` (a ``WalkLut`` subclass), and
    the events per column counted over both sides."""
    import scipy.sparse as sp
    blocks = table.model.blocks
    template = table.empty(edges)
    (i1, _), (i2, _) = template.bins(side1), template.bins(side2)
    sizes = [template.offset_ps[b].size for b in blocks]
    starts = np.cumsum([0] + sizes[:-1])
    n = len(side1[table.model.inputs[0]])
    columns = [i1[b] + s for b, s in zip(blocks, starts)] + [i2[b] + s for b, s in zip(blocks, starts)]
    cols = np.concatenate(columns)
    rows = np.tile(np.arange(n), 2 * len(blocks))
    vals = np.repeat([1.0, -1.0], len(blocks) * n)
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n, sum(sizes)))
    matrix.sum_duplicates()
    return matrix, np.bincount(cols, minlength=sum(sizes)), starts, sizes


def fit_tables(side1, side2, residual, train, bins, *, core_ps, iterations, minimum, manifest, table):
    """Joint least squares of residual = W(side 1) - W(side 2) over the blocks on ``train`` events.

    Edges are quantiles of the train events (both sides pooled); iterated core window
    |residual - model| < ``core_ps`` (first pass |residual| < core_ps); a 1e-6 relative ridge fixes the
    per-block constants, then ``table.from_fit`` fills sparse bins and sets the gauge.
    """
    blocks = table.model.blocks
    edges = {b: [quantile_edges(np.concatenate([side1[name][train], side2[name][train]]), bins[name])
                 for name in blocks[b]] for b in blocks}
    matrix, _, starts, sizes = design(edges, side1, side2, table=table)
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
    _, counts, _, _ = design(edges, {k: v[train] for k, v in side1.items()}, {k: v[train] for k, v in side2.items()},
                             table=table)
    offsets, events = {}, {}
    for b, start, size in zip(blocks, starts, sizes):
        shape = tuple(len(e) - 1 for e in edges[b])
        offsets[b] = solution[start:start + size].reshape(shape)
        events[b] = counts[start:start + size].reshape(shape)
    lut = table.from_fit(edges, offsets, events, dict(manifest, iterations=history, core_ps=core_ps), minimum=minimum)
    return lut, history


def walk_difference(lut, side1, side2):
    """W(side 1) - W(side 2) per event (ps) and per side the clamp flags."""
    w1, out1 = lut.correction(side1)
    w2, out2 = lut.correction(side2)
    return w1 - w2, out1, out2


def fit_two_branch(side1, side2, dt, near, far, train, bins, *, rounds, core_ps, iterations, minimum, manifest,
                   table):
    """``fit_tables`` against the nearer of two expected ``dt`` branches, re-chosen after each walk update
    (``rounds`` times, from no walk). Returns the last LUT, its per-event ``W1 - W2`` and per round
    {near_fraction, fit}, plus per round the events whose branch changed from the previous round."""
    model = np.zeros(len(dt))
    history, flips, previous = [], [], None
    for _ in range(rounds):
        branch = np.where(np.abs(dt - model - near) <= np.abs(dt - model - far), near, far)
        residual = dt - branch
        lut, fit_history = fit_tables(side1, side2, residual, train, bins, core_ps=core_ps, iterations=iterations,
                                      minimum=minimum, manifest=manifest, table=table)
        model, _, _ = walk_difference(lut, side1, side2)
        history.append({"near_fraction": float(np.mean(branch == near)), "fit": fit_history})
        flips.append(None if previous is None else int(np.count_nonzero(branch != previous)))
        previous = branch
    return lut, model, history, flips


def check_pairing(skew_lut, skew_sha256, walk_lut):
    """A walk LUT is applied with the skew LUT it was fitted with, or with a skew LUT solved with it (round
    trip: skew re-solved with walk 1, walk 2 refitted with that skew); a skew LUT solved with a walk LUT is
    never applied without walk."""
    skew_walk = (skew_lut.manifest.get("provenance") or {}).get("walk_lut_sha256")
    if walk_lut is None:
        if skew_walk:
            raise WalkError(f"the skew LUT was solved with walk LUT {skew_walk[:12]}...: apply a walk LUT paired with it")
        return
    if walk_lut.manifest.get("skew_lut_sha256") != skew_sha256 and walk_lut.sha256 != skew_walk:
        raise WalkError(f"walk LUT {walk_lut.sha256[:12]}... was fitted with skew LUT "
                        f"{str(walk_lut.manifest.get('skew_lut_sha256'))[:12]}..., not with {skew_sha256[:12]}..., "
                        "and that skew LUT was not solved with it")


def compare_tables(first, second, side1, side2):
    """Per-event difference of two walk LUTs' corrections W(1) - W(2) on the same events (ps): mean and rms."""
    a, _, _ = walk_difference(first, side1, side2)
    b, _, _ = walk_difference(second, side1, side2)
    d = (a - b)[np.isfinite(a) & np.isfinite(b)]
    return {"events": int(len(d)), "mean_ps": float(d.mean()), "rms_ps": float(np.sqrt(np.mean(d ** 2))),
            "p95_abs_ps": float(np.percentile(np.abs(d), 95))}
