"""Two-branch shell constraints per TIME-channel pair (spec 008 T6, FR-9).

Each canonical observation gives ``dt = t_i - t_j = tof + d`` with
``d = b_i - b_j``. Its LOR crosses the annular shell in two segments (near
and far along i -> j); the event is born in one of them. Owner decisions
(2026-10-13): per channel pair, an unbinned likelihood keeps every event's
own segment TOF intervals (no geometry compression), so

    p(dt) = (1 - f_bg) [w K(dt - d; near) + (1 - w) K(dt - d; far)] + f_bg / window

where ``K`` is a Gaussian timing kernel of global width ``sigma`` averaged
uniformly over the segment interval (finite wall thickness, axial clipping),
``w`` is a free branch fraction (unequal populations do not move ``d``) and
the flat background covers the legacy ``|dt| < 15 ns`` window. ``sigma`` and
``f_bg`` are global and fixed in the pair fits.

``d`` is scanned over the window, so the ±separation alias is resolved by the
data: a pair is usable only when its branches are resolved (median branch
separation at least ``min_separation_sigmas`` kernel widths), each branch
holds enough expected events and the best fit beats the best alias (a local
maximum at least half a branch separation away) by a log-likelihood margin.
Otherwise it is unavailable, never a zero shift. The scan uses an evenly strided subsample of at most
``scan_events`` events per pair; the leading candidates are refined with all
events, so cost grows linearly with larger acquisitions. No event is ever
assigned to its nearest branch.

The free-separation fit adds ``s``: the near interval moves by ``-s/2`` and
the far one by ``+s/2``. ``s`` is offset-free and tests geometry (handedness)
against measured peak separation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numba
import numpy as np
from scipy.optimize import minimize

from src.cornell.skew.ingest import LEGACY_COINCIDENCE_WINDOW_PS

STATUSES = ("usable", "one branch", "alias unresolved", "merged branches", "scan edge", "fit failed")
USABLE, ONE_BRANCH, ALIAS_UNRESOLVED, MERGED, SCAN_EDGE, FIT_FAILED = range(len(STATUSES))
PAIR_FIT_DTYPE = np.dtype([
    ("channel_i", "<i4"), ("channel_j", "<i4"), ("role", "i1"), ("events", "<i8"), ("status", "i1"),
    ("shift_ps", "<f8"), ("shift_sigma_ps", "<f8"), ("branch_fraction", "<f8"),
    ("near_events", "<f8"), ("far_events", "<f8"), ("alias_margin", "<f8"), ("log_likelihood", "<f8"),
    ("half_separation_ps", "<f8"), ("separation_ps", "<f8"), ("separation_sigma_ps", "<f8")])
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
_GOLDEN = 0.5 * (3.0 - math.sqrt(5.0))


@dataclass(frozen=True)
class PairModel:
    """Global timing kernel and flat background, fixed within pair fits."""
    sigma_ps: float
    background_fraction: float
    window_ps: float = 2.0 * LEGACY_COINCIDENCE_WINDOW_PS

    def __post_init__(self):
        if not (self.sigma_ps > 0 and 0 <= self.background_fraction < 1 and self.window_ps > 0):
            raise ValueError("pair model needs sigma > 0, background fraction in [0, 1) and window > 0")


@dataclass(frozen=True)
class FitOptions:
    min_branch_events: float = 3.0
    min_alias_margin: float = 3.0
    min_separation_sigmas: float = 2.0
    scan_ps: float = LEGACY_COINCIDENCE_WINDOW_PS - 1000.0
    step_ps: float = 25.0
    scan_events: int = 256
    candidates: int = 4


@numba.njit(cache=True)
def _kernel(x, a, b, sigma):
    if b - a < 1e-3:
        z = (x - a) / sigma
        return math.exp(-0.5 * z * z) / (sigma * _SQRT2PI)
    return 0.5 * (math.erf((x - a) / (_SQRT2 * sigma)) - math.erf((x - b) / (_SQRT2 * sigma))) / (b - a)


@numba.njit(cache=True)
def _branches(dt, tof, index, d, s, sigma, fb, near, far):
    for k in range(len(index)):
        e = index[k]
        x = dt[e] - d
        near[k] = (1.0 - fb) * _kernel(x, tof[e, 0] - 0.5 * s, tof[e, 1] - 0.5 * s, sigma)
        far[k] = (1.0 - fb) * _kernel(x, tof[e, 2] + 0.5 * s, tof[e, 3] + 0.5 * s, sigma)


@numba.njit(cache=True)
def _profile(dt, tof, index, d, s, sigma, fb, window, near, far):
    """ln L at shift d, separation change s, maximised over branch fraction w (concave in w)."""
    n = len(index)
    _branches(dt, tof, index, d, s, sigma, fb, near, far)
    flat = fb / window
    w = 0.5
    for _ in range(100):
        g = 0.0
        h = 0.0
        for k in range(n):
            u = (near[k] - far[k]) / (w * near[k] + (1.0 - w) * far[k] + flat)
            g += u
            h -= u * u
        if h == 0.0:
            break
        step = -g / h
        new = min(1.0, max(0.0, w + step))
        if abs(new - w) < 1e-10:
            w = new
            break
        w = new
    total = 0.0
    for k in range(n):
        total += math.log(w * near[k] + (1.0 - w) * far[k] + flat)
    return total, w


@numba.njit(cache=True)
def _golden_shift(dt, tof, index, lo, hi, s, sigma, fb, window, near, far):
    a, b = lo, hi
    c = b - (1.0 - _GOLDEN) * (b - a)
    e = a + (1.0 - _GOLDEN) * (b - a)
    fc = _profile(dt, tof, index, c, s, sigma, fb, window, near, far)[0]
    fe = _profile(dt, tof, index, e, s, sigma, fb, window, near, far)[0]
    while b - a > 0.01:
        if fc > fe:
            b, e, fe = e, c, fc
            c = b - (1.0 - _GOLDEN) * (b - a)
            fc = _profile(dt, tof, index, c, s, sigma, fb, window, near, far)[0]
        else:
            a, c, fc = c, e, fe
            e = a + (1.0 - _GOLDEN) * (b - a)
            fe = _profile(dt, tof, index, e, s, sigma, fb, window, near, far)[0]
    return 0.5 * (a + b)


@numba.njit(cache=True)
def _golden_separation(dt, tof, index, d, lo, hi, sigma, fb, window, near, far):
    a, b = lo, hi
    c = b - (1.0 - _GOLDEN) * (b - a)
    e = a + (1.0 - _GOLDEN) * (b - a)
    fc = _profile(dt, tof, index, d, c, sigma, fb, window, near, far)[0]
    fe = _profile(dt, tof, index, d, e, sigma, fb, window, near, far)[0]
    while b - a > 0.01:
        if fc > fe:
            b, e, fe = e, c, fc
            c = b - (1.0 - _GOLDEN) * (b - a)
            fc = _profile(dt, tof, index, d, c, sigma, fb, window, near, far)[0]
        else:
            a, c, fc = c, e, fe
            e = a + (1.0 - _GOLDEN) * (b - a)
            fe = _profile(dt, tof, index, d, e, sigma, fb, window, near, far)[0]
    return 0.5 * (a + b)


@numba.njit(cache=True)
def _curvature_sigma(dt, tof, index, d, s, which, sigma, fb, window, near, far):
    h = 2.0
    centre = _profile(dt, tof, index, d, s, sigma, fb, window, near, far)[0]
    if which == 0:
        plus = _profile(dt, tof, index, d + h, s, sigma, fb, window, near, far)[0]
        minus = _profile(dt, tof, index, d - h, s, sigma, fb, window, near, far)[0]
    else:
        plus = _profile(dt, tof, index, d, s + h, sigma, fb, window, near, far)[0]
        minus = _profile(dt, tof, index, d, s - h, sigma, fb, window, near, far)[0]
    c = (plus - 2.0 * centre + minus) / (h * h)
    return math.sqrt(-1.0 / c) if c < 0 else math.nan


@numba.njit(cache=True)
def _fit_one(dt, tof, start, stop, grid, step, sigma, fb, window, scan_events, candidates,
             min_branch, min_margin, min_separation, free_separation, out):
    n = stop - start
    everything = np.arange(start, stop)
    stride = max(1, (n + scan_events - 1) // scan_events)
    sample = everything[::stride]
    near = np.empty(n)
    far = np.empty(n)
    half = np.empty(n)
    for k in range(n):
        e = start + k
        half[k] = 0.25 * (tof[e, 2] + tof[e, 3] - tof[e, 0] - tof[e, 1])
    out[0] = np.median(half)
    scan = np.empty(len(grid))
    for g in range(len(grid)):
        scan[g] = _profile(dt, tof, sample, grid[g], 0.0, sigma, fb, window, near, far)[0]
    # leading local maxima of the subsample scan, refined with every event
    peaks = np.empty(0, np.int64)
    for g in range(len(grid)):
        left = scan[g - 1] if g > 0 else -np.inf
        right = scan[g + 1] if g + 1 < len(grid) else -np.inf
        if scan[g] >= left and scan[g] > right:
            peaks = np.append(peaks, g)
    order = np.argsort(-scan[peaks])
    best_d = math.nan
    best = -np.inf
    best_edge = False
    for q in range(min(candidates, len(peaks))):
        g = peaks[order[q]]
        d = _golden_shift(dt, tof, everything, grid[g] - step, grid[g] + step, 0.0, sigma, fb, window, near, far)
        value = _profile(dt, tof, everything, d, 0.0, sigma, fb, window, near, far)[0]
        if value > best:
            best, best_d, best_edge = value, d, g == 0 or g == len(grid) - 1
    alias = -np.inf
    for q in range(len(peaks)):
        g = peaks[order[q]]
        if abs(grid[g] - best_d) < out[0]:
            continue
        d = _golden_shift(dt, tof, everything, grid[g] - step, grid[g] + step, 0.0, sigma, fb, window, near, far)
        if abs(d - best_d) >= out[0]:
            alias = _profile(dt, tof, everything, d, 0.0, sigma, fb, window, near, far)[0]
            break
    value, w = _profile(dt, tof, everything, best_d, 0.0, sigma, fb, window, near, far)
    flat = fb / window
    n_near = 0.0
    n_far = 0.0
    for k in range(n):
        p = w * near[k] + (1.0 - w) * far[k] + flat
        n_near += w * near[k] / p
        n_far += (1.0 - w) * far[k] / p
    out[1] = best_d
    out[2] = _curvature_sigma(dt, tof, everything, best_d, 0.0, 0, sigma, fb, window, near, far)
    out[3] = w
    out[4] = n_near
    out[5] = n_far
    out[6] = value - alias
    out[7] = value
    out[9] = math.nan
    out[10] = math.nan
    if not math.isfinite(best_d):
        out[8] = FIT_FAILED
    elif 2.0 * out[0] < min_separation * sigma:
        out[8] = MERGED
    elif min(n_near, n_far) < min_branch:
        out[8] = ONE_BRANCH
    elif not out[6] >= min_margin:
        out[8] = ALIAS_UNRESOLVED
    elif best_edge:
        out[8] = SCAN_EDGE
    elif not (math.isfinite(value) and math.isfinite(out[2])):
        out[8] = FIT_FAILED
    else:
        out[8] = USABLE
        if free_separation:
            d, s = best_d, 0.0
            for _ in range(6):
                s = _golden_separation(dt, tof, everything, d, s - 300.0, s + 300.0, sigma, fb, window, near, far)
                d = _golden_shift(dt, tof, everything, d - 100.0, d + 100.0, s, sigma, fb, window, near, far)
            out[9] = s
            out[10] = _curvature_sigma(dt, tof, everything, d, s, 1, sigma, fb, window, near, far)


@numba.njit(cache=True, parallel=True)
def _fit_all(dt, tof, starts, grid, step, sigma, fb, window, scan_events, candidates, min_branch, min_margin,
             min_separation, free_separation, out):
    for p in numba.prange(len(starts) - 1):
        _fit_one(dt, tof, starts[p], starts[p + 1], grid, step, sigma, fb, window, scan_events, candidates,
                 min_branch, min_margin, min_separation, free_separation, out[p])


def _pair_keys(rows):
    for name in ("channel_i", "channel_j"):
        if len(rows) and not (0 <= rows[name].min() and rows[name].max() < 1 << 25):
            raise ValueError(f"{name} outside the supported TIME-channel id range")
    return (rows["role"].astype(np.int64) << 50) | (rows["channel_i"].astype(np.int64) << 25) | rows["channel_j"]


def _groups(rows):
    if len(rows) and not np.isfinite(rows["shell_tof_ps"]).all():
        raise ValueError("pair fits need two finite shell segments per observation")
    key = _pair_keys(rows)
    order = np.lexsort((rows["record"], rows["file"], key))
    rows = rows[order]
    _, starts = np.unique(key[order], return_index=True)
    return rows, np.append(starts, len(rows)).astype(np.int64)


def fit_pairs(rows, model, options=FitOptions(), *, free_separation=False):
    """One ``PAIR_FIT_DTYPE`` row per (role, channel_i, channel_j) in canonical observation rows."""
    rows, starts = _groups(rows)
    out = np.full((len(starts) - 1, 11), np.nan)
    if len(out):
        grid = np.arange(-options.scan_ps, options.scan_ps + 0.5 * options.step_ps, options.step_ps)
        _fit_all(rows["dt_ps"].astype(np.float64), np.ascontiguousarray(rows["shell_tof_ps"], np.float64),
                 starts, grid, float(options.step_ps), float(model.sigma_ps), float(model.background_fraction),
                 float(model.window_ps), int(options.scan_events), int(options.candidates),
                 float(options.min_branch_events), float(options.min_alias_margin),
                 float(options.min_separation_sigmas), bool(free_separation), out)
    result = np.zeros(len(out), PAIR_FIT_DTYPE)
    first = rows[starts[:-1]]
    for name in ("channel_i", "channel_j", "role"):
        result[name] = first[name]
    result["events"] = np.diff(starts)
    result["half_separation_ps"], result["shift_ps"], result["shift_sigma_ps"] = out[:, 0], out[:, 1], out[:, 2]
    result["branch_fraction"], result["near_events"], result["far_events"] = out[:, 3], out[:, 4], out[:, 5]
    result["alias_margin"], result["log_likelihood"] = out[:, 6], out[:, 7]
    result["status"] = out[:, 8].astype(np.int8)
    result["separation_ps"], result["separation_sigma_ps"] = out[:, 9], out[:, 10]
    unavailable = result["status"] != USABLE
    for name in ("shift_ps", "shift_sigma_ps", "separation_ps", "separation_sigma_ps"):
        result[name][unavailable] = np.nan  # insufficient information, never a zero shift
    return result


def fit_observation_set(observations, model, options=FitOptions(), *, roles=(0,), free_separation=False):
    """Pair fits over an observation set, one pair partition in memory at a time."""
    parts = []
    for p in range(observations.partitions):
        rows = observations.partition(p)
        parts.append(fit_pairs(rows[np.isin(rows["role"], roles)], model, options, free_separation=free_separation))
    return np.concatenate(parts) if parts else np.zeros(0, PAIR_FIT_DTYPE)


@numba.njit(cache=True)
def _total_profile(dt, tof, starts, shifts, sigma, fb, window):
    total = 0.0
    for p in range(len(starts) - 1):
        index = np.arange(starts[p], starts[p + 1])
        near, far = np.empty(len(index)), np.empty(len(index))
        total += _profile(dt, tof, index, shifts[p], 0.0, sigma, fb, window, near, far)[0]
    return total



def estimate_model(rows, start=PairModel(200.0, 0.1), options=FitOptions(), *, min_events=30, max_pairs=2000,
                   iterations=6):
    """Global kernel width and background from well-populated pairs (training rows only).

    Alternates pair shifts (full scan) with a joint (sigma, f_bg) likelihood
    maximum at fixed usable shifts, until both settle.
    """
    rows, starts = _groups(rows)
    counts = np.diff(starts)
    chosen = np.flatnonzero(counts >= min_events)[:max_pairs]
    if not len(chosen):
        raise ValueError(f"no pair has at least {min_events} events")
    subset = np.concatenate([rows[starts[p]:starts[p + 1]] for p in chosen])
    model, history = start, []
    for _ in range(iterations):
        fits = fit_pairs(subset, model, options)
        usable = fits["status"] == USABLE
        if not usable.any():
            raise ValueError("no usable pair for kernel estimation")
        sub = subset[np.isin(_pair_keys(subset), _pair_keys(fits[usable]))]
        sub, sub_starts = _groups(sub)
        dt, tof = sub["dt_ps"].astype(np.float64), np.ascontiguousarray(sub["shell_tof_ps"], np.float64)
        shifts = fits["shift_ps"][usable]

        def negative(x):
            return -_total_profile(dt, tof, sub_starts, shifts, math.exp(x[0]), 1 / (1 + math.exp(-x[1])),
                                   model.window_ps)

        x0 = [math.log(model.sigma_ps), math.log(model.background_fraction / (1 - model.background_fraction))]
        best = minimize(negative, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-3})
        new = PairModel(math.exp(best.x[0]), 1 / (1 + math.exp(-best.x[1])), model.window_ps)
        history.append({"sigma_ps": new.sigma_ps, "background_fraction": new.background_fraction,
                        "pairs": int(usable.sum()), "log_likelihood": float(-best.fun)})
        settled = abs(new.sigma_ps - model.sigma_ps) < 0.5 and abs(new.background_fraction - model.background_fraction) < 1e-3
        model = new
        if settled:
            break
    return model, history

