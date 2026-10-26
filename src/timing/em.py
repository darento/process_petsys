"""Global EM estimator: one delay per TIME channel from every observation (spec 011 T9, FR-7, FR-8).

Model (spec 008's two-branch shell mixture, all channels at once): an
observation of channels ``i < j`` has ``x = dt - (b_i - b_j)`` with density

    p(x) = (1 - f_bg) [w K(x; near) + (1 - w) K(x; far)] + f_bg / window

where ``K`` is a Gaussian of width ``sigma`` averaged uniformly over the
segment's TOF interval. ``sigma``, ``f_bg`` and ``w`` are global.

Iteration (owner decision 2026-10-26):

* **E-step:** branch and background responsibilities per observation and the
  truncated-normal posterior of the TOF inside each segment. Their score
  ``d ln p / dx = sum_b r_b (E_b[t] - x) / sigma^2`` is exact.
* **M-step:** ``sigma``, ``f_bg`` and ``w`` in closed form (EM). Then one
  weighted least-squares (scoring) step for the delays on the group graph. It
  uses the moment-matched weights ``sum_b r_b / (sigma^2 + L_b^2 / 12)`` and is
  solved by preconditioned conjugate gradients with zero mean per connected
  component. Its fixed point is the exact likelihood maximum. A step that
  lowers the likelihood is halved.
* **Data-only coarse start:** delays are first solved per SM with ``sigma``
  annealed from wide to free, then per minimodule, then per channel; each
  stage starts from the previous one.
* **Alias test** after each stage: every group is shifted by plus or minus its
  mean branch separation, others held fixed. A better alias is taken. A group
  whose best alias is within ``min_alias_margin`` log-likelihood units is
  ``ambiguous``.
* **Uncertainty:** the diagonal of the pseudo-inverse of the information
  Laplacian built from the per-observation squared scores (outer product of
  gradients). Exact (dense) up to ``dense_limit`` channels per component,
  otherwise Hutchinson's stochastic diagonal with ``probes`` Rademacher
  vectors.

Statuses and the output form are those of ``src.timing.solver``:

* Only the largest connected component is applied; the others are
  ``disconnected``.
* A channel with fewer than ``min_events`` observations is ``insufficient
  statistics``.
* A channel with no observation is ``unobserved``.
* Missing delays are NaN, never 0 or a sentinel.

Observations are held as chunks of ``(pair id, dt, 4 TOF bounds)``.
``EmData.from_observation_set`` can write them as memory-mapped ``.npy``
files, so the process holds per-channel and per-pair arrays, not per-event
ones.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from pathlib import Path

import numba
import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from src.timing import solver as ts

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
_TINY = 1e-300
STAT_FIELDS = ("log_likelihood", "near", "far", "background", "square", "observations", "underflow",
               "scale_numerator", "scale_denominator")


@dataclass(frozen=True)
class EmOptions:
    sigma_ps: float = 250.0                 # start (and fixed value when not fitted)
    background_fraction: float = 0.1
    branch_fraction: float = 0.5
    window_ps: float = 20000.0              # flat background support (IMAS: |dt| < 10 ns)
    fit_sigma: bool = True
    fit_background: bool = True
    fit_branch_fraction: bool = True
    separation_scale: float = 1.0           # global branch-separation scale (sign study, spec 011 T11)
    fit_separation_scale: bool = False
    anneal_ps: tuple = (4000.0, 2000.0, 1000.0, 500.0)
    tolerance_ps: float = 0.01
    max_iterations: int = 300
    anneal_iterations: int = 50             # an anneal stage only brings the delays into the basin
    max_halvings: int = 8
    alias_rounds: int = 3
    min_alias_margin: float = 3.0
    min_events: int = 10
    precision_gate_ps: float = 10.0
    cg_tolerance: float = 1e-10
    dense_limit: int = 3000
    probes: int = 64
    seed: int = 0


# --- data -------------------------------------------------------------------------------------------------------

class EmData:
    """Observations as chunks over a fixed channel list and pair table (local channel indices, i < j)."""

    def __init__(self, channels, pair_i, pair_j, chunks):
        self.channels = np.asarray(channels, np.int64)
        self.pair_i = np.asarray(pair_i, np.int64)
        self.pair_j = np.asarray(pair_j, np.int64)
        self.chunks = chunks
        self.pair_events = np.zeros(len(self.pair_i), np.int64)
        for pid, _, _ in chunks:
            self.pair_events += np.bincount(pid, minlength=len(self.pair_i))
        self.channel_events = (np.bincount(self.pair_i, self.pair_events, len(self.channels))
                               + np.bincount(self.pair_j, self.pair_events, len(self.channels))).astype(np.int64)

    @property
    def observations(self):
        return int(self.pair_events.sum())

    @staticmethod
    def _local(channels, values, name):
        position = np.searchsorted(channels, values)
        position = np.minimum(position, len(channels) - 1)
        if len(values) and (channels[position] != values).any():
            raise ValueError(f"{name} outside the declared TIME-channel list")
        return position

    @classmethod
    def from_arrays(cls, channel_i, channel_j, dt_ps, shell_tof_ps, channels=None, dtype=np.float64):
        """One in-memory chunk; endpoints are made canonical (lower channel first)."""
        ci, cj = np.asarray(channel_i, np.int64), np.asarray(channel_j, np.int64)
        dt = np.asarray(dt_ps, np.float64).copy()
        tof = np.asarray(shell_tof_ps, np.float64).copy()
        if (ci == cj).any():
            raise ValueError("an observation needs two different channels")
        swap = ci > cj
        ci, cj = np.where(swap, cj, ci), np.where(swap, ci, cj)
        dt[swap], tof[swap] = -dt[swap], -tof[swap][:, ::-1]
        channels = np.unique(np.r_[ci, cj]) if channels is None else np.unique(np.asarray(channels, np.int64))
        li, lj = cls._local(channels, ci, "channel_i"), cls._local(channels, cj, "channel_j")
        keys, pid = np.unique(li * len(channels) + lj, return_inverse=True)
        chunk = (pid.astype(np.int32), dt.astype(dtype), np.ascontiguousarray(tof, dtype))
        return cls(channels, keys // len(channels), keys % len(channels), [chunk])

    @classmethod
    def from_observation_set(cls, observations, channels, *, roles=(0,), where=None, directory=None,
                             dtype=np.float32):
        """One chunk per pair partition (pairs never span partitions); ``directory`` keeps them memory-mapped.

        ``where(rows)`` returns a mask restricting the rows further (a held-out split, common events).
        """
        channels = np.unique(np.asarray(channels, np.int64))
        pair_i, pair_j, chunks, offset = [], [], [], 0
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=False)
        for p in range(observations.partitions):
            rows = observations.partition(p)
            keep = np.isin(rows["role"], roles)
            rows = rows[keep & where(rows) if where is not None else keep]
            li = cls._local(channels, rows["channel_i"].astype(np.int64), "channel_i")
            lj = cls._local(channels, rows["channel_j"].astype(np.int64), "channel_j")
            if (li >= lj).any():
                raise ValueError("observations must be canonical (channel_i < channel_j)")
            keys, pid = np.unique(li * len(channels) + lj, return_inverse=True)
            pair_i.append(keys // len(channels))
            pair_j.append(keys % len(channels))
            arrays = ((pid + offset).astype(np.int32), rows["dt_ps"].astype(dtype),
                      np.ascontiguousarray(rows["shell_tof_ps"], dtype))
            offset += len(keys)
            if directory is not None:
                names = [directory / f"p{p:03d}_{name}.npy" for name in ("pair", "dt", "tof")]
                for name, array in zip(names, arrays):
                    np.save(name, array)
                arrays = tuple(np.load(name, mmap_mode="r") for name in names)
            chunks.append(arrays)
        return cls(channels, np.concatenate(pair_i) if pair_i else np.empty(0, np.int64),
                   np.concatenate(pair_j) if pair_j else np.empty(0, np.int64), chunks)


# --- kernels ----------------------------------------------------------------------------------------------------

@numba.njit(cache=True)
def _moments(x, a, c, sigma):
    """Segment-averaged kernel K(x) and the mean and variance of the TOF t in [a, c] given x."""
    width = c - a
    if width < 1e-3:
        z = (x - a) / sigma
        return math.exp(-0.5 * z * z) / (sigma * _SQRT2PI), 0.5 * (a + c), width * width / 12.0
    al = (a - x) / sigma
    be = (c - x) / sigma
    if al > 0.0:
        z = 0.5 * (math.erfc(al / _SQRT2) - math.erfc(be / _SQRT2))
    elif be < 0.0:
        z = 0.5 * (math.erfc(-be / _SQRT2) - math.erfc(-al / _SQRT2))
    else:
        z = 0.5 * (math.erf(be / _SQRT2) - math.erf(al / _SQRT2))
    if not z > _TINY:
        return 0.0, (a if x < a else c), 0.0
    pa = math.exp(-0.5 * al * al) / _SQRT2PI
    pb = math.exp(-0.5 * be * be) / _SQRT2PI
    q = (pa - pb) / z
    mean = min(c, max(a, x + sigma * q))
    variance = max(0.0, sigma * sigma * (1.0 + (al * pa - be * pb) / z - q * q))
    return z / width, mean, variance


@numba.njit(cache=True)
def _density(x, tof, e, sigma, fb, w, flat, scale):
    d = (scale - 1.0) * 0.25 * (tof[e, 2] + tof[e, 3] - tof[e, 0] - tof[e, 1])
    kn = _moments(x, tof[e, 0] - d, tof[e, 1] - d, sigma)[0]
    kf = _moments(x, tof[e, 2] + d, tof[e, 3] + d, sigma)[0]
    return max(_TINY, (1.0 - fb) * (w * kn + (1.0 - w) * kf) + flat)


@numba.njit(cache=True)
def _estep(pid, dt, tof, pi, pj, b, sigma, fb, w, scale, window, grad, wpair, spair, score, csq, cn, stats):
    flat = fb / window
    s2 = sigma * sigma
    for e in range(len(pid)):
        p = pid[e]
        i = pi[p]
        j = pj[p]
        x = dt[e] - (b[i] - b[j])
        h = 0.25 * (tof[e, 2] + tof[e, 3] - tof[e, 0] - tof[e, 1])
        d = (scale - 1.0) * h
        kn, mn, vn = _moments(x, tof[e, 0] - d, tof[e, 1] - d, sigma)
        kf, mf, vf = _moments(x, tof[e, 2] + d, tof[e, 3] + d, sigma)
        an = (1.0 - fb) * w * kn
        af = (1.0 - fb) * (1.0 - w) * kf
        total = an + af + flat
        if not total > _TINY:
            stats[6] += 1.0
            stats[0] += math.log(_TINY)
            continue
        rn = an / total
        rf = af / total
        stats[0] += math.log(total)
        stats[1] += rn
        stats[2] += rf
        stats[3] += flat / total
        square = rn * ((x - mn) ** 2 + vn) + rf * ((x - mf) ** 2 + vf)
        stats[4] += square
        stats[5] += 1.0
        s = (rn * (mn - x) + rf * (mf - x)) / s2
        grad[i] -= s
        grad[j] += s
        ln = tof[e, 1] - tof[e, 0]
        lf = tof[e, 3] - tof[e, 2]
        wpair[p] += rn / (s2 + ln * ln / 12.0) + rf / (s2 + lf * lf / 12.0)
        spair[p] += s * s
        score[p] += s
        csq[i] += square / s2
        csq[j] += square / s2
        cn[i] += rn + rf
        cn[j] += rn + rf
        stats[7] += h * (rn * (x - mn) - rf * (x - mf))
        stats[8] += h * h * (rn + rf)


@numba.njit(cache=True)
def _alias(pid, dt, tof, pi, pj, b, group, shift, sigma, fb, w, scale, window, plus, minus):
    flat = fb / window
    for e in range(len(pid)):
        p = pid[e]
        gi = group[pi[p]]
        gj = group[pj[p]]
        if gi == gj:
            continue
        x = dt[e] - (b[pi[p]] - b[pj[p]])
        base = math.log(_density(x, tof, e, sigma, fb, w, flat, scale))
        plus[gi] += math.log(_density(x - shift[gi], tof, e, sigma, fb, w, flat, scale)) - base
        minus[gi] += math.log(_density(x + shift[gi], tof, e, sigma, fb, w, flat, scale)) - base
        plus[gj] += math.log(_density(x + shift[gj], tof, e, sigma, fb, w, flat, scale)) - base
        minus[gj] += math.log(_density(x - shift[gj], tof, e, sigma, fb, w, flat, scale)) - base


@numba.njit(cache=True)
def _separation(pid, tof, pi, pj, group, total, count):
    for e in range(len(pid)):
        p = pid[e]
        gi = group[pi[p]]
        gj = group[pj[p]]
        if gi == gj:
            continue
        s = 0.5 * (tof[e, 2] + tof[e, 3] - tof[e, 0] - tof[e, 1])
        total[gi] += s
        total[gj] += s
        count[gi] += 1.0
        count[gj] += 1.0


# --- one pass, linear algebra -----------------------------------------------------------------------------------

@dataclass
class _Theta:
    sigma: float
    background: float
    branch: float
    scale: float = 1.0


@dataclass
class _Pass:
    stats: dict
    grad: np.ndarray
    wpair: np.ndarray
    spair: np.ndarray
    score: np.ndarray
    channel_square: np.ndarray
    channel_weight: np.ndarray


def _estep_all(data, b, theta, options):
    n, m = len(data.channels), len(data.pair_i)
    grad, wpair, spair, score = np.zeros(n), np.zeros(m), np.zeros(m), np.zeros(m)
    csq, cn, stats = np.zeros(n), np.zeros(n), np.zeros(len(STAT_FIELDS))
    for pid, dt, tof in data.chunks:
        _estep(pid, dt, tof, data.pair_i, data.pair_j, b, theta.sigma, theta.background, theta.branch, theta.scale,
               float(options.window_ps), grad, wpair, spair, score, csq, cn, stats)
    return _Pass(dict(zip(STAT_FIELDS, stats.tolist())), grad, wpair, spair, score, csq, cn)


def _laplacian(u, v, weight, n):
    adjacency = sp.coo_matrix((np.r_[weight, weight], (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sp.diags(degree) - adjacency).tocsr(), degree


def _center(x, labels, members=None):
    """Subtract each component's mean (over ``members`` when given)."""
    mask = np.ones(len(x), bool) if members is None else members
    total = np.bincount(labels[mask], x[mask], labels.max() + 1 if len(labels) else 0)
    count = np.bincount(labels[mask], minlength=labels.max() + 1 if len(labels) else 0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return x - mean[labels]


def _pcg(matrix, rhs, degree, labels, tolerance, max_iterations=None):
    """Jacobi-preconditioned CG on a Laplacian; right side and solution with zero mean per component."""
    rhs = _center(rhs, labels)
    x = np.zeros_like(rhs)
    norm = float(np.linalg.norm(rhs))
    if norm == 0.0:
        return x, 0
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    r = rhs.copy()
    z = inverse * r
    p = z.copy()
    rz = float(r @ z)
    iterations = 0
    for iterations in range(1, (max_iterations or 10 * len(rhs) + 100) + 1):
        ap = matrix @ p
        curvature = float(p @ ap)
        if not curvature > 0:
            break
        alpha = rz / curvature
        x += alpha * p
        r -= alpha * ap
        if np.linalg.norm(r) <= tolerance * norm:
            break
        z = inverse * r
        new = float(r @ z)
        p = z + (new / rz) * p
        rz = new
    return _center(x, labels), iterations


def _components(u, v, n):
    graph = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


# --- stages -----------------------------------------------------------------------------------------------------

def _stage(data, b, theta, group, n_groups, options, *, fixed_sigma=None, free=True, label="",
           max_iterations=None, progress=None):
    """Iterate E-step + M-step at one grouping until delays and global parameters settle.

    ``progress(text)`` gets one line per iteration and one at the end of the stage."""
    gu, gv = group[data.pair_i], group[data.pair_j]
    inter = gu != gv
    labels = _components(gu[inter], gv[inter], n_groups)
    if fixed_sigma is not None:
        theta = _Theta(fixed_sigma, theta.background, theta.branch, theta.scale)
    previous, b_previous, step, halvings, converged = -np.inf, b, None, 0, False
    iterations = 0
    record = None
    started = time.perf_counter()
    limit = max_iterations or options.max_iterations
    for iterations in range(1, limit + 1):
        current = _estep_all(data, b, theta, options)
        ll = current.stats["log_likelihood"]
        if step is not None and ll < previous - 1e-9 * max(1.0, abs(previous)) and halvings < options.max_halvings:
            step = 0.5 * step
            b = b_previous + step[group]
            halvings += 1
            continue
        halvings = 0
        branch_events = current.stats["near"] + current.stats["far"]
        new = _Theta(theta.sigma, theta.background, theta.branch, theta.scale)
        if free and fixed_sigma is None and options.fit_sigma and branch_events > 0:
            new.sigma = math.sqrt(current.stats["square"] / branch_events)
        if free and options.fit_background and current.stats["observations"] > 0:
            new.background = current.stats["background"] / current.stats["observations"]
        if free and options.fit_branch_fraction and branch_events > 0:
            new.branch = current.stats["near"] / branch_events
        if free and options.fit_separation_scale and current.stats["scale_denominator"] > 0:
            # exact EM step: the segments move rigidly with the scale
            new.scale = theta.scale - current.stats["scale_numerator"] / current.stats["scale_denominator"]
        matrix, degree = _laplacian(gu[inter], gv[inter], current.wpair[inter], n_groups)
        delta, cg_iterations = _pcg(matrix, np.bincount(group, current.grad, n_groups), degree, labels,
                                    options.cg_tolerance)
        settled = (np.abs(delta).max(initial=0.0) < options.tolerance_ps
                   and abs(new.sigma - theta.sigma) < options.tolerance_ps
                   and abs(new.background - theta.background) < 1e-6 and abs(new.branch - theta.branch) < 1e-6
                   and abs(new.scale - theta.scale) < 1e-6)
        record = {"stage": label, "iterations": iterations, "log_likelihood": ll, "sigma_ps": theta.sigma,
                  "background_fraction": theta.background, "branch_fraction": theta.branch,
                  "separation_scale": theta.scale, "last_step_ps": float(np.abs(delta).max(initial=0.0)), "cg_iterations": cg_iterations,
                  "underflow": current.stats["underflow"]}
        if progress is not None:
            progress(f"{label} {iterations}/{limit}: log-likelihood {ll:.1f}, sigma {theta.sigma:.1f} ps, "
                     f"background {theta.background:.3f}, max step {record['last_step_ps']:.3f} ps, "
                     f"{time.perf_counter() - started:.0f} s")
        theta = new
        if settled:
            converged = True
            break
        previous, b_previous, step = ll, b, delta
        b = b + delta[group]
    record = dict(record or {"stage": label, "iterations": iterations}, converged=converged)
    if progress is not None:
        progress(f"{label}: {'converged' if converged else 'stopped at the iteration limit'} after {iterations} "
                 f"iterations ({time.perf_counter() - started:.0f} s)")
    return b, theta, record


def _alias_test(data, b, theta, group, n_groups, options):
    total, count = np.zeros(n_groups), np.zeros(n_groups)
    for pid, _, tof in data.chunks:
        _separation(pid, tof, data.pair_i, data.pair_j, group, total, count)
    shift = theta.scale * np.divide(total, count, out=np.zeros(n_groups), where=count > 0)
    plus, minus = np.zeros(n_groups), np.zeros(n_groups)
    for pid, dt, tof in data.chunks:
        _alias(pid, dt, tof, data.pair_i, data.pair_j, b, group, shift, theta.sigma, theta.background, theta.branch,
               theta.scale,
               float(options.window_ps), plus, minus)
    return shift, plus, minus, count > 0


def _stage_with_alias(data, b, theta, group, n_groups, options, label, history, progress=None):
    b, theta, record = _stage(data, b, theta, group, n_groups, options, label=label, progress=progress)
    history.append(record)
    moved = 0
    for _ in range(options.alias_rounds):
        shift, plus, minus, seen = _alias_test(data, b, theta, group, n_groups, options)
        gain = np.maximum(plus, minus)
        better = seen & (gain > 1e-6 * max(1.0, abs(record.get("log_likelihood", 1.0))) / max(1, n_groups)) & (gain > 0.5)
        if not better.any():
            break
        before = _estep_all(data, b, theta, options).stats["log_likelihood"]
        trial = b + np.where(better, np.where(plus >= minus, shift, -shift), 0.0)[group]
        trial, trial_theta, trial_record = _stage(data, trial, theta, group, n_groups, options,
                                                  label=label + " (alias moved)", progress=progress)
        if trial_record.get("log_likelihood", -np.inf) <= before:
            break
        b, theta, record = trial, trial_theta, trial_record
        history.append(record)
        moved += int(better.sum())
    shift, plus, minus, seen = _alias_test(data, b, theta, group, n_groups, options)
    margin = np.where(seen, -np.maximum(plus, minus), np.nan)
    history[-1] = dict(history[-1], alias_moved_groups=moved,
                       ambiguous_groups=int((seen & (margin < options.min_alias_margin)).sum()),
                       mean_branch_separation_ps=float(np.mean(shift[seen])) if seen.any() else None)
    return b, theta, margin


# --- uncertainty ------------------------------------------------------------------------------------------------

def _pinv_diagonal(matrix, degree, labels, options, rng):
    """diag(pinv(L)) per component: dense when small, Hutchinson otherwise; returns (values, methods)."""
    out = np.full(matrix.shape[0], np.nan)
    methods = {}
    large = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) == 1:
            continue
        if len(members) <= options.dense_limit:
            # pinvh also covers a vanishing score (noiseless data): zero information gives zero variance there
            out[members] = np.clip(np.diag(scipy.linalg.pinvh(matrix[members][:, members].toarray())), 0, None)
            methods["dense"] = methods.get("dense", 0) + 1
        else:
            large.append(members)
    if large:
        mask = np.zeros(len(out), bool)
        for members in large:
            mask[members] = True
        total = np.zeros(len(out))
        for _ in range(options.probes):
            z = np.where(mask, rng.choice((-1.0, 1.0), len(out)), 0.0)
            x, _ = _pcg(matrix, z, degree, labels, options.cg_tolerance)
            total += z * x
        out[mask] = np.clip(total[mask] / options.probes, 0, None)
        methods["hutchinson"] = {"components": len(large), "probes": options.probes}
    return out, methods


# --- driver -----------------------------------------------------------------------------------------------------

def solve_em(data, coarse=(), options=EmOptions(), *, progress=None):
    """Channel delays (``src.timing.solver.Solution``) from an ``EmData`` set.

    ``coarse`` lists ``(name, labels)`` groupings applied before the channel
    stage, coarsest first; ``labels`` gives each of ``data.channels`` its group
    (for IMAS: SM, then SM x 16 + minimodule). ``sigma`` is annealed at the first
    stage only. ``progress(text)`` receives a line per iteration.
    """
    n = len(data.channels)
    theta = _Theta(float(options.sigma_ps), float(options.background_fraction), float(options.branch_fraction),
                   float(options.separation_scale))
    b = np.zeros(n)
    history = []
    margins = {}
    stages = list(coarse) + [("channel", np.arange(n))]
    for number, (name, labels) in enumerate(stages):
        labels = np.asarray(labels)
        if len(labels) != n:
            raise ValueError(f"grouping {name!r} must label every channel")
        _, group = np.unique(labels, return_inverse=True)
        n_groups = int(group.max()) + 1 if n else 0
        if number == 0:
            for width in options.anneal_ps:
                if options.fit_sigma and width <= options.sigma_ps:
                    continue
                b, kept, record = _stage(data, b, theta, group, n_groups, options, fixed_sigma=float(width),
                                         free=False, label=f"{name} anneal {width:g} ps",
                                         max_iterations=options.anneal_iterations, progress=progress)
                record["anneal"] = True
                history.append(record)
        b, theta, margin = _stage_with_alias(data, b, theta, group, n_groups, options, name, history, progress)
        margins[name] = margin
    final = _estep_all(data, b, theta, options)
    return _solution(data, b, theta, final, margins["channel"], history, options)


def _solution(data, b, theta, final, alias_margin, history, options):
    n = len(data.channels)
    observed = data.pair_events > 0
    labels = _components(data.pair_i[observed], data.pair_j[observed], n)
    events = data.channel_events
    sizes = np.bincount(labels, events > 0, labels.max() + 1 if n else 0)
    order = np.lexsort((np.arange(len(sizes)), -sizes))
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    component = np.where(events > 0, rank[labels], -1)

    status = np.full(n, ts.UNOBSERVED, np.int8)
    seen = events > 0
    status[seen & (component == 0)] = ts.CALIBRATED
    status[seen & (component > 0)] = ts.DISCONNECTED
    status[seen & (events < options.min_events)] = ts.INSUFFICIENT
    aliased = seen & ~(alias_margin >= options.min_alias_margin)
    # disconnected channels are never applied; their status stays disconnected (solver precedence)
    status[aliased & (component == 0)] = ts.AMBIGUOUS
    gauged = np.isin(status, (ts.CALIBRATED, ts.DISCONNECTED))
    delay = _center(b, np.where(component >= 0, component, 0), gauged & (component >= 0))

    matrix, degree = _laplacian(data.pair_i[observed], data.pair_j[observed], final.spair[observed], n)
    rng = np.random.default_rng(options.seed)
    variance, methods = _pinv_diagonal(matrix, degree, labels, options, rng)
    sigma = np.sqrt(variance)
    status[(status == ts.CALIBRATED) & ~(sigma <= options.precision_gate_ps)] = ts.LOW_STATISTICS

    channels = np.zeros(n, ts.CHANNEL_DTYPE)
    channels["time_channel"] = data.channels
    channels["delay_ps"], channels["sigma_ps"], channels["raw_sigma_ps"] = delay, sigma, sigma
    channels["status"], channels["component"] = status, component
    channels["apply"] = np.isin(status, ts.APPLIED_STATUSES)
    pairs_per_channel = np.bincount(data.pair_i[observed], minlength=n) + np.bincount(data.pair_j[observed],
                                                                                         minlength=n)
    channels["pairs"] = channels["usable_pairs"] = pairs_per_channel
    channels["usable_events"] = events
    channels["residual_rms"] = np.sqrt(np.divide(final.channel_square, final.channel_weight,
                                                 out=np.full(n, np.nan), where=final.channel_weight > 0))
    unsolved = ~np.isin(status, ts.APPLIED_STATUSES + (ts.DISCONNECTED,))
    for name in ("delay_ps", "sigma_ps", "raw_sigma_ps"):
        channels[name][unsolved] = np.nan

    index = np.flatnonzero(observed)
    pairs = np.zeros(len(index), ts.PAIR_RESIDUAL_DTYPE)
    pi, pj = data.pair_i[index], data.pair_j[index]
    pairs["channel_i"], pairs["channel_j"] = data.channels[pi], data.channels[pj]
    pairs["component"] = component[pi]
    pairs["shift_ps"] = b[pi] - b[pj]
    information = final.spair[index]
    pairs["shift_sigma_ps"] = np.divide(1.0, np.sqrt(information), out=np.full(len(index), np.nan),
                                        where=information > 0)
    pairs["residual_ps"] = np.divide(final.score[index], final.wpair[index], out=np.full(len(index), np.nan),
                                     where=final.wpair[index] > 0)
    pairs["normalized_residual"] = np.divide(final.score[index], np.sqrt(information),
                                             out=np.full(len(index), np.nan), where=information > 0)
    pairs["huber_weight"] = 1.0
    diagnostics = {
        "estimator": "global EM (spec 011 FR-7)",
        "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(options).items()},
        "model": {"sigma_ps": theta.sigma, "background_fraction": theta.background, "branch_fraction": theta.branch,
                  "separation_scale": theta.scale, "window_ps": options.window_ps},
        "log_likelihood": final.stats["log_likelihood"], "observations": data.observations,
        "pairs": int(observed.sum()), "history": history,
        "gauge": "zero mean over each component's calibrated / low-statistics channels",
        "uncertainty": {"method": "diag pinv of the outer-product-of-scores Laplacian", **methods},
        "components": int(sizes.size), "largest_component_channels": int(sizes.max(initial=0)),
        "status_counts": {name: int((status == k).sum()) for k, name in enumerate(ts.CHANNEL_STATUSES)},
        "aliased_disconnected_channels": int((aliased & (component > 0)).sum()),
        "converged": all(record.get("converged", False) for record in history if not record.get("anneal")),
    }
    return ts.Solution(channels, pairs, diagnostics)


def evaluate(data, delays_ps, options=EmOptions(), *, max_iterations=500, channel_shifts=False):
    """Global parameters fitted with fixed delays (held-out check): sigma, background, branch fraction.

    ``delays_ps`` (one per ``data.channels``) must be finite for every observed channel; a smaller ``sigma``
    and a higher log-likelihood per event mean better delays on the same events. ``channel_shifts`` adds,
    per channel, the one-step delay change these events ask for with every other delay held fixed
    (``shift_ps``) and its uncertainty (``shift_sigma_ps``, from the squared scores): a diagnostic of
    residual consistency, not a fit.
    """
    b = np.asarray(delays_ps, np.float64)
    if not np.isfinite(b[data.channel_events > 0]).all():
        raise ValueError("held-out evaluation needs a finite delay for every observed channel")
    b = np.where(np.isfinite(b), b, 0.0)
    theta = _Theta(float(options.sigma_ps), float(options.background_fraction), float(options.branch_fraction),
                   float(options.separation_scale))
    iterations, stats = 0, {}
    for iterations in range(1, max_iterations + 1):
        stats = _estep_all(data, b, theta, options).stats
        branch_events = stats["near"] + stats["far"]
        if not (branch_events > 0 and stats["observations"] > 0):
            break
        new = _Theta(math.sqrt(stats["square"] / branch_events) if options.fit_sigma else theta.sigma,
                     stats["background"] / stats["observations"] if options.fit_background else theta.background,
                     stats["near"] / branch_events if options.fit_branch_fraction else theta.branch, theta.scale)
        settled = (abs(new.sigma - theta.sigma) < options.tolerance_ps and abs(new.background - theta.background) < 1e-7
                   and abs(new.branch - theta.branch) < 1e-7)
        theta = new
        if settled:
            break
    observations = stats.get("observations", 0.0)
    result = {"observations": int(observations), "sigma_ps": theta.sigma, "background_fraction": theta.background,
              "branch_fraction": theta.branch, "log_likelihood": stats.get("log_likelihood"),
              "log_likelihood_per_event": stats["log_likelihood"] / observations if observations else None,
              "iterations": iterations}
    if channel_shifts:
        final = _estep_all(data, b, theta, options)
        n = len(data.channels)
        degree = np.bincount(data.pair_i, final.wpair, n) + np.bincount(data.pair_j, final.wpair, n)
        information = np.bincount(data.pair_i, final.spair, n) + np.bincount(data.pair_j, final.spair, n)
        result["shift_ps"] = np.divide(final.grad, degree, out=np.full(n, np.nan), where=degree > 0)
        result["shift_sigma_ps"] = np.divide(1.0, np.sqrt(information), out=np.full(n, np.nan),
                                             where=information > 0)
    return result
