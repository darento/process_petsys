"""Sparse global TIME-channel delay solve from shell pair constraints (spec 008 T7, FR-10/FR-12).

Each usable T6 pair gives ``d_ij = b_i - b_j +- sigma_ij``. The incidence
graph (+1 for i, -1 for j) is split into connected components; each is solved
by weighted least squares with its own gauge. Owner decisions (2026-10-13):

* Huber reweighting (k = 3 normalized residuals) against contaminated pair
  constraints; it reduces to plain weighted least squares when every
  residual is within k.
* Zero-mean gauge over all channels that receive a delay in a component.
  Only the largest component is applied; others are ``disconnected``: their
  relative origin is unobservable and they are never globally aligned.
* Uncertainty from the gauge-fixed covariance, inflated per component by the
  Birge ratio sqrt(chi2/dof) when above 1; raw values are kept.
* Solved channels above the precision gate keep their delay with status
  ``low statistics`` and are applied.

Statuses are explicit. Missing delays are NaN in memory and never 0 or a
sentinel; correction code must check ``apply``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from src.cornell import timing_pairs as tp

CHANNEL_STATUSES = ("calibrated", "low statistics", "insufficient statistics", "disconnected", "ambiguous",
                    "failed fit", "unobserved", "hardware unavailable")
(CALIBRATED, LOW_STATISTICS, INSUFFICIENT, DISCONNECTED, AMBIGUOUS, FAILED_FIT, UNOBSERVED,
 HARDWARE_UNAVAILABLE) = range(len(CHANNEL_STATUSES))
APPLIED_STATUSES = (CALIBRATED, LOW_STATISTICS)
CHANNEL_DTYPE = np.dtype([
    ("time_channel", "<i4"), ("delay_ps", "<f8"), ("sigma_ps", "<f8"), ("raw_sigma_ps", "<f8"),
    ("status", "i1"), ("apply", "?"), ("component", "<i4"), ("usable_pairs", "<i8"), ("pairs", "<i8"),
    ("usable_events", "<i8"), ("residual_rms", "<f8")])
PAIR_RESIDUAL_DTYPE = np.dtype([
    ("channel_i", "<i4"), ("channel_j", "<i4"), ("component", "<i4"), ("shift_ps", "<f8"),
    ("shift_sigma_ps", "<f8"), ("residual_ps", "<f8"), ("normalized_residual", "<f8"), ("huber_weight", "<f8")])


@dataclass(frozen=True)
class SolveOptions:
    huber_k: float | None = 3.0
    precision_gate_ps: float = 10.0
    flag_normalized_residual: float = 5.0
    tolerance_ps: float = 1e-7
    max_iterations: int = 200


@dataclass
class Solution:
    channels: np.ndarray
    pairs: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def delays(self):
        """{channel: delay_ps} for applied channels only."""
        rows = self.channels[self.channels["apply"]]
        return dict(zip(rows["time_channel"].tolist(), rows["delay_ps"].tolist()))


def _solve_component(n, i, j, shift, sigma, options):
    """Zero-mean gauge IRLS on one connected component of n channels; local indices."""
    m = len(shift)
    A = sp.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[i, j])),
                      shape=(m, n))
    ones = np.full((n, n), 1.0 / n)
    huber = np.ones(m)
    b = np.zeros(n)
    iterations = 0
    for iterations in range(1, options.max_iterations + 1):
        w = huber / sigma ** 2
        # L + 11^T/n is positive definite on a connected component; its solution has zero mean.
        factor = scipy.linalg.cho_factor((A.T @ sp.diags(w) @ A).toarray() + ones)
        new = scipy.linalg.cho_solve(factor, A.T @ (w * shift))
        change = np.abs(new - b).max()
        b = new
        r = (shift - (b[i] - b[j])) / sigma
        if options.huber_k is None:
            break
        updated = np.where(np.abs(r) <= options.huber_k, 1.0, options.huber_k / np.maximum(np.abs(r), 1e-300))
        # unchanged weights reproduce this solution exactly
        if np.abs(updated - huber).max() < 1e-12 or (change < options.tolerance_ps
                                                      and np.abs(updated - huber).max() < 1e-9):
            break
        huber = updated
    # gauge-fixed covariance pinv(L) = (L + 11^T/n)^-1 - 11^T/n; only its diagonal is needed
    inverse, info = scipy.linalg.lapack.dpotri(factor[0], lower=factor[1])
    if info:
        raise np.linalg.LinAlgError(f"covariance inversion failed ({info})")
    raw = np.sqrt(np.clip(np.diag(inverse) - 1.0 / n, 0, None))
    dof = m - (n - 1)
    chi2 = float((huber * r ** 2).sum())
    birge = float(np.sqrt(chi2 / dof)) if dof > 0 else 1.0
    return b, raw, raw * max(1.0, birge), r, huber, {"channels": n, "pairs": m, "chi2": chi2, "dof": dof,
                                                       "birge_ratio": birge, "iterations": iterations}


def solve(pair_fits, operational_channels, unavailable_channels=(), options=SolveOptions()):
    """Channel delays from T6 pair fits (one role); every operational/unavailable channel gets a status."""
    operational = np.unique(np.asarray(list(operational_channels), np.int64))
    unavailable = np.unique(np.asarray(list(unavailable_channels), np.int64))
    if np.intersect1d(operational, unavailable).size:
        raise ValueError("a channel cannot be both operational and hardware unavailable")
    if len(np.unique(pair_fits["role"])) > 1:
        raise ValueError("solve one role at a time")
    known = np.union1d(operational, unavailable)
    ends = np.concatenate([pair_fits["channel_i"], pair_fits["channel_j"]]).astype(np.int64)
    stray = np.setdiff1d(ends, known)
    if stray.size:
        raise ValueError(f"pair fits reference channels outside the operational/unavailable TIME list: {stray[:5]}")
    if (pair_fits["channel_i"] >= pair_fits["channel_j"]).any():
        raise ValueError("pair fits must be canonical (channel_i < channel_j)")

    channels = np.zeros(len(known), CHANNEL_DTYPE)
    channels["time_channel"] = known
    for name in ("delay_ps", "sigma_ps", "raw_sigma_ps", "residual_rms"):
        channels[name] = np.nan
    channels["component"] = -1
    pos_i, pos_j = np.searchsorted(known, pair_fits["channel_i"]), np.searchsorted(known, pair_fits["channel_j"])
    np.add.at(channels["pairs"], np.r_[pos_i, pos_j], 1)

    down = np.isin(known, unavailable)
    usable = (pair_fits["status"] == tp.USABLE) & ~down[pos_i] & ~down[pos_j]
    np.add.at(channels["usable_pairs"], np.r_[pos_i[usable], pos_j[usable]], 1)
    events = pair_fits["events"][usable]
    np.add.at(channels["usable_events"], np.r_[pos_i[usable], pos_j[usable]], np.r_[events, events])

    # statuses for channels without a usable constraint, most informative first
    status = np.full(len(known), UNOBSERVED, np.int8)
    for code, kinds in ((INSUFFICIENT, (tp.ONE_BRANCH,)), (FAILED_FIT, (tp.SCAN_EDGE, tp.FIT_FAILED)),
                        (AMBIGUOUS, (tp.ALIAS_UNRESOLVED, tp.MERGED))):
        hit = np.isin(pair_fits["status"], kinds)
        status[np.r_[pos_i[hit], pos_j[hit]]] = code

    fits = pair_fits[usable]
    gi, gj = pos_i[usable], pos_j[usable]
    graph = sp.csr_matrix((np.ones(len(fits)), (gi, gj)), shape=(len(known), len(known)))
    _, labels = connected_components(graph, directed=False)
    solved = np.zeros(len(known), bool)
    solved[np.r_[gi, gj]] = True
    comps = np.unique(labels[solved])
    sizes = np.array([(labels[solved] == c).sum() for c in comps])
    order = comps[np.lexsort((comps, -sizes))]  # largest first, ties by label
    residuals = np.zeros(len(fits), PAIR_RESIDUAL_DTYPE)
    residuals["channel_i"], residuals["channel_j"] = fits["channel_i"], fits["channel_j"]
    residuals["shift_ps"], residuals["shift_sigma_ps"] = fits["shift_ps"], fits["shift_sigma_ps"]
    components = []
    for number, label in enumerate(order):
        members = np.flatnonzero(solved & (labels == label))
        local = np.full(len(known), -1)
        local[members] = np.arange(len(members))
        edges = np.flatnonzero(labels[gi] == label)
        b, raw, sigma, r, huber, info = _solve_component(len(members), local[gi[edges]], local[gj[edges]],
                                                         fits["shift_ps"][edges], fits["shift_sigma_ps"][edges],
                                                         options)
        channels["delay_ps"][members], channels["raw_sigma_ps"][members] = b, raw
        channels["sigma_ps"][members], channels["component"][members] = sigma, number
        residuals["component"][edges] = number
        residuals["residual_ps"][edges] = r * fits["shift_sigma_ps"][edges]
        residuals["normalized_residual"][edges], residuals["huber_weight"][edges] = r, huber
        square = np.zeros(len(members))
        count = np.zeros(len(members))
        np.add.at(square, np.r_[local[gi[edges]], local[gj[edges]]], np.r_[r, r] ** 2)
        np.add.at(count, np.r_[local[gi[edges]], local[gj[edges]]], 1)
        channels["residual_rms"][members] = np.sqrt(square / count)
        if number == 0:
            status[members] = np.where(sigma <= options.precision_gate_ps, CALIBRATED, LOW_STATISTICS)
        else:
            status[members] = DISCONNECTED
        flagged = np.abs(r) > options.flag_normalized_residual
        components.append(info | {"component": number, "applied": number == 0,
                                  "gauge": "zero mean over the component's solved channels",
                                  "flagged_pairs": int(flagged.sum()),
                                  "downweighted_pairs": int((huber < 1).sum())})
    status[down] = HARDWARE_UNAVAILABLE
    channels["status"] = status
    channels["apply"] = np.isin(status, APPLIED_STATUSES)
    unsolved = ~np.isin(status, APPLIED_STATUSES + (DISCONNECTED,))
    for name in ("delay_ps", "sigma_ps", "raw_sigma_ps"):
        channels[name][unsolved] = np.nan
    diagnostics = {
        "options": {k: getattr(options, k) for k in SolveOptions.__dataclass_fields__},
        "components": components,
        "status_counts": {name: int((status == k).sum()) for k, name in enumerate(CHANNEL_STATUSES)},
        "operational_channels": int(len(operational)), "unavailable_channels": int(len(unavailable)),
        "usable_pairs": int(usable.sum()), "pairs": int(len(pair_fits)),
    }
    return Solution(channels, residuals, diagnostics)
