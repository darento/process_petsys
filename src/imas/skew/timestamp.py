"""IMAS side timestamp rule (spec 014 FR-1, FR-3).

Each side is timestamped by the highest-charge TIME hit of the selected minimodule (the top). The other TIME
hits of that minimodule above the hit energy floor are candidates, carried as ``CANDIDATE_DTYPE`` slots (at most
``SLOTS`` per side; channel -1 marks an empty slot; ``lag_ps`` = candidate timestamp - top timestamp). Each
candidate is corrected as if it were the timestamped hit: its own channel's skew delay and the walk with its own
charge and share (charge over the side's summed TIME charge); DOI ratio and keV stay the side's. The earliest
is the candidate with the smallest corrected time; a tie keeps the top. Rules (``RULES``):

* ``top``: the top's corrected time (spec 011/012);
* ``top_earliest_mean``: the mean of the top and earliest corrected times (the top's when it is earliest);
* ``earliest``: the earliest corrected time; skew-only tables only (spec 014 clarify 4).

``side_offset`` returns the side time minus the top's corrected time, so a corrected ``t_1 - t_2`` under a rule
is the top-only correction plus ``offset_1 - offset_2``. A candidate without an applied delay or without every
walk input is excluded and counted; an event whose top cannot be corrected keeps offset 0 (its handling is
the top's, spec 012 FR-6). Tables record the rule they were fitted under (``timestamp_rule``, absent = ``top``);
``check_rule`` refuses another rule unless the mismatch is named.
"""
from __future__ import annotations

import numpy as np

from src.imas.skew import walk as iw

RULES = ("top", "top_earliest_mean", "earliest")
SLOTS = 7                                   # 8 TIME channels per minimodule, the top excluded
CANDIDATE_DTYPE = np.dtype([("channel", "<i4"), ("lag_ps", "<i4"), ("charge", "<f4")])
COUNTS = ("rule acts", "candidates", "candidates excluded (no delay)", "candidates excluded (walk input)")


class RuleError(ValueError):
    """A timestamp rule that is unknown, unsupported for the tables, or not the one a table was fitted under."""


def _known(rule):
    if rule not in RULES:
        raise RuleError(f"timestamp rule {rule!r}: one of {RULES}")
    return rule


def table_rule(manifest):
    """The rule a table was fitted under, from its manifest (top level or ``provenance``); absent = ``top``."""
    manifest = manifest or {}
    rule = manifest.get("timestamp_rule", (manifest.get("provenance") or {}).get("timestamp_rule", "top"))
    return _known(rule)


def check_rule(fitted, rule, *, mismatch=False):
    """Refuse applying a table fitted under ``fitted`` with ``rule`` unless ``mismatch`` names the difference
    (the frozen-table comparison of FR-5). Returns the record for reports."""
    _known(fitted), _known(rule)
    if fitted != rule and not mismatch:
        raise RuleError(f"table fitted under timestamp rule {fitted!r}, applied under {rule!r}: name the mismatch "
                        "to compare it frozen")
    return {"table_rule": fitted, "rule": rule, "mismatch": fitted != rule}


def empty_candidates(n):
    out = np.zeros((n, SLOTS), CANDIDATE_DTYPE)
    out["channel"] = -1
    return out


def _walk(walk, charge, side, shape):
    """Walk correction (ps, NaN where an input is missing) and clamp flags for charges of shape ``shape``
    (events x slots or events) with the side's q_time, DOI ratio and keV broadcast."""
    def spread(name):
        return np.broadcast_to(np.asarray(side[name], np.float64).reshape((-1,) + (1,) * (len(shape) - 1)),
                               shape).ravel()
    values, status = iw.walk_inputs(np.asarray(charge, np.float64).ravel(), spread("q_time"), spread("doi_ratio"),
                                    spread("kev"))
    correction, outside = walk.correction(values)
    return np.where(status == 0, correction, np.nan).reshape(shape), {k: v.reshape(shape) for k, v in outside.items()}


def side_offset(top_channel, candidates, side, skew, walk, rule):
    """Per event the side time under ``rule`` minus the top's corrected time (float64 ps), whether the rule
    acts (an earlier candidate replaced or joined the top), and counts (``COUNTS`` plus clamped candidate walk
    inputs). ``side`` = {q_top, q_time, doi_ratio, kev} of the side; ``skew`` has ``lookup(channels)``;
    ``walk`` is an IMAS walk LUT or None (skew only). ``top`` never reads ``candidates``."""
    _known(rule)
    top_channel = np.asarray(top_channel, np.int64)
    n = len(top_channel)
    counts = dict.fromkeys(COUNTS, 0)
    if rule == "top":
        return np.zeros(n), np.zeros(n, bool), counts
    if rule == "earliest" and walk is not None:
        raise RuleError("timestamp rule 'earliest' is for skew-only tables (spec 014 clarify 4)")
    if candidates is None or candidates.shape != (n, SLOTS) or candidates.dtype != CANDIDATE_DTYPE:
        raise RuleError(f"rule {rule!r} needs the candidates: ({n}, {SLOTS}) {CANDIDATE_DTYPE}")
    b_top, top_found = skew.lookup(top_channel)
    c_top = -b_top
    present = candidates["channel"] >= 0
    b, found = skew.lookup(np.where(present, candidates["channel"], -1))
    c = candidates["lag_ps"].astype(np.float64) - b
    usable = present & found
    if walk is not None:
        w_top, _ = _walk(walk, side["q_top"], side, (n,))
        c_top = c_top - w_top
        w, outside = _walk(walk, candidates["charge"], side, (n, SLOTS))
        walk_ok = np.isfinite(w)
        counts["candidates excluded (walk input)"] = int(np.count_nonzero(usable & ~walk_ok))
        usable &= walk_ok
        c = c - np.where(walk_ok, w, 0.0)
        for name in iw.INPUTS:
            counts[f"candidates clamped {name}"] = int(np.count_nonzero(usable & outside[name]))
    counts["candidates"] = int(np.count_nonzero(present))
    counts["candidates excluded (no delay)"] = int(np.count_nonzero(present & ~found))
    earliest = np.min(np.where(usable, c, np.inf), axis=1)
    acts = top_found & np.isfinite(c_top) & (earliest < c_top)   # a tie keeps the top
    lead = np.where(acts, earliest - c_top, 0.0)
    counts["rule acts"] = int(np.count_nonzero(acts))
    return (lead / 2 if rule == "top_earliest_mean" else lead), acts, counts
