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
from src.timing.cache import ROLES

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


def skew_rule(skew):
    """The rule of a skew ``DelayTable`` (its LUT provenance; legacy and provenance-less tables are ``top``)."""
    return table_rule({"provenance": getattr(skew, "provenance", None)})


def check_rule(fitted, rule, *, mismatch=False):
    """Refuse applying a table fitted under ``fitted`` with ``rule`` unless ``mismatch`` names the difference
    (the frozen-table comparison of FR-5). Returns the record for reports."""
    _known(fitted), _known(rule)
    if fitted != rule and not mismatch:
        raise RuleError(f"table fitted under timestamp rule {fitted!r}, applied under {rule!r}: name the mismatch "
                        "to compare it frozen")
    return {"table_rule": fitted, "rule": rule, "mismatch": fitted != rule}


def require_candidates(observations, rule):
    """Refuse an observation set (or a view with ``carries_candidates``) without candidates for a rule other
    than ``top`` (FR-2)."""
    if _known(rule) != "top" and not getattr(observations, "carries_candidates", False):
        raise RuleError(f"timestamp rule {rule!r} needs observation rows with the candidates; stream the set "
                        "with carry_candidates in a new directory")


def observation_candidates(rows, suffix):
    """``CANDIDATE_DTYPE`` slots of one canonical side (``i``/``j``) of candidate-carrying observation rows."""
    out = np.empty(rows[f"cand_channel_{suffix}"].shape, CANDIDATE_DTYPE)
    out["channel"], out["lag_ps"], out["charge"] = (rows[f"cand_channel_{suffix}"], rows[f"cand_lag_{suffix}"],
                                                    rows[f"cand_charge_{suffix}"])
    return out


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
    n = len(top_channel)
    if rule == "top":
        return np.zeros(n), np.zeros(n, bool), dict.fromkeys(COUNTS, 0)
    if candidates is None or candidates.shape != (n, SLOTS) or candidates.dtype != CANDIDATE_DTYPE:
        raise RuleError(f"rule {rule!r} needs the candidates: ({n}, {SLOTS}) {CANDIDATE_DTYPE}")
    relative, top_found, counts = skew_relative(top_channel, candidates, skew)
    offset, acts, found = relative_offset(relative, candidates["charge"], top_found, side, walk, rule)
    return offset, acts, counts | found


def skew_relative(top_channel, candidates, skew):
    """Candidate times after their skew delay relative to the top's (``lag + b_top - b``, float64 ps; NaN for an
    empty slot or a candidate without a delay), whether the top has a delay, and the candidate counts. The walk
    fit (FR-4) keeps these fixed while the walk changes."""
    b_top, top_found = skew.lookup(np.asarray(top_channel, np.int64))
    present = candidates["channel"] >= 0
    b, found = skew.lookup(np.where(present, candidates["channel"], -1))
    relative = np.where(present & found, candidates["lag_ps"].astype(np.float64) - b + b_top[:, None], np.nan)
    return relative, top_found, {"candidates": int(np.count_nonzero(present)),
                                 "candidates excluded (no delay)": int(np.count_nonzero(present & ~found))}


def relative_offset(relative, charge, top_found, side, walk, rule):
    """``side_offset`` from ``skew_relative`` times and the candidate charges: the walk (None: skew only) of each
    candidate and of the top, the earliest (a tie keeps the top) and the offset under ``rule``. Counts: ``rule
    acts``, ``candidates excluded (walk input)`` and, with walk, the clamped candidate inputs."""
    if _known(rule) == "earliest" and walk is not None:
        raise RuleError("timestamp rule 'earliest' is for skew-only tables (spec 014 clarify 4)")
    n = len(relative)
    c, c_top = relative, np.zeros(n)
    usable = np.isfinite(relative)
    counts = {"candidates excluded (walk input)": 0}
    if walk is not None:
        w_top, _ = _walk(walk, side["q_top"], side, (n,))
        c_top = c_top - w_top
        w, outside = _walk(walk, charge, side, (n, SLOTS))
        walk_ok = np.isfinite(w)
        counts["candidates excluded (walk input)"] = int(np.count_nonzero(usable & ~walk_ok))
        usable &= walk_ok
        c = c - np.where(walk_ok, w, 0.0)
        for name in iw.INPUTS:
            counts[f"candidates clamped {name}"] = int(np.count_nonzero(usable & outside[name]))
    earliest = np.min(np.where(usable, c, np.inf), axis=1)
    acts = top_found & np.isfinite(c_top) & (earliest < c_top)   # a tie keeps the top
    lead = np.where(acts, earliest - c_top, 0.0)
    counts["rule acts"] = int(np.count_nonzero(acts))
    return (lead / 2 if rule == "top_earliest_mean" else lead), acts, counts


def observation_sides(rows, suffix):
    """Walk ingredients {q_top, q_time, doi_ratio, kev} of one canonical side of walk-carrying observation rows."""
    return {"q_top": rows[f"q_top_{suffix}"], "q_time": rows[f"q_time_{suffix}"],
            "doi_ratio": rows[f"doi_ratio_{suffix}"], "kev": rows[f"energy_kev_{suffix}"]}


def rule_offsets(rows, skew, walk, rule):
    """Per row ``offset_i - offset_j`` (float ps) of candidate-carrying observation rows under ``rule``, whether
    the rule acts on either side, and the summed counts."""
    offset_i, acts_i, counts_i = side_offset(rows["channel_i"], observation_candidates(rows, "i"),
                                             observation_sides(rows, "i"), skew, walk, rule)
    offset_j, acts_j, counts_j = side_offset(rows["channel_j"], observation_candidates(rows, "j"),
                                             observation_sides(rows, "j"), skew, walk, rule)
    counts = {key: counts_i[key] + counts_j[key] for key in counts_i if key != "rule acts"}
    acts = acts_i | acts_j
    counts["rule acts"] = int(np.count_nonzero(acts))
    return offset_i - offset_j, acts, counts


class RuleAdjusted:
    """FR-4 fixed point: an observation set (or view) whose partitions get ``dt_ps += rint(offset_i - offset_j)``
    under ``rule`` with the previous pass's ``skew`` and ``walk`` (None: skew only), so the EM and the walk fit,
    which see one delay and one walk per side, solve for the top's channel and inputs while the rule's offset
    from the previous tables is held fixed. No rule check: the fixed point starts from tables fitted under
    ``top``. ``counts()`` sums per role the partitions read (each counted once)."""

    def __init__(self, observations, skew, walk, rule):
        require_candidates(observations, rule)
        self.observations, self.skew, self.walk, self.rule = observations, skew, walk, rule
        self.partitions = observations.partitions
        self._counts = {}

    @property
    def carries_candidates(self):
        return True

    def partition(self, p):
        rows = self.observations.partition(p)
        if self.rule == "top" or not len(rows):
            self._counts[p] = {}
            return rows
        difference, counts = np.zeros(len(rows)), {}
        for code, role in enumerate(ROLES):
            mine = rows["role"] == code
            if mine.any():
                difference[mine], _, found = rule_offsets(rows[mine], self.skew, self.walk, self.rule)
                counts[role] = {"rows": int(mine.sum()), **found}
        self._counts[p] = counts
        rows = rows.copy()
        rows["dt_ps"] += np.rint(difference).astype(np.int64)
        return rows

    def counts(self):
        total = {}
        for counts in self._counts.values():
            for role, entry in counts.items():
                mine = total.setdefault(role, {})
                for key, n in entry.items():
                    mine[key] = mine.get(key, 0) + n
        return total
