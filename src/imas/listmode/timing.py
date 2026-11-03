"""Listmode ``dt`` from the delivered IMAS timing tables (spec 015 T2, FR-4).

``dt`` = side 1 minus side 2 in LDAT order (``records.build`` flips the sign of swapped pairs), corrected by
``src.imas.skew.walk.correct_dt`` under the tables' timestamp rule. The walk keV is the minimodule ENERGY sum
through the walk tables' own calibration (``walk_mu``), not the LM energy calibration. Modes:

* ``em``: an IMAS canonical skew LUT and (unless ``walk`` is None) its paired walk LUT, checked together
  (map SHA-256, pairing, fitted rule == applied rule);
* ``legacy``: a ``ch skew`` file (``t + skew``), rule ``top``, no walk;
* ``none``: no correction; every ``dt`` is ``UNCORRECTED_DT`` (the legacy LM today).

An event without both delays or without every walk input, or whose corrected ``dt`` leaves int16, gets
``UNCORRECTED_DT`` and is counted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.imas.skew import outputs, timestamp
from src.imas.skew import walk as iw
from src.timing.lut import DelayTable

UNCORRECTED_DT = -32700
MODES = ("em", "legacy", "none")
STATUS = ("corrected", "no applied delay", "walk input missing", "outside int16")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Timing:
    """A frozen correction: ``dt(features)`` gives int16 ``dt`` and counts; ``describe()`` its provenance."""

    def __init__(self, mode, skew=None, walk=None, rule="top", provenance=None):
        if mode not in MODES:
            raise ValueError(f"timing mode {mode!r}: one of {MODES}")
        self.mode, self.skew, self.walk, self.rule = mode, skew, walk, rule
        self.provenance = provenance or {}

    @classmethod
    def em(cls, setup, skew_path, walk_path, rule):
        """Canonical IMAS skew LUT (+ walk LUT or None), both checked against ``setup`` and ``rule``."""
        table = outputs.read_imas_lut(skew_path, setup)
        skew_sha = sha256(skew_path)
        walk = iw.WalkLut.read(walk_path) if walk_path is not None else None
        iw.check_tables(table, skew_sha, walk, setup)
        delays = DelayTable.from_lut(table)
        record = {"mode": "em", "rule": rule, "skew": str(skew_path), "skew_sha256": skew_sha,
                  "skew_applied_channels": len(delays.delays),
                  "skew_rule": timestamp.check_rule(timestamp.skew_rule(delays), rule)["table_rule"]}
        if walk is not None:
            record |= {"walk": str(walk_path), "walk_sha256": sha256(walk_path), "walk_lut_sha256": walk.sha256,
                       "walk_rule": timestamp.check_rule(timestamp.table_rule(walk.manifest), rule)["table_rule"]}
        return cls("em", delays, walk, rule, record)

    @classmethod
    def legacy(cls, setup, path):
        delays = outputs.read_legacy(path, setup)
        return cls("legacy", delays, None, "top", {"mode": "legacy", "rule": "top", "legacy_skew": str(path),
                                                   "legacy_skew_sha256": sha256(path),
                                                   "skew_applied_channels": len(delays.delays)})

    @classmethod
    def none(cls):
        return cls("none", provenance={"mode": "none", "dt": UNCORRECTED_DT})

    @property
    def needs_walk_kev(self):
        return self.walk is not None

    def describe(self):
        return dict(self.provenance)

    def dt(self, features, walk_mu):
        """int16 ``dt`` (side 1 - side 2, LDAT order) of ``LmFeatures`` and counts by ``STATUS``."""
        n = len(features)
        counts = dict.fromkeys(STATUS, 0)
        if self.mode == "none":
            counts["no applied delay"] = n
            return np.full(n, UNCORRECTED_DT, np.int16), counts
        ch1, ch2 = features.column("time_channel_1"), features.column("time_channel_2")

        def side(s, channels):
            kev = (511.0 / walk_mu[channels] * features.column(f"e_raw_{s}")) if self.walk is not None \
                else np.zeros(n)
            return {"q_top": features.column(f"q_top_{s}"), "q_time": features.column(f"q_time_{s}"),
                    "doi_ratio": features.column(f"doi_ratio_{s}"), "kev": kev}

        rule_on = self.rule != "top"
        value, status, details = iw.correct_dt(
            features.column("dt_ps"), ch1, ch2, side(1, ch1), side(2, ch2), self.skew, self.walk, policy="drop",
            rule=self.rule, candidates_1=features.candidates[:, 0] if rule_on else None,
            candidates_2=features.candidates[:, 1] if rule_on else None)
        rounded = np.rint(np.where(np.isfinite(value), value, 0.0))
        inside = np.isfinite(value) & (np.abs(rounded) <= 32767)      # symmetric: a swapped pair negates dt
        counts["corrected"] = int(np.count_nonzero(inside))
        counts["no applied delay"] = int(np.count_nonzero(status == 1))
        counts["walk input missing"] = int(np.count_nonzero(status == 2))
        counts["outside int16"] = int(np.count_nonzero((status == 0) & ~inside))
        for key, number in details.items():
            if key.startswith("clamped") or key.startswith("candidates") or key == "rule acts":
                counts[key] = int(number)
        return np.where(inside, rounded, UNCORRECTED_DT).astype(np.int16), counts


def merge_counts(total, counts):
    for key, value in counts.items():
        total[key] = total.get(key, 0) + int(value)
    return total


def from_options(setup, *, skew=None, walk=None, rule="top_earliest_mean", legacy=None, none=False):
    """The run's ``Timing`` from command-line choices (exactly one mode)."""
    chosen = sum((bool(none), legacy is not None, skew is not None))
    if chosen != 1:
        raise ValueError("choose exactly one timing: a skew LUT (with its walk LUT), a legacy skew file, or none")
    if none:
        return Timing.none()
    if legacy is not None:
        return Timing.legacy(setup, Path(legacy))
    return Timing.em(setup, Path(skew), None if walk is None else Path(walk), rule)
