"""Toolkit-free run-progress logic for the manager window (spec 005).

Estimates use only the running stage's own counters and rate (FR-1); the GUI
supplies its monotonic clock as ``now``. Nothing here imports Tk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Progress:
    fraction: float | None      # None: unknown total
    remaining_s: float | None   # None: no estimate


class RateEstimate:
    """Time remaining from this stage's progress rate so far."""

    def __init__(self, min_elapsed_s=5.0, min_fraction=0.01):
        self.min_elapsed_s, self.min_fraction = min_elapsed_s, min_fraction
        self._started = self._phase = self._last = None

    def start(self, now, phase=None):
        self._started, self._phase, self._last = now, phase, now

    def update(self, now, done, total, phase=None):
        """A different phase restarts the clock where the previous phase was last seen: its first
        event already carries some of its work, so starting at ``now`` would overstate its rate."""
        if self._started is None:   # no start(): the clock begins at this event
            self.start(now, phase)
        elif phase != self._phase:
            self.start(self._last, phase)
        self._last = now
        if not total:
            return Progress(None, None)
        fraction = min(1.0, done / total)   # an overshooting counter means done, never a negative estimate
        elapsed = now - self._started
        if elapsed < self.min_elapsed_s or fraction < self.min_fraction:
            return Progress(fraction, None)
        return Progress(fraction, elapsed * (1 - fraction) / fraction)


def split_remaining(splits, split_durations_s):
    """Conversion time remaining: splits not yet closed x mean time per closed split (FR-1).

    None with a single split (its total is unknown) or before the first split has closed."""
    if splits < 2 or not split_durations_s:
        return None
    left = max(0, splits - len(split_durations_s))     # never negative if the converter wrote more
    return left * sum(split_durations_s) / len(split_durations_s)
