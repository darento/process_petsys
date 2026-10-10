"""Toolkit-free run-progress logic for the manager window (spec 005).

Estimates use only the running stage's own counters and rate (FR-1); the GUI
supplies its monotonic clock as ``now``. ``RunTracker`` turns RunEvents into what
the run panel shows and ``StageOverview`` lists a multi-stage run's stages (FR-2).
Nothing here imports Tk.
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


@dataclass(frozen=True)
class RunView:
    title: str
    fraction: float | None      # None: indeterminate bar
    counter: str
    elapsed_s: float
    remaining_s: float | None   # None: no estimate
    finished: bool
    stages: tuple = ()          # StageRow per stage of a multi-stage run (FR-2); empty for a single stage
    idle: bool = False          # no run requested yet this session


class RunTracker:
    """What the run panel shows for the one foreground workflow: its stage, counter, elapsed time and
    estimate (FR-1). Elapsed time counts from ``workflow_started`` on the GUI clock; each stage has its
    own estimate."""

    def __init__(self, stage_names=None):
        self._names = dict(stage_names or {})
        self.reset(None, "")

    def reset(self, now, action):
        """A new run was requested: forget the previous run's final state."""
        self._action, self._stages, self._stage = action, (), None
        self._origin, self._ended, self._status = None, None, None
        self._overview = None
        self._new_stage(None)

    def _new_stage(self, stage, now=None):
        self._stage, self._estimate = stage, RateEstimate()
        if now is not None:
            self._estimate.start(now)      # the first phase's clock starts with the stage
        self._fraction, self._counter, self._remaining = None, "", None

    def event(self, now, kind, stage, payload):
        if kind == "workflow_started":
            self._origin, self._stages = now, tuple(payload.get("stages") or ())
            self._overview = StageOverview(self._stages) if len(self._stages) > 1 else None
            return
        if self._overview is not None:
            self._overview.event(now, kind, stage, payload)
        if kind == "workflow_finished":
            self.finish(now, payload.get("status") or "finished")
            return
        if stage != "workflow" and stage != self._stage:
            self._new_stage(stage, now)
        if kind == "stage_progress":
            self._progress(now, payload)
        elif kind == "acquisition_rawf_progress" and isinstance(payload.get("size"), int):
            self._counter = f"RAW {payload['size'] / 1e6:,.1f} MB written"

    def finish(self, now, status):
        """The run ended: elapsed time stops and the final state stays until ``reset``. A succeeded run
        fills the bar; any other keeps it where its stage stopped."""
        if self._ended is not None:
            return
        self._ended, self._status, self._remaining = now, status, None
        if status == "succeeded":
            self._fraction = 1.0

    def _progress(self, now, payload):
        phase, phases = payload.get("phase"), list(payload.get("phases") or ())
        if phase == "convert":
            self._conversion(payload)
            return
        step = f"phase {phases.index(phase) + 1} of {len(phases)}" if len(phases) > 1 and phase in phases else ""
        if phase == "fits":
            done, total = payload.get("keys_done"), payload.get("keys_total")
            known = isinstance(total, int) and total > 0
            self._counter = f"fits {done or 0:,} / {total:,} keys" if known else "fits"
            self._counter += f" ({step})" if step else ""
            self._show(self._estimate.update(now, done or 0, total if known else None, phase))
            return
        read, total = payload.get("bytes_read"), payload.get("bytes_total")
        if not isinstance(read, int):
            return
        known = isinstance(total, int) and total > 0
        self._counter = f"{read / 1e9:.2f} / {total / 1e9:.2f} GB input read" if known else \
            f"{read / 1e9:.2f} GB input read"
        self._counter += f" ({phase}, {step})" if step else ""
        self._show(self._estimate.update(now, read, total if known else None, phase))

    def _conversion(self, payload):
        """LDAT bytes written beside the RAW size, never as a fraction; the bar and estimate count closed
        splits, and a single split has no known total (FR-1)."""
        ldat, raw = payload.get("ldat_bytes") or 0, payload.get("rawf_bytes")
        splits, durations = payload.get("splits") or 0, list(payload.get("split_durations_s") or ())
        self._counter = f"LDAT {ldat / 1e9:.1f} GB written, " + \
            (f"RAW {raw / 1e9:.1f} GB" if isinstance(raw, int) else "RAW size unknown") + \
            f"; splits {len(durations)} / {splits} closed"
        self._remaining = split_remaining(splits, durations)
        self._fraction = None if self._remaining is None else min(1.0, len(durations) / splits)

    def _show(self, progress):
        """The bar stays indeterminate until the estimate's gate opens (FR-1)."""
        self._remaining = progress.remaining_s
        self._fraction = None if progress.remaining_s is None else progress.fraction

    def view(self, now):
        end = self._ended if self._ended is not None else now
        elapsed = 0.0 if self._origin is None else max(0.0, end - self._origin)
        title = self._action
        if self._stage in self._stages and len(self._stages) > 1:
            title += f" - step {self._stages.index(self._stage) + 1}/{len(self._stages)}: " \
                     f"{self._names.get(self._stage, self._stage)}"
        if self._status is not None:
            title += f" {self._status}"
        rows = self._overview.rows(end) if self._overview is not None else ()
        return RunView(title, self._fraction, self._counter, elapsed, self._remaining, self._ended is not None, rows,
                       idle=not self._action)


@dataclass(frozen=True)
class StageRow:
    stage: str
    state: str                  # pending, running, succeeded, failed, stopped
    elapsed_s: float | None     # recorded once the stage reports it; the GUI clock until then


STAGE_STATES = {"succeeded": "succeeded", "cancelled": "stopped"}     # any other result: failed


class StageOverview:
    """The stages of a multi-stage run and their states (FR-2), from its RunEvents. A stage runs from its
    first event; acquisition has no stage events of its own, so the next stage's start ends it as succeeded
    (a stage only starts after its predecessor succeeded)."""

    def __init__(self, stages):
        self._stages = list(stages)
        self._state = {stage: "pending" for stage in self._stages}
        self._started, self._elapsed = {}, {}

    def event(self, now, kind, stage, payload):
        if kind == "workflow_finished":
            self._finished(now, payload.get("status"))
            return
        if stage not in self._state:
            return
        if self._state[stage] == "pending":
            for earlier in self._stages[:self._stages.index(stage)]:
                if self._state[earlier] == "running":
                    self._end(now, earlier, "succeeded")
            self._state[stage], self._started[stage] = "running", now
        if kind == "stage_finished":
            self._end(now, stage, payload.get("status"), payload.get("elapsed_s"))

    def _finished(self, now, status):
        """A stage still running ends with the workflow; after a STOP the stages not run are stopped,
        otherwise they stay pending (shown as not run)."""
        for stage in self._stages:
            if self._state[stage] == "running":
                self._end(now, stage, status)
            elif self._state[stage] == "pending" and status == "cancelled":
                self._state[stage] = "stopped"

    def _end(self, now, stage, status, elapsed=None):
        self._state[stage] = STAGE_STATES.get(status, "failed")
        self._elapsed[stage] = elapsed if isinstance(elapsed, (int, float)) else now - self._started[stage]

    def rows(self, now):
        return tuple(StageRow(stage, self._state[stage], now - self._started[stage] if self._state[stage] == "running"
                              else self._elapsed.get(stage)) for stage in self._stages)
