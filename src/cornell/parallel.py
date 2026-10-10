"""Ordered process pool for the Cornell processing stages (FR-15, spec 003 T25).

Results are delivered in input order whatever order the workers finish in, so a
stage merging them in order gives the same output for any worker count. Workers
are ``spawn`` processes (as the LDAT Inspector's pool); each receives the shared
cancellation event as the first initializer argument, so a STOP reaches running
workers at their next batch. One worker runs everything in-process (no pool).

Progress slots (spec 005 FR-1): with ``progress_slots=n`` a task reports its own
counter with ``report_progress(index, value)``; the parent's ``on_tick(values)``
sees all slots from the wait loop (pool) or at each report (in-process), and
once after the last result. Slots are counters only, never data.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
import threading

POLL_S = 0.2
_SLOTS = None   # this process's progress slots: a shared array in a worker, _LocalSlots in-process


class _LocalSlots(list):
    """In-process slots: the task blocks the parent's thread, so each report ticks at once."""
    on_tick = None

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        if self.on_tick is not None:
            self.on_tick(tuple(self))


def report_progress(index, value):
    """Set task ``index``'s progress counter; a no-op when the pool has no slots."""
    if _SLOTS is not None:
        _SLOTS[index] = value


def slot_progress(index):
    """A stage's progress callback inside pool task ``index``: its ``bytes_read`` goes to the slot."""
    def progress(*args, bytes_read=None, **extra):
        if bytes_read is not None:
            report_progress(index, bytes_read)
    return progress


def tick_progress(paths, progress):
    """``on_tick`` for a pool over ``paths``: each changed slot becomes ``progress(index, path, bytes_read=...)``."""
    seen = [0] * len(paths)

    def on_tick(values):
        for index, value in enumerate(values):
            if value != seen[index]:
                seen[index] = value
                progress(index, paths[index], bytes_read=value)
    return on_tick


def _init_with_slots(slots, initializer, event, *initargs):
    global _SLOTS
    _SLOTS = slots
    initializer(event, *initargs)


class PoolCancelled(Exception):
    """The parent's cancellation was seen while tasks were pending."""


def resolve_workers(workers):
    """0 = automatic: CPU count - 2, at least 1 (as the Inspector)."""
    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be a non-negative integer (0 = automatic)")
    return max(1, (os.cpu_count() or 1) - 2) if workers == 0 else workers


class OrderedPool:
    """``with OrderedPool(workers, initializer, initargs, cancelled) as pool: pool.run(fn, tasks, on_result)``.

    ``initializer(event, *initargs)`` runs once per worker (in-process for one worker); ``on_result(index,
    value)`` is called in input order. A worker exception is raised in the parent (the lowest-index one among
    the tasks finished so far); pending tasks are cancelled and running ones see the event."""

    def __init__(self, workers, initializer, initargs=(), cancelled=None, progress_slots=0):
        self.workers, self.initializer, self.initargs, self.cancelled = workers, initializer, tuple(initargs), cancelled
        self.progress_slots = progress_slots
        self.executor = self.event = self.slots = None
        self._previous_slots = None

    def __enter__(self):
        global _SLOTS
        if self.workers > 1:
            context = multiprocessing.get_context("spawn")
            self.event = context.Event()
            initializer, initargs = self.initializer, (self.event, *self.initargs)
            if self.progress_slots:
                self.slots = context.Array("q", self.progress_slots, lock=False)
                initializer, initargs = _init_with_slots, (self.slots, initializer, *initargs)
            self.executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=context,
                                                initializer=initializer, initargs=initargs)
        else:
            self.event = threading.Event()
            if self.progress_slots:
                self.slots, self._previous_slots = _LocalSlots([0] * self.progress_slots), _SLOTS
                _SLOTS = self.slots
            self.initializer(self.event, *self.initargs)
        return self

    def __exit__(self, kind, value, traceback):
        global _SLOTS
        if kind is not None:
            self.event.set()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        elif self.slots is not None:
            _SLOTS = self._previous_slots
        return False

    def _tick(self, on_tick):
        if on_tick is not None and self.slots is not None:
            on_tick(tuple(self.slots[:]))

    def _check(self):
        if self.cancelled is not None and self.cancelled():
            self.event.set()
            raise PoolCancelled("Cancelled")

    def run(self, fn, tasks, on_result, on_tick=None):
        tasks = list(tasks)
        if self.executor is None:
            if self.slots is not None:
                self.slots.on_tick = on_tick
            try:
                for index, task in enumerate(tasks):
                    self._check()
                    on_result(index, fn(*task))
            finally:
                if self.slots is not None:
                    self.slots.on_tick = None
            self._tick(on_tick)
            return
        futures = [self.executor.submit(fn, *task) for task in tasks]
        position = {future: index for index, future in enumerate(futures)}
        pending, delivered = set(futures), 0
        try:
            while delivered < len(futures):
                self._check()
                finished, pending = wait(pending, timeout=POLL_S, return_when=FIRST_COMPLETED)
                self._tick(on_tick)
                failed = sorted((position[f] for f in finished if f.exception() is not None))
                if failed:
                    # STOP sends SIGTERM to the whole process group: a worker killed by it is the cancellation.
                    self._check()
                    raise futures[failed[0]].exception()
                while delivered < len(futures) and futures[delivered].done():
                    on_result(delivered, futures[delivered].result())
                    delivered += 1
        except BaseException:
            self.event.set()
            for future in pending:
                future.cancel()
            raise
        self._tick(on_tick)
