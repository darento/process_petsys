"""Ordered process pool for the Cornell processing stages (FR-15, spec 003 T25).

Results are delivered in input order whatever order the workers finish in, so a
stage merging them in order gives the same output for any worker count. Workers
are ``spawn`` processes (as the LDAT Inspector's pool); each receives the shared
cancellation event as the first initializer argument, so a STOP reaches running
workers at their next batch. One worker runs everything in-process (no pool).
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
import threading

POLL_S = 0.2


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

    def __init__(self, workers, initializer, initargs=(), cancelled=None):
        self.workers, self.initializer, self.initargs, self.cancelled = workers, initializer, tuple(initargs), cancelled
        self.executor = self.event = None

    def __enter__(self):
        if self.workers > 1:
            context = multiprocessing.get_context("spawn")
            self.event = context.Event()
            self.executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=context,
                                                initializer=self.initializer, initargs=(self.event, *self.initargs))
        else:
            self.event = threading.Event()
            self.initializer(self.event, *self.initargs)
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.event.set()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        return False

    def _check(self):
        if self.cancelled is not None and self.cancelled():
            self.event.set()
            raise PoolCancelled("Cancelled")

    def run(self, fn, tasks, on_result):
        tasks = list(tasks)
        if self.executor is None:
            for index, task in enumerate(tasks):
                self._check()
                on_result(index, fn(*task))
            return
        futures = [self.executor.submit(fn, *task) for task in tasks]
        position = {future: index for index, future in enumerate(futures)}
        pending, delivered = set(futures), 0
        try:
            while delivered < len(futures):
                self._check()
                finished, pending = wait(pending, timeout=POLL_S, return_when=FIRST_COMPLETED)
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
