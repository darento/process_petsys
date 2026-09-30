"""Memory figures and the pre-processing estimate for LDATInspector (spec 002, FR-2).

The estimate is advisory: the operator sees it before processing and is warned
when it exceeds the free physical memory. There is no fixed memory cap; whole
files over the per-file pair cap (FR-22) are listed so they can be refused.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import ctypes
import os
from pathlib import Path
import sys

import numpy as np

from .engine import MAX_PAIRS_PER_FILE

# Main-process bytes per detector side at the merge/calibration peak: 54 B of
# stored columns, the merged copy of the column being filled, the int64 row map
# and partner rows, and the calibrated energy column. Checked against the
# measured peak on real Cornell data in spec 002 T5.
PEAK_BYTES_PER_SIDE = 80
# Private memory a worker adds while reading one file, per accepted side of that
# file (chunk outputs, their concatenation and the SM sort). Measured on Cornell
# 00000003: 0.68 GB for 3,853,684 sides (177 B). The memory-mapped file itself
# is reclaimable page cache and is not counted.
WORKER_BYTES_PER_SIDE = 180
SAMPLE_PAIRS = 20_000
_MEMINFO = Path("/proc/meminfo")
_PROC_STATUS = Path("/proc/self/status")


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _ProcessMemory(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def available_bytes() -> int | None:
    """Free physical memory, or None when unknown."""
    if sys.platform == "win32":
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    # Linux: MemAvailable counts reclaimable page cache (e.g. a memory-mapped
    # LDAT just read); SC_AVPHYS_PAGES is MemFree and excludes it.
    try:
        for line in _MEMINFO.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def _process_counters():
    if sys.platform != "win32":
        return None
    counters = _ProcessMemory()
    counters.cb = ctypes.sizeof(counters)
    # Declare the types: the pseudo-handle is a 64-bit pointer and would be
    # truncated as a default int argument.
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = ctypes.c_void_p
    info = ctypes.windll.psapi.GetProcessMemoryInfo
    info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemory), ctypes.c_ulong]
    info.restype = ctypes.c_int
    return counters if info(current(), ctypes.byref(counters), counters.cb) else None


def working_set(peak: bool = False) -> int | None:
    """This process's current (or peak) working set in bytes, or None when unknown."""
    if sys.platform.startswith("linux"):
        key = "VmHWM:" if peak else "VmRSS:"
        try:
            for line in _PROC_STATUS.read_text().splitlines():
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return None
    counters = _process_counters()
    if counters is None:
        return None
    return int(counters.PeakWorkingSetSize if peak else counters.WorkingSetSize)


def private_bytes(peak: bool = False) -> int | None:
    """This process's current (or peak) private committed memory, or None when unknown."""
    counters = _process_counters()
    if counters is None:
        return None
    return int(counters.PeakPagefileUsage if peak else counters.PagefileUsage)


@dataclass
class MemoryEstimate:
    files: int
    pairs: int            # coincidence pairs that will be read
    sides: int            # expected accepted detector sides
    bytes: int            # expected main-process peak (baseline + sides)
    baseline: int         # this process now (includes any loaded dataset)
    available: int | None
    acceptance_sampled: bool
    workers: int = 0
    worker_bytes: int = 0  # concurrent workers' private memory while reading
    over_cap: tuple = ()   # whole files only: (path, estimated pairs) above MAX_PAIRS_PER_FILE

    @property
    def total(self) -> int:
        return self.bytes + self.worker_bytes

    @property
    def exceeds_available(self) -> bool:
        return self.available is not None and self.total > self.available

    def text(self) -> str:
        free = f"; free RAM {self.available / 1e9:.1f} GB" if self.available is not None else ""
        basis = "" if self.acceptance_sampled else " (upper bound: no config yet)"
        workers = (f" + {self.worker_bytes / 1e9:.1f} GB in {self.workers} workers while reading"
                   if self.workers else "")
        return (f"Estimated memory: {self.bytes / 1e9:.1f} GB{workers} for ~{self.pairs / 1e6:.2f} M pairs, "
                f"~{self.sides / 1e6:.2f} M sides{basis}{free}")

    def over_cap_text(self) -> str:
        files = ", ".join(f"{Path(path).name} ~{pairs / 1e6:.0f} M" for path, pairs in self.over_cap)
        return (f"Whole files unavailable above {MAX_PAIRS_PER_FILE / 1e6:.0f} M pairs per file ({files}): "
                f"read a prefix or split the acquisition into several files")


def _sample(path, settings):
    """(bytes per pair, accepted fraction) from the file's first records."""
    from .fastread import _scan_headers, process_file_fast

    buf = np.memmap(path, dtype=np.uint8, mode="r")
    _, _, _, count, end, _ = _scan_headers(buf, 0, SAMPLE_PAIRS)
    if count == 0:
        return None, None
    per_pair = end / count
    if settings is None:
        return per_pair, 1.0
    result = process_file_fast(path, replace(settings, max_pairs=SAMPLE_PAIRS))
    if not result.success or result.pairs_read == 0:
        return per_pair, 1.0
    return per_pair, result.pairs_accepted / result.pairs_read


def estimate_memory(paths, settings=None, max_pairs: int | None = None,
                    workers: int | None = None) -> MemoryEstimate:
    """Expected peak for reading ``paths`` (``max_pairs`` per file, None = whole files).

    With ``settings`` (a valid config) the accepted fraction is sampled from
    each file's first records; without, every pair is assumed accepted.
    ``workers`` defaults to the GUI's pool size, min(files, CPU cores - 2); the
    worker figure assumes the largest files are read at the same time.
    """
    pairs = sides = 0
    per_file, over_cap = [], []
    sampled = settings is not None
    for path in paths:
        size = Path(path).stat().st_size
        if size == 0:
            continue
        per_pair, accepted = _sample(path, settings)
        if per_pair is None:
            continue
        in_file = size / per_pair
        if max_pairs is None and in_file > MAX_PAIRS_PER_FILE:
            over_cap.append((str(path), int(in_file)))
        read = in_file if max_pairs is None else min(in_file, max_pairs)
        pairs += read
        sides += 2 * read * accepted
        per_file.append(2 * read * accepted)
    if workers is None:
        workers = max(1, min(len(paths), (os.cpu_count() or 1) - 2)) if paths else 0
    workers = min(workers, len(per_file))
    concurrent = sum(sorted(per_file, reverse=True)[:workers])
    baseline = working_set() or 0
    return MemoryEstimate(len(paths), int(pairs), int(sides),
                          int(baseline + sides * PEAK_BYTES_PER_SIDE), int(baseline),
                          available_bytes(), sampled, workers, int(concurrent * WORKER_BYTES_PER_SIDE),
                          tuple(over_cap))
