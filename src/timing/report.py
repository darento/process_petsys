"""Job report helpers shared by scanner front ends (spec 011, FR-6): peak memory, directory bytes.

``peak_memory`` moved from ``src.cornell.skew.job`` (spec 008).
"""
from __future__ import annotations

from pathlib import Path
import sys


def peak_memory():
    """Peak resident memory in bytes for this process (and finished children where the OS reports it)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return {"parent_bytes": int(counters.PeakWorkingSetSize), "largest_child_bytes": None}
    import resource
    scale = 1 if sys.platform == "darwin" else 1024
    return {"parent_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale,
            "largest_child_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale}


def tree_bytes(directory):
    """Total bytes of the files under ``directory`` (0 when it does not exist)."""
    directory = Path(directory)
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) if directory.exists() else 0
