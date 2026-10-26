"""Moved to ``src.timing.pairs`` (spec 011 T1, FR-1).

This name is the same module object, so Cornell imports and patches keep working.
"""
import sys

from src.timing import pairs as _module

sys.modules[__name__] = _module
