"""Positive subtractive channel delays in ps (spec 008 T2a).

This is an offline difference contract, not native PETsys export qualification.
Channel status, gauge and correction-state checks belong to the LUT adapter.
"""
import math
from numbers import Integral, Real


def corrected_dt_ps(timestamp_i, timestamp_j, delay_i_ps, delay_j_ps):
    """y_corrected = (t_i-t_j) - b_i + b_j; subtract integers before floats."""
    if any(isinstance(t, bool) or not isinstance(t, Integral) for t in (timestamp_i, timestamp_j)):
        raise ValueError("raw compact timestamps must be integers in ps")
    if any(isinstance(b, bool) or not isinstance(b, Real) or not math.isfinite(b)
           for b in (delay_i_ps, delay_j_ps)):
        raise ValueError("explicit finite delays in ps required")
    # Python ints also avoid signed int64 overflow for an extreme difference.
    return math.fsum((float(int(timestamp_i) - int(timestamp_j)), -float(delay_i_ps), float(delay_j_ps)))
