"""Offline sign and real shared consumer checks; native PETsys remains pending."""
import numpy as np
import pytest

from src.cornell.timing_convention import corrected_dt_ps
from src.cornell.timing_geometry import C_MM_PER_PS, geometric_tof_ps
from src.mapping_generator import ChannelType
from src.utils import apply_skew

pytestmark = pytest.mark.fr("008-FR-3")


def test_positive_delay_and_endpoint_swap():
    assert corrected_dt_ps(1300, 1000, 300, 0) == 0
    assert corrected_dt_ps(1300, 1000, 200, -100) == 0
    assert corrected_dt_ps(1307, 1000, 200, -100) == 7
    assert corrected_dt_ps(1000, 1307, -100, 200) == -7
    assert corrected_dt_ps(1307, 1000, 2200, 1900) == 7  # common gauge doesn't affect differences
    assert corrected_dt_ps(1007, 1000, 1e18, 1e18) == 7  # summation preserves a small residual
    types = {11: [ChannelType.TIME], 12: [ChannelType.TIME], 13: [ChannelType.ENERGY]}
    hits = [(1307, 1., 11), (1000, 1., 12), (1307, 2., 13)]
    original = list(hits)
    corrected = apply_skew(hits, types, {11: 200, 12: -100, 13: 500})
    assert corrected[0][0] - corrected[1][0] == 7
    assert corrected[2] == original[2] and hits == original
    # Old additive corrections a=-b require explicit migration before this consumer.
    additive = {11: -150, 12: 150}
    migrated = apply_skew([(1300, 1., 11), (1000, 1., 12)], types,
                          {ch: -value for ch, value in additive.items()})
    assert migrated[0][0] - migrated[1][0] == 0


def test_large_integer_timestamps_and_geometric_tof():
    base = np.int64(2**60)
    assert corrected_dt_ps(base + 301, base + 1, 299.75, -0.25) == 0
    assert corrected_dt_ps(np.int64(2**63 - 1), np.int64(-2**63), 0, 0) == float(2**64 - 1)
    i, j, source = (160, 0, 0), (-160, 0, 0), (30, 0, 0)
    tof = geometric_tof_ps(source, i, j)
    assert tof == pytest.approx(-60 / C_MM_PER_PS)
    # Exact known-position flight difference plus injected delays; source branch
    # is provided by truth rather than chosen using measured timing.
    raw_dt = round(tof) + 300
    residual = corrected_dt_ps(10**15 + raw_dt, 10**15, 200, -100) - tof
    assert abs(residual) <= 0.5
    assert geometric_tof_ps(source, j, i) == pytest.approx(-tof)
    assert corrected_dt_ps(10**15, 10**15 + raw_dt, -100, 200) + tof == pytest.approx(-residual)


@pytest.mark.parametrize("values", [(1000.0, 900, 0, 0), (True, 900, 0, 0),
                                   (1000, 900, None, 0), (1000, 900, float('nan'), 0),
                                   (1000, 900, float('inf'), 0)])
def test_unknown_timing_convention_is_rejected(values):
    with pytest.raises(ValueError):
        corrected_dt_ps(*values)
