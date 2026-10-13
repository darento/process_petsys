"""Frozen legacy defects, without importing untracked reference scripts."""
import hashlib
import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "data/golden/timing/cornell_legacy_probes.json"


@pytest.mark.fr("008-FR-2", "008-FR-3")
def test_frozen_legacy_probe_integrity():
    provenance = json.loads(FIXTURE.with_name(FIXTURE.name + ".provenance.json").read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    data = json.loads(FIXTURE.read_text())
    assert data["geometry"]["original_xyz_mm"] == [160, -8, 90]
    assert data["geometry"]["optimized_xyz_mm"] == [160, 12, 110]
    assert data["thin_cylinder"]["original_result"] == [True, -933, 933]


@pytest.mark.fr("bug-cornell-skew-geometry")
@pytest.mark.xfail(strict=True, reason="bug-cornell-skew-geometry: optimized transform is not equivalent; legacy scripts unfixed")
def test_legacy_transform_non_equivalence():
    geometry = json.loads(FIXTURE.read_text())["geometry"]
    assert geometry["optimized_xyz_mm"] == geometry["original_xyz_mm"]


@pytest.mark.fr("bug-cornell-skew-sign")
@pytest.mark.xfail(strict=True, reason="bug-cornell-skew-sign: additive legacy LUT passed to subtractive consumer; migration pending")
def test_legacy_additive_table_consumer_mismatch():
    sign = json.loads(FIXTURE.read_text())["additive_table_to_subtractive_consumer"]
    assert sign["consumer_dt_ps"] == sign["correct_dt_ps"]


@pytest.mark.fr("bug-cornell-ctr-sign")
@pytest.mark.xfail(strict=True, reason="bug-cornell-ctr-sign: pointSource uses mixed signs; separate script fix pending")
def test_legacy_point_source_mixed_sign():
    sign = json.loads(FIXTURE.read_text())["point_source"]
    assert sign["legacy_corrected_dt_ps"] == sign["correct_dt_ps"]


@pytest.mark.fr("bug-cornell-ctr-sign")
@pytest.mark.xfail(strict=True, reason="bug-cornell-ctr-sign: mixed signs violate endpoint swap; separate script fix pending")
def test_legacy_point_source_endpoint_swap():
    sign = json.loads(FIXTURE.read_text())["point_source"]
    assert sign["legacy_swapped_dt_ps"] == -sign["legacy_corrected_dt_ps"]
