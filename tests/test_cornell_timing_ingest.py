"""Cornell timing ingest: legacy parity (T4a/T4b), production holdout and geometry (T4c).

Fixtures use the tracked full-system map/config and a synthetic `.encal`
(mu = 511 a.u., so keV equals a.u. exactly). Every expected outcome is pinned by
hand from the legacy definitions: each physical record counts once.
"""
from collections import Counter
import json
import math

import numpy as np
import pytest

from src.cornell.timing_ingest import (ACCEPTED, FEATURE_INT, REJECTIONS, LegacySetup, legacy_cylinder,
                                       legacy_local_to_global, reference_counts, reference_outcomes,
                                       slab_bit)
from cornell_timing_helpers import (CONFIG, DRAWING_SOURCE, FULL, INVALID_SM, MISSING_SM, ROOT, SUMMARY, T0, accepted_pair,
                                    channels, make_setup, production, random_records, shifted, side, write)

pytestmark = pytest.mark.fr("008-FR-5", "008-FR-6")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("encal"))


def run(setup, path, **kwargs):
    return list(reference_outcomes(path, setup, file_index=0, **kwargs))


def test_reference_pins_each_legacy_outcome(tmp_path, setup):
    unmapped = next(ch for ch in range(10 ** 6) if ch not in setup.chtype_map)
    det1, det2 = accepted_pair(setup)
    t_ch0, e_ch0 = channels(setup, 1, 0)
    records = [
        ("accepted", (det1, det2)),
        ("unmapped channel", (det1 + [(T0, 5.0, unmapped)], det2)),
        ("side min channels", (det1, side(setup, 16, energy=FULL[:3]))),
        # Two minimodules, all ENERGY a.u. zero: legacy returns no minimodule (crash path).
        ("no positive minimodule energy",
         (side(setup, 1, energy=[0.0] * 4) + [(T0, 0.0, e_ch0[0]), (T0, 0.0, e_ch0[1])], det2)),
        # Highest-energy minimodule has ENERGY hits only.
        ("minimodule min channels",
         (side(setup, 1, energy=[1.0] * 4) + [(T0, 50.0, ch) for ch in e_ch0[:4]], det2)),
        # Max TIME hits share slab position 3: |dt| = 15,000 ps exactly (strict window).
        ("coincidence window", (det1, side(setup, 16, stamp=T0 - 15000))),
        ("accepted", (det1, side(setup, 16, stamp=T0 - 14999))),
        ("unresolved slab", (side(setup, 1, time={2: 12.0, 5: 6.0}), det2)),
        ("missing energy calibration", (side(setup, MISSING_SM), det2)),
        ("invalid energy calibration", (det1, side(setup, INVALID_SM))),
        ("energy window", (det1, side(setup, 16, energy=[300.0, 30.0, 20.0, 7.0]))),
        # Same SM, transversely separated minimodules: XY line x=160 misses r=140.
        ("no cylinder crossing", (det1, side(setup, 1, mm=0))),
        # Same XY, axially separated minimodules: purely axial LOR, angle 0.
        ("opposition angle", (det1, side(setup, 1, mm=6))),
        ("opposition angle", (det1, side(setup, 7))),
    ]
    path = write(tmp_path / "pins.ldat", [record for _, record in records])
    outcomes = run(setup, path)
    assert [label for _, label, _ in outcomes] == [label for label, _ in records]
    assert [index for index, _, _ in outcomes] == list(range(len(records)))
    counts = reference_counts(path, setup, file_index=0)
    assert sum(counts.values()) == len(records) and set(counts) == {ACCEPTED, *REJECTIONS}
    features = outcomes[0][2]
    t_ch1, _ = channels(setup, 1, 5)
    t_ch16, _ = channels(setup, 16, 5)
    # Max-energy TIME hits are slab position 3; position 4 neighbour -> slab 2*3+1.
    assert features.time_channels == (t_ch1[3], t_ch16[3]) and features.slabs == (7, 7)
    assert features.dt_ps == 250 and type(features.dt_ps) is int
    assert features.energy_kev == (511.0, 511.0)
    assert features.modules == ((1, 5), (16, 5))
    # Original planar transform: z = sm_z + 48 - x; both endpoints at radius 160.
    for (x, y), xyz in zip(features.local_mm, features.global_mm):
        assert xyz[2] == pytest.approx(48 - x)
        assert math.hypot(xyz[0], xyz[1]) == pytest.approx(math.hypot(160, y - 48))


def test_energy_cut_is_explicit(tmp_path, setup):
    det1, det2 = accepted_pair(setup)
    cases = [
        ([300.0, 30.0, 20.0, 7.0], "energy window"),        # 357 keV: strict lower bound
        ([300.0, 30.0, 20.0, 7.25], ACCEPTED),
        ([400.0, 200.0, 50.0, 15.0], "energy window"),      # 665 keV: strict upper bound
        ([400.0, 200.0, 50.0, 14.75], ACCEPTED),
    ]
    records = []
    for energy, _ in cases:
        records += [(side(setup, 1, energy=energy), det2), (det1, side(setup, 16, energy=energy))]
    # en_min_ch (0.2) is read but not applied: a 0.1 a.u. ENERGY hit still counts.
    records.append((side(setup, 1, energy=[200.0, 150.0, 160.9, 0.1]), det2))
    # The legacy reader drops negative/NaN hits before any cut: 3 ENERGY hits remain.
    records.append((side(setup, 1, energy=[300.0, 150.0, 61.0, -1.0]), det2))
    records.append((side(setup, 1, energy=[300.0, 150.0, 61.0, float("nan")]), det2))
    labels = [label for _, label, _ in run(setup, write(tmp_path / "energy.ldat", records))]
    assert labels == [label for _, label in cases for _ in range(2)] + [
        ACCEPTED, "side min channels", "side min channels"]
    assert setup.energy_range == (357.0, 665.0) and setup.en_min_ch == 0.2
    assert setup.provenance["en_min_ch_applied"] is False


def test_missing_calibration_and_unmapped_channels(tmp_path, setup):
    det1, det2 = accepted_pair(setup)
    unmapped = next(ch for ch in range(10 ** 6) if ch not in setup.chtype_map)
    records = [(det1, [(T0, 5.0, unmapped)] + det2), (side(setup, MISSING_SM), det2),
               (det1, side(setup, MISSING_SM)), (side(setup, INVALID_SM), det2)]
    labels = [label for _, label, _ in run(setup, write(tmp_path / "missing.ldat", records))]
    assert labels == ["unmapped channel", "missing energy calibration",
                      "missing energy calibration", "invalid energy calibration"]
    # A truncated record is a reasoned failure after complete records, never a partial outcome.
    path = tmp_path / "trunc.ldat"
    path.write_bytes(write(tmp_path / "t0.ldat", [(det1, det2)]).read_bytes() + b"\x02")
    seen = []
    with pytest.raises(ValueError, match="truncated LDAT pair header"):
        for outcome in reference_outcomes(path, setup, file_index=0):
            seen.append(outcome[1])
    assert seen == [ACCEPTED]


def test_full_file_and_explicit_prefix(tmp_path, setup):
    det1, det2 = accepted_pair(setup)
    # More records than the legacy hidden 500,000 stop; empty sides fail fast.
    count = 500_010
    path = tmp_path / "long.ldat"
    with open(path, "wb") as handle:
        handle.write(b"\x00\x00" * (count - 1))
    with open(path, "ab") as handle:
        handle.write(write(tmp_path / "last.ldat", [(det1, det2)]).read_bytes())
    counts = reference_counts(path, setup, file_index=0)
    assert sum(counts.values()) == count
    assert counts["side min channels"] == count - 1 and counts[ACCEPTED] == 1
    prefix = reference_counts(path, setup, file_index=0, max_records=7)
    assert sum(prefix.values()) == 7 and prefix[ACCEPTED] == 0


def test_legacy_angle_boundary_and_axial_obliquity():
    probes = json.loads((ROOT / "tests/data/golden/timing/cornell_legacy_probes.json").read_text())
    geometry = probes["geometry"]
    assert legacy_local_to_global(*geometry["local_xy_mm"], geometry["sm_parameters"]) == pytest.approx(
        geometry["original_xyz_mm"])
    thin = probes["thin_cylinder"]
    label, tof1, tof2 = legacy_cylinder(np.array(thin["endpoint_i_mm"]), np.array(thin["endpoint_j_mm"]),
                                        thin["radius_mm"])
    assert (label is None, tof1, tof2) == tuple(thin["original_result"])
    i = np.array([160.0, 0.0, 0.0])
    at = lambda deg: np.array([160 * math.cos(math.radians(deg)), 160 * math.sin(math.radians(deg)), 0.0])
    assert legacy_cylinder(i, at(165.001), 140)[0] is None
    assert legacy_cylinder(i, at(164.999), 140)[0] == "opposition angle"
    # 3D angle: same-z outer SMs (115 degrees) fail although XY-opposite; opposite z passes.
    assert legacy_cylinder(np.array([160.0, 0, 102]), np.array([-160.0, 0, 102]), 140)[0] == "opposition angle"
    assert legacy_cylinder(np.array([160.0, 0, -102]), np.array([-160.0, 0, 102]), 140)[0] is None
    # A crossing beyond the segment (endpoint inside the cylinder) is its own label.
    assert legacy_cylinder(np.array([-139.0, 0, 0]), np.array([160.0, 0, 0]), 140)[0] == "crossing outside segment"
    assert legacy_cylinder(np.array([160.0, 0, 0]), np.array([150.0, 30.0, 0]), 140)[0] == "no cylinder crossing"


def test_seeded_slab_is_record_keyed(tmp_path, setup):
    bits = [slab_bit(0, record, side) for record in range(200) for side in (0, 1)]
    assert set(bits) == {0, 1} and 150 < sum(bits) < 250
    assert slab_bit(3, 12, 1) == slab_bit(3, 12, 1) and slab_bit(3, 12, 1, seed=1) in (0, 1)
    for bad in ((-1, 0, 0), (0, -1, 0), (0, 0, 2), (2 ** 20, 0, 0)):
        with pytest.raises(ValueError):
            slab_bit(*bad)
    # One interior TIME channel (position 3): legacy coin flip picks slab 6 or 7.
    det1 = side(setup, 1, time={3: 12.0})
    det2 = side(setup, 16, time={3: 12.0}, stamp=T0 - 100)
    path = write(tmp_path / "coin.ldat", [(det1, det2)] * 64)
    outcomes = run(setup, path)
    expected = [(6 + slab_bit(0, r, 0), 6 + slab_bit(0, r, 1)) for r in range(64)]
    assert [f.slabs for _, _, f in outcomes] == expected
    assert len(set(expected)) > 1
    # Independent of block size and of the prefix that contains the record.
    assert [f.slabs for _, _, f in run(setup, path, block_bytes=37)] == expected
    assert [f.slabs for _, _, f in run(setup, path, max_records=9)] == expected[:9]
    # Edge positions are deterministic regardless of the bit.
    edge = write(tmp_path / "edge.ldat", [(side(setup, 1, time={0: 12.0}), det2)] * 8)
    assert {f.slabs[0] for _, _, f in run(setup, edge)} == {0}


# --- T4b: compiled kernels reproduce the reference ---------------------------
def _worker_outcomes(path, config, encal):
    from src.cornell.timing_ingest import CompiledLookup, compiled_legacy_outcomes
    setup = LegacySetup.from_files(config, encal, root=ROOT)
    lookup = CompiledLookup.from_setup(setup)
    return [(r, label) for r, label, _ in compiled_legacy_outcomes(path, setup, lookup, file_index=0)]


def compiled(setup, path, lookup=None, **kwargs):
    from src.cornell.timing_ingest import CompiledLookup, compiled_legacy_outcomes
    lookup = lookup or CompiledLookup.from_setup(setup)
    return list(compiled_legacy_outcomes(path, setup, lookup, file_index=0, **kwargs))


def test_reference_compiled_selection_parity(tmp_path, setup):
    N = 6000
    from src.cornell.timing_ingest import CompiledLookup, compiled_batches
    lookup = CompiledLookup.from_setup(setup)
    det1, det2 = accepted_pair(setup)
    _, e_ch0 = channels(setup, 1, 0)
    # Equal-energy minimodules: compiled takes the first seen and flags the tie.
    tie = side(setup, 1, energy=[100.0] * 4) + [(T0, 100.0, ch) for ch in e_ch0[:4]] + [
        (T0 + 9, 1.0, channels(setup, 1, 0)[0][2])]
    path = write(tmp_path / "rand.ldat", random_records(setup, N, seed=11) + [(det1, det2), (tie, det2)])
    reference = run(setup, path)
    fast = compiled(setup, path, lookup)
    ties = set()
    for features in compiled_batches(path, lookup, file_index=0):
        ties |= set(features.column("record")[features.column("module_tie") == 1].tolist())
    assert N + 1 in ties
    labels = Counter(label for _, label, _ in reference)
    # Fixtures reach every pre-geometry label and the legacy geometry labels.
    assert set(labels) >= set(REJECTIONS[:11]) | {ACCEPTED}, labels
    assert labels[ACCEPTED] >= 100
    mismatches = [(r, f) for r, f in zip(reference, fast) if r != f and r[0] not in ties]
    assert not mismatches, mismatches[:3]
    assert len(reference) == len(fast) == N + 2
    # Exact equality includes keV, centroid and int64 dt on every accepted record.
    assert fast[N] == reference[N] and fast[N][1] == ACCEPTED


def test_seed_independent_of_chunks_and_workers(tmp_path, setup):
    from concurrent.futures import ProcessPoolExecutor
    from src.cornell.timing_ingest import CompiledLookup
    lookup = CompiledLookup.from_setup(setup)
    det1 = side(setup, 1, time={3: 12.0})
    det2 = side(setup, 16, time={4: 12.0}, stamp=T0 - 100)
    records = random_records(setup, 600, seed=5) + [(det1, det2)] * 50
    paths = [write(tmp_path / f"f{i}.ldat", records) for i in range(2)]
    base = compiled(setup, paths[0], lookup)
    for options in ({"block_bytes": 97, "batch_records": 7}, {"batch_records": 1}, {"block_bytes": 1 << 20}):
        assert compiled(setup, paths[0], lookup, **options) == base
    assert compiled(setup, paths[0], lookup, max_records=300) == base[:300]
    coin = [f.slabs for _, label, f in base[600:] if label == ACCEPTED]
    assert len(set(coin)) > 1
    # Separate worker processes return the same outcomes for the same file index.
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_worker_outcomes, [str(p) for p in paths], [str(CONFIG)] * 2,
                                [str(setup.provenance["encal"])] * 2))
    assert results[0] == results[1] == [(r, label) for r, label, _ in base]


# --- T4c: holdout roles, separate geometry stage, operational coverage -------
@pytest.mark.fr("008-FR-16")
def test_holdout_excludes_energy_calibration_prefix(tmp_path, setup):
    from src.cornell.timing_ingest import (PASSED_PRE_GEOMETRY, RUN7_ENERGY_CALIBRATION_PREFIX,
                                           CompiledLookup, Partition, code_label)
    lookup = CompiledLookup.from_setup(setup)
    det1, det2 = accepted_pair(setup)
    start, end = T0 + 10_000, T0 + 20_000
    # Anchor = first hit of detector 1 (detector 2 when detector 1 is empty).
    at = lambda anchor, d1=det1, d2=det2: (shifted(d1, anchor - det1[0][0]), shifted(d2, anchor - det1[0][0]))
    cases = [
        ("energy calibration prefix", at(start + 1)),
        ("energy calibration prefix", at(end + 5)),        # prefix wins over the edge guard
        ("energy calibration prefix", at(start)),
        ("file edge guard", at(start - 1)),                # guards are inclusive bounds
        (PASSED_PRE_GEOMETRY, at(start)),
        (PASSED_PRE_GEOMETRY, at(end)),
        ("file edge guard", at(end + 1)),
        ("side min channels", ([], shifted(det2, start + 7 - det2[0][0]))),
        ("file edge guard", ([], shifted(det2, end + 7 - det2[0][0]))),
        ("file edge guard", ([], [])),                     # no hit, no anchor
        ("energy window", at(start + 9, d1=side(setup, 1, energy=[300.0, 30.0, 20.0, 7.0]))),
    ]
    path = write(tmp_path / "holdout.ldat", [record for _, record in cases])
    test = Partition(3, "test", start, end, 3)
    features = production(path, lookup, test)
    assert [code_label(code) for code in features.codes] == [label for label, _ in cases]
    assert features.column("record").tolist() == [4, 5]
    for options in ({"batch_records": 2}, {"block_bytes": 41}):
        again = production(path, lookup, test, **options)
        assert np.array_equal(again.codes, features.codes) and np.array_equal(again.ints, features.ints)
    # A train role keeps its prefix; the 1 us edge guards still apply.
    train = production(path, lookup, Partition(0, "train", start, end, 0))
    assert [code_label(code) for code in train.codes[:4]] == [
        PASSED_PRE_GEOMETRY, "file edge guard", PASSED_PRE_GEOMETRY, "file edge guard"]
    # Frozen run7 roles: exactly the validation/test files carry the calibrate-run prefix bound.
    holdout = json.loads(SUMMARY.read_text(encoding="utf-8"))["holdout"]
    partitions = [Partition.from_holdout(holdout, index, RUN7_ENERGY_CALIBRATION_PREFIX) for index in range(16)]
    assert {p.file_index for p in partitions if p.role != "train"} == set(RUN7_ENERGY_CALIBRATION_PREFIX)
    assert {p.file_index: p.excluded_prefix_records for p in partitions if p.excluded_prefix_records} == \
        RUN7_ENERGY_CALIBRATION_PREFIX
    assert partitions[5].role == "validation" and partitions[3].role == "test"
    assert partitions[0].anchor_start_ps == holdout["partitions"][0]["anchor_start_ps"]
    with pytest.raises(ValueError, match="prefix bound required"):
        Partition.from_holdout(holdout, 11, {5: 1})
    with pytest.raises(ValueError, match="qualified"):
        Partition.from_holdout(holdout | {"qualified": False}, 0, RUN7_ENERGY_CALIBRATION_PREFIX)


def _scalar_geometry(frames, source, row_i, row_f):
    """Expected label/endpoints from the tracked scalar geometry functions."""
    from src.cornell.timing_geometry import legacy_opposition_accepts, shell_intervals
    p = frames[row_i[FEATURE_INT.index("sm_1")]].to_global(row_f[2], row_f[3])
    q = frames[row_i[FEATURE_INT.index("sm_2")]].to_global(row_f[4], row_f[5])
    segments = shell_intervals(p, q, source)
    label = ("no shell crossing" if not segments else "opposition angle" if not legacy_opposition_accepts(p, q)
             else "one shell segment" if len(segments) != 2 else ACCEPTED)
    return label, np.concatenate([p, q]), segments


@pytest.mark.fr("008-FR-2", "008-FR-5")
def test_geometry_stage_hypotheses_without_reread(tmp_path, setup):
    from src.cornell.timing_geometry import AnnularSource, GeometryError, build_detector_frames
    from src.cornell.timing_ingest import (GEOMETRY_REJECTIONS, CompiledLookup, FeatureBatch,
                                           GeometryHypothesis, Partition, apply_geometry)
    lookup = CompiledLookup.from_setup(setup)
    det1, det2 = accepted_pair(setup)
    path = write(tmp_path / "geometry.ldat", random_records(setup, 3000, seed=23) + [(det1, det2)])
    features = production(path, lookup, Partition(0, "train", -2 ** 62, 2 ** 62, 0), batch_records=500)
    path.unlink()                       # hypotheses below cannot re-read LDAT
    source = AnnularSource(**DRAWING_SOURCE)
    hypotheses = {sign: GeometryHypothesis.faceted(f"faceted {sign:+d}", CONFIG, setup, axial_sign=-1,
                                                   transverse_sign=sign, source=source) for sign in (1, -1)}
    results = {sign: apply_geometry(features, hypothesis) for sign, hypothesis in hypotheses.items()}
    labels_of = lambda result: [((ACCEPTED,) + GEOMETRY_REJECTIONS)[code] for code in result.codes]
    for sign, result in results.items():
        expected = [_scalar_geometry(hypotheses[sign].frames, source, row_i, row_f)
                    for row_i, row_f in zip(features.ints.tolist(), features.floats.tolist())]
        assert labels_of(result) == [label for label, _, _ in expected]
        assert np.allclose(result.endpoints_mm, [xyz for _, xyz, _ in expected], rtol=0, atol=1e-9)
        for k, (_, _, segments) in enumerate(expected):
            assert result.shell_segments[k] == len(segments)
            fractions = [x for s in segments for x in (s.fraction_start, s.fraction_end)]
            tof = [x for s in segments for x in (s.tof_start_ps, s.tof_end_ps)]
            assert np.allclose(result.shell_fractions[k, :len(fractions)], fractions, rtol=0, atol=1e-12)
            assert np.allclose(result.shell_tof_ps[k, :len(tof)], tof, rtol=0, atol=1e-6)
            assert np.isnan(result.shell_tof_ps[k, len(tof):]).all()
        counts = result.counts()
        assert sum(counts.values()) == len(features.ints)
        assert counts[ACCEPTED] >= 50 and counts["no shell crossing"] and counts["opposition angle"]
    # Handedness candidates mirror each SM: different endpoints from the same stored rows.
    assert not np.allclose(results[1].endpoints_mm, results[-1].endpoints_mm)
    # Pinned diametric pair (last row): two shell segments, one per side of the phantom.
    last = results[1]
    assert last.codes[-1] == 0 and last.shell_segments[-1] == 2
    assert last.shell_tof_ps[-1, 0] < 0 < last.shell_tof_ps[-1, 3]
    # A transversely displaced shell crosses this LOR once; bad endpoints are labelled.
    pinned = FeatureBatch(0, np.zeros(3, np.int8), features.ints[[-1, -1, -1]].copy(),
                          features.floats[[-1, -1, -1]].copy())
    pinned.ints[1, FEATURE_INT.index("sm_2")] = 99
    pinned.floats[2, 2] = 1e6
    shifted_source = AnnularSource(**DRAWING_SOURCE | {"center_mm": (100.0, 0.0, 0.0)})
    moved = GeometryHypothesis("displaced shell", hypotheses[1].frames, shifted_source)
    assert labels_of(apply_geometry(pinned, moved)) == ["one shell segment", "invalid endpoint", "invalid endpoint"]
    assert _scalar_geometry(moved.frames, shifted_source, pinned.ints[0].tolist(),
                            pinned.floats[0].tolist())[0] == "one shell segment"
    config = __import__("yaml").safe_load(CONFIG.read_text(encoding="utf-8"))
    planar = build_detector_frames(config, setup.sm_mM_map, local_origin_mm=(48, 48), local_axes="time-axial",
                                   axial_sign=-1, crystal_thickness_mm=20)
    with pytest.raises(GeometryError, match="faceted"):
        apply_geometry(features, GeometryHypothesis("planar", planar, source))


@pytest.mark.fr("008-FR-6", "008-FR-16")
def test_operational_time_coverage_and_down_regions(tmp_path, setup):
    from src.cornell.timing_geometry import AnnularSource
    from src.cornell.timing_ingest import (CompiledLookup, GeometryHypothesis, Partition, apply_geometry,
                                           time_channel_coverage)
    down = json.loads(SUMMARY.read_text(encoding="utf-8"))["known_unavailable_regions"]
    assert down == [[14, 0], [14, 1], [14, 4], [14, 5]]
    time_ok, time_down, time_unpopulated = channels(setup, 1, 5)[0][3], channels(setup, 14, 0)[0][3], \
        channels(setup, 2, 2)[0][0]
    coverage = time_channel_coverage({time_ok: 3, time_down: 2, time_unpopulated: 1}, setup, down)
    # 30 SMs x 16 minimodules x 8 TIME; 10 half-populated SMs; 4 down minimodules in SM14.
    assert (coverage["operational_time_channels"], coverage["unpopulated_time_channels"],
            coverage["down_time_channels"]) == (3840 - 640 - 32, 640, 32)
    per_channel = coverage["endpoints_per_operational_channel"]
    assert per_channel[time_ok] == 3 and time_down not in per_channel and time_unpopulated not in per_channel
    assert (coverage["down_region_endpoints"], coverage["unpopulated_endpoints"]) == (2, 1)
    assert len(coverage["operational_without_endpoints"]) == 3168 - 1
    with pytest.raises(ValueError, match="non-TIME"):
        time_channel_coverage({channels(setup, 1, 5)[1][0]: 1}, setup, down)
    with pytest.raises(ValueError, match="not in the selected map"):
        time_channel_coverage({}, setup, [(99, 0)])
    # End to end: accepted endpoints per selected TIME channel; a down-region hit is reported apart.
    lookup = CompiledLookup.from_setup(setup)
    det1, det2 = accepted_pair(setup)
    # Down SM14 MM5 (cassette 4, outer z) against SM27 MM5 (cassette 9, opposite z): oblique, accepted.
    down_pair = (side(setup, 14, mm=5), side(setup, 27, stamp=T0 - 80))
    path = write(tmp_path / "coverage.ldat", [(det1, det2)] * 3 + [down_pair])
    features = production(path, lookup, Partition(0, "train", -2 ** 62, 2 ** 62, 0))
    hypothesis = GeometryHypothesis.faceted("faceted +1", CONFIG, setup, axial_sign=-1, transverse_sign=1,
                                            source=AnnularSource(**DRAWING_SOURCE))
    accepted = apply_geometry(features, hypothesis).codes == 0
    assert accepted.tolist() == [True] * 4
    endpoints = Counter(features.column("time_channel_1")[accepted].tolist()
                        + features.column("time_channel_2")[accepted].tolist())
    coverage = time_channel_coverage(endpoints, setup, down)
    assert coverage["down_region_endpoints"] == 1 and coverage["unpopulated_endpoints"] == 0
    assert sum(coverage["endpoints_per_operational_channel"].values()) == 2 * 4 - 1
