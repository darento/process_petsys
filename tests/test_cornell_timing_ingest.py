"""Legacy-parity reference ingest for Cornell timing (spec 008 T4a).

Fixtures use the tracked full-system map/config and a synthetic `.encal`
(mu = 511 a.u., so keV equals a.u. exactly). Every expected outcome is pinned by
hand from the legacy definitions: each physical record counts once.
"""
from collections import Counter
import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest

from src.cornell.timing_ingest import (ACCEPTED, REJECTIONS, LegacySetup, legacy_cylinder,
                                       legacy_local_to_global, reference_counts, reference_outcomes,
                                       slab_bit)
from src.mapping_generator import ChannelType

pytestmark = pytest.mark.fr("008-FR-5", "008-FR-6")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cornell_full_system.yaml"
HIT = struct.Struct("<qfi")
T0 = 1_000_000_000_000
MISSING_SM, INVALID_SM = 13, 8     # SMs whose TIME channels lack / have invalid keV factors
FULL = [200.0, 150.0, 100.0, 61.0]  # 511 keV with mu = 511


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    base = tmp_path_factory.mktemp("encal")
    probe = LegacySetup.from_files(CONFIG, _encal(base / "probe.encal", {}), root=ROOT)
    rows = []
    for ch, (sm, _) in probe.sm_mM_map.items():
        if ChannelType.TIME in probe.chtype_map[ch] and sm != MISSING_SM:
            for slab in range(16):
                rows.append(((ch, slab), 0.0 if sm == INVALID_SM else 511.0))
    return LegacySetup.from_files(CONFIG, _encal(base / "r.encal", dict(rows)), root=ROOT)


def _encal(path, factors):
    lines = ["ID(t_ch, slab)\tmu\tsigma"] + [f"({k[0]}, {k[1]})\t{v}\t1.0" for k, v in factors.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def channels(setup, sm, mm):
    keyed = [ch for ch, key in setup.sm_mM_map.items() if tuple(key) == (sm, mm)]
    time = sorted((ch for ch in keyed if ChannelType.TIME in setup.chtype_map[ch]),
                  key=lambda ch: setup.local_map[ch][2])
    energy = sorted((ch for ch in keyed if ChannelType.ENERGY in setup.chtype_map[ch]),
                    key=lambda ch: setup.local_map[ch][1])
    return time, energy


def side(setup, sm, mm=5, time=None, energy=FULL, stamp=T0, extra=()):
    """One detector side: TIME hits {slab position: a.u.} then ENERGY hits."""
    time = {3: 12.0, 4: 6.0} if time is None else time
    t_ch, e_ch = channels(setup, sm, mm)
    hits = [(stamp + pos, value, t_ch[pos]) for pos, value in time.items()]
    hits += [(stamp + 500 + i, value, e_ch[i]) for i, value in enumerate(energy)]
    return hits + list(extra)


def write(path, records):
    with open(path, "wb") as handle:
        for det1, det2 in records:
            handle.write(struct.pack("<BB", len(det1), len(det2)))
            for hit in det1 + det2:
                handle.write(HIT.pack(*hit))
    return path


def run(setup, path, **kwargs):
    return list(reference_outcomes(path, setup, file_index=0, **kwargs))


def accepted_pair(setup, dt=250):
    # SM 1 (cassette 0, z=0) against SM 16 (cassette 5, z=0): diametric LOR.
    return side(setup, 1), side(setup, 16, stamp=T0 - dt)


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
def random_records(setup, count, seed):
    """Plausible opposed sides with randomized defects; reaches every label."""
    rng = np.random.default_rng(seed)
    unmapped = next(ch for ch in range(10 ** 6) if ch not in setup.chtype_map)
    cache = {}

    def module_hits(sm, mm, stamp, total, n_time, zero):
        if (sm, mm) not in cache:
            cache[sm, mm] = channels(setup, sm, mm)
        t_ch, e_ch = cache[sm, mm]
        p = int(rng.integers(0, 8))
        positions = [] if n_time == 0 else [p] if n_time == 1 else [p, p + int(rng.choice([-1, 1, 2]))]
        hits = [(stamp + q, 7.0 if rng.random() < 0.15 else float(rng.uniform(1, 20)), t_ch[q])
                for q in positions if 0 <= q < 8]
        n_energy = int(rng.integers(3, 7))
        for i, value in zip(rng.choice(len(e_ch), n_energy, replace=False), rng.dirichlet(np.ones(n_energy)) * total):
            draw = rng.random()
            value = 0.0 if zero else -1.0 if draw < 0.03 else float("nan") if draw < 0.05 else float(value)
            hits.append((stamp + 500, value, e_ch[i]))
        return hits

    def one_side(sm, stamp):
        zero = rng.random() < 0.03
        hits = module_hits(sm, int(rng.integers(0, 16)), stamp, float(rng.normal(511, 110)),
                           int(rng.integers(1, 3)), zero)
        if rng.random() < 0.25:
            hits += module_hits(sm, int(rng.integers(0, 16)), stamp, float(rng.uniform(0, 500)),
                                int(rng.integers(0, 3)), zero)
        if rng.random() < 0.02:
            hits.append((stamp, 5.0, unmapped))
        return [hits[i] for i in rng.permutation(len(hits))]

    out = []
    for _ in range(count):
        cassette, z = int(rng.integers(0, 10)), int(rng.integers(0, 3))
        draw = rng.random()
        opposite = 3 * ((cassette + 5) % 10)
        sm1 = MISSING_SM if draw < 0.03 else INVALID_SM if draw < 0.06 else 3 * cassette + z
        draw = rng.random()
        sm2 = (opposite + z if draw < 0.75 else opposite + int(rng.integers(0, 3)) if draw < 0.85
               else 3 * ((cassette + int(rng.integers(1, 10))) % 10) + z if draw < 0.95 else sm1)
        stamp = T0 + int(rng.integers(0, 10 ** 9))
        out.append((one_side(sm1, stamp), one_side(sm2, stamp - int(rng.integers(-16000, 16000)))))
    return out


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
