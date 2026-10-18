"""Shared synthetic Cornell timing fixtures (spec 008 T4/T5 tests; T7a/T9 job runs)."""
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from src.cornell.skew import job as tj
from src.cornell.skew.ingest import LegacySetup
from src.mapping_generator import ChannelType

ROOT = Path(__file__).resolve().parents[1]
# tracked copy of the local configs/cornell_full_system.yaml (September map); configs/ is gitignored
CONFIG = ROOT / "tests/data/configs/cornell_september.yaml"
HIT = struct.Struct("<qfi")
T0 = 1_000_000_000_000
MISSING_SM, INVALID_SM = 13, 8     # SMs whose TIME channels lack / have invalid keV factors
FULL = [200.0, 150.0, 100.0, 61.0]  # 511 keV with mu = 511


def make_setup(base):
    """Full-system map/config with synthetic `.encal`: mu 511, SM 13 missing, SM 8 invalid."""
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


def accepted_pair(setup, dt=250):
    # SM 1 (cassette 0, z=0) against SM 16 (cassette 5, z=0): diametric LOR.
    return side(setup, 1), side(setup, 16, stamp=T0 - dt)


# T4b randomized records
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


# T4c/T5 production helpers
SUMMARY = ROOT / "specs/008-cornell-skew-calibration/run7-dataset-summary.json"
# Owner drawing Ph Norm D275L300 r01: fluid annulus, centered at YAML z=0.
DRAWING_SOURCE = {"inner_radius_mm": 137.5, "outer_radius_mm": 142.5, "axial_min_mm": -150.0,
                  "axial_max_mm": 150.0, "center_mm": (0.0, 0.0, 0.0), "axis": (0.0, 0.0, 1.0)}


def production(path, lookup, partition, **kwargs):
    from src.cornell.skew.ingest import FeatureBatch, production_batches
    batches = list(production_batches(path, lookup, partition, **kwargs))
    return FeatureBatch(0, np.concatenate([b.codes for b in batches]),
                        np.concatenate([b.ints for b in batches]), np.concatenate([b.floats for b in batches]))


def shifted(hits, delta):
    return [(t + delta, e, c) for t, e, c in hits]


# synthetic pooled-job runs (spec 008 T7a): SM 1 vs SM 16 with a known pair shift
FILES = 3
TOF_PS = 934.0  # diametric branch TOF for the 140 mm mid-radius shell


def coincidence_records(setup, rng, count, shift, mms=(5, 6)):
    """SM 1 vs SM 16, alternating minimodules: dt = shift +- branch TOF + kernel noise."""
    out = []
    for k in range(count):
        mm = mms[k % len(mms)]
        dt = int(round(shift + rng.choice((-TOF_PS, TOF_PS)) + rng.normal(0, 150)))
        stamp = T0 + k * 1_000_000
        out.append((side(setup, 1, mm=mm, stamp=stamp), side(setup, 16, mm=mm, stamp=stamp - dt)))
    return out


def make_run(base, setup, name, *, seed, accepted=70, shift=600.0, mms=(5, 6)):
    """FILES chronological splits plus a succeeded conversion ledger."""
    directory = base / name
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    prefix = f"norm_test_{name}_coincCompact_"
    artifacts = []
    for index in range(FILES):
        records = coincidence_records(setup, rng, accepted, shift, mms)
        offset = index * 10 ** 12  # one second per split, chronological
        records = [(shifted(a, offset), shifted(b, offset)) for a, b in records]
        records.sort(key=lambda r: min(h[0] for h in r[0] + r[1]))
        path = write(directory / f"{prefix}{index:08d}.ldat", records)
        artifacts.append({"kind": "ldat", "cleanup": "retained", "path": str(path), "size_bytes": path.stat().st_size,
                          "input_descriptor": {"format": "compact", "population": "coincidence"}})
    ledger = {"status": "succeeded", "attempts": [{"stage_id": "conversion", "status": "succeeded", "exit_code": 0,
                                                  "artifacts": artifacts}]}
    (directory / "run.json").write_text(json.dumps(ledger))
    return {"name": name, "directory": str(directory), "ledger": str(directory / "run.json"), "prefix": prefix,
            "energy_calibration_prefix": "none" if name != "ra" else {"1": 5, "2": 0}}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("job")
    setup = make_setup(base)
    return base, [make_run(base, setup, "ra", seed=1), make_run(base, setup, "rb", seed=2)]


def job_file(path, base, run_items, **overrides):
    spec = {"schema": tj.JOB_SCHEMA, "config": CONFIG.relative_to(ROOT).as_posix(), "encal": str(base / "r.encal"),
            "reference_run": run_items[0]["name"], "down_regions": [[14, 0]], "runs": run_items,
            "roles": {"validation": [1], "test": [2]}, "geometry": "selected", "work_directory": "work",
            "lut": "out/lut.csv", "workers": 1} | overrides
    path.write_text(json.dumps(spec))
    return path
