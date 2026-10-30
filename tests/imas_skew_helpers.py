"""Shared synthetic IMAS skew fixtures (spec 011): config copies, sides and compact LDAT records."""
import json
from pathlib import Path
import struct

import numpy as np
import yaml

from src.mapping_generator import ChannelType
from src.timing.cache import (OBSERVATION_DTYPE, PAIR_COUNT_DTYPE, ObservationSet, observation_dtype,
                              pair_partition, walk_fingerprint)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/imas_2DAQ.yaml"
HIT = struct.Struct("<qfi")
T0 = 1_000_000_000_000
FULL = [200.0, 150.0, 100.0, 40.0, 21.0]     # five ENERGY hits (min_ch 5), 511 a.u. = 511 keV at mu 511


def config_copy(path, **changes):
    """``imas_2DAQ.yaml`` with ``changes`` applied and an absolute map path."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["map_file"] = str(ROOT / config["map_file"])
    config.update(changes)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


_MODULES = {}


def channels(setup, sm, mm):
    """(TIME channels by slab index, ENERGY channels by axial y) of one minimodule."""
    if id(setup.sm_mM_map) not in _MODULES:
        index = {}
        for ch, key in setup.sm_mM_map.items():
            index.setdefault(tuple(key), []).append(ch)
        _MODULES[id(setup.sm_mM_map)] = (setup.sm_mM_map, index)
    keyed = _MODULES[id(setup.sm_mM_map)][1].get((sm, mm), [])
    time = sorted((ch for ch in keyed if ChannelType.TIME in setup.chtype_map[ch]),
                  key=lambda ch: setup.local_map[ch][2])
    energy = sorted((ch for ch in keyed if ChannelType.ENERGY in setup.chtype_map[ch]),
                    key=lambda ch: setup.local_map[ch][1])
    return time, energy


def side(setup, sm, mm=5, time=None, energy=FULL, stamp=T0, extra=()):
    """One detector side: TIME hits {slab: a.u.} (first listed slab gets ``stamp``) then ENERGY hits."""
    time = {3: 12.0, 4: 6.0} if time is None else time
    t_ch, e_ch = channels(setup, sm, mm)
    hits = [(stamp + 10 * i, value, t_ch[slab]) for i, (slab, value) in enumerate(time.items())]
    hits += [(stamp + 500 + i, value, e_ch[i]) for i, value in enumerate(energy)]
    return hits + list(extra)


def write(path, records):
    with open(path, "wb") as handle:
        for det1, det2 in records:
            handle.write(struct.pack("<BB", len(det1), len(det2)))
            for hit in det1 + det2:
                handle.write(HIT.pack(*hit))
    return path


def mu_file(path, values):
    """``ID\\tmu`` energy calibration (KevConverter ``mu``)."""
    lines = ["ID\tmu"] + [f"{ch}\t{mu}" for ch, mu in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- synthetic IMAS-geometry observations (spec 011 T9-T11) --------------------------------------------------

class Columns:
    """``column(name)`` view of plain arrays (geometry adapter input)."""

    def __init__(self, **values):
        self.values = values

    def column(self, name):
        return self.values[name]


def channel_table(setup, sms, modules):
    """TIME channels of ``modules`` in ``sms``: channel, SM, minimodule, slab x and the minimodule's y range."""
    rows = []
    for sm in sms:
        for mm in modules:
            time, energy = channels(setup, sm, mm)
            ys = [setup.local_map[ch][1] for ch in energy]
            rows += [(ch, sm, mm, setup.local_map[ch][0], min(ys), max(ys)) for ch in time]
    return np.array(rows, dtype=[("ch", "i8"), ("sm", "i8"), ("mm", "i8"), ("x", "f8"), ("lo", "f8"),
                                 ("hi", "f8")])


def sides(pool, count, rng, *, partner_pool=None, rings=None, n_azimuth=24):
    """Random side pairs from ``pool`` to the opposite SM (+-1) in ``partner_pool`` (same ring, or one of
    ``rings``), lower channel first, with random axial positions: Columns for a geometry adapter."""
    partner_pool = pool if partner_pool is None else partner_pool
    a = pool[rng.integers(0, len(pool), count)]
    ring = a["sm"] // n_azimuth if rings is None else np.asarray(rings)[rng.integers(0, len(rings), count)]
    k = (a["sm"] % n_azimuth + n_azimuth // 2 + rng.integers(-1, 2, count)) % n_azimuth + n_azimuth * ring
    b = np.empty(count, pool.dtype)
    for n, sm in enumerate(k):
        options = partner_pool[partner_pool["sm"] == sm]
        if not len(options):
            options = partner_pool
        b[n] = options[rng.integers(0, len(options))]
    swap = a["ch"] > b["ch"]
    first, second = np.where(swap, b, a), np.where(swap, a, b)
    keep = first["ch"] != second["ch"]
    first, second = first[keep], second[keep]
    return Columns(ch_1=first["ch"], ch_2=second["ch"], sm_1=first["sm"], x_1=first["x"],
                   y_1=rng.uniform(first["lo"], first["hi"]), sm_2=second["sm"], x_2=second["x"],
                   y_2=rng.uniform(second["lo"], second["hi"]))


def lors(geometry, pool, count, rng, *, partner_pool=None, n_azimuth=24):
    """Accepted LORs from ``pool`` to the opposite SM (+-1) in ``partner_pool``: (channel_i < channel_j, TOF)."""
    columns = sides(pool, count, rng, partner_pool=partner_pool, n_azimuth=n_azimuth)
    result = geometry.apply(columns)
    ok = result.codes == 0
    return columns.column("ch_1")[ok], columns.column("ch_2")[ok], result.shell_tof_ps[ok]


def skew_events(ci, cj, tof, delays, rng, *, noise=200.0, background=0.1, window=10_000.0, zero_width=False,
                branch=None):
    """``dt = t + b_i - b_j + noise`` with t uniform in one branch segment; flat background inside the window."""
    near = rng.random(len(ci)) < 0.5 if branch is None else np.full(len(ci), branch == "near")
    lo, hi = np.where(near, tof[:, 0], tof[:, 2]), np.where(near, tof[:, 1], tof[:, 3])
    if zero_width:
        point = 0.5 * (tof[:, [0, 2]] + tof[:, [1, 3]])
        tof = np.repeat(point, 2, axis=1)
        lo = hi = np.where(near, point[:, 0], point[:, 1])
    shift = np.array([delays[c] for c in ci]) - np.array([delays[c] for c in cj])
    dt = rng.uniform(lo, hi) + shift + rng.normal(0, noise, len(ci)) if noise else lo + shift
    flat = rng.random(len(ci)) < background
    dt[flat] = rng.uniform(-window, window, flat.sum())
    inside = np.abs(dt) < window
    return ci[inside], cj[inside], dt[inside], tof[inside]


def observation_set(directory, ci, cj, dt, tof, partitions=2, files=None, walk=None, roles=None, candidates=None):
    """Minimal canonical ``ObservationSet`` (train role) from arrays with ``channel_i < channel_j``; ``files``
    gives each row its input-file index (default 0); ``walk`` ({column: array}, spec 012) makes a walk-carrying
    set; ``roles`` gives each row its role code; ``candidates`` ({cand_*_i/j: events x slots}, spec 014, with
    ``walk``) makes a candidate-carrying set."""
    slots = 0 if candidates is None else candidates["cand_channel_i"].shape[1]
    rows = np.zeros(len(ci), OBSERVATION_DTYPE if walk is None else observation_dtype(True, slots))
    for name, values in {**(walk or {}), **(candidates or {})}.items():
        rows[name] = values
    if roles is not None:
        rows["role"] = roles
    rows["record"] = np.arange(len(ci))
    rows["channel_i"], rows["channel_j"], rows["dt_ps"] = ci, cj, np.round(dt)
    rows["shell_tof_ps"] = tof
    if files is not None:
        rows["file"] = files
    part = pair_partition(rows["channel_i"], rows["channel_j"], partitions)
    directory.mkdir(parents=True)
    for p in range(partitions):
        np.save(directory / f"p{p:03d}_c00000.npy", rows[part == p])
    keys, counts = np.unique(np.stack([rows["channel_i"], rows["channel_j"]]), axis=1, return_counts=True)
    table = np.zeros(keys.shape[1], PAIR_COUNT_DTYPE)
    table["channel_i"], table["channel_j"], table["count"] = keys[0], keys[1], counts
    np.save(directory / "pair_counts.npy", table)
    (directory / "manifest.json").write_text(json.dumps({"fingerprint": walk_fingerprint({"partitions": partitions},
                                                                                         walk is not None, slots),
                                                         "chunks": [1] * partitions}), encoding="utf-8")
    return ObservationSet(directory)


SIDES = [(mm, slab) for mm in (5, 6) for slab in range(4)]     # 8 TIME channels per SM, SMs 50 and 62


def job_splits(base):
    """Two runs x three splits of diametric ring-2 coincidences with a 60 a.u. photopeak (spec 011 job tests)."""
    from src.imas.skew.setup import ImasSetup
    setup = ImasSetup.from_files(CONFIG)
    rng = np.random.default_rng(13)

    def energy():
        return rng.normal(60.0, 4.0) if rng.random() < 0.85 else rng.uniform(5.0, 120.0)

    paths = []
    for index in range(6):
        records = []
        for _ in range(3200):
            (mm_a, slab_a), (mm_b, slab_b) = SIDES[rng.integers(8)], SIDES[rng.integers(8)]
            sm_a, sm_b = 50, 62
            ea, eb = energy(), energy()
            records.append((side(setup, sm_a, mm_a, time={slab_a: 9.0}, energy=[ea / 5] * 5),
                            side(setup, sm_b, mm_b, time={slab_b: 9.0}, energy=[eb / 5] * 5,
                                 stamp=T0 - int(rng.integers(-900, 900)))))
        paths.append(write(base / f"run{index // 3}_s{index % 3}.ldat", records))
    return paths


def job_file(path, splits, **changes):
    """Job file over ``job_splits``: per run encal+train, train, heldout."""
    from src.imas.skew import job as ij
    runs = []
    for r in range(2):
        a, b, c = splits[3 * r:3 * r + 3]
        runs.append({"name": f"r{r + 1}", "splits": [{"path": str(a), "roles": ["encal", "train"]},
                                                      {"path": str(b), "roles": ["train"]},
                                                      {"path": str(c), "roles": ["heldout"]}]})
    data = {"schema": ij.SCHEMA, "config": str(CONFIG), "runs": runs, "estimators": ["em", "pairs"],
            "per_run_solves": True, **changes}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
