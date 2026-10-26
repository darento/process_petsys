"""Shared synthetic IMAS skew fixtures (spec 011): config copies, sides and compact LDAT records."""
from pathlib import Path
import struct

import yaml

from src.mapping_generator import ChannelType

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
