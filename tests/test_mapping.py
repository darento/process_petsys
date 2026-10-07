"""src.mapping_generator.map_factory on every tracked map (spec 006 FR-7).

Expectations are computed from each map's own YAML (mod_feb_map, channel lists,
FEM, mM_channels, sum_rows_cols), never from per-map hard-coded counts, so a map
edit changes the expectation instead of silently passing. IMAS and Cornell
numbering differ through their mod_feb_map; nothing here assumes 48 modules or
256 channels per SuperModule.
"""

from collections import Counter, defaultdict
from functools import lru_cache

import pytest
import yaml

from helpers import MAPS
from src.mapping_generator import ChannelType, map_factory
from src.utils import get_electronics_nums

MAP_FILES = ["imas1DAQ_map.yaml", "imas2DAQ_map.yaml", "default_map.yaml", "cpp_map.yaml",
             "erc_map.yaml", "cornell_map_full_system.yaml", "cornell_map_full_system_old.yaml"]
TIME, ENERGY = ChannelType.TIME, ChannelType.ENERGY


@lru_cache(maxsize=None)
def _load(name):
    path = MAPS / name
    with open(path, encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    group = ("Time", "Energy") if "Time" in spec else ("channels_j1", "channels_j2")
    return spec, spec[group[0]], spec[group[1]], map_factory(str(path))


@pytest.fixture(params=MAP_FILES)
def mapped(request):
    return _load(request.param)


@pytest.mark.fr("006-FR-7")
def test_every_channel_mapped_once(mapped):
    spec, first, second, (local, sm_mm, types, _) = mapped
    assert len(set(first)) == len(first) and len(set(second)) == len(second)
    assert not set(first) & set(second)
    assert local.keys() == sm_mm.keys() == types.keys()
    # No channel overwritten by another module or by the other channel list.
    assert len(sm_mm) == len(spec["mod_feb_map"]) * (len(first) + len(second))


@pytest.mark.fr("006-FR-7")
def test_channel_ids_follow_mod_feb_map(mapped):
    spec, first, second, (_, sm_mm, _, fem) = mapped
    assert fem.channels == spec["channels"]
    for module, (port, slave, feb_port) in spec["mod_feb_map"].items():
        base = 131072 * port + 4096 * slave + fem.channels * feb_port
        for channel in (*first, *second):
            absolute = base + channel
            assert sm_mm[absolute][0] == module, absolute
            assert get_electronics_nums(absolute)[:2] == (port, slave), absolute


@pytest.mark.fr("006-FR-7")
def test_channel_types_follow_sum_rows_cols(mapped):
    spec, first, second, (_, sm_mm, types, _) = mapped
    for module, (port, slave, feb_port) in spec["mod_feb_map"].items():
        base = 131072 * port + 4096 * slave + spec["channels"] * feb_port
        for channel in first:
            assert types[base + channel] == ([TIME] if spec["sum_rows_cols"] else [TIME, ENERGY])
        for channel in second:
            assert types[base + channel] == ([ENERGY] if spec["sum_rows_cols"] else [TIME, ENERGY])


@pytest.mark.fr("006-FR-7")
def test_minimodules_are_complete(mapped):
    spec, first, second, (local, sm_mm, types, _) = mapped
    per_mm = defaultdict(lambda: defaultdict(list))
    for channel, (module, minimodule) in sm_mm.items():
        per_mm[(module, minimodule)][tuple(types[channel])].append(local[channel])
    n_mm = (len(first) + len(second)) // (2 * spec["mM_channels"]) if spec["sum_rows_cols"] \
        else (len(first) + len(second)) // spec["mM_channels"]
    assert {mm for _, mm in per_mm} == set(range(n_mm))
    assert len(per_mm) == len(spec["mod_feb_map"]) * n_mm
    for key, by_type in per_mm.items():
        if spec["sum_rows_cols"]:
            # mM_channels time + mM_channels energy channels, at positions 0..mM_channels-1 each.
            assert set(by_type) == {(TIME,), (ENERGY,)}, key
            for coords in by_type.values():
                assert all(len(c) == 3 for c in coords), key
                assert sorted(c[2] for c in coords) == list(range(spec["mM_channels"])), key
        else:
            assert {k: len(v) for k, v in by_type.items()} == {(TIME, ENERGY): spec["mM_channels"]}, key
            assert all(len(c) == 2 for coords in by_type.values() for c in coords), key


@pytest.mark.fr("006-FR-7")
def test_imas_and_cornell_numbering_distinct():
    channels = {name: set(_load(name)[3][1]) for name in ("imas1DAQ_map.yaml", "cornell_map_full_system.yaml")}
    modules = {name: Counter(sm for sm, _ in _load(name)[3][1].values()) for name in channels}
    imas, cornell = channels.values()
    assert imas != cornell and not cornell <= imas
    assert len(modules["imas1DAQ_map.yaml"]) != len(modules["cornell_map_full_system.yaml"])
