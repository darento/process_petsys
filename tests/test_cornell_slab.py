"""Cornell slab convention (spec 002, task B1), migrated from scripts/cornell_slab_convention_check.py.

Owner-confirmed convention: time channel at position p covers slab 2p at
X_p - 0.8 mm and slab 2p+1 at X_p + 0.8 mm. When p fires strongest and its
neighbour p-1 (lower X) also fires, the event is in slab 2p.

Both the scalar ``src.utils.get_slab_cornell`` and the vectorized
``src.utils_fixed.get_slab_cornell_vectorized`` must agree with that convention
on edge, one-channel, adjacent and non-adjacent cases, on the tracked
``maps/cornell_map_full_system.yaml``. Energies are PETsys a.u.
"""

import random

import numpy as np
import pytest

from helpers import load_map
from src.mapping_generator import ChannelType
from src.utils import get_slab_cornell
from src.utils_fixed import get_slab_cornell_vectorized, get_time_channel_mask

MAP = "cornell_map_full_system.yaml"
HALF_SLAB = 0.8
HIT = np.dtype([("timestamp", "<i8"), ("energy", "<f4"), ("channelID", "<i4")])
# Time hits in SuperModule 0, minimodule 0: position -> energy (a.u.), plus the
# expected slab (None = unresolved; "random" = either slab of the pair).
CASES = [
    ("edge 0, one channel", {0: 14.0}, 0),
    ("edge 0, two channels", {0: 14.0, 1: 5.0}, 1),
    ("edge 7, one channel", {7: 14.0}, 15),
    ("edge 7, two channels", {7: 14.0, 6: 5.0}, 14),
    ("middle, one channel", {3: 14.0}, "random"),
    ("adjacent, neighbour p-1", {3: 12.0, 2: 5.0}, 6),
    ("adjacent, neighbour p+1", {3: 12.0, 4: 5.0}, 7),
    ("non-adjacent top two", {3: 12.0, 6: 6.0}, None),
]


@pytest.fixture(scope="module")
def geometry():
    coords, sm_mm, types, _ = load_map(MAP)
    by_pos = {coords[ch][2]: ch for ch, (sm, mm) in sm_mm.items()
              if sm == 0 and mm == 0 and ChannelType.TIME in types[ch]}
    assert sorted(by_pos) == list(range(8)), by_pos
    return coords, sm_mm, types, by_pos


def _expected_x(coords, channel, slab):
    return coords[channel][0] + (-HALF_SLAB if slab % 2 == 0 else HALF_SLAB)


def _runner(name, hits, geometry):
    coords, sm_mm, types, by_pos = geometry
    det = [[0, energy, by_pos[pos]] for pos, energy in hits.items()]
    if name == "scalar":
        return lambda: get_slab_cornell(det, types, coords)
    chunk = np.zeros(1, dtype=[("hits", HIT, (16,))])
    chunk["hits"]["channelID"] = -1
    for i, (_, energy, channel) in enumerate(det):
        chunk["hits"]["energy"][0, i] = energy
        chunk["hits"]["channelID"][0, i] = channel
    mm_id = np.array([0 * 1000 + 0], dtype=np.int32)
    time_mask = get_time_channel_mask(types)
    return lambda: tuple(v[0] for v in get_slab_cornell_vectorized(
        chunk, mm_id, sm_mm, time_mask, coords))[:3]


@pytest.mark.fr("bug-cornell-slab-convention", "002-FR-17")
@pytest.mark.parametrize("name", ["scalar", "vectorized"])
@pytest.mark.parametrize("label, hits, expected", CASES, ids=[c[0] for c in CASES])
def test_slab_convention(geometry, name, label, hits, expected):
    coords, _, _, by_pos = geometry
    top = max(hits, key=hits.get)
    top_ch = by_pos[top]
    run = _runner(name, hits, geometry)
    trials = 200 if expected == "random" else 1
    want = {2 * top, 2 * top + 1} if expected == "random" else {expected}
    seen = set()
    random.seed(1)
    np.random.seed(1)
    for _ in range(trials):
        slab, _flag, x = run()
        slab = None if slab is None or slab == -1 else int(slab)
        seen.add(slab)
        if expected is None:
            assert slab is None, f"slab {slab} (expected unresolved)"
            continue
        assert slab in want, f"slab {slab}, expected {sorted(want)}"
        assert x is not None, f"slab {slab}, x None"
        assert abs(float(x) - _expected_x(coords, top_ch, slab)) < 1e-4, \
            f"slab {slab}, x {float(x):.2f} mm, expected {_expected_x(coords, top_ch, slab):.2f}"
    if expected == "random":
        assert seen == want, f"slabs seen over {trials} trials: {sorted(seen)}"
