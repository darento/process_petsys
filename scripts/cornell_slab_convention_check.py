#!/usr/bin/env python3
"""Regression check for the Cornell slab convention (spec 002, task B1).

Owner-confirmed convention: time channel at position p covers slab 2p at
X_p - 0.8 mm and slab 2p+1 at X_p + 0.8 mm. When p fires strongest and its
neighbour p-1 (lower X) also fires, the event is in slab 2p.

Both the scalar ``src.utils.get_slab_cornell`` and the vectorized
``src.utils_fixed.get_slab_cornell_vectorized`` must agree with that
convention on edge, one-channel, adjacent and non-adjacent cases.

Run from the repo root:
    python scripts/cornell_slab_convention_check.py
"""

from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mapping_generator import ChannelType, map_factory  # noqa: E402
from src.utils import get_slab_cornell  # noqa: E402
from src.utils_fixed import get_slab_cornell_vectorized, get_time_channel_mask  # noqa: E402

MAP = "maps/cornell_map_full_system.yaml"
HALF_SLAB = 0.8
# Time hits in the selected minimodule: position -> energy (a.u.), plus the
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


def _geometry():
    coords, sm_mm, types, _ = map_factory(MAP)
    by_pos = {coords[ch][2]: ch for ch, (sm, mm) in sm_mm.items()
              if sm == 0 and mm == 0 and ChannelType.TIME in types[ch]}
    assert sorted(by_pos) == list(range(8)), by_pos
    return coords, sm_mm, types, by_pos


def _expected_x(coords, channel, slab):
    return coords[channel][0] + (-HALF_SLAB if slab % 2 == 0 else HALF_SLAB)


def main():
    coords, sm_mm, types, by_pos = _geometry()
    time_mask = get_time_channel_mask(types)
    hit = np.dtype([("timestamp", "<i8"), ("energy", "<f4"), ("channelID", "<i4")])
    failures = []

    def check(label, ok, detail):
        print(f"[{'ok' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            failures.append(label)

    for label, hits, expected in CASES:
        top = max(hits, key=hits.get)
        top_ch = by_pos[top]
        det = [[0, energy, by_pos[pos]] for pos, energy in hits.items()]
        chunk = np.zeros(1, dtype=[("hits", hit, (16,))])
        chunk["hits"]["channelID"] = -1
        for i, (_, energy, channel) in enumerate(det):
            chunk["hits"]["energy"][0, i] = energy
            chunk["hits"]["channelID"][0, i] = channel
        mm_id = np.array([0 * 1000 + 0], dtype=np.int32)

        for name, run in (
            ("scalar", lambda: get_slab_cornell(det, types, coords)),
            ("vectorized", lambda: tuple(v[0] for v in get_slab_cornell_vectorized(
                chunk, mm_id, sm_mm, time_mask, coords))[:3]),
        ):
            trials = 200 if expected == "random" else 1
            seen = set()
            random.seed(1)
            np.random.seed(1)
            for _ in range(trials):
                slab, _flag, x = run()
                slab = None if slab is None or slab == -1 else int(slab)
                seen.add(slab)
                if expected is None:
                    ok = slab is None
                    detail = f"slab {slab} (expected unresolved)"
                else:
                    want = {2 * top, 2 * top + 1} if expected == "random" else {expected}
                    ok = (slab in want and x is not None
                          and abs(float(x) - _expected_x(coords, top_ch, slab)) < 1e-4)
                    detail = (f"slab {slab}, x {float(x):.2f} mm; expected slab "
                              f"{sorted(want)} at x {_expected_x(coords, top_ch, slab if slab in want else min(want)):.2f}"
                              if x is not None else f"slab {slab}, x None")
                if not ok:
                    break
            if ok and expected == "random":
                ok = seen == {2 * top, 2 * top + 1}
                detail = f"both slabs {sorted(seen)} seen over {trials} trials, x consistent"
            check(f"{name}: {label}", ok, detail)

    total = 2 * len(CASES)
    print(f"{'PASS' if not failures else 'FAIL'}: {total - len(failures)}/{total} slab convention checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
