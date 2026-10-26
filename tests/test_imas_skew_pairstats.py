"""FR-7 pair statistics on synthetic observation sets with known pair counts (spec 011 T7)."""
import json

import numpy as np
import pytest

from imas_skew_helpers import CONFIG
from src.imas.skew import pairstats as ps
from src.imas.skew.setup import ImasSetup
from src.timing.cache import OBSERVATION_DTYPE, PAIR_COUNT_DTYPE, ObservationSet, pair_partition

pytestmark = pytest.mark.fr("011-FR-7")


@pytest.fixture(scope="module")
def setup():
    return ImasSetup.from_files(CONFIG)


def observation_set(directory, rows, partitions=3):
    """Level-2 layout written directly (as ``build_observations`` leaves it)."""
    directory.mkdir()
    part = pair_partition(rows["channel_i"], rows["channel_j"], partitions)
    keys, counts = np.unique(np.stack([rows["channel_i"], rows["channel_j"]]), axis=1, return_counts=True)
    table = np.zeros(keys.shape[1], PAIR_COUNT_DTYPE)
    table["channel_i"], table["channel_j"], table["count"] = keys[0], keys[1], counts
    np.save(directory / "pair_counts.npy", table)
    chunks = []
    for p in range(partitions):
        selected = rows[part == p]
        chunks.append(1 if len(selected) else 0)
        if len(selected):
            np.save(directory / f"p{p:03d}_c00000.npy", selected)
    (directory / "manifest.json").write_text(json.dumps({"fingerprint": {"partitions": partitions},
                                                         "chunks": chunks}), encoding="utf-8")
    return ObservationSet(directory)


def rows_for(pairs, rng, shift=0.0, sigma=100.0):
    """``pairs``: {(i, j): events}; branches at TOF [-1100, -1000] and [1000, 1100] ps, half each."""
    out = []
    for (i, j), count in pairs.items():
        r = np.zeros(count, OBSERVATION_DTYPE)
        r["channel_i"], r["channel_j"] = i, j
        r["shell_tof_ps"] = [-1100.0, -1000.0, 1000.0, 1100.0]
        near = np.arange(count) % 2 == 0
        tof = np.where(near, rng.uniform(-1100, -1000, count), rng.uniform(1000, 1100, count))
        r["dt_ps"] = np.rint(tof + shift + rng.normal(0, sigma, count)).astype(np.int64)
        out.append(r)
    return np.concatenate(out)


def test_counts_components_grouping_and_projection(tmp_path, setup):
    t = setup.time_channels
    # channel 0 lies in ASIC t[0] // 64; pairs: a 4-channel chain, an isolated pair, and counts 1..40
    pairs = {(t[0], t[1]): 1, (t[1], t[2]): 2, (t[2], t[3]): 5, (t[100], t[200]): 40}
    observations = observation_set(tmp_path / "obs", rows_for(pairs, np.random.default_rng(1)))
    report = ps.pair_statistics(observations, setup, ldat_bytes=10 ** 11, project_bytes=1e12, fit=False)
    assert report["accepted_observations"] == 48 and report["distinct_pairs"] == 4
    bins = {(b["from"], b["below"]): b["count"] for b in report["observations_per_pair_histogram"]}
    assert bins[(1, 2)] == 1 and bins[(2, 3)] == 1 and bins[(5, 10)] == 1 and bins[(30, 50)] == 1
    assert report["pairs_with_at_least"] == {"3": 2, "6": 1, "10": 1, "30": 1, "100": 0}
    assert report["time_channels_observed"] == 6
    assert report["partners_per_channel"]["max"] == 2 and report["partners_per_channel"]["min"] == 1
    assert report["components_all_pairs"] == {"count": 2, "sizes_top5": [4, 2]}
    assert report["components_pairs_ge_30"] == {"count": 1, "sizes_top5": [2]}
    projection = report["projection"]
    assert projection["scale"] == 10.0 and projection["accepted_observations"] == 480
    assert projection["observed_pairs_reaching"] == {"6": 4, "30": 2, "100": 1}
    lookup = setup.lookup()
    sm0 = lookup.sm[t[0]]
    assert report["coverage"]["observations_per_ring"][sm0 // 24] >= 1 + 1
    assert sum(report["coverage"]["observations_per_ring"]) == 2 * 48


def test_branch_resolution_uses_spec008_pair_fit(tmp_path, setup):
    t = setup.time_channels
    pairs = {(t[0], t[1]): 80, (t[2], t[3]): 4, (t[4], t[5]): 1}
    observations = observation_set(tmp_path / "obs", rows_for(pairs, np.random.default_rng(2), shift=300.0))
    resolution = ps.branch_resolution(observations)
    assert resolution["model"]["source"].startswith("estimated")
    assert resolution["status_counts"]["usable"] == 1 and resolution["observations_in_usable_pairs"] == 80
    assert resolution["usable_by_pair_events"]["1-6"] == {"pairs": 2, "usable": 0}
    assert resolution["median_branch_separation_ps"] == pytest.approx(2100.0, abs=150.0)
