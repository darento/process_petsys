"""Bounded numeric cache and canonical observations for Cornell timing (spec 008 T5).

Level 1 caches every holdout-kept record passing the pre-geometry cuts; level 2
holds one canonical row per geometry-accepted physical coincidence. Expected
rows come from the T4c in-memory production/geometry path on the same files.
"""
import json
import os
import tracemalloc

import numpy as np
import pytest

from cornell_timing_helpers import (CONFIG, DRAWING_SOURCE, ROOT, T0, accepted_pair, make_setup, production,
                                    random_records, side, write)
from src.cornell import timing_cache as tc
from src.cornell.timing_geometry import AnnularSource
from src.cornell.timing_ingest import (PASSED_PRE_GEOMETRY, CompiledLookup, GeometryHypothesis, Partition,
                                       apply_geometry)

pytestmark = pytest.mark.fr("008-FR-7", "008-FR-8")
EVERYTHING = (-2 ** 62, 2 ** 62)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return make_setup(tmp_path_factory.mktemp("encal"))


@pytest.fixture(scope="module")
def hypotheses(setup):
    source = AnnularSource(**DRAWING_SOURCE)
    return {sign: GeometryHypothesis.faceted(f"faceted {sign:+d}", CONFIG, setup, axial_sign=-1,
                                             transverse_sign=sign, source=source) for sign in (1, -1)}


def holdout(count, validation=()):
    return {"qualified": True, "roles_frozen_before_fitting": True,
            "partitions": [{"index": i, "role": "validation" if i in validation else "train",
                            "anchor_start_ps": EVERYTHING[0], "anchor_end_ps": EVERYTHING[1]}
                           for i in range(count)]}


def build(directory, setup, paths, *, validation=(), prefix=None, **options):
    inputs = [{"index": i, "path": path} for i, path in enumerate(paths)]
    return tc.build_feature_cache(directory, inputs, config_path=CONFIG, encal_path=setup.provenance["encal"],
                                  root=ROOT, holdout=holdout(len(paths), validation),
                                  energy_calibration_prefix=prefix or {}, **options)


def every_row(observations):
    return np.concatenate([observations.partition(p) for p in range(observations.partitions)])


def test_one_observation_per_coincidence(tmp_path, setup, hypotheses):
    det1, det2 = accepted_pair(setup)
    records = [random_records(setup, 1500, seed=31) + [(det1, det2)] * 5, random_records(setup, 1500, seed=32)]
    paths = [write(tmp_path / f"run_{i}.ldat", r) for i, r in enumerate(records)]
    cache = build(tmp_path / "cache", setup, paths, validation=(1,), prefix={1: 2}, block_rows=97, workers=2)
    assert cache.complete and len(cache.files[0]["blocks"]) > 2
    lookup = CompiledLookup.from_setup(setup)
    expected = set()
    for index, path in enumerate(paths):
        done = cache.files[index]
        assert done["records"] == len(records[index]) == sum(done["first_failure"].values())
        partition = Partition(index, done["role"], *EVERYTHING, 2 if index else 0)
        features = production(path, lookup, partition)
        assert done["rows"] == done["first_failure"][PASSED_PRE_GEOMETRY] == len(features.ints)
        accepted = apply_geometry(features, hypotheses[1]).codes == 0
        expected |= {(index, record) for record in features.column("record")[accepted].tolist()}
    assert cache.files[1]["first_failure"]["energy calibration prefix"] == 2
    observations = tc.build_observations(cache, hypotheses[1], tmp_path / "obs", partitions=4, buffer_rows=50,
                                         slice_rows=64)
    rows = every_row(observations)
    identities = list(zip(rows["file"].tolist(), rows["record"].tolist()))
    # Each accepted physical coincidence exactly once: no reversed view, no per-branch duplicate.
    assert len(identities) == len(set(identities)) and set(identities) == expected and len(expected) >= 50
    assert {index for index, _ in expected} == {0, 1}
    assert (rows["channel_i"] < rows["channel_j"]).all()
    for p in range(observations.partitions):
        part = observations.partition(p)
        assert (tc.pair_partition(part["channel_i"], part["channel_j"], 4) == p).all()
    counts = observations.pair_counts()
    assert counts["count"].dtype == np.uint64 and int(counts["count"].sum()) == len(rows)
    keys, n = np.unique(np.stack([rows["channel_i"], rows["channel_j"], rows["role"]]), axis=1, return_counts=True)
    assert np.array_equal(np.stack([counts["channel_i"], counts["channel_j"], counts["role"]]), keys)
    assert np.array_equal(counts["count"], n)
    manifest = observations.manifest
    assert manifest["observations"]["train"] + manifest["observations"]["validation"] == len(rows)
    assert manifest["geometry"]["train"]["accepted"] == manifest["observations"]["train"]
    assert sum(manifest["geometry"]["train"].values()) == cache.files[0]["rows"]


def test_orientation_swaps_features_and_dt(tmp_path, setup, hypotheses):
    det1, det2 = accepted_pair(setup, dt=250)
    path = write(tmp_path / "pair.ldat", [(det1, det2), (det2, det1)])
    cache = build(tmp_path / "cache", setup, [path])
    rows = every_row(tc.build_observations(cache, hypotheses[1], tmp_path / "obs", partitions=2))
    rows = rows[np.argsort(rows["record"])]
    assert rows["record"].tolist() == [0, 1] and sorted(rows["swapped"].tolist()) == [0, 1]
    for name in ("channel_i", "channel_j", "dt_ps", "sm_i", "mm_i", "sm_j", "mm_j", "slab_i", "slab_j",
                 "energy_kev_i", "energy_kev_j", "endpoint_i_mm", "endpoint_j_mm"):
        assert np.array_equal(rows[name][0], rows[name][1]), name
    assert np.allclose(rows["shell_tof_ps"][0], rows["shell_tof_ps"][1], rtol=0, atol=1e-6)
    # dt = t_i - t_j for the lower TIME channel i; SM1 fired 250 ps after SM16.
    sm1_first = det1[0][2] < det2[0][2]
    assert rows["dt_ps"][0] == (250 if sm1_first else -250)
    assert rows["sm_i"][0] == (1 if sm1_first else 16)
    # TOF = (|x - i| - |x - j|)/c: the crossing nearer i is negative; intervals ascend.
    tof = rows["shell_tof_ps"][0]
    assert tof[0] < tof[1] < 0 < tof[2] < tof[3]
    assert np.isclose(tof[0] + tof[3], 0, atol=40) and abs(tof[0]) == pytest.approx(1000, abs=100)


def test_restart_fingerprints_and_partial_blocks(tmp_path, setup, hypotheses, monkeypatch):
    det1, det2 = accepted_pair(setup)
    paths = [write(tmp_path / f"run_{i}.ldat", random_records(setup, 400, seed=40 + i) + [(det1, det2)])
             for i in range(2)]
    directory = tmp_path / "cache"
    cache = build(directory, setup, paths, block_rows=50)
    first = directory / "features" / "f000" / "done.json"
    stamp = first.stat().st_mtime_ns
    blocks = cache.files[1]["blocks"]
    # Interrupted file: its directory without the completing rename is not part of the cache.
    os.rename(directory / "features" / "f001", directory / "features" / "f001.partial")
    (directory / "features" / "f001.partial" / "done.json").unlink()
    interrupted = tc.FeatureCache(directory)
    assert not interrupted.complete and set(interrupted.files) == {0}
    with pytest.raises(tc.CacheMismatch, match="pending"):
        tc.build_observations(interrupted, hypotheses[1], tmp_path / "early")
    assert not (tmp_path / "early").exists()
    resumed = build(directory, setup, paths, block_rows=50)
    assert resumed.complete and first.stat().st_mtime_ns == stamp            # file 0 reused, not redone
    assert resumed.files[1]["blocks"] == blocks and not (directory / "features" / "f001.partial").exists()
    # Different seed, calibration or changed input is refused without touching the cache.
    manifest = (directory / "manifest.json").read_bytes()
    with pytest.raises(tc.CacheMismatch, match="fingerprint"):
        build(directory, setup, paths, seed=1)
    os.utime(paths[0], ns=(stamp, stamp + 10 ** 9))
    with pytest.raises(tc.CacheMismatch, match="fingerprint"):
        build(directory, setup, paths)
    assert (directory / "manifest.json").read_bytes() == manifest
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "skew_iteration_0.csv").write_text("user output\n")
    with pytest.raises(tc.CacheMismatch, match="not a timing cache"):
        build(unrelated, setup, paths)
    assert (unrelated / "skew_iteration_0.csv").read_text() == "user output\n"
    # Solver-side work reads only the cache: LDAT gone and the reader disabled.
    for path in paths:
        path.unlink()

    def no_reads(*args, **kwargs):
        raise AssertionError("LDAT read during cache-only stage")

    monkeypatch.setattr("src.ldat_stream.LdatStream.__init__", no_reads)
    observations = tc.build_observations(tc.FeatureCache(directory), hypotheses[1], tmp_path / "obs", partitions=3)
    built = (tmp_path / "obs" / "manifest.json").stat().st_mtime_ns
    again = tc.build_observations(tc.FeatureCache(directory), hypotheses[1], tmp_path / "obs", partitions=3)
    assert (tmp_path / "obs" / "manifest.json").stat().st_mtime_ns == built
    assert np.array_equal(every_row(again), every_row(observations))
    with pytest.raises(tc.CacheMismatch, match="different inputs"):
        tc.build_observations(tc.FeatureCache(directory), hypotheses[-1], tmp_path / "obs", partitions=3)
    # A stale partial build of the same set is replaced; an unrelated one is refused.
    mirrored = tc.build_observations(tc.FeatureCache(directory), hypotheses[-1], tmp_path / "mirror")
    mirrored_rows = every_row(mirrored)
    os.rename(tmp_path / "mirror", tmp_path / "stale.partial")
    with pytest.raises(tc.CacheMismatch, match="unrelated partial"):
        tc.build_observations(tc.FeatureCache(directory), hypotheses[-1], tmp_path / "stale")
    (tmp_path / "stale.partial" / "partial.json").write_text(
        json.dumps(mirrored.manifest["fingerprint"]), encoding="utf-8")
    rebuilt = tc.build_observations(tc.FeatureCache(directory), hypotheses[-1], tmp_path / "stale")
    assert np.array_equal(every_row(rebuilt), mirrored_rows)
    assert not (tmp_path / "stale.partial").exists()


def test_counts_above_int16_limit(tmp_path, setup, hypotheses):
    # The original stored per-pair counts in int16; 70,000 exceeds int16 and uint16.
    det1, det2 = accepted_pair(setup)
    pair = write(tmp_path / "one.ldat", [(det1, det2)]).read_bytes()
    path = tmp_path / "many.ldat"
    path.write_bytes(pair * 70_000)
    cache = build(tmp_path / "cache", setup, [path], block_rows=9_000)
    observations = tc.build_observations(cache, hypotheses[1], tmp_path / "obs", partitions=2, buffer_rows=8_000,
                                         slice_rows=5_000)
    counts = observations.pair_counts()
    assert len(counts) == 1 and counts["count"].dtype == np.uint64 and int(counts["count"][0]) == 70_000
    assert sum(observations.manifest["chunks"]) > 5 and observations.manifest["observations"]["train"] == 70_000


def _synthetic_cache(directory, hypothesis, rows, pairs, seed):
    """Level-1 layout written directly: opposed middle-ring SMs, many distinct channel pairs."""
    rng = np.random.default_rng(seed)
    centers = next(iter(hypothesis.frames.values())).map_centers_mm
    features = np.zeros(rows, tc.FEATURE_DTYPE)
    cassette = rng.integers(0, 10, rows)
    features["sm_1"], features["sm_2"] = 3 * cassette + 1, 3 * ((cassette + 5) % 10) + 1
    for side_ in ("1", "2"):
        features["x_" + side_] = np.asarray(centers[0])[rng.integers(1, 3, rows)] + rng.uniform(-5, 5, rows)
        features["y_" + side_] = np.asarray(centers[1])[rng.integers(1, 3, rows)] + rng.uniform(-5, 5, rows)
    pair = rng.integers(0, pairs, rows)
    features["time_channel_1"], features["time_channel_2"] = pair, pairs + pair
    features["record"] = np.arange(rows)
    features["dt_ps"] = rng.integers(-2000, 2000, rows)
    (directory / "features" / "f000").mkdir(parents=True)
    np.save(directory / "features" / "f000" / "b00000.npy", features)
    done = {"file": 0, "role": "train", "records": rows, "rows": rows, "first_failure": {},
            "blocks": [{"name": "b00000.npy", "rows": rows, "sha256": ""}]}
    (directory / "features" / "f000" / "done.json").write_text(json.dumps(done))
    (directory / "manifest.json").write_text(json.dumps({"fingerprint": {"inputs": [{"index": 0}]},
                                                         "fingerprint_sha256": f"synthetic {rows} {pairs}"}))
    return tc.FeatureCache(directory)


def test_memory_bound_with_more_events_and_pairs(tmp_path, hypotheses):
    peaks, observed = [], []
    for rows, pairs in ((50_000, 1_000), (200_000, 8_000)):
        cache = _synthetic_cache(tmp_path / f"cache_{rows}", hypotheses[1], rows, pairs, seed=rows)
        tracemalloc.start()
        observations = tc.build_observations(cache, hypotheses[1], tmp_path / f"obs_{rows}", partitions=8,
                                             buffer_rows=20_000, slice_rows=10_000)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        observed.append(sum(observations.manifest["observations"].values()))
        assert len(observations.pair_counts()) > 0.9 * pairs
    # Four times the events and eight times the pairs: peak set by slice/buffer bounds, not data size.
    assert observed[1] > 3.5 * observed[0] > 0
    assert peaks[1] < 1.5 * peaks[0] and peaks[1] < 64 * 2 ** 20, peaks
