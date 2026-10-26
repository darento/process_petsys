"""Streamed observation sets equal feature cache + build_observations; resume and refusal (spec 011 T8)."""
import os
import shutil

import numpy as np
import pytest

from imas_skew_helpers import CONFIG, T0, config_copy, mu_file, side, write
from src.imas.skew import cache as ic
from src.imas.skew import geometry as imas_geometry
from src.imas.skew.geometry import ImasGeometry
from src.imas.skew.setup import ImasSetup
from src.timing import cache as core_cache

pytestmark = pytest.mark.fr("011-FR-6")


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    base = tmp_path_factory.mktemp("imas_stream")
    probe = ImasSetup.from_files(CONFIG)
    encal = mu_file(base / "mu.tsv", {ch: 511.0 for ch in probe.time_channels.tolist()})
    setup = ImasSetup.from_files(CONFIG, encal)
    rng = np.random.default_rng(11)
    records = []
    for i in range(360):
        k, ring = int(rng.integers(0, 24)), int(rng.integers(0, 5))
        a, b = 24 * ring + k, 24 * (4 - ring) + (k + 12) % 24
        records.append((side(setup, a, int(rng.integers(0, 16))),
                        side(setup, b, int(rng.integers(0, 16)), stamp=T0 - int(rng.integers(-900, 900)))))
        if i % 12 == 0:
            records.append((side(setup, a, 3, energy=[1.0] * 4), side(setup, b, 3)))      # side min channels
        if i % 15 == 0:
            records.append((side(setup, a, 2), side(setup, a + 1, 5)))                     # no shell crossing
    paths = [write(base / f"split_{i}.ldat", records[i::3]) for i in range(3)]
    return setup, encal, paths


def inputs(paths):
    return [{"index": i, "path": p} for i, p in enumerate(paths)]


def stream(directory, encal, paths, geometry, config=CONFIG, **options):
    return ic.stream_observations(directory, inputs(paths), geometry, config_path=config, encal_path=encal,
                                  holdout=ic.all_train(range(len(paths))), energy_calibration_prefix={},
                                  partitions=3, buffer_rows=40, **options)


def every_row(observations):
    rows = np.concatenate([observations.partition(p) for p in range(observations.partitions)])
    return rows[np.lexsort((rows["record"], rows["file"]))]


@pytest.fixture(scope="module")
def cached(files, tmp_path_factory):
    setup, encal, paths = files
    base = tmp_path_factory.mktemp("imas_cached")
    cache = ic.build_feature_cache(base / "features", inputs(paths), config_path=CONFIG, encal_path=encal,
                                   holdout=ic.all_train(range(3)), energy_calibration_prefix={}, block_rows=50)
    return ic.build_observations(cache, ImasGeometry.nominal(setup), base / "obs", partitions=3, buffer_rows=40)


@pytest.mark.parametrize("workers", [1, 2])
def test_streamed_equals_cached(tmp_path, files, cached, workers):
    setup, encal, paths = files
    streamed = stream(tmp_path / "s", encal, paths, ImasGeometry.nominal(setup), workers=workers)
    rows, expected = every_row(streamed), every_row(cached)
    assert len(rows) > 300 and rows.tobytes() == expected.tobytes()
    assert streamed.pair_counts().tobytes() == cached.pair_counts().tobytes()
    for key in ("observations", "geometry", "first_failure"):
        assert streamed.manifest[key] == cached.manifest[key], key
    assert streamed.manifest["first_failure"]["train"]["side min channels"] == 30
    assert streamed.manifest["geometry"]["train"]["no shell crossing"] > 0
    for p in range(3):          # every row sits in its pair partition
        part = streamed.partition(p)
        assert (core_cache.pair_partition(part["channel_i"], part["channel_j"], 3) == p).all()


def test_resume_redoes_only_interrupted_files(tmp_path, files, cached):
    setup, encal, paths = files
    directory = tmp_path / "s"
    stream(directory, encal, paths, ImasGeometry.nominal(setup))
    kept = directory / "files" / "f000" / "done.json"
    stamp = kept.stat().st_mtime_ns
    # file 1 interrupted mid-way, set never finished
    os.rename(directory / "files" / "f001", directory / "files" / "f001.partial")
    (directory / "files" / "f001.partial" / "done.json").unlink()
    (directory / "manifest.json").unlink()
    (directory / "pair_counts.npy").unlink()
    with pytest.raises(core_cache.CacheMismatch, match="incomplete"):
        core_cache.ObservationSet(directory)
    seen = []
    resumed = stream(directory, encal, paths, ImasGeometry.nominal(setup), progress=lambda r: seen.append(r["file"]))
    assert seen == [1] and kept.stat().st_mtime_ns == stamp
    assert not (directory / "files" / "f001.partial").exists()
    assert every_row(resumed).tobytes() == every_row(cached).tobytes()
    assert resumed.pair_counts().tobytes() == cached.pair_counts().tobytes()
    # a complete set is reopened without work
    again = stream(directory, encal, paths, ImasGeometry.nominal(setup), progress=lambda r: seen.append(r["file"]))
    assert seen == [1] and again.manifest == resumed.manifest


def test_changed_inputs_are_refused(tmp_path, files, monkeypatch):
    setup, encal, paths = files
    directory = tmp_path / "s"
    stream(directory, encal, paths, ImasGeometry.nominal(setup))
    snapshot = {p: p.read_bytes() for p in (directory / "fingerprint.json", directory / "manifest.json")}
    other_config = config_copy(tmp_path / "c.yaml", min_ch=4)
    other_encal = shutil.copy(encal, tmp_path / "mu2.tsv")
    with open(other_encal, "a", encoding="utf-8") as handle:
        handle.write("999999\t1.0\n")
    nominal = ImasGeometry.nominal(setup)
    attempts = [dict(config=other_config), dict(encal=other_encal), dict(window_ps=9_000),
                dict(geometry=ImasGeometry.from_setup(setup, axial_sign=1)), dict(paths=paths[:2])]
    for attempt in attempts:
        with pytest.raises(core_cache.CacheMismatch, match="different inputs"):
            stream(directory, attempt.get("encal", encal), attempt.get("paths", paths),
                   attempt.get("geometry", nominal), config=attempt.get("config", CONFIG),
                   **{k: v for k, v in attempt.items() if k == "window_ps"})
    edited = tmp_path / "geometry.py"
    edited.write_text(open(imas_geometry.__file__, encoding="utf-8").read() + "\n# edited\n", encoding="utf-8")
    monkeypatch.setattr(imas_geometry, "__file__", str(edited))
    with pytest.raises(core_cache.CacheMismatch, match="different inputs"):
        stream(directory, encal, paths, nominal)
    monkeypatch.undo()
    # an input rewritten in place (new size/mtime) is a different input
    copies = [shutil.copy(p, tmp_path / os.path.basename(p)) for p in paths]
    fresh = tmp_path / "s2"
    stream(fresh, encal, copies, nominal)
    with open(copies[2], "ab") as handle:
        handle.write(b"\0" * 16)
    with pytest.raises(core_cache.CacheMismatch, match="different inputs"):
        stream(fresh, encal, copies, nominal)
    assert all(p.read_bytes() == data for p, data in snapshot.items())
    # an unrelated non-empty directory is never used
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(core_cache.CacheMismatch, match="not empty"):
        stream(unrelated, encal, paths, nominal)
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "user data"
