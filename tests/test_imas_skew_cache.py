"""IMAS adapters for the core timing cache: resume, refusal and level-2 rows (spec 011 T6)."""
import os
import shutil

import numpy as np
import pytest

from imas_skew_helpers import CONFIG, T0, config_copy, mu_file, side, write
from src.imas.skew import cache as ic
from src.imas.skew import select as sel
from src.imas.skew.geometry import ImasGeometry
from src.imas.skew.setup import ImasSetup
from src.timing import cache as core_cache

pytestmark = pytest.mark.fr("011-FR-6", "011-FR-1")


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    base = tmp_path_factory.mktemp("imas")
    probe = ImasSetup.from_files(CONFIG)
    encal = mu_file(base / "mu.tsv", {ch: 511.0 for ch in probe.time_channels.tolist()})
    setup = ImasSetup.from_files(CONFIG, encal)
    rng = np.random.default_rng(5)
    records = []
    for i in range(300):
        k, ring = int(rng.integers(0, 24)), int(rng.integers(0, 5))
        a, b = 24 * ring + k, 24 * (4 - ring) + (k + 12) % 24          # diametric through the bar
        records.append((side(setup, a, int(rng.integers(0, 16))),
                        side(setup, b, int(rng.integers(0, 16)), stamp=T0 - int(rng.integers(-900, 900)))))
        if i % 10 == 0:
            records.append((side(setup, a, 3, energy=[1.0] * 4), side(setup, b, 3)))      # side min channels
    paths = [write(base / f"split_{i}.ldat", records[i::2]) for i in range(2)]
    return setup, encal, paths


def build(directory, encal, paths, config=CONFIG, **options):
    inputs = [{"index": i, "path": p} for i, p in enumerate(paths)]
    return ic.build_feature_cache(directory, inputs, config_path=config, encal_path=encal,
                                  holdout=ic.all_train(range(len(paths))), energy_calibration_prefix={}, **options)


def every_row(observations):
    return np.concatenate([observations.partition(p) for p in range(observations.partitions)])


def test_resume_refusal_and_cache_only_observations(tmp_path, files, monkeypatch):
    setup, encal, paths = files
    directory = tmp_path / "cache"
    cache = build(directory, encal, paths, block_rows=40)
    assert cache.complete and cache.manifest["fingerprint"]["kind"] == "imas timing features"
    counts = {}
    for done in cache.files.values():
        for label, n in done["first_failure"].items():
            counts[label] = counts.get(label, 0) + n
    assert counts == {"passed pre-geometry": 300, "side min channels": 30}
    first = directory / "features" / "f000" / "done.json"
    stamp = first.stat().st_mtime_ns
    blocks = cache.files[1]["blocks"]
    os.rename(directory / "features" / "f001", directory / "features" / "f001.partial")
    (directory / "features" / "f001.partial" / "done.json").unlink()
    assert set(core_cache.FeatureCache(directory).files) == {0}
    resumed = build(directory, encal, paths, block_rows=40)
    assert resumed.complete and first.stat().st_mtime_ns == stamp and resumed.files[1]["blocks"] == blocks
    # changed config, calibration, cuts or source: refused, cache untouched
    manifest = (directory / "manifest.json").read_bytes()
    other_config = config_copy(tmp_path / "c.yaml", min_ch=4)
    other_encal = shutil.copy(encal, tmp_path / "mu2.tsv")
    with open(other_encal, "a", encoding="utf-8") as handle:
        handle.write("999999\t1.0\n")
    for options in ({"config": other_config}, {"encal": other_encal}, {"window_ps": 9_000}):
        with pytest.raises(core_cache.CacheMismatch, match="fingerprint"):
            build(directory, options.get("encal", encal), paths, config=options.get("config", CONFIG),
                  **{k: v for k, v in options.items() if k == "window_ps"})
    edited = tmp_path / "select.py"
    edited.write_text(open(sel.__file__, encoding="utf-8").read() + "\n# edited\n", encoding="utf-8")
    monkeypatch.setattr(sel, "__file__", str(edited))
    with pytest.raises(core_cache.CacheMismatch, match="fingerprint"):
        build(directory, encal, paths)
    monkeypatch.undo()
    assert (directory / "manifest.json").read_bytes() == manifest

    # level 2 from the cache alone, equal to selection + geometry on the files
    geometry = ImasGeometry.nominal(setup)
    lookup = setup.lookup()
    expected = set()
    for index, path in enumerate(paths):
        for features in sel.feature_batches(path, lookup, sel.Selection.production(lookup)):
            accepted = geometry.apply(features).codes == 0
            expected |= {(index, r) for r in features.column("record")[accepted].tolist()}

    def no_reads(*args, **kwargs):
        raise AssertionError("LDAT read during cache-only stage")

    monkeypatch.setattr("src.ldat_stream.LdatStream.__init__", no_reads)
    observations = ic.build_observations(core_cache.FeatureCache(directory), geometry, tmp_path / "obs",
                                         partitions=3, buffer_rows=50, slice_rows=64)
    rows = every_row(observations)
    assert set(zip(rows["file"].tolist(), rows["record"].tolist())) == expected and len(expected) > 250
    assert (rows["channel_i"] < rows["channel_j"]).all()
    assert observations.manifest["fingerprint"]["kind"] == "imas timing observations"
    assert observations.manifest["geometry"]["train"]["accepted"] == len(expected)
    flipped = ImasGeometry.from_setup(setup, axial_sign=1)
    with pytest.raises(core_cache.CacheMismatch, match="different inputs"):
        ic.build_observations(core_cache.FeatureCache(directory), flipped, tmp_path / "obs", partitions=3)
