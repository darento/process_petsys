"""Scanner-neutral timing core moved out of the Cornell skew code (spec 011 T1-T2, FR-1).

The expected values are frozen from the Cornell code before the move (golden and
baseline files with provenance); after the move the same calls must give equal arrays
and file bytes.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cornell_timing_helpers import CONFIG, ROOT, make_setup
from src.cornell.skew.ingest import GeometryHypothesis, apply_geometry

pytestmark = pytest.mark.fr("011-FR-1")
GOLDEN = ROOT / "tests/data/golden/timing/011_t1_cornell_geometry.npz"
BASELINE = ROOT / "tests/data/baselines/cornell_run7_pipeline_011.json"
GEOMETRY_FIELDS = ("codes", "endpoints_mm", "shell_segments", "shell_fractions", "shell_tof_ps")


class Columns:
    """Feature stand-in: ``apply_geometry`` only reads ``column(name)``."""

    def __init__(self, values):
        self.values = values

    def column(self, name):
        return self.values[name]


def synthetic_columns(setup, count=5_000, seed=11):
    """Random Cornell sides over every SM plus an unknown one, local positions 5 mm past the map."""
    rng = np.random.default_rng(seed)
    sms = np.array(sorted({sm for sm, _ in setup.sm_mM_map.values()}) + [max(sm for sm, _ in setup.sm_mM_map.values()) + 7])
    xs = np.array([setup.local_map[ch][0] for ch in setup.local_map])
    ys = np.array([setup.local_map[ch][1] for ch in setup.local_map])
    values = {}
    for side in (1, 2):
        values[f"sm_{side}"] = rng.choice(sms, count).astype(np.int64)
        values[f"x_{side}"] = rng.uniform(xs.min() - 5, xs.max() + 5, count)
        values[f"y_{side}"] = rng.uniform(ys.min() - 5, ys.max() + 5, count)
    return Columns(values)


def geometry_arrays(setup):
    """Selected hypothesis and its transverse-sign alternative on the synthetic columns."""
    columns = synthetic_columns(setup)
    hypotheses = {"selected": GeometryHypothesis.selected(CONFIG, setup),
                  "transverse_-1": GeometryHypothesis.faceted("faceted -1", CONFIG, setup, axial_sign=-1,
                                                              transverse_sign=-1,
                                                              source=GeometryHypothesis.selected(CONFIG, setup).source)}
    arrays = {}
    for name, hypothesis in hypotheses.items():
        result = apply_geometry(columns, hypothesis)
        for field in GEOMETRY_FIELDS:
            arrays[f"{name}/{field}"] = getattr(result, field)
    return arrays


def test_lor_stage_matches_cornell_geometry(tmp_path):
    """The geometry stage gives the arrays frozen before the core move, exactly (NaN in place)."""
    expected = np.load(GOLDEN)
    actual = geometry_arrays(make_setup(tmp_path))
    assert sorted(actual) == sorted(expected.files)
    for key, value in actual.items():
        assert value.dtype == expected[key].dtype, key
        assert np.array_equal(value, expected[key], equal_nan=True), key
    # the fixture reaches the common geometry outcomes (one shell segment needs a tangent LOR)
    assert {0, 1, 2, 3} <= set(np.unique(actual["selected/codes"]).tolist())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lut_digest(path, decimals=6):
    """SHA-256 of a canonical LUT CSV with floats rounded to ``decimals`` (ps).

    The global solve goes through BLAS/LAPACK, whose summation order follows the
    thread count; its delays differ by ~1e-11 ps between processes. Statuses,
    flags and counts stay exact.
    """
    import csv
    rows = list(csv.reader(Path(path).read_text(encoding="utf-8").splitlines()))
    floats = {"delay_ps", "sigma_ps", "raw_sigma_ps", "residual_rms"}
    header = rows[0]
    text = [",".join(header)]
    for row in rows[1:]:
        text.append(",".join(f"{float(v):.{decimals}f}" if name in floats and v else v
                             for name, v in zip(header, row)))
    return hashlib.sha256("\n".join(text).encode()).hexdigest()


def run7_pipeline_hashes(work, data_dir, cal_dir, *, files=tuple(range(8)), max_records=None):
    """Bounded Cornell run7 pipeline: feature cache, observations, pair fits, solve and LUT.

    Roles are all train (no held-out prefix), so the result is a fixed-input regression
    value, not a calibration. Returns SHA-256 of every numeric output.
    """
    from src.cornell.skew import cache as tc
    from src.cornell.skew import job as tj
    from src.cornell.skew import pairs as tp
    from src.cornell.skew import solver as ts
    from src.cornell.skew.ingest import LegacySetup, time_channel_classes
    from src.cornell.skew.lut import write_lut

    work = Path(work)
    encal = Path(cal_dir) / "norm_10082026_r7_coincCompact_resolved.encal"
    run = Path(data_dir) / "Cornell/full_system/skew"
    inputs = [{"index": i, "path": run / f"norm_10082026_r7_coincCompact_{i:08d}.ldat"} for i in files]
    for path in [encal] + [item["path"] for item in inputs]:
        if not path.is_file():
            pytest.skip(f"real_data: {path} is missing")
    holdout = {"qualified": True, "roles_frozen_before_fitting": True,
               "partitions": [{"index": i, "role": "train", "anchor_start_ps": 0, "anchor_end_ps": 2 ** 62}
                              for i in files]}
    setup = LegacySetup.from_files(CONFIG, encal, root=ROOT)
    _, operational, down = time_channel_classes(setup, [[14, 0], [14, 1], [14, 4], [14, 5]])
    features = tc.build_feature_cache(work / "features", inputs, config_path=CONFIG, encal_path=encal, root=ROOT,
                                      holdout=holdout, energy_calibration_prefix={}, seed=0,
                                      max_records=max_records, workers=4)
    features.require_complete()
    observations = tc.build_observations(features, GeometryHypothesis.selected(CONFIG, setup), work / "observations",
                                         partitions=4)
    pooled = tj.PooledObservations([observations])
    model, _ = tp.estimate_model(tj.kernel_rows(pooled))
    fits = tp.fit_observation_set(pooled, model, roles=(0,))
    solution = ts.solve(fits, operational, down)
    lut = work / "lut.csv"
    write_lut(lut, solution, {"test": "011 T2 run7 pipeline regression"})
    hashes = {str(path.relative_to(work)).replace("\\", "/"): _sha256(path)
              for folder in ("features", "observations") for path in sorted((work / folder).rglob("*.npy"))}
    hashes["pair_fits"] = hashlib.sha256(np.ascontiguousarray(fits).tobytes()).hexdigest()
    hashes["lut.csv"] = lut_digest(lut)
    counts = {"observations": int(sum(len(observations.partition(p)) for p in range(observations.partitions))),
              "pairs": int(len(fits)), "usable_pairs": int((fits["status"] == tp.USABLE).sum()),
              "applied_channels": int(solution.channels["apply"].sum())}
    return hashes, counts


@pytest.mark.real_data
@pytest.mark.slow
def test_run7_pipeline_bytes_unchanged(tmp_path, real_data_dir, real_cal_dir):
    """A bounded run7 pipeline after the core move writes the same bytes as before it."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    hashes, counts = run7_pipeline_hashes(tmp_path, real_data_dir, real_cal_dir, files=tuple(baseline["files"]),
                                          max_records=baseline["max_records"])
    assert counts == baseline["counts"]
    assert hashes == baseline["sha256"]
