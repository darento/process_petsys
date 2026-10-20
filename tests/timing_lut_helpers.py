"""Synthetic skew and walk LUTs for a fixture map (spec 010).

Tables follow the spec 008/009 contracts but carry random values: they test which tables a listmode
run accepts, never timing numbers.
"""

import hashlib

import numpy as np

from src.cornell import listmode as lm
from src.cornell.skew import solver as ts
from src.cornell.skew import walk
from src.cornell.skew.lut import write_lut


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_skew_lut(path, mapping, *, seed=41, channels=None, provenance=None):
    """Canonical skew LUT over every TIME channel of ``mapping`` (or ``channels``), random delays."""
    rng = np.random.default_rng(seed)
    time = np.flatnonzero(lm.CalibrationMaps.from_mapping(mapping).time_mask) if channels is None else channels
    rows = np.zeros(len(time), ts.CHANNEL_DTYPE)
    rows["time_channel"] = time
    rows["delay_ps"] = rng.normal(0, 300, len(time))
    rows["sigma_ps"] = rows["raw_sigma_ps"] = 5.0
    rows["status"] = ts.CALIBRATED
    rows["apply"] = True
    write_lut(path, ts.Solution(rows, np.zeros(0, ts.PAIR_RESIDUAL_DTYPE), {}), provenance or {"job": "synthetic"})
    return path


def write_walk_lut(path, skew_path, calibration, cog_limits, doi_limits, *, seed=42):
    """Walk LUT 'fitted' with ``skew_path`` and these loaded listmode files (random offsets)."""
    rng = np.random.default_rng(seed)
    edges = {"walk2d": [np.linspace(0, 1000, 21), np.linspace(0.0, 1.0, 11)], "doi": [np.linspace(0, 20, 21)],
             "kev": [np.linspace(300, 700, 21)]}
    offsets = {"walk2d": rng.normal(0, 50, (20, 10)), "doi": rng.normal(0, 10, 20), "kev": rng.normal(0, 10, 20)}
    events = {"walk2d": np.full((20, 10), 500), "doi": np.full(20, 500), "kev": np.full(20, 500)}
    table = walk.WalkLut.from_fit(edges, offsets, events, {"fit_kind": "synthetic"})
    files = {name: {"path": str(item.path), "sha256": item.sha256}
             for name, item in (("cog_limits", cog_limits), ("doi_limits", doi_limits))}
    table.manifest.update(listmode_files={**files, "calibration": lm.calibration_entry(calibration)},
                          skew_lut_sha256=sha256(skew_path))
    table.write(path)
    return path
