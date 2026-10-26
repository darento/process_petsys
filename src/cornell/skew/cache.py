"""Cornell adapters for the bounded timing cache (spec 008 T5; seam spec 011 T2, FR-1).

The cache itself (feature blocks, observation sets, manifests and refusal
rules) is ``src.timing.cache``. This module supplies the Cornell pieces: the
selection lookup built from config, map and per-(TIME channel, slab)
``.encal``; the faceted geometry hypothesis; and the spec 009 walk correction.
``build_feature_cache`` and ``build_observations`` keep their spec 008
signatures, and the fingerprints keep their fields, so existing caches stay
reusable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.cornell.skew import geometry as skew_geometry, ingest
from src import ldat_stream
from src.timing import cache as core_cache, kernels as core_kernels, shell as core_shell
from src.timing.cache import (FEATURE_DTYPE, OBSERVATION_DTYPE, PAIR_COUNT_DTYPE, ROLES,  # noqa: F401
                              SCHEMA_VERSION, CacheMismatch, FeatureCache, ObservationSet, _Columns, _digest,
                              _prepare, _sha256, canonical_observations, feature_rows, pair_partition)


def walk_rows(rows, walk):
    """Spec 009 FR-4: rows with every walk input on both sides, their ``dt_ps`` corrected by
    round(W(side 1) - W(side 2)); and the counts (missing inputs, clamped per input)."""
    from src.cornell.skew import walk as tw
    values, statuses = {}, []
    for s in (1, 2):
        side, status = tw.inputs(walk["context"], rows[f"time_channel_{s}"], rows[f"slab_{s}"], rows[f"q_top_{s}"],
                                 rows[f"q_time_{s}"], rows[f"e_raw_{s}"], rows[f"doi_ratio_{s}"], rows[f"y_{s}"])
        values[s] = side
        statuses.append(status)
    keep = (statuses[0] == 0) & (statuses[1] == 0)
    difference, out1, out2 = tw.walk_difference(walk["lut"], values[1], values[2])
    rows = rows[keep].copy()
    rows["dt_ps"] -= np.rint(difference[keep]).astype(np.int64)
    counts = {"walk input missing": int(np.count_nonzero(~keep))}
    for name in tw.INPUTS:
        counts[f"clamped {name}"] = int(np.count_nonzero(keep & out1[name])) + int(np.count_nonzero(keep & out2[name]))
    return rows, counts


_LOOKUP = {}


def _compiled_lookup(config_path, encal_path, root):
    """Per-process selection lookup, rebuilt whenever config, map or ``.encal`` changes."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    key = (str(root),) + tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns)
                               for path in (Path(config_path), Path(root) / config["map_file"], Path(encal_path)))
    if key not in _LOOKUP:
        _LOOKUP.clear()
        setup = ingest.LegacySetup.from_files(config_path, encal_path, root=root)
        _LOOKUP[key] = ingest.CompiledLookup.from_setup(setup)
    return _LOOKUP[key]


@dataclass(frozen=True)
class CornellFeatures:
    """Feature adapter: Cornell production selection; plain paths so worker processes can unpickle it."""
    config_path: str
    encal_path: str
    root: str
    kind = "cornell timing features"
    labels = ingest.PRODUCTION_LABELS

    def fingerprint(self):
        config = yaml.safe_load(Path(self.config_path).read_text(encoding="utf-8"))
        return {"config_sha256": _sha256(self.config_path), "map_sha256": _sha256(Path(self.root) / config["map_file"]),
                "encal_sha256": _sha256(self.encal_path),
                "sources_sha256": {name: _sha256(module.__file__) for name, module in
                                   (("ingest", ingest), ("kernels", core_kernels), ("ldat_stream", ldat_stream))}}

    @staticmethod
    def code_label(code):
        return ingest.code_label(code)

    def feature_batches(self, path, partition, *, max_records, seed):
        lookup = _compiled_lookup(self.config_path, self.encal_path, self.root)
        return ingest.production_batches(path, lookup, partition, max_records=max_records, seed=seed)


def describe_hypothesis(hypothesis):
    """Stable description of every geometry input of a ``GeometryHypothesis``."""
    frames = {str(sm): repr(frame) for sm, frame in sorted(hypothesis.frames.items())}
    return {"name": hypothesis.name, "frames_sha256": _digest(frames), "source": repr(hypothesis.source),
            "opposition_tolerance_deg": hypothesis.opposition_tolerance_deg}


@dataclass(frozen=True)
class CornellGeometry:
    """Geometry adapter: one faceted ``GeometryHypothesis``."""
    hypothesis: object
    kind = "cornell timing observations"

    def describe(self):
        return describe_hypothesis(self.hypothesis)

    @staticmethod
    def sources():
        return {name: module.__file__ for name, module in
                (("ingest", ingest), ("geometry", skew_geometry), ("kernels", core_kernels), ("shell", core_shell))}

    def apply(self, columns):
        return ingest.apply_geometry(columns, self.hypothesis)


@dataclass(frozen=True)
class CornellWalk:
    """Walk adapter for ``{"lut": WalkLut, "context": WalkContext}`` (spec 009 FR-4)."""
    walk: dict

    @property
    def source(self):
        from src.cornell.skew import walk as tw
        return tw.__file__

    def fingerprint(self):
        return {"walk_lut_sha256": self.walk["lut"].sha256, "listmode_files": self.walk["context"].files}

    def rows(self, rows):
        return walk_rows(rows, self.walk)


def build_feature_cache(directory, inputs, *, config_path, encal_path, root, holdout,
                        energy_calibration_prefix, seed=0, max_records=None, workers=1, block_rows=1_000_000):
    """``src.timing.cache.build_feature_cache`` with the Cornell selection from config, map and ``.encal``."""
    adapter = CornellFeatures(str(config_path), str(encal_path), str(Path(root)))
    return core_cache.build_feature_cache(directory, inputs, adapter, holdout=holdout,
                                          energy_calibration_prefix=energy_calibration_prefix, seed=seed,
                                          max_records=max_records, workers=workers, block_rows=block_rows)


def build_observations(feature_cache, hypothesis, directory, *, partitions=16, buffer_rows=250_000,
                       slice_rows=250_000, walk=None):
    """``src.timing.cache.build_observations`` for a Cornell ``GeometryHypothesis``.

    ``walk`` is ``{"lut": WalkLut, "context": WalkContext}`` (spec 009 FR-4) or None.
    """
    return core_cache.build_observations(feature_cache, CornellGeometry(hypothesis), directory,
                                         partitions=partitions, buffer_rows=buffer_rows, slice_rows=slice_rows,
                                         walk=None if walk is None else CornellWalk(walk))
