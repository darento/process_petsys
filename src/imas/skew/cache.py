"""IMAS adapters for the bounded timing cache (spec 011 T6, FR-1, FR-6).

``src.timing.cache`` holds the feature blocks, observation sets and refusal
rules. This module supplies the IMAS selection (``select``) over a lookup
built from config, map and the ``mu`` energy calibration, and the IMAS
geometry hypothesis (``geometry.ImasGeometry``). Fingerprints hash the
config, map and calibration bytes, the declared cuts and the IMAS and core
sources, so a cache from different inputs, cuts or code is refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src import ldat_stream
from src.imas.skew import geometry as imas_geometry, select as sel, setup as imas_setup
from src.imas.skew.setup import ImasSetup
from src.timing import cache as core_cache, kernels as core_kernels, shell as core_shell, stream as core_stream
from src.timing.cache import _sha256

_LOOKUP = {}


def _lookup(config_path, encal_path, root):
    """Per-process lookup, rebuilt whenever config, map or calibration changes."""
    setup = ImasSetup.from_files(config_path, encal_path, root=root)
    key = tuple(sorted(setup.provenance.items()))
    if key not in _LOOKUP:
        _LOOKUP.clear()
        _LOOKUP[key] = setup.lookup()
    return _LOOKUP[key]


@dataclass(frozen=True)
class ImasFeatures:
    """Feature adapter: IMAS production selection; plain values so worker processes can unpickle it."""
    config_path: str
    encal_path: str
    root: str
    window_ps: int = sel.COINCIDENCE_WINDOW_PS
    kev_window: tuple = field(default=sel.KEV_WINDOW)
    kind = "imas timing features"
    labels = sel.PRODUCTION_LABELS

    def _selection(self, lookup):
        return sel.Selection.production(lookup, window_ps=self.window_ps, kev_window=self.kev_window)

    def fingerprint(self):
        setup = ImasSetup.from_files(self.config_path, self.encal_path, root=self.root)
        return {"config_sha256": setup.provenance["config_sha256"], "map_sha256": setup.provenance["map_sha256"],
                "encal_sha256": setup.provenance["encal_sha256"],
                "selection": self._selection(setup.lookup()).describe(),
                "sources_sha256": {name: _sha256(module.__file__) for name, module in
                                   (("setup", imas_setup), ("select", sel), ("kernels", core_kernels),
                                    ("ldat_stream", ldat_stream))}}

    @staticmethod
    def code_label(code):
        return sel.code_label(code)

    def feature_batches(self, path, partition, *, max_records, seed):
        lookup = _lookup(self.config_path, self.encal_path, self.root)
        return sel.feature_batches(path, lookup, self._selection(lookup), partition, max_records=max_records)


@dataclass(frozen=True)
class ImasObservations:
    """Geometry adapter for one ``ImasGeometry`` hypothesis."""
    geometry: imas_geometry.ImasGeometry
    kind = "imas timing observations"

    def describe(self):
        return self.geometry.describe()

    @staticmethod
    def sources():
        return {name: module.__file__ for name, module in (("geometry", imas_geometry), ("shell", core_shell))}

    def apply(self, columns):
        return self.geometry.apply(columns)


def all_train(indices):
    """Holdout with every file in the train role over its whole time range (development runs)."""
    return {"qualified": True, "roles_frozen_before_fitting": True,
            "partitions": [{"index": int(i), "role": "train", "anchor_start_ps": -2 ** 62, "anchor_end_ps": 2 ** 62}
                           for i in indices]}


def build_feature_cache(directory, inputs, *, config_path, encal_path, root=".", holdout, energy_calibration_prefix,
                        window_ps=sel.COINCIDENCE_WINDOW_PS, kev_window=sel.KEV_WINDOW, max_records=None, workers=1,
                        block_rows=1_000_000):
    """``src.timing.cache.build_feature_cache`` with the IMAS selection."""
    adapter = ImasFeatures(str(config_path), str(encal_path), str(Path(root)), int(window_ps),
                           tuple(float(v) for v in kev_window))
    return core_cache.build_feature_cache(directory, inputs, adapter, holdout=holdout,
                                          energy_calibration_prefix=energy_calibration_prefix, seed=0,
                                          max_records=max_records, workers=workers, block_rows=block_rows)


def build_observations(feature_cache, geometry, directory, *, partitions=16, buffer_rows=250_000,
                       slice_rows=250_000):
    """``src.timing.cache.build_observations`` for an ``ImasGeometry``."""
    return core_cache.build_observations(feature_cache, ImasObservations(geometry), directory,
                                         partitions=partitions, buffer_rows=buffer_rows, slice_rows=slice_rows)


def stream_observations(directory, inputs, geometry, *, config_path, encal_path, root=".", holdout,
                        energy_calibration_prefix, window_ps=sel.COINCIDENCE_WINDOW_PS, kev_window=sel.KEV_WINDOW,
                        max_records=None, workers=1, partitions=16, buffer_rows=250_000, progress=None):
    """``src.timing.stream.stream_observations`` with the IMAS selection and an ``ImasGeometry`` (no feature cache)."""
    adapter = ImasFeatures(str(config_path), str(encal_path), str(Path(root)), int(window_ps),
                           tuple(float(v) for v in kev_window))
    return core_stream.stream_observations(directory, inputs, adapter, ImasObservations(geometry), holdout=holdout,
                                           energy_calibration_prefix=energy_calibration_prefix, seed=0,
                                           max_records=max_records, workers=workers, partitions=partitions,
                                           buffer_rows=buffer_rows, progress=progress)
