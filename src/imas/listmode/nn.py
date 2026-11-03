"""NN axial position and DOI for IMAS listmode (spec 015 T3, FR-5).

The combined Y + DOI model of ``scripts_imas/imas_listmode.py`` (``src.slab_nn.SlabNN.build_combined``) with
categories 0. Input per side: the 8 ENERGY charges of the selected minimodule by map row (float32). Output:
the axial y relative to the slab (mm) and the DOI value of the DOI model (not a calibrated depth).
TensorFlow is imported only when a ``KerasPredictor`` is built, i.e. in the NN process.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BATCH = 16384           # SlabNN.predict default, as the legacy


class KerasPredictor:
    def __init__(self, y_file, doi_file, *, batch=DEFAULT_BATCH):
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from src.slab_nn import SlabNN, SlabSystem
        self.batch = int(batch)
        self.nn = SlabNN(SlabSystem("IMAS"))
        self.nn.build_combined(str(y_file), str(doi_file), print_model=False)
        self.describe = {"kind": "keras combined", "y_model": str(y_file), "doi_model": str(doi_file),
                         "batch": self.batch, "categories": 0}

    def predict(self, rows):
        """``rows``: (n, 8) float32. Returns (y_mm, doi) float32 arrays of length n."""
        rows = np.ascontiguousarray(rows, np.float32)
        if not len(rows):
            return np.empty(0, np.float32), np.empty(0, np.float32)
        categories = np.zeros(len(rows), np.int32)
        y, doi = self.nn.predict([categories, rows, categories, rows], batch=self.batch)
        return np.asarray(y, np.float32).reshape(-1), np.asarray(doi, np.float32).reshape(-1)


class LinearPredictor:
    """Deterministic stand-in for tests: y = rows @ wy + by, doi = rows @ wd + bd."""

    def __init__(self, wy=None, by=0.0, wd=None, bd=0.0):
        self.wy = np.linspace(-1.0, 1.0, 8, dtype=np.float32) if wy is None else np.asarray(wy, np.float32)
        self.wd = np.full(8, 0.05, np.float32) if wd is None else np.asarray(wd, np.float32)
        self.by, self.bd = np.float32(by), np.float32(bd)
        self.describe = {"kind": "linear test predictor"}

    def predict(self, rows):
        rows = np.asarray(rows, np.float32)
        return (rows @ self.wy + self.by).astype(np.float32), (rows @ self.wd + self.bd).astype(np.float32)


def make_predictor(spec):
    """A predictor from a picklable spec: ("keras", y, doi[, batch]) or ("linear", kwargs)."""
    kind = spec[0]
    if kind == "keras":
        return KerasPredictor(spec[1], spec[2], **({"batch": spec[3]} if len(spec) > 3 else {}))
    if kind == "linear":
        return LinearPredictor(**(spec[1] if len(spec) > 1 else {}))
    raise ValueError(f"unknown predictor {kind!r}")
