"""IMAS LM V5 records and header (spec 015 T3, FR-5, FR-6).

The record layout is ``src.listmode.CoincidenceV5`` (24 B). Positions are pixel indices over the SM face:
``searchsorted(PIXEL_EDGES, v, "right") - 1`` of the local x (the top TIME channel's map x) and the local y
(NN y + the top TIME channel's map y); z is the NN DOI truncated to int8 (the legacy numpy assignment). The
header is the legacy script's ``write_header`` with acquisition time, activity and isotope from the run.
"""
from __future__ import annotations

import ctypes

import numpy as np

from src.listmode import CoincidenceV5, LMHeader

LM_DTYPE = np.dtype(CoincidenceV5)
PIXEL_EDGES = np.linspace(0, 103.22, 101)
ISOTOPES = ("F18", "Na22")


def pixels(values):
    """Pixel index per value (int8 storage; -1 / 100 outside the face, as the legacy)."""
    return (np.searchsorted(PIXEL_EDGES, np.asarray(values, np.float64), side="right") - 1).astype(np.int8)


def header_bytes(acq_time, activity, isotope="F18"):
    """The legacy ``LMHeader`` (176 B)."""
    if isotope not in ISOTOPES:
        raise ValueError(f"isotope {isotope!r}: one of {ISOTOPES}")
    step = float(np.diff(PIXEL_EDGES)[0])
    header = LMHeader(identifier=b"IMAS", acqTime=float(acq_time), activity=float(activity),
                      isotope=isotope.encode("utf-8"), detectorSizeX=103.22, detectorSizeY=103.22, startTime=0,
                      measurementTime=10, moduleNumber=120, ringNumber=5, ringDistance=820,
                      detectorPixelSizeX=step, detectorPixelSizeY=step, version=(9, 5),
                      detectorPixelsX=PIXEL_EDGES.size - 1, detectorPixelsY=PIXEL_EDGES.size - 1)
    return ctypes.string_at(ctypes.addressof(header), ctypes.sizeof(header))


def build_chunk(chunk, y_nn, doi, t0):
    """LM records of one ``job.CHUNK_DTYPE`` chunk in LDAT record order.

    ``chunk["dt"]``: int16 (side 1 - side 2, LDAT order); ``y_nn``/``doi``: (n, 2) NN outputs per side in LDAT
    order; ``t0``: raw ps of the run's first timestamp. Swapped pairs exchange the sides and negate ``dt``.
    """
    n = len(chunk)
    out = np.zeros(n, LM_DTYPE)
    swapped = chunk["swapped"].astype(bool)
    first = np.where(swapped, 1, 0)
    second = 1 - first
    rows = np.arange(n)
    px = pixels(chunk["x_mm"])
    py = pixels(chunk["y0_mm"] + np.asarray(y_nn, np.float64).reshape(n, 2))
    pz = np.asarray(doi, np.float32).reshape(n, 2).astype(np.int8)
    out["time"] = np.round((chunk["t_min"] - np.int64(t0)) * 1e-12, 4)
    out["energy1"] = np.rint(chunk["kev"][rows, first])
    out["energy2"] = np.rint(chunk["kev"][rows, second])
    out["amount"] = 1.0
    out["xPosition1"], out["yPosition1"], out["zPosition1"] = px[rows, first], py[rows, first], pz[rows, first]
    out["xPosition2"], out["yPosition2"], out["zPosition2"] = px[rows, second], py[rows, second], pz[rows, second]
    out["pair"] = chunk["pair"]
    dt = chunk["dt"].astype(np.int32)
    out["dt"] = np.where(swapped & (dt != -32700), -dt, dt).astype(np.int16)
    return out
