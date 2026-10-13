"""Canonical Cornell TIME-channel delay LUT (spec 008 T7, FR-12/FR-17).

The canonical table is the deliverable before any native PETsys adapter
(T9a): one row per operational or hardware-unavailable TIME channel of the
selected map, a positive subtractive delay ``b`` in ps
(``t_corrected = t - b``, so ``dt_corrected = dt - b_i + b_j``), its
uncertainty, explicit status, component and an ``apply`` flag. A missing
delay is an empty field, never 0 or a sentinel. Channels absent from the
table (unpopulated, ENERGY, unmapped) are never corrected. A JSON manifest
next to the CSV records units, sign, gauge, solve diagnostics, provenance and
the CSV's SHA-256; reading checks all of it.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from src.cornell import timing_solver as ts
from src.cornell.timing_convention import corrected_dt_ps

SCHEMA = "cornell-time-delay-lut/1"
COLUMNS = ("time_channel", "delay_ps", "sigma_ps", "raw_sigma_ps", "status", "apply", "component",
           "usable_pairs", "pairs", "usable_events", "residual_rms")
CONVENTION = "subtractive: t_corrected = t - delay_ps; dt_corrected = (t_i - t_j) - delay_i + delay_j"
SCOPE = ("operational and hardware-unavailable TIME channels of the selected map; channels absent from the "
         "table are never corrected")


class LutError(ValueError):
    """The table or manifest is inconsistent, or a correction was requested for a non-applied channel."""


def _number(value):
    return "" if not math.isfinite(value) else repr(float(value))


def write_lut(path, solution, provenance):
    """Write ``path`` (CSV) and ``path.with_suffix('.json')``; returns the manifest."""
    path = Path(path)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in solution.channels:
        writer.writerow([int(row["time_channel"]), _number(row["delay_ps"]), _number(row["sigma_ps"]),
                         _number(row["raw_sigma_ps"]), ts.CHANNEL_STATUSES[row["status"]], int(row["apply"]),
                         int(row["component"]), int(row["usable_pairs"]), int(row["pairs"]),
                         int(row["usable_events"]), _number(row["residual_rms"])])
    text = buffer.getvalue().encode("utf-8")
    applied = solution.channels[solution.channels["apply"]]
    manifest = {
        "schema": SCHEMA, "units": "ps", "convention": CONVENTION, "scope": SCOPE,
        "statuses": list(ts.CHANNEL_STATUSES),
        "applied_statuses": [ts.CHANNEL_STATUSES[k] for k in ts.APPLIED_STATUSES],
        "gauge": {"definition": "zero mean over each component's solved channels; only component 0 is applied",
                  "applied_mean_delay_ps": float(applied["delay_ps"].mean()) if len(applied) else None},
        "diagnostics": solution.diagnostics, "provenance": provenance,
        "table": {"name": path.name, "rows": int(len(solution.channels)),
                  "sha256": hashlib.sha256(text).hexdigest()},
    }
    path.write_bytes(text)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


class CanonicalLut:
    """Validated canonical table; corrections only for channels flagged ``apply``."""

    def __init__(self, rows, manifest):
        self.rows, self.manifest = rows, manifest
        self._delay = {int(r["time_channel"]): float(r["delay_ps"]) for r in rows if r["apply"]}

    @classmethod
    def read(cls, path):
        path = Path(path)
        text = path.read_bytes()
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or manifest.get("units") != "ps" or manifest.get("convention") != CONVENTION:
            raise LutError("unknown LUT schema, units or sign convention")
        if manifest["table"]["sha256"] != hashlib.sha256(text).hexdigest():
            raise LutError("LUT table does not match its manifest")
        reader = csv.reader(io.StringIO(text.decode("utf-8")))
        if tuple(next(reader)) != COLUMNS:
            raise LutError("unexpected LUT columns")
        rows = np.zeros(manifest["table"]["rows"], ts.CHANNEL_DTYPE)
        count = 0
        for count, fields in enumerate(reader, 1):
            if count > len(rows) or len(fields) != len(COLUMNS):
                raise LutError("LUT row count or width mismatch")
            row = rows[count - 1]
            if fields[4] not in ts.CHANNEL_STATUSES:
                raise LutError(f"unknown status {fields[4]!r}")
            status = ts.CHANNEL_STATUSES.index(fields[4])
            row["time_channel"], row["status"], row["apply"] = int(fields[0]), status, fields[5] == "1"
            for name, text_value in zip(("delay_ps", "sigma_ps", "raw_sigma_ps"), fields[1:4]):
                row[name] = float(text_value) if text_value else np.nan
            row["component"], row["usable_pairs"], row["pairs"], row["usable_events"] = map(int, fields[6:10])
            row["residual_rms"] = float(fields[10]) if fields[10] else np.nan
            solved = status in ts.APPLIED_STATUSES + (ts.DISCONNECTED,)
            if fields[5] not in ("0", "1") or row["apply"] != (status in ts.APPLIED_STATUSES):
                raise LutError(f"channel {row['time_channel']}: apply flag disagrees with status")
            values = (row["delay_ps"], row["sigma_ps"], row["raw_sigma_ps"])
            if solved != all(math.isfinite(v) for v in values) or (not solved and any(fields[1:4])):
                raise LutError(f"channel {row['time_channel']}: delay present/absent contradicts status")
        if count != len(rows) or len(np.unique(rows["time_channel"])) != len(rows):
            raise LutError("LUT rows missing or duplicated")
        return cls(rows, manifest)

    def delay(self, channel):
        try:
            return self._delay[int(channel)]
        except KeyError:
            match = self.rows[self.rows["time_channel"] == int(channel)]
            state = ts.CHANNEL_STATUSES[match["status"][0]] if len(match) else "not in table"
            raise LutError(f"no applicable delay for TIME channel {channel} ({state})") from None

    def corrected_dt_ps(self, timestamp_i, timestamp_j, channel_i, channel_j):
        return corrected_dt_ps(timestamp_i, timestamp_j, self.delay(channel_i), self.delay(channel_j))
