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

Native adapter (T9a, owner decisions 2026-10-13): ``write_native_table``
exports every applied TIME channel as a PETsys ``time_offset_calibration_table``
row ``port slave chip channel offset_ps``; other TIME channels and all ENERGY
channels are absent, so PETsys leaves them at its default 0. Installed
contract (``SystemConfig.cpp``/``ProcessHit.cpp``, recorded in T2b/T9a): lines
are whitespace/comment normalised and parsed as ``%u %u %u %d %f`` (anything
else silently skipped), ``MAKE_GID = channel | chip<<6 | slave<<12 |
port<<17`` and, with TDC calibration active, every hit loses ``t0/clockPeriod``
clock units before grouping, i.e. ``t0`` is the same positive subtractive
delay in ps as the canonical LUT. ``read_native_table`` mirrors that parser;
``activated_ini`` writes an INI copy that activates one table.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

from src.cornell import timing_solver as ts
from src.cornell.timing_convention import corrected_dt_ps
from src.utils import get_absolute_id, get_electronics_nums

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


# --- Native PETsys time-offset table (T9a) ------------------------------------

NATIVE_SCHEMA = "cornell-petsys-time-offset/1"
NATIVE_CONTRACT = ("PETsys [main] time_offset_calibration_table: 'port slave chip channel t0' per line; with TDC "
                   "calibration active ProcessHit subtracts t0/clockPeriod from every hit before grouping, so t0 is "
                   "a positive subtractive delay in ps; unlisted channels keep t0 = 0")
CORRECTION_STATE = ("apply to an uncorrected RAWF conversion only; LDAT converted with this table is already "
                    "corrected and must not receive the same LUT offline")
_OFFSET_DECIMALS = 3


def _normalize(line):
    """``normalizeLine`` of PETsys SystemConfig.cpp: CR, comments, outer and inner whitespace."""
    line = line.replace("\r", "")
    line = re.sub(r"\s*#.*", "", line)
    return re.sub(r"\s+", "\t", line.strip())


def read_native_table(path):
    """Offsets as PETsys would load them: ({global channel: float32 ps}, skipped lines, duplicate channels)."""
    offsets, skipped, duplicates = {}, [], []
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = _normalize(raw)
        if not line:
            continue
        fields = line.split("\t")
        try:
            port, slave, chip, channel = (int(v) for v in fields[:4])
            offset = np.float32(float(fields[4]))
            if min(port, slave, chip, channel) < 0 or channel > 63 or chip > 63 or slave > 31:  # stricter than sscanf
                raise ValueError
        except (ValueError, IndexError):
            skipped.append(number)
            continue
        gid = get_absolute_id(port, slave, chip, channel)
        if gid in offsets:
            duplicates.append(gid)
        offsets[gid] = float(offset)
    return offsets, skipped, duplicates


def write_native_table(lut_path, table_path):
    """Write the native TSV and ``table_path.with_suffix('.json')`` from a validated canonical LUT."""
    lut_path, table_path = Path(lut_path), Path(table_path)
    manifest_path = table_path.with_suffix(".json")
    for path in (table_path, manifest_path):
        if path.exists():
            raise LutError(f"native output exists: {path}; choose a new name")
    lut = CanonicalLut.read(lut_path)
    rows = lut.rows[lut.rows["apply"]]
    rows = rows[np.argsort(rows["time_channel"], kind="stable")]
    lines = [f"# PETsys time_offset_calibration_table from {lut_path.name} "
             f"(LUT SHA-256 {lut.manifest['table']['sha256']})",
             "# columns: portID slaveID chipID channelID t0_ps; t0 = positive subtractive delay (t - t0)",
             "# only applied TIME channels are listed; other TIME channels and all ENERGY channels keep PETsys t0 = 0",
             f"# {CORRECTION_STATE}"]
    for row in rows:
        port, slave, chip, channel = get_electronics_nums(int(row["time_channel"]))
        value = round(float(row["delay_ps"]), _OFFSET_DECIMALS)
        if abs(float(np.float32(value)) - float(row["delay_ps"])) > 0.01:
            raise LutError(f"channel {row['time_channel']}: delay not representable within 0.01 ps as float32")
        lines.append(f"{port}\t{slave}\t{chip}\t{channel}\t{value:.{_OFFSET_DECIMALS}f}")
    text = ("\n".join(lines) + "\n").encode("utf-8")
    table_path.write_bytes(text)
    offsets, skipped, duplicates = read_native_table(table_path)
    expected = {int(r["time_channel"]): float(r["delay_ps"]) for r in rows}
    if skipped or duplicates or set(offsets) != set(expected) or \
            max((abs(offsets[c] - expected[c]) for c in expected), default=0.0) > 0.01:
        table_path.unlink()
        raise LutError("native table does not parse back to the applied LUT delays")
    statuses = {name: int((lut.rows["status"] == k).sum()) for k, name in enumerate(ts.CHANNEL_STATUSES)}
    provenance = lut.manifest.get("provenance", {})
    manifest = {
        "schema": NATIVE_SCHEMA, "units": "ps", "contract": NATIVE_CONTRACT, "correction_state": CORRECTION_STATE,
        "source_lut": {"name": lut_path.name, "sha256": lut.manifest["table"]["sha256"], "convention": CONVENTION},
        "policy": {"included": "applied TIME channels (calibrated and low statistics), listed outliers included "
                               "(owner decision 2026-10-13)",
                   "rows": int(len(rows)), "lut_status_counts": statuses,
                   "not_listed": "non-applied TIME channels and every ENERGY channel: PETsys default t0 = 0",
                   "listed_cross_run_outliers": len(provenance.get("cross_run_outliers") or []),
                   "offset_decimals": _OFFSET_DECIMALS},
        "table": {"name": table_path.name, "rows": int(len(rows)), "sha256": hashlib.sha256(text).hexdigest()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


_TABLE_KEY = re.compile(r"^\s*time_offset_calibration_table\s*=", re.IGNORECASE | re.MULTILINE)


def activated_ini(source_ini, table_path, output_ini):
    """INI copy activating ``table_path``: ``%CDIR%`` made absolute, one active table line in ``[main]``."""
    source_ini, table_path, output_ini = Path(source_ini), Path(table_path).resolve(), Path(output_ini)
    if output_ini.exists():
        raise LutError(f"INI output exists: {output_ini}; choose a new name")
    text = source_ini.read_text(encoding="utf-8")
    if _TABLE_KEY.search(text):
        raise LutError(f"{source_ini} already activates a time-offset table; refusing a second correction")
    cdir = source_ini.resolve().parent.as_posix()
    text = re.sub(r"%CDIR%", lambda _: cdir, text, flags=re.IGNORECASE)
    lines = text.splitlines(keepends=True)
    try:
        main = next(k for k, line in enumerate(lines) if line.strip().lower() == "[main]")
    except StopIteration:
        raise LutError(f"{source_ini}: no [main] section") from None
    end = next((k for k in range(main + 1, len(lines)) if lines[k].lstrip().startswith("[")), len(lines))
    commented = [k for k in range(main + 1, end)
                 if re.match(r"^\s*#\s*time_offset_calibration_table\s*=", lines[k], re.IGNORECASE)]
    at = commented[0] + 1 if commented else end
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    lines.insert(at, f"time_offset_calibration_table = {table_path.as_posix()}{newline}")
    output_ini.write_text("".join(lines), encoding="utf-8", newline="")
    return {"source_ini": str(source_ini), "source_sha256": hashlib.sha256(source_ini.read_bytes()).hexdigest(),
            "cdir": cdir, "table": table_path.as_posix(), "output_ini": str(output_ini), "line": at + 1}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Export a canonical Cornell LUT as a PETsys time-offset table.")
    parser.add_argument("lut", type=Path, help="canonical LUT CSV (manifest next to it)")
    parser.add_argument("table", type=Path, help="native TSV to write (manifest .json next to it)")
    parser.add_argument("--ini", type=Path, help="PETsys config.ini to copy with the table activated")
    parser.add_argument("--out-ini", type=Path, help="activated INI copy to write (required with --ini)")
    args = parser.parse_args(argv)
    if (args.ini is None) != (args.out_ini is None):
        parser.error("--ini and --out-ini go together")
    try:
        manifest = write_native_table(args.lut, args.table)
        result = {"table": manifest["table"], "policy": manifest["policy"]}
        if args.ini is not None:
            result["ini"] = activated_ini(args.ini, args.table, args.out_ini)
    except (LutError, OSError) as error:
        print(f"native export failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
