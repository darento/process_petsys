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

Second pass (T9c, FR-20): a conversion made with a native table is declared
by ``write_conversion_manifest`` (it has no Manager ledger) and read back by
``conversion_inputs``. A LUT solved from such data is a residual LUT: its
manifest's ``input_correction`` names the native table, and it is never
exported natively. ``compose_lut`` adds the residual to the native offsets,
giving a LUT for uncorrected data again.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
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


UNCORRECTED = {"state": "uncorrected"}


class LutError(ValueError):
    """The table or manifest is inconsistent, or a correction was requested for a non-applied channel."""


def _number(value):
    return "" if not math.isfinite(value) else repr(float(value))


def write_lut(path, solution, provenance, input_correction=UNCORRECTED):
    """Write ``path`` (CSV) and ``path.with_suffix('.json')``; returns the manifest.

    ``input_correction`` states the data the delays were solved from: uncorrected conversions, or
    ``{"state": "native", ...}`` naming the native table already applied (a residual LUT).
    """
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
        "diagnostics": solution.diagnostics, "provenance": provenance, "input_correction": input_correction,
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

    @property
    def input_correction(self):
        """Correction already in the data the delays were solved from; LUTs before T9c are uncorrected."""
        return self.manifest.get("input_correction", UNCORRECTED)

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
    if lut.input_correction != UNCORRECTED:
        raise LutError(f"{lut_path.name} is a residual LUT of natively corrected data; compose it with its "
                       "source LUT first")
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


# --- Second pass on natively corrected conversions (T9c) ---------------------

CONVERSION_SCHEMA = "cornell-native-conversion/1"
_ACTIVE_TABLE = re.compile(r"^[ \t]*time_offset_calibration_table[ \t]*=([^\r\n]*)\r?$", re.IGNORECASE | re.MULTILINE)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_manifest(table_path):
    """Manifest of a native table written by ``write_native_table``, checked against the table bytes."""
    table_path = Path(table_path)
    manifest = json.loads(table_path.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest.get("schema") != NATIVE_SCHEMA or manifest["table"]["sha256"] != _sha256(table_path):
        raise LutError(f"{table_path.name}: not a native table matching its manifest")
    return manifest


def write_conversion_manifest(directory, prefix, table_path, ini_path, output, *, raw_source, command):
    """Declare a PETsys conversion made with ``table_path`` activated in ``ini_path`` (no Manager ledger).

    Lists every ``<prefix>NNNNNNNN.ldat`` split in ``directory`` (consecutive from 0) with its size; the
    INI must activate exactly this table. ``raw_source`` and ``command`` are recorded as declared.
    """
    directory, table_path, ini_path, output = Path(directory), Path(table_path), Path(ini_path), Path(output)
    if output.exists():
        raise LutError(f"conversion manifest exists: {output}; choose a new name")
    native = _native_manifest(table_path)
    text = ini_path.read_text(encoding="utf-8")
    values = [m.group(1).split("#", 1)[0].strip() for m in _ACTIVE_TABLE.finditer(text)]
    if len(values) != 1 or "%" in values[0] or Path(values[0]).resolve() != table_path.resolve():
        raise LutError(f"{ini_path} must activate exactly {table_path} (found {values})")
    pattern = re.compile(re.escape(prefix) + r"(\d{8})\.ldat")
    files = sorted(({"index": int(m[1]), "name": p.name, "size_bytes": p.stat().st_size}
                    for p in directory.iterdir() if (m := pattern.fullmatch(p.name)) and p.is_file()),
                   key=lambda item: item["index"])
    if not files or [f["index"] for f in files] != list(range(len(files))):
        raise LutError(f"{directory}: no {prefix}NNNNNNNN.ldat splits, or indices not consecutive from 0")
    manifest = {
        "schema": CONVERSION_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "correction_state": "native time offsets applied by PETsys during this conversion",
        "directory": str(directory.resolve()), "prefix": prefix, "files": files,
        "native_table": {"path": str(table_path.resolve()), "sha256": native["table"]["sha256"],
                         "manifest_sha256": _sha256(table_path.with_suffix(".json")),
                         "source_lut_sha256": native["source_lut"]["sha256"]},
        "ini": {"path": str(ini_path.resolve()), "sha256": _sha256(ini_path),
                "time_offset_calibration_table": values[0]},
        "declared": {"raw_source": str(raw_source), "converter_command": str(command)},
    }
    with open(output, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=1) + "\n")
    return manifest


def conversion_inputs(manifest_path, directory, prefix):
    """``(inputs, input_correction)`` of a declared corrected conversion; inputs as ``ledger_inputs`` returns."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CONVERSION_SCHEMA:
        raise LutError(f"{manifest_path}: conversion manifest schema must be {CONVERSION_SCHEMA!r}")
    if manifest["prefix"] != prefix:
        raise LutError(f"{manifest_path}: prefix {manifest['prefix']!r} differs from the run's {prefix!r}")
    inputs = []
    for item in manifest["files"]:
        path = Path(directory) / item["name"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise LutError(f"missing or size-mismatched conversion output: {item['name']}")
        inputs.append({"index": int(item["index"]), "path": path, "size_bytes": int(item["size_bytes"])})
    if not inputs or [item["index"] for item in inputs] != list(range(len(inputs))):
        raise LutError(f"{manifest_path}: split indices are empty or not consecutive from zero")
    table = manifest["native_table"]
    if not Path(table["path"]).is_file() or _sha256(table["path"]) != table["sha256"]:
        raise LutError(f"{manifest_path}: native table {table['path']} missing or changed")
    correction = {"state": "native", "native_table": table["path"], "native_table_sha256": table["sha256"],
                  "source_lut_sha256": table["source_lut_sha256"], "conversion_manifest": str(manifest_path),
                  "conversion_manifest_sha256": _sha256(manifest_path)}
    return inputs, correction


COMPOSED_SOURCES = ("residual plus native", "native only", "residual only", "not applied")


def compose_lut(residual, native_offsets, source):
    """Channels of a LUT for uncorrected data: ``b_native + b_residual`` (FR-20).

    ``residual``: CHANNEL_DTYPE rows solved from data corrected with ``native_offsets`` ({channel: ps}
    as PETsys loaded them); ``source``: the ``CanonicalLut`` the native table came from. Returns
    ``(channels, counts per COMPOSED_SOURCES)``. A residual-applied channel takes the second pass's
    status and uncertainty; a channel only the first pass applied keeps its source row at its native offset.
    """
    if source.input_correction != UNCORRECTED:
        raise LutError("the source LUT must itself be for uncorrected data")
    src = {int(r["time_channel"]): r for r in source.rows}
    applied = {c for c, r in src.items() if r["apply"]}
    if set(native_offsets) != applied or any(abs(native_offsets[c] - float(src[c]["delay_ps"])) > 0.01
                                             for c in applied):
        raise LutError("native offsets are not the applied delays of the source LUT")
    if set(src) != {int(c) for c in residual["time_channel"]}:
        raise LutError("residual and source LUTs cover different TIME channels")
    out = residual.copy()
    counts = dict.fromkeys(COMPOSED_SOURCES, 0)
    for k, row in enumerate(residual):
        channel = int(row["time_channel"])
        if row["apply"]:
            out[k]["delay_ps"] = native_offsets.get(channel, 0.0) + float(row["delay_ps"])
            counts["residual plus native" if channel in native_offsets else "residual only"] += 1
        elif channel in native_offsets:
            out[k] = src[channel]
            out[k]["delay_ps"] = native_offsets[channel]
            counts["native only"] += 1
        else:
            counts["not applied"] += 1
    return out, counts


def _conversion_main(argv):
    import argparse

    parser = argparse.ArgumentParser(prog="python -m src.cornell.timing_petsys conversion",
                                     description="Declare a PETsys conversion made with a native time-offset table.")
    parser.add_argument("directory", type=Path, help="conversion output directory")
    parser.add_argument("--prefix", required=True, help="split prefix before NNNNNNNN.ldat")
    parser.add_argument("--table", type=Path, required=True, help="native TSV used (manifest .json next to it)")
    parser.add_argument("--ini", type=Path, required=True, help="activated INI the conversion used")
    parser.add_argument("--raw-source", required=True, help="RAWF path converted (recorded as declared)")
    parser.add_argument("--command", required=True, help="converter command line (recorded as declared)")
    parser.add_argument("--output", type=Path, required=True, help="manifest JSON to write")
    args = parser.parse_args(argv)
    try:
        manifest = write_conversion_manifest(args.directory, args.prefix, args.table, args.ini, args.output,
                                             raw_source=args.raw_source, command=args.command)
    except (LutError, OSError) as error:
        print(f"conversion manifest failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"files": len(manifest["files"]), "native_table": manifest["native_table"]}, indent=1))
    return 0


def main(argv=None):
    import argparse

    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["conversion"]:
        return _conversion_main(argv[1:])

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
