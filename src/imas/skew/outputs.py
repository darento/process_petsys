"""IMAS LUT outputs: canonical LUT, PETsys native table, legacy skew file (spec 011 T12, FR-9).

Every IMAS output is tied to its map. The canonical LUT's manifest records
``scanner: imas`` and the map's SHA-256, and every reader here refuses a LUT
or legacy file made for another map. The checks: scanner, map fingerprint,
and channels that are not TIME channels of the selected map.

* **Canonical:** ``src.timing.lut`` CSV and manifest; ``t_corrected = t - b``.
* **Native:** one ``time_offset_calibration_table`` for both DAQ boards; rows
  ``port slave chip channel t0`` decoded from the IMAS global IDs for applied
  channels only (``src.timing.lut.write_native_table``).
* **Legacy:** ``ch skew`` lines as ``src/utils.py:read_skew`` reads them. The
  legacy consumers add the value (``t + skew``), so ``skew = -b``. Channels
  without an applied delay are omitted, and legacy readers then fall back to
  ``9999``. On import, ``9999`` means no delay, never a value.
* ``DelayTable`` / ``correct_timestamps`` (``src.timing.lut``) apply either
  form to hits.

``python -m src.imas.skew export LUT --native T.tsv --legacy S.txt`` writes
the requested forms.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from src.imas.skew.setup import ImasSetup
from src.timing import lut
from src.utils import read_skew

SCANNER = "imas"
LEGACY_MISSING = 9999.0
LEGACY_CONVENTION = ("additive: t_corrected = t + skew, skew = -delay_ps; channels absent from the file have no "
                     "delay (legacy readers use 9999)")


def write_imas_lut(path, solution, provenance, setup):
    """Canonical LUT whose manifest names the scanner and the map it belongs to."""
    unknown = np.setdiff1d(solution.channels["time_channel"], setup.time_channels)
    if unknown.size:
        raise lut.LutError(f"LUT rows that are not TIME channels of the selected map: {unknown[:5].tolist()}")
    return lut.write_lut(path, solution, {**provenance, "scanner": SCANNER,
                                          "map_sha256": setup.provenance["map_sha256"],
                                          "config_sha256": setup.provenance["config_sha256"]})


def read_imas_lut(path, setup):
    """``CanonicalLut`` checked against the selected IMAS map."""
    table = lut.CanonicalLut.read(path)
    provenance = table.manifest.get("provenance", {})
    if provenance.get("scanner") != SCANNER:
        raise lut.LutError(f"{Path(path).name}: not an IMAS LUT (scanner {provenance.get('scanner')!r})")
    if provenance.get("map_sha256") != setup.provenance["map_sha256"]:
        raise lut.LutError(f"{Path(path).name}: made for another map (map SHA-256 differs)")
    unknown = np.setdiff1d(table.rows["time_channel"], setup.time_channels)
    if unknown.size:
        raise lut.LutError(f"{Path(path).name}: channels outside the map's TIME channels: {unknown[:5].tolist()}")
    return table


def delay_table(path, setup):
    return lut.DelayTable.from_lut(read_imas_lut(path, setup))


def write_native(lut_path, table_path, setup):
    """PETsys native table of the applied delays (one table, both DAQ boards)."""
    read_imas_lut(lut_path, setup)
    manifest = lut.write_native_table(lut_path, table_path)
    manifest.update({"scanner": SCANNER, "map_sha256": setup.provenance["map_sha256"]})
    Path(table_path).with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def write_legacy(lut_path, out_path, setup):
    """Legacy ``ch skew`` file (``skew = -b``), applied channels only, and a manifest next to it."""
    out_path = Path(out_path)
    manifest_path = out_path.with_suffix(".json")
    for path in (out_path, manifest_path):
        if path.exists():
            raise lut.LutError(f"legacy output exists: {path}; choose a new name")
    table = read_imas_lut(lut_path, setup)
    rows = table.rows[table.rows["apply"]]
    rows = rows[np.argsort(rows["time_channel"], kind="stable")]
    out_path.write_text("".join(f"{int(r['time_channel'])} {repr(-float(r['delay_ps']))}\n" for r in rows),
                        encoding="utf-8")
    back = read_skew(str(out_path))
    expected = {int(r["time_channel"]): -float(r["delay_ps"]) for r in rows}
    if back != expected:
        out_path.unlink()
        raise lut.LutError("legacy file does not read back to the applied delays")
    manifest = {"schema": "imas-legacy-skew/1", "convention": LEGACY_CONVENTION, "scanner": SCANNER,
                "map_sha256": setup.provenance["map_sha256"],
                "source_lut": {"name": Path(lut_path).name, "sha256": table.manifest["table"]["sha256"]},
                "rows": int(len(rows)), "omitted_time_channels": int(len(setup.time_channels) - len(rows))}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def read_legacy(path, setup):
    """Legacy ``ch skew`` file as a ``DelayTable`` (b = -skew); ``9999`` and absent TIME channels have no delay."""
    values = read_skew(str(path))
    channels = np.array(sorted(values), np.int64)
    unknown = np.setdiff1d(channels, setup.time_channels)
    if unknown.size:
        raise lut.LutError(f"{Path(path).name}: channels outside the map's TIME channels: {unknown[:5].tolist()}")
    delays = {c: -s for c, s in values.items() if s != LEGACY_MISSING}
    if any(not math.isfinite(b) for b in delays.values()):
        raise lut.LutError(f"{Path(path).name}: non-finite skew value")
    missing = set(setup.time_channels.tolist()) - set(delays)
    return lut.DelayTable(delays, missing, {"legacy": str(path), "convention": LEGACY_CONVENTION,
                                            "listed_9999": int(sum(s == LEGACY_MISSING for s in values.values()))})


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew export", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lut", type=Path, help="canonical IMAS LUT CSV (manifest next to it)")
    parser.add_argument("--native", type=Path, help="PETsys time-offset table to write")
    parser.add_argument("--legacy", type=Path, help="legacy ch-skew file to write")
    parser.add_argument("--config", type=Path, default=Path("configs/imas_2DAQ.yaml"))
    args = parser.parse_args(argv)
    if args.native is None and args.legacy is None:
        parser.error("nothing to export: give --native and/or --legacy")
    setup = ImasSetup.from_files(args.config)
    result = {}
    try:
        if args.native is not None:
            result["native"] = write_native(args.lut, args.native, setup)["table"]
        if args.legacy is not None:
            result["legacy"] = {k: v for k, v in write_legacy(args.lut, args.legacy, setup).items()
                                if k in ("rows", "omitted_time_channels")}
    except (lut.LutError, OSError) as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
