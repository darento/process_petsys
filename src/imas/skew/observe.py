"""IMAS feature cache and observation set over declared splits, with a stage report (spec 011 T6, FR-6).

``python -m src.imas.skew observe SPLITS... --encal ENCAL.tsv --work DIR`` builds
``DIR/features`` (selection, one block set per split) and ``DIR/observations``
(nominal geometry, bar shell), resuming completed files, and writes
``DIR/observe_report.json``: per-cut counts, accepted observations, distinct
pairs, cache bytes per LDAT byte, stage times and peak memory. Every split is
a train file here (development); held-out roles come with the job (T11).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from src.imas.skew import cache as ic
from src.imas.skew import select as sel
from src.imas.skew.geometry import ImasGeometry
from src.imas.skew.setup import ImasSetup
from src.timing.report import peak_memory, tree_bytes


def observe(splits, work, *, config_path, encal_path, root=".", workers=1, max_records=None, partitions=16,
            window_ps=sel.COINCIDENCE_WINDOW_PS, kev_window=sel.KEV_WINDOW):
    work = Path(work)
    setup = ImasSetup.from_files(config_path, encal_path, root=root)
    inputs = [{"index": i, "path": Path(p)} for i, p in enumerate(splits)]
    started = time.perf_counter()
    features = ic.build_feature_cache(work / "features", inputs, config_path=config_path, encal_path=encal_path,
                                      root=root, holdout=ic.all_train(range(len(inputs))),
                                      energy_calibration_prefix={}, window_ps=window_ps, kev_window=kev_window,
                                      max_records=max_records, workers=workers)
    features.require_complete()
    feature_seconds = time.perf_counter() - started
    geometry = ImasGeometry.nominal(setup)
    observations = ic.build_observations(features, geometry, work / "observations", partitions=partitions)
    observation_seconds = time.perf_counter() - started - feature_seconds
    pair_counts = observations.pair_counts()
    first_failure = {}
    for done in features.files.values():
        for label, n in done["first_failure"].items():
            first_failure[label] = first_failure.get(label, 0) + n
    ldat_bytes = sum(Path(p).stat().st_size for p in splits)
    feature_bytes, observation_bytes = tree_bytes(work / "features"), tree_bytes(work / "observations")
    records = sum(done["records"] for done in features.files.values())
    report = {
        "schema_version": 1, "scanner": "imas", "requirement": "011-FR-6",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "development: every split in the train role; nominal geometry and source",
        "inputs": [{"index": i["index"], "name": Path(i["path"]).name, "bytes": Path(i["path"]).stat().st_size}
                   for i in inputs],
        "max_records": max_records, "provenance": setup.provenance,
        "selection": sel.Selection.production(setup.lookup(), window_ps=window_ps, kev_window=kev_window).describe(),
        "geometry": geometry.describe(),
        "records": records, "first_failure": first_failure,
        "geometry_counts": observations.manifest["geometry"]["train"],
        "accepted_observations": int(pair_counts["count"].sum()), "distinct_pairs": int(len(pair_counts)),
        "bytes": {"ldat": ldat_bytes, "features": feature_bytes, "observations": observation_bytes,
                  "features_per_ldat_byte": feature_bytes / ldat_bytes if ldat_bytes else None},
        "seconds": {"features": feature_seconds, "observations": observation_seconds},
        "peak_memory": peak_memory(),
        "peak_memory_note": None if workers <= 1 else "feature workers are child processes; not included on Windows",
        "workers": workers,
    }
    (work / "observe_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.imas.skew observe", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("splits", type=Path, nargs="+")
    parser.add_argument("--encal", type=Path, required=True, help="encal.tsv from the encal command")
    parser.add_argument("--work", type=Path, required=True, help="cache directory (resumed when it matches)")
    parser.add_argument("--config", type=Path, default=Path("configs/imas_2DAQ.yaml"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--partitions", type=int, default=16)
    args = parser.parse_args(argv)
    report = observe(args.splits, args.work, config_path=args.config, encal_path=args.encal, workers=args.workers,
                     max_records=args.max_records, partitions=args.partitions)
    print(json.dumps({k: report[k] for k in ("records", "first_failure", "geometry_counts", "accepted_observations",
                                             "distinct_pairs", "bytes", "seconds")}, indent=1))
    print(f"peak memory {report['peak_memory']['parent_bytes'] / 2 ** 30:.2f} GiB -> {args.work}")
    return 0
