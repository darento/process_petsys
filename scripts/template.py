"""Template for a local analysis script built on the `src` modules.

Copy this file to a new name in scripts/ and edit ``analyse``. Everything in
scripts/ except this template stays untracked, so copies can use hardcoded
paths.

The population is detector sides of accepted coincidence pairs (LDAT compact
files): there are no singles. Energies are PETsys a.u. unless a calibration is
given; slabs or channels without a keV factor stay NaN.

Run from the repo root, for example:
    python scripts/template.py configs/cornell_full_system.yaml run_00000003.ldat run_00000004.ldat \
        --system CORNELL --calibration encal_files/<name>_resolved.encal --max-pairs 200000
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # makes `from src ... import` work from scripts/

import numpy as np

from src.ldat_inspector.engine import (Selection, Settings, fit_peak, merge_results, process_file,
                                system_channel_findings)


def read(settings, files, workers):
    """Read the files in parallel (one process per file) and merge them."""
    with ProcessPoolExecutor(max_workers=min(len(files), workers)) as pool:
        results = list(pool.map(process_file, files, [settings] * len(files), range(len(files))))
    for result in results:
        status = "ok" if result.success else f"FAILED: {result.error}"
        print(f"{Path(result.path).name}: {result.pairs_accepted:,}/{result.pairs_read:,} pairs accepted ({status})")
    # consume=True frees each file's table while merging (lower peak memory)
    return merge_results(settings, results, consume=True)


def analyse(dataset):
    """Replace with your analysis. ``dataset.modules`` maps SM -> ModuleEvents.

    Per-side columns of a ModuleEvents: energy (keV if calibrated, else a.u.),
    raw_energy, x, y, doi (light-sharing ratio, not mm), timestamp (ps), mm,
    file_index, calibration_key, and partner_* for the other side of each pair.
    """
    calibrated = dataset.settings.calibrated
    unit = "keV" if calibrated else "a.u."
    print(f"\n{len(dataset.table):,} detector sides in {len(dataset.modules)} SuperModules ({unit})")

    # Example 1: channel occupancy findings, one row per mapped SM.
    for row in system_channel_findings(dataset):
        if row["state"] != "OK":
            flagged = {kind: {s: n for s, n in row[kind]["counts"].items() if s != "OK"}
                       for kind in ("time", "energy")}
            print(f"SM {row['sm']:3d} {row['state']:<20} {row['events']:>10,} sides  {flagged}")

    # Example 2: photopeak per SM on the DOI/XY selection (energy window off, as in uniformity).
    if calibrated:
        selection = Selection()
        for sm, data in sorted(dataset.modules.items()):
            energies = data.energy[selection.mask(data, energy=False)]
            fit = fit_peak(energies[np.isfinite(energies)])
            if fit["status"] == "FIT":
                print(f"SM {sm:3d}: photopeak {fit['mu']:.1f} keV, resolution {fit['resolution']:.1f} %")
            else:
                print(f"SM {sm:3d}: {fit['status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="system config YAML (selects the map)")
    parser.add_argument("files", nargs="+", help=".ldat compact coincidence files")
    parser.add_argument("--system", choices=("IMAS", "CORNELL"), default="CORNELL")
    parser.add_argument("--calibration", default="", help="energy calibration; omit for raw a.u.")
    parser.add_argument("--max-pairs", type=int, default=None, help="prefix per file; default whole files")
    parser.add_argument("--min-channels", type=int, default=4)
    parser.add_argument("--min-channel-energy", type=float, default=0.2, help="per-channel cut in a.u.")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    settings = Settings(args.config, args.calibration, args.system, max_pairs=args.max_pairs,
                        min_channels=args.min_channels, min_channel_energy=args.min_channel_energy,
                        calibrated=bool(args.calibration))
    scope = "whole files" if args.max_pairs is None else f"first {args.max_pairs:,} pairs per file"
    print(f"{args.system}, {Path(args.config).name}, {scope}, >= {args.min_channels} channels, "
          f">= {args.min_channel_energy} a.u. per channel, calibration: {args.calibration or 'none'}")
    analyse(read(settings, args.files, args.workers))
    return 0


if __name__ == "__main__":  # required: worker processes re-import this file on Windows
    sys.exit(main())
