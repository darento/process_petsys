"""IMAS compact LDAT -> LM V5 with the delivered timing tables (spec 015).

Usage:
  imas_ldat2lm [options] YAMLCONF SLAB_EN_MAP INFILES...
  imas_ldat2lm -h | --help

Arguments:
  YAMLCONF      IMAS config (min_ch, en_min_ch, energy_range, map_file), e.g. configs/imas_2DAQ.yaml.
  SLAB_EN_MAP   Energy calibration "ID<tab>mu" per TIME channel (keV = 511 / mu x minimodule ENERGY sum).
  INFILES       PETsys compact coincidence .ldat files (natural-sorted; the first sets the LM time origin).

Options:
  -o DIR --out DIR         Output directory [default: auto], i.e. <lm-root>/lm-<date>-<experiment>/.
  -c DIR --continue DIR    Continue the run in DIR: files whose segment completed are skipped.
  --lm-root DIR            Root of the automatic output directory
                           [default: /mnt/data/000_samba_lin/LaFe_acquisitions_win/LMs].
  --luts DIR               The delivered I3M_newSkew_walk/luts directory
                           [default: /scratch/imas_files_cal/I3M_newSkew_walk/luts].
  --rule RULE              Side timestamp rule: top_earliest_mean (EM4 + walk_rule4) or top (EM2 + walk3)
                           [default: top_earliest_mean].
  --skew FILE              Canonical IMAS skew LUT (.csv with its .json); default from --luts and --rule.
  --walk FILE              Its walk LUT (imas_walk_lut.json), or "none" for a skew-only canonical LUT;
                           default from --luts and --rule.
  --walk-encal FILE        Energy calibration of the walk keV axis; default --luts energy_calibration.
  --legacy-skew FILE       Skew-only legacy "ch skew" file (t + skew) instead of the LUTs (rule top, no walk).
  --no-timing              Write dt = -32700 for every event (no timing correction).
  --nn-y FILE              NN Y model [default: /scratch/imas_files_cal/IMASNoCatY.h5].
  --nn-doi FILE            NN DOI model [default: /scratch/imas_files_cal/IMASNoCatDOI.h5].
  --nn-batch N             Keras predict batch [default: 16384].
  --pairs FILE             SM pair file "p s0 s1" [default: /scratch/imas_files_cal/pares_5SR.txt].
  -w N --workers N         File worker processes; 0 runs in this process [default: auto].
  --acq-time S             LM header acquisition time in s [default: 300].
  --activity MCI           LM header activity in mCi [default: 0].
  --isotope ISO            LM header isotope, F18 or Na22 [default: F18].
  -a --ask                 Ask the acquisition time and activity on the command line (legacy -a).
  -d --debug               Save total_energy.png and sm_hits.png in the output directory.
  --keep-segments          Keep the per-file .lm segments after the merge.
  --max-records N          Read only the first N records of each file (tests, quick looks).
  -h --help                Show this screen.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

from docopt import docopt

from src.imas.listmode import job

REPO = Path(__file__).resolve().parents[3]
PAIRS_BY_RULE = {"top_earliest_mean": ("skew_em4/lut_em_walk-f5d870aa6390_rule-mean-freeze200.csv",
                                       "walk_rule4/imas_walk_lut.json"),
                 "top": ("skew_em2/lut_em_walk-dd68b558604c.csv", "walk3/imas_walk_lut.json")}
WALK_ENCAL = "energy_calibration/encal_skew_run7.tsv"


def _ask(prompt, default, kind):
    text = input(prompt)
    return default if text.strip() == "" else kind(text)


def settings_from_args(args, *, now=None):
    """``job.Settings`` from parsed docopt ``args`` (raises ``job.JobError`` naming a bad option)."""
    inputs = [str(p) for p in args["INFILES"]]
    rule = args["--rule"]
    timing = {"no_timing": bool(args["--no-timing"]), "legacy_skew": args["--legacy-skew"], "rule": rule}
    if timing["no_timing"] and timing["legacy_skew"]:
        raise job.JobError("--no-timing and --legacy-skew exclude each other")
    if not timing["no_timing"] and not timing["legacy_skew"]:
        luts = Path(args["--luts"])
        if rule not in PAIRS_BY_RULE and not (args["--skew"] and args["--walk"]):
            raise job.JobError(f"--rule {rule!r}: give --skew and --walk for it, or use {sorted(PAIRS_BY_RULE)}")
        default_skew, default_walk = PAIRS_BY_RULE.get(rule, (None, None))
        timing["skew"] = args["--skew"] or str(luts / default_skew)
        walk = args["--walk"] or str(luts / default_walk)
        timing["walk"] = None if walk == "none" else walk
        if timing["walk"] is not None:
            timing["walk_encal"] = args["--walk-encal"] or str(luts / WALK_ENCAL)
    elif rule not in ("top", "top_earliest_mean"):
        raise job.JobError("--legacy-skew and --no-timing use rule top only")
    if timing.get("legacy_skew") or timing["no_timing"]:
        timing["rule"] = "top"
    acq_time, activity = float(args["--acq-time"]), float(args["--activity"])
    if args["--ask"]:
        acq_time = _ask("Enter the acquisition time of the measurement (total) in seconds: ", acq_time, int)
        activity = _ask("Enter the activity of the source in mCi: ", activity, float)
    resume = args["--continue"]
    if resume and args["--out"] != "auto":
        raise job.JobError("give either --out or --continue")
    if resume:
        out_dir = resume
    elif args["--out"] != "auto":
        out_dir = args["--out"]
    else:
        from natsort import natsorted
        out_dir = str(job.default_out_dir(args["--lm-root"], natsorted(inputs)[0], now))
    workers = args["--workers"]
    workers = max(1, min(len(inputs), (os.cpu_count() or 2) - 1)) if workers == "auto" else int(workers)
    max_records = None if args["--max-records"] is None else int(args["--max-records"])
    return job.Settings(config=args["YAMLCONF"], slab_en_map=args["SLAB_EN_MAP"], inputs=inputs, out_dir=out_dir,
                        nn=("keras", args["--nn-y"], args["--nn-doi"], int(args["--nn-batch"])),
                        pairs=args["--pairs"], workers=workers, acq_time=acq_time, activity=activity,
                        isotope=args["--isotope"], debug=bool(args["--debug"]), resume=bool(resume),
                        keep_segments=bool(args["--keep-segments"]), max_records=max_records, root=str(REPO),
                        **timing)


def main(argv=None):
    args = docopt(__doc__, argv=argv)
    try:
        settings = settings_from_args(args)
        provenance = job.run(settings, log=lambda message: print(message, flush=True))
    except (job.JobError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Directory: {Path(provenance['output']).parent}")
    print(f"Provenance: {Path(provenance['output']).with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
