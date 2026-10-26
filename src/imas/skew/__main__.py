"""``python -m src.imas.skew COMMAND ...``: the IMAS skew command (spec 011).

Each command keeps its own options (``COMMAND -h``); modules load only when
their command runs.
"""
import sys


def _inventory(argv):
    from src.imas.skew import inventory
    return inventory.main(argv)


def _encal(argv):
    from src.imas.skew import encal
    return encal.main(argv)


def _observe(argv):
    from src.imas.skew import observe
    return observe.main(argv)


def _pairstats(argv):
    from src.imas.skew import pairstats
    return pairstats.main(argv)


def _solve(argv):
    from src.imas.skew import solve
    return solve.main(argv)


def _study(argv):
    from src.imas.skew import study
    return study.main(argv)


def _run(argv):
    from src.imas.skew import job
    return job.main(argv)


def _plot(argv):
    from src.imas.skew import plots
    return plots.main(argv)


def _ctr(argv):
    from src.imas.skew import ctr
    return ctr.main(argv)


def _export(argv):
    from src.imas.skew import outputs
    return outputs.main(argv)


COMMANDS = {
    "inventory": (_inventory, "raw hits per TIME channel, SM and minimodule of skew splits"),
    "encal": (_encal, "per-TIME-channel 511 keV calibration from dedicated skew splits"),
    "observe": (_observe, "feature cache and observation set of skew splits, with a stage report"),
    "pairstats": (_pairstats, "FR-7 pair statistics of an observation set"),
    "solve": (_solve, "global EM and/or spec 008 pair estimator: LUTs, held-out check, comparisons"),
    "study": (_study, "frame-sign and opening-angle studies from a feature cache (EM)"),
    "run": (_run, "pooled job: energy calibration, streamed observations, both estimators, checks"),
    "plot": (_plot, "diagnostics of an IMAS LUT: per ASIC, ring x azimuth, legacy, train/held-out"),
    "ctr": (_ctr, "point-source CTR: uncorrected, legacy and new LUTs on the same 22Na events (FR-10)"),
    "export": (_export, "PETsys native table and/or legacy skew file from a canonical IMAS LUT"),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: python -m src.imas.skew COMMAND [options]   (COMMAND -h for its options)\n\ncommands:")
        for name, (_, text) in COMMANDS.items():
            print(f"  {name:<11} {text}")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return COMMANDS[argv[0]][0](argv[1:]) or 0


if __name__ == "__main__":
    sys.exit(main())
