"""``python -m src.cornell.skew COMMAND ...``: the one Cornell skew command (spec 008 FR-15).

Each command keeps its own options (``COMMAND -h``); modules load only when
their command runs.
"""
import sys


def _run(argv):
    from src.cornell.skew import job
    return job.main(argv)


def _validate(argv):
    from src.cornell.skew import validation
    return validation.main(argv)


def _plot(argv):
    from src.cornell.skew import plots
    return plots.main(argv)


def _export(argv):
    from src.cornell.skew import lut
    return lut.export_main(argv)


def _conversion(argv):
    from src.cornell.skew import lut
    return lut.conversion_main(argv)


def _roundtrip(argv):
    from src.cornell.skew import roundtrip
    return roundtrip.main(argv)


def _ctr(argv):
    from src.cornell.skew import ctr
    return ctr.main(argv)


def _inventory(argv):
    from src.cornell.skew import dataset
    return dataset.main(argv)


def _benchmark(argv):
    from src.cornell.skew import benchmark
    return benchmark.main(argv)


COMMANDS = {
    "run": (_run, "pooled calibration job: JOB.json -> canonical LUT and report"),
    "validate": (_validate, "frozen-LUT test evaluation: VALIDATION.json"),
    "plot": (_plot, "per-ASIC plots of a canonical LUT"),
    "export": (_export, "canonical LUT -> native PETsys time-offset table (+ activated INI)"),
    "conversion": (_conversion, "declare a conversion made with a native table (manifest JSON)"),
    "roundtrip": (_roundtrip, "matched-event check of a native conversion"),
    "ctr": (_ctr, "independent CTR from known-position point sources: CTR.json"),
    "inventory": (_inventory, "raw LDAT inventory and holdout (development)"),
    "benchmark": (_benchmark, "bounded raw-decoder benchmark (development)"),
}


def usage():
    width = max(map(len, COMMANDS))
    lines = ["usage: python -m src.cornell.skew COMMAND [options]   (COMMAND -h for its options)", "", "commands:"]
    lines += [f"  {name:<{width}}  {summary}" for name, (_, summary) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage(), file=sys.stdout if argv else sys.stderr)
        return 0 if argv else 2
    if argv[0] not in COMMANDS:
        print(f"unknown command {argv[0]!r}\n\n{usage()}", file=sys.stderr)
        return 2
    result = COMMANDS[argv[0]][0](argv[1:])
    return 0 if result is None else result


if __name__ == "__main__":
    sys.exit(main())
