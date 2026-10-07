"""Bounded storage: each processing action end to end under tracemalloc (spec 003 T17; spec 007 T12).

Moved from scripts/petsys_manager_bounded_check.py. Each processing action runs end to end through
``src.cornell.cli.main`` (request -> validation -> processing -> plots/debug -> published result) in-process under
tracemalloc, on BASE and then COPIES byte-identical copies of one fixture file.
Limits fixed before the first run (2026-10-02): peak growth <= BUDGET, while the
added copies hold >= SENSITIVITY input bytes, so keeping even the raw input would
exceed the budget. Per-file counts must be equal and totals scale exactly.
Deviation (2026-10-02): the first run compared 1 file with 8. Calibration peaks
are deterministic after warm-up: 1 file 51.65 MiB, 2 files 58.46, 3 58.59,
8 58.61, 16 58.63. That is a one-time step at the second file, not growth, so the
baseline is BASE = 2 files. The budget is unchanged; the 1-file peak is printed.
Synthetic fixtures only.
"""

import gc
from pathlib import Path
import shutil
import sys
import tracemalloc
import unittest

import pytest

from manager_helpers import P3_SPECS, CLIFixtures, entry, pairs, sides_for
from src.cornell import listmode as lm
from src.petsys_manager.contracts import DataFormat, InputDescriptor

BASE = 2
COPIES = 8
SCALE = COPIES // BASE
BUDGET = 2 * 1024 * 1024          # peak growth allowed for COPIES - BASE extra files
SENSITIVITY = 8 * 1024 * 1024     # minimum input bytes in those extra files


@pytest.mark.fr("003-FR-15", "003-FR-16")  # spec 003 T17
class BoundedChecks(CLIFixtures, unittest.TestCase):
    """Reuses the CLI fixtures and in-process runner (``CLIFixtures``)."""

    fixture_prefix = "petsys-manager-bounded-"

    def setUp(self):
        self.work = self.output / self._testMethodName.replace("test_bounded_", "b_")[:14] / "bounded ; & [x]"
        self.work.mkdir(parents=True)
        self.count = 0
        self.helpers = 0

    def copies(self, descriptor, count):
        """``count`` byte-identical copies, distinct basenames in natural order."""
        out = []
        for index in range(1, count + 1):
            path = descriptor.path.with_name(f"bounded_{count}x_{index}.ldat")
            shutil.copyfile(descriptor.path, path)
            out.append(InputDescriptor(path, descriptor.format, descriptor.population))
        return out

    def measure(self, action, request, inputs, outputs):
        run = dict(request, inputs=[entry(d) for d in inputs], outputs=outputs)
        gc.collect()
        tracemalloc.start()
        try:
            code, result, _, stderr = self.in_process(action, run)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(code, 0, stderr[-4000:])
        self.assertEqual(result["status"], "succeeded")
        return peak, result

    def bounded(self, action, request, base, outputs, *, budget=BUDGET):
        """Warm-up, 1 copy (recorded only), then BASE vs COPIES copies; returns (base, many) summaries.
        ``budget=None`` records the peaks without asserting a growth bound."""
        added = (COPIES - BASE) * base.path.stat().st_size
        self.assertGreaterEqual(added, SENSITIVITY, "fixture too small to detect per-event retention")
        self.measure(action, request, self.copies(base, 1), outputs("warm"))   # imports, fonts, caches
        peak_single, _ = self.measure(action, request, self.copies(base, 1), outputs("single"))
        peak_one, one = self.measure(action, request, self.copies(base, BASE), outputs("one"))
        peak_many, many = self.measure(action, request, self.copies(base, COPIES), outputs("many"))
        print(f"\n  {action}: 1 file {peak_single / 2**20:.2f} MiB; {BASE} -> {COPIES} files (+{added / 2**20:.1f} "
              f"MiB input): peak {peak_one / 2**20:.2f} -> {peak_many / 2**20:.2f} MiB, growth "
              f"{(peak_many - peak_one) / 2**10:.0f} KiB, budget "
              f"{f'{budget / 2**10:.0f} KiB' if budget else 'none'}", file=sys.stderr)
        if budget is not None:
            self.assertLessEqual(peak_many - peak_one, budget, (peak_one, peak_many))
        self.assertEqual((len(one["summary"]["inputs"]), len(many["summary"]["inputs"])), (BASE, COPIES))
        return one["summary"], many["summary"]

    @pytest.mark.slow  # ~8 s
    def test_bounded_calibration_end_to_end(self):
        fx, _, _, request = self.calibration_request(positions=5)
        # One file with every seeded side (half a fixture leaves keys below the fit minimum).
        base = fx.write("all.ldat", pairs(sides_for(fx.geometry, P3_SPECS, seed=5, depth_spread=1.0)),
                        DataFormat.FIXED)

        def outputs(tag):
            folder = self.work / f"encal {tag}"
            folder.mkdir()
            return {"encal": str(folder / "b.encal"), "sidecar": str(folder / "b.encal.json"),
                    "status": str(folder / "b_status.txt"), "plot": str(folder / "b.png")}
        one, many = self.bounded("calibrate", request, base, outputs)
        for key in ("records_read", "events_passed", "accepted_sides"):
            self.assertEqual([item[key] for item in many["inputs"]], [one["inputs"][0][key]] * COPIES, key)
        self.assertEqual((many["keys"], many["layout"]), (one["keys"], one["layout"]))   # keys, not events

    @pytest.mark.slow  # ~65 s
    def test_bounded_listmode_with_debug_end_to_end(self):
        h, descriptors, _, request = self.listmode_request(count=3000, debug=True)
        outputs = lambda tag: {"directory": str(h.lm_parent / f"bounded {tag}")}
        one, many = self.bounded("listmode", request, descriptors[0], outputs)
        self.assertGreater(one["records_written"], 0)
        self.assertEqual(many["records_written"], SCALE * one["records_written"])
        self.assertEqual({k: v * SCALE for k, v in one["rejected"].items()}, many["rejected"])
        self.assertEqual([item["records_written"] for item in many["inputs"]],
                         [one["inputs"][0]["records_written"]] * COPIES)
        job = Path(h.lm_parent / "bounded many")
        written = [p for p in job.iterdir() if p.suffix == ".lm"]
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].stat().st_size,
                         lm.HEADER_BYTES + many["records_written"] * lm.RECORD_DTYPE.itemsize)

    @pytest.mark.slow  # ~92 s
    def test_bounded_qc_with_plots_and_slabs_end_to_end(self):
        h, descriptors, request = self.qc_request(count=7000, plots=True, slabs=True)
        outputs = lambda tag: {"directory": str(h.results / f"bounded {tag}")}
        # Plot memory is set by image size, not events, but denser histograms move the plotting peak by up to
        # ~3 MiB (owner 2026-10-07, spec 007 T12): the budget applies to the same run without plots and slabs.
        one, many = self.bounded("qc", request, descriptors[0], outputs, budget=None)
        for key in ("records_read", "occupancy_pairs", "accepted_pairs"):
            self.assertEqual([item[key] for item in many["inputs"]], [one["inputs"][0][key]] * COPIES, key)
        self.assertGreater(one["inputs"][0]["accepted_pairs"], 0)
        self.assertEqual(many["totals"]["accepted_pairs"], SCALE * one["totals"]["accepted_pairs"])
        bare = dict(request, options=dict(request["options"], plots=False, slabs=False))
        outputs = lambda tag: {"directory": str(h.results / f"bounded bare {tag}")}
        bare_one, bare_many = self.bounded("qc", bare, descriptors[0], outputs)
        self.assertEqual((bare_one["totals"], bare_many["totals"]), (one["totals"], many["totals"]))
