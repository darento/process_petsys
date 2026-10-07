"""Energy calibration: per-slab and position calibration, limits, workers, outputs (spec 003 T20, T25; spec 007 T9).

Moved from scripts/petsys_manager_calibration_check.py. The reference outputs (scripts_cornell/cornell_slab_en_cal.py)
are the golden files under tests/data/golden/calibration/ (T8); no test loads the reference.
"""

import ast
import io
import json
import os
from pathlib import Path
import struct
import time
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from helpers import DATA, REPO
from manager_helpers import (A, B, HIT, P1_SPECS, P3_SPECS, CalibrationFixtures, ListmodeFixtures, PrivateOutput,
                             encode, pairs, sides_for)
from src.cornell import calibration as cal
from src.cornell.inputs import InputError, load_calibration, load_channel_map, validate_ldat
from src.mapping_generator import ChannelType
from src.petsys_manager.contracts import DataFormat, InputDescriptor, Population
from src.utils import KevConverter

GOLDEN = DATA / "golden" / "calibration"
OWNER_ENCAL = "20260119_2NaSourcesAxialSeparated_vBiasCompDiscCalibAdjusted2hits_300s_coincCompact11s_resolved.encal"


def die_after_marker(path):
    """Pool task: leave a marker, then die abruptly, as a worker killed by the STOP's group SIGTERM."""
    Path(path).write_text("died", encoding="utf-8")
    os._exit(1)


def golden(name):
    """A reference output frozen at T8 (FR-3): text byte for byte, or parsed JSON."""
    text = (GOLDEN / name).read_bytes().decode("utf-8")
    return json.loads(text) if name.endswith(".json") else text


@pytest.mark.fr("003-FR-10", "003-FR-12", "003-FR-15", "003-FR-16", "003-FR-21")  # spec 003 T20
class CalibrationChecks(PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-calibration-"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fx = CalibrationFixtures(cls.output / "shared")
        records = pairs(sides_for(cls.fx.geometry, P1_SPECS, seed=11))
        half = len(records) // 2
        cls.p1 = {fmt: [cls.fx.write(f"p1_{fmt.value}_{i}.ldat", part, fmt)
                        for i, part in enumerate((records[:half], records[half:]))]
                  for fmt in (DataFormat.COMPACT, DataFormat.FIXED)}
        cls.p1_records = records
        records3 = pairs(sides_for(cls.fx.geometry, P3_SPECS, seed=5, depth_spread=1.0))
        cls.p3 = {fmt: [cls.fx.write(f"p3_{fmt.value}.ldat", records3, fmt)]
                  for fmt in (DataFormat.COMPACT, DataFormat.FIXED)}
        cls.limits = cls.fx.limits(missing={(B, 7)})

    def setUp(self):
        self.root = self.output / self._testMethodName[5:40]
        self.root.mkdir()

    def ours(self, descriptors, positions=1, **options):
        limits = self.limits if positions > 1 else None
        return cal.calibrate(descriptors, self.fx.config, limits, positions=positions, **options)

    # Exact binning and fit entry point ----------------------------------------------------------

    def test_bin_index_equals_numpy_histogram_including_edges(self):
        rng = np.random.default_rng(1)
        edges = np.linspace(0, 220, 151)
        values = np.concatenate([rng.uniform(-5, 230, 20000), edges, np.nextafter(edges, -np.inf),
                                 np.nextafter(edges, np.inf), [220.0, 0.0]])
        index = cal.bin_index(values, 0.0, 220.0, edges, 150)
        expected, _ = np.histogram(values, bins=150, range=(0, 220))
        self.assertEqual(np.bincount(index[index >= 0], minlength=150).tolist(), expected.tolist())
        anchors = rng.uniform(40, 120, 300)
        lows, highs = 0.55 * anchors, 1.5 * anchors
        rows = rng.integers(0, 300, 30000)
        table = np.array([np.linspace(lo, hi, 101) for lo, hi in zip(lows, highs)])
        values = np.concatenate([rng.uniform(0, 200, 30000), table[rows[:3000], rng.integers(0, 101, 3000)]])
        rows = np.concatenate([rows, rows[:3000]])
        index = cal.bin_index(values, lows[rows], highs[rows], table[rows], 100)
        for row in range(0, 300, 7):
            mine = np.bincount(index[(rows == row) & (index >= 0)], minlength=100)
            numpy, _ = np.histogram(values[rows == row], bins=100, range=(lows[row], highs[row]))
            self.assertEqual(mine.tolist(), numpy.tolist(), row)

    def test_stand_in_values_give_the_raw_value_fit_exactly(self):
        from src.ldat_inspector.engine import fit_peak_background
        rng = np.random.default_rng(3)
        for values in (np.concatenate([rng.normal(90, 6, 900), rng.uniform(10, 200, 300)]),
                       rng.uniform(10, 200, 400), rng.normal(90, 6, 150), rng.normal(70, 4, 5000)):
            for model in ("linear", "constant"):
                interval = (0.55 * 90.0, 1.5 * 90.0)
                settings = dict(interval=interval, search=(72.0, 108.0), sigma0=6.3, mu_halfwidth=13.5)
                raw = fit_peak_background(values, background_model=model, bins=100, **settings)
                edges = np.linspace(*interval, 101)
                counts = np.bincount(cal.bin_index(values, *interval, edges, 100)[
                    cal.bin_index(values, *interval, edges, 100) >= 0], minlength=100)
                stand_in = cal.stand_in_values(counts, edges, len(values))
                self.assertEqual(np.histogram(stand_in, bins=100, range=interval)[0].tolist(),
                                 np.histogram(values, bins=100, range=interval)[0].tolist())
                fitted = fit_peak_background(stand_in, background_model=model, bins=100, **settings)
                self.assertEqual(raw.keys(), fitted.keys())
                for key in raw:
                    self.assertTrue(np.array_equal(np.asarray(raw[key]), np.asarray(fitted[key])), (key, model))

    # Reference parity (positions = 1) -----------------------------------------------------------------

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_per_slab_outputs_byte_identical_to_reference_script(self):
        result = self.ours(self.p1[DataFormat.COMPACT])
        self.assertEqual(cal.encal_text(result), golden("p1_per_slab.encal"))
        self.assertEqual(cal.status_text(result), golden("p1_per_slab_status.txt"))
        texts = set(result.statuses.values())
        for expected in ("fit", "borrowed from slab 1", "borrowed from slab 14"):
            self.assertIn(expected, texts)
        self.assertTrue(any(t.startswith("no fit: ") and t.endswith(" events (< 200)") for t in texts), texts)
        self.assertTrue(any(t.startswith("fit; check: higher-energy peak") for t in texts), texts)
        self.assertTrue(any(t.startswith("estimated from neighbour slabs [6, 8]") for t in texts), texts)
        self.assertTrue(any(t.startswith("estimated from minimodule median") for t in texts), texts)
        self.assertTrue(any(t.startswith("no fit: ") and "events" not in t for t in texts), texts)
        counts = result.status_counts()
        self.assertGreater(counts["no_values"], 0)
        rejected = {k: sum(f.rejected[k] for f in result.files) for k in cal.REJECTIONS}
        for reason in ("min_channels", "minimodule_channels", "unresolved_slab", "one_time_channel"):
            self.assertGreater(rejected[reason], 0, (reason, rejected))
        self.assertEqual(sum(f.accepted_sides for f in result.files), golden("p1_per_slab.json")["accepted_sides"])

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_passing_event_limit_matches_reference_semantics(self):
        passed = golden("p1_limit700.json")["events_passed"]
        encal, stat = golden("p1_limit700.encal"), golden("p1_limit700_status.txt")
        for batch in (64, 5000):
            result = self.ours(self.p1[DataFormat.COMPACT], event_limit=700, batch_records=batch)
            self.assertEqual([f.events_passed for f in result.files], passed, batch)
            self.assertEqual(passed, [701, 701])                # stops once more than 700 have passed
            self.assertTrue(all(f.stopped_at_limit for f in result.files))
            self.assertLess(result.files[0].records_read, result.files[0].records_validated)
            self.assertEqual((cal.encal_text(result), cal.status_text(result)), (encal, stat), batch)

    @pytest.mark.slow  # ~23 s
    def test_target_limit_counts_kept_sides(self):
        """FR-21 (amended 2026-10-05, T25.2): target mode S = K x P x T kept sides, ceil(S / n) per file; a file
        stops once its kept sides exceed the share. The result equals calibrating, as whole files, exactly the
        records each file read; one file stops at S; the library default stays the reference."""
        files = self.p1[DataFormat.COMPACT]
        keys = len(cal._mapped_keys(self.fx.mapping, 1))
        whole = min(f.accepted_sides for f in self.ours(files, event_limit=None).files)

        def plan(target, n=2, positions=1):
            return cal.event_limit_plan(self.fx.mapping, positions, n, limit_mode="target", target_per_key=target)
        target = max(t for t in range(1, 100_000) if plan(t)["limit_per_file"] <= whole // 2)
        share, total = plan(target)["limit_per_file"], plan(target)["limit_total"]
        self.assertEqual((total, share, plan(target)["limit_unit"]), (keys * target, -(-keys * target // 2), "kept sides"))
        self.assertEqual(plan(target, positions=3)["limit_total"], keys * 3 * target)
        result = self.ours(files, limit_mode="target", target_per_key=target)
        self.assertTrue(all(f.stopped_at_limit for f in result.files))
        for sample in result.files:                           # within one record past the share
            self.assertGreater(sample.accepted_sides, share)
            self.assertLessEqual(sample.accepted_sides, share + 2)
        halves = (self.p1_records[:len(self.p1_records) // 2], self.p1_records[len(self.p1_records) // 2:])
        prefixes = [self.fx.write(f"prefix_{os.getpid()}_{i}.ldat", part[:sample.records_read], DataFormat.COMPACT)
                    for i, (part, sample) in enumerate(zip(halves, result.files))]
        whole_prefix = self.ours(prefixes, event_limit=None)
        self.assertEqual((cal.encal_text(result), cal.status_text(result)),
                         (cal.encal_text(whole_prefix), cal.status_text(whole_prefix)))
        self.assertEqual([f.accepted_sides for f in result.files], [f.accepted_sides for f in whole_prefix.files])
        self.assertEqual((result.event_limit, result.limit_plan["limit_mode"], result.limit_plan["mapped_slab_keys"]),
                         (None, "target", keys))
        one = self.ours(files[:1], limit_mode="target", target_per_key=target)
        self.assertEqual(one.limit_plan["limit_per_file"], total)                          # one file: the total
        self.assertEqual(one.files[0].stopped_at_limit, one.files[0].accepted_sides > total)
        sampling = cal.sidecar(result, "0" * 64)["sampling"]
        self.assertEqual({k: sampling[k] for k in ("limit_mode", "limit_unit", "target_per_key", "mapped_slab_keys",
                                                   "limit_total", "limit_per_file", "passing_event_limit_per_file")},
                         {"limit_mode": "target", "limit_unit": "kept sides", "target_per_key": target,
                          "mapped_slab_keys": keys, "limit_total": total, "limit_per_file": share,
                          "passing_event_limit_per_file": None})
        default = self.ours(files)                                   # the library default stays the reference
        self.assertEqual((default.event_limit, default.limit_plan["limit_mode"], default.limit_plan["limit_unit"],
                          default.limit_plan["limit_total"]), (cal.EVENT_LIMIT, "reference", "passing events", None))
        for options in ({"limit_mode": "guess"}, {"limit_mode": "target"}, {"limit_mode": "target", "target_per_key": 0},
                        {"limit_mode": "target", "target_per_key": 1.5}, {"event_limit": 0}):
            with self.subTest(options=options), self.assertRaises(InputError):
                self.ours(files, **options)
        with self.assertRaises(InputError):
            cal.event_limit_plan(self.fx.mapping, 1, 0, limit_mode="target", target_per_key=3)

    @pytest.mark.slow  # ~26 s
    def test_one_decoding_equals_two_and_reads_each_file_once(self):
        """FR-15 (T25.3): target mode within the memory budget decodes each file once and gives the same files,
        samples and coverage as two decodings; above the budget, in reference mode or without one: twice."""
        readers = []
        real = cal._Reader

        class Counted(real):
            def __init__(self, descriptor, *args, **kwargs):
                readers.append(Path(descriptor.path).name)
                super().__init__(descriptor, *args, **kwargs)
        for positions, files in ((1, self.p1[DataFormat.COMPACT]), (3, self.p3[DataFormat.COMPACT])):
            whole = min(f.accepted_sides for f in self.ours(files, positions, event_limit=None).files)
            for target in (1, 1_000_000):          # binding share / larger than the files
                with self.subTest(positions=positions, target=target):
                    options = dict(limit_mode="target", target_per_key=target)
                    if target == 1:
                        options["target_per_key"] = max(t for t in range(1, 50_000) if cal.event_limit_plan(
                            self.fx.mapping, positions, len(files), **options | {"target_per_key": t}
                        )["limit_per_file"] <= whole // 2)
                    readers.clear()
                    with patch.object(cal, "_Reader", Counted):
                        once = self.ours(files, positions, memory_budget=2 ** 62, **options)   # the bound uses the limit
                        reads_once = list(readers)
                        readers.clear()
                        twice = self.ours(files, positions, **options)
                    self.assertEqual(reads_once, [Path(d.path).name for d in files])            # one reading each
                    self.assertEqual(readers, [Path(d.path).name for d in files] * 2)
                    self.assertEqual((cal.encal_text(once), cal.status_text(once)),
                                     (cal.encal_text(twice), cal.status_text(twice)))
                    self.assertEqual((once.files, once.coverage), (twice.files, twice.coverage))
                    self.assertEqual((once.decoding["decodings"], twice.decoding["decodings"]), (1, 2))
                    kept = once.decoding["pairs_kept_bytes"]
                    self.assertEqual(kept, sum(f.accepted_sides for f in once.files) * cal.PAIR_BYTES)
                    self.assertLessEqual(kept, once.decoding["pair_storage_bound_bytes"])
                    bound = once.decoding["pair_storage_bound_bytes"]
                    self.assertEqual(bound, len(files) * (once.limit_plan["limit_per_file"] + 2) * cal.PAIR_BYTES)
                    tight = self.ours(files, positions, memory_budget=bound - 1, **options)
                    self.assertEqual((tight.decoding["decodings"], cal.encal_text(tight)), (2, cal.encal_text(once)))
                    self.assertEqual(self.ours(files, positions, memory_budget=bound, **options).decoding["decodings"], 1)
        reference = self.ours(self.p1[DataFormat.COMPACT], memory_budget=10 ** 9)
        self.assertEqual((reference.decoding["decodings"], reference.decoding["pairs_kept_bytes"]), (2, 0))
        self.assertEqual(cal.sidecar(once, "0" * 64)["sampling"]["decoding"], once.decoding)
        for budget in (0, -1, 1.5):
            with self.subTest(budget=budget), self.assertRaises(InputError):
                self.ours(self.p1[DataFormat.COMPACT], limit_mode="target", target_per_key=5, memory_budget=budget)
        plan = {"limit_mode": "target", "limit_per_file": 10, "files": 3}
        self.assertEqual((cal.pair_storage_bound(plan, Population.GROUP),
                          cal.pair_storage_bound(plan, Population.COINCIDENCE),
                          cal.pair_storage_bound(plan | {"limit_mode": "reference"}, Population.COINCIDENCE),
                          cal.pair_storage_bound(plan | {"limit_per_file": None}, Population.COINCIDENCE)),
                         (3 * 11 * 16, 3 * 12 * 16, 2 * 3 * 11 * 16, None))

    def test_one_decoding_cancellation_during_replay_publishes_nothing(self):
        """FR-15 (T25.3): STOP after reading, during the kept-pair accumulation, is a cancellation."""
        files, calls = self.p1[DataFormat.COMPACT], [0]
        options = dict(limit_mode="target", target_per_key=1_000_000, memory_budget=2 ** 62)

        def counting():
            calls[0] += 1
            return False
        self.assertEqual(self.ours(files, cancelled=counting, **options).decoding["decodings"], 1)
        total, seen = calls[0], [0]

        def at_the_last_check():                   # the last check of a run is in the replay loop
            seen[0] += 1
            return seen[0] >= total
        with self.assertRaises(cal.CalibrationCancelled):
            self.ours(files, cancelled=at_the_last_check, **options)
        self.assertEqual(seen[0], total)

    def comparable(self, result):
        """Everything a calibration publishes except the recorded worker count."""
        sidecar = cal.sidecar(result, "0" * 64)
        sidecar["sampling"].pop("workers")
        return cal.encal_text(result), cal.status_text(result), sidecar, result.files, result.coverage

    @pytest.mark.slow  # ~37 s
    def test_workers_give_identical_outputs(self):
        """FR-15 (T25.4): 2 and 3 worker processes give the same .encal, status and sidecar as one, in reference
        mode (binding per-file limit), in target mode read once and read twice, for P = 1 and P = 3."""
        p1, p3 = self.p1[DataFormat.COMPACT], self.p3[DataFormat.COMPACT]
        whole = min(f.accepted_sides for f in self.ours(p1, event_limit=None).files)
        target = max(t for t in range(1, 100_000) if cal.event_limit_plan(
            self.fx.mapping, 1, 2, limit_mode="target", target_per_key=t)["limit_per_file"] <= whole // 2)
        cases = {"reference 700": (p1, 1, dict(event_limit=700)),
                 "target once": (p1, 1, dict(limit_mode="target", target_per_key=target, memory_budget=2 ** 62)),
                 "target twice": (p1, 1, dict(limit_mode="target", target_per_key=target)),
                 "P = 3 whole files": (p3, 3, dict(event_limit=None)),
                 "P = 3 target once": (p3, 3, dict(limit_mode="target", target_per_key=1_000, memory_budget=2 ** 62))}
        for name, (files, positions, options) in cases.items():
            with self.subTest(name):
                serial = self.ours(files, positions, **options)
                for workers in (2, 3):
                    parallel = self.ours(files, positions, workers=workers, **options)
                    self.assertEqual(self.comparable(parallel), self.comparable(serial), workers)
                    self.assertEqual((parallel.workers, cal.sidecar(parallel, "0" * 64)["sampling"]["workers"]),
                                     (workers, workers))
                    self.assertEqual(parallel.decoding["decodings"], serial.decoding["decodings"])
        for workers in (0, -1, 1.5):
            with self.subTest(workers=workers), self.assertRaises(InputError):
                self.ours(p1, workers=workers)

    @pytest.mark.slow  # ~5 s
    @pytest.mark.fr("003-FR-1")  # T25.4
    def test_parallel_progress_cancellation_and_worker_errors(self):
        """FR-1/FR-15 (T25.4): progress phases read, pass 2 and fits; a STOP reaches the workers; a worker's
        validation error is the stage's error and nothing is published."""
        files = self.p1[DataFormat.COMPACT]
        for workers in (1, 2):
            with self.subTest(workers=workers):
                seen = []
                self.ours(files, workers=workers, progress=lambda path, records, **extra: seen.append(
                    (extra["phase"], path, records, extra.get("keys_done"), extra.get("keys_total"))))
                phases = [item[0] for item in seen]
                self.assertEqual(sorted(set(phases), key=phases.index), ["read", "pass 2", "fits"])
                self.assertTrue({Path(d.path) for d in files} <= {Path(item[1]) for item in seen
                                                                   if item[0] == "pass 2"})
                fits = [item for item in seen if item[0] == "fits"]
                done = [item[3] for item in fits]
                self.assertEqual(done, sorted(done))
                self.assertEqual((done[-1], fits[-1][4]), (fits[-1][4], fits[-1][4]))   # ends at keys_total
        calls = []
        with self.assertRaises(cal.CalibrationCancelled):
            self.ours(files, workers=2, cancelled=lambda: calls.append(1) or len(calls) > 2)
        bad = self.root / "unmapped_in_list.ldat"
        good = self.p1_records[:30]
        encode(bad, [tuple([[(1, 1.0, 999999)] * 5, good[0][1]])] + good, DataFormat.COMPACT)
        listed = [files[0], InputDescriptor(bad, DataFormat.COMPACT, Population.COINCIDENCE), files[1]]
        for workers in (1, 2):
            with self.subTest(workers=workers), self.assertRaisesRegex(InputError, "unmapped channel 999999"):
                self.ours(listed, workers=workers)

    @pytest.mark.fr("003-FR-1")  # T25.4
    def test_worker_killed_by_the_stop_is_a_cancellation(self):
        """T25.4 (Cornell 2026-10-06, STOP during the fits): STOP sends SIGTERM to the whole process group, so
        the workers die while the parent marks itself cancelled; a dead worker seen after the cancellation is
        the cancellation, not a BrokenProcessPool failure. Without a STOP the dead worker stays an error."""
        from concurrent.futures.process import BrokenProcessPool
        from src.cornell.parallel import OrderedPool, PoolCancelled
        marker = self.root / "worker_died"
        tasks = [(str(marker),)] * 2
        with self.assertRaises(PoolCancelled):
            with OrderedPool(2, cal._plain_init, (), marker.exists) as pool:
                pool.run(die_after_marker, tasks, lambda index, value: None)
        marker.unlink()
        with self.assertRaises(BrokenProcessPool):
            with OrderedPool(2, cal._plain_init, (), lambda: False) as pool:
                pool.run(die_after_marker, tasks, lambda index, value: None)

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_coverage_reports_sides_per_key(self):
        """FR-21 (T25.2): sides each key received, over keys with sides, equal the reference's per-key values."""
        self.assertEqual(cal.key_coverage([0, 5, 300, 1000, 250], 400),
                         {"keys_with_sides": 4, "min_sides": 5, "median_sides": 275.0, "keys_below_target": 3,
                          "keys_below_fit_minimum": 1, "fit_minimum": cal.MIN_EVENTS, "target_per_key": 400})
        self.assertEqual(cal.key_coverage([], None)["keys_with_sides"], 0)
        self.assertIsNone(cal.key_coverage([3], None)["keys_below_target"])
        sides = [count for _, count in golden("p1_limit700.json")["sides_per_key"]]
        result = self.ours(self.p1[DataFormat.COMPACT], event_limit=700, target_per_key=250)
        self.assertEqual(result.coverage, cal.key_coverage(sides, 250))
        self.assertGreater(result.coverage["keys_with_sides"], 0)
        self.assertEqual(cal.sidecar(result, "0" * 64)["coverage"], result.coverage)

    # Positions >= 2 -----------------------------------------------------------------------------------

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_position_calibration_matches_per_event_oracle(self):
        encal, stat = golden("p3_positions3.encal"), golden("p3_positions3_status.txt")
        result = self.ours(self.p3[DataFormat.COMPACT], positions=3)
        self.assertEqual(cal.encal_text(result), encal)
        self.assertEqual(cal.status_text(result), stat)
        regions = {key[2] for key, (mu, _) in result.factors.items() if result.statuses[key] == "fit"}
        self.assertEqual(regions, {0, 1, 2})
        rejected = {k: sum(f.rejected[k] for f in result.files) for k in cal.REJECTIONS}
        self.assertGreater(rejected["missing_cog_limits"], 0)    # slab 7 of (7, 1) has no limits
        self.assertGreater(rejected["cog_out_of_range"], 0)
        self.assertTrue(encal.startswith("# Position-dependent energy calibration (3 regions per slab)\n"))
        with self.assertRaises(InputError):
            cal.calibrate(self.p3[DataFormat.COMPACT], self.fx.config, None, positions=3)

    # Fixed == compact -----------------------------------------------------------------------------------

    def test_fixed_and_compact_encodings_give_identical_outputs(self):
        for positions, files in ((1, self.p1), (5, self.p3), (3, self.p3)):
            compact = self.ours(files[DataFormat.COMPACT], positions=positions)
            for batch in (37, 5000):
                fixed = self.ours(files[DataFormat.FIXED], positions=positions, batch_records=batch)
                self.assertEqual(cal.encal_text(fixed), cal.encal_text(compact), (positions, batch))
                self.assertEqual(cal.status_text(fixed), cal.status_text(compact), (positions, batch))
                self.assertEqual([f.rejected for f in fixed.files], [f.rejected for f in compact.files])
            self.assertEqual(fixed.data_format, DataFormat.FIXED)

    def test_fixed_group_calibrates_each_group_as_one_side(self):
        sides = sides_for(self.fx.geometry, P1_SPECS, seed=11)
        group = self.fx.write_group(f"group_{os.getpid()}.ldat", sides)
        result = self.ours([group])
        self.assertEqual(result.population, Population.GROUP)
        self.assertGreater(len(result.factors), 10)
        fitted = sum(1 for t in result.statuses.values() if t == "fit")
        self.assertGreater(fitted, 10)
        with self.assertRaises(InputError):
            self.ours([group, self.p1[DataFormat.FIXED][0]])      # no mixing

    # Bounds, cancellation, validation -----------------------------------------------------------------

    def test_storage_bounded_by_keys_not_events(self):
        sizes = []
        for repeat in (1, 4):
            records = self.p1_records * repeat
            descriptor = self.fx.write(f"repeat_{repeat}.ldat", records, DataFormat.COMPACT)
            accumulator = cal._Accumulator()
            context = cal._Context(self.fx.config, None, 1)
            cal._sample(descriptor, context, accumulator.first, None, 500, None, None, validate=True)
            accumulator.prepare_fit()
            sizes.append((len(accumulator.codes), accumulator.nbytes, int(accumulator.counts.sum())))
        self.assertEqual(sizes[0][:2], sizes[1][:2])
        self.assertEqual(sizes[1][2], 4 * sizes[0][2])

    def test_cancel_and_changed_input_fail(self):
        calls = []
        with self.assertRaises(cal.CalibrationCancelled):
            self.ours(self.p1[DataFormat.COMPACT], cancelled=lambda: calls.append(1) or len(calls) > 3)
        copy = self.root / "changing.ldat"
        copy.write_bytes(self.p1[DataFormat.COMPACT][0].path.read_bytes())
        descriptor = InputDescriptor(copy, DataFormat.COMPACT, Population.COINCIDENCE)
        state = {"done": False}

        def mutate(path, records, **extra):
            if not state["done"]:
                state["done"] = True
                with open(copy, "ab") as out:
                    out.write(b"")
                os.utime(copy, ns=(time.time_ns(), time.time_ns() + 10_000_000_000))
        with self.assertRaises(InputError):
            self.ours([descriptor], progress=mutate)

    def test_fused_validation_rejects_what_t5_rejects(self):
        good = self.p1_records[:20]
        cases = {}
        compact = bytearray()
        for record in good:
            compact += bytes(len(s) for s in record) + b"".join(HIT.pack(*h) for s in record for h in s)
        cases["truncated compact"] = (bytes(compact[:-3]), DataFormat.COMPACT)
        cases["zero count"] = (b"\x00\x02" + bytes(compact[2:]), DataFormat.COMPACT)
        bad = [[(1, 1.0, 999999)] * 5, good[0][1]]
        cases["unmapped"] = (None, DataFormat.COMPACT, [tuple(bad)] + good)
        nan_side = [(1, float("nan"), good[0][0][0][2])] + list(good[0][0][1:])
        cases["nonfinite"] = (None, DataFormat.COMPACT, [(nan_side, good[0][1])] + good)
        fixed = io.BytesIO()
        fixed.write(struct.pack("<i", 300))
        cases["fixed limit"] = (fixed.getvalue() + b"\0" * 64, DataFormat.FIXED)
        cases["empty"] = (b"", DataFormat.COMPACT)
        for name, case in cases.items():
            path = self.root / f"{name.replace(' ', '_')}.ldat"
            if case[0] is None:
                encode(path, case[2], case[1])
            else:
                path.write_bytes(case[0])
            descriptor = InputDescriptor(path, case[1], Population.COINCIDENCE)
            with self.subTest(name), self.assertRaises(InputError):
                self.ours([descriptor])
            with self.subTest(f"T5 {name}"), self.assertRaises(InputError):
                validate_ldat(descriptor, self.fx.mapping.modules)
        fixed_bad = self.root / "fixed_remainder.ldat"
        fixed_bad.write_bytes(self.p1[DataFormat.FIXED][0].path.read_bytes()[:-7])
        with self.assertRaises(InputError):
            self.ours([InputDescriptor(fixed_bad, DataFormat.FIXED, Population.COINCIDENCE)])

    def test_side_without_time_channel_is_rejected_not_a_crash(self):
        """The reference would raise IndexError; on summed Cornell maps the filters reject it first."""
        rng = np.random.default_rng(9)
        geometry = self.fx.geometry
        records = [(geometry.side(rng, A, 4, 90.0, kind="notime"), geometry.side(rng, A, 5, 90.0)) for _ in range(30)]
        descriptor = self.fx.write(f"notime_{os.getpid()}.ldat", records, DataFormat.COMPACT)
        result = self.ours([descriptor])
        self.assertEqual(result.files[0].rejected["min_channels"], 30)   # no time hit: n_energy == n_hits
        self.assertEqual(result.files[0].accepted_sides, 0)

    # Outputs and consumers -----------------------------------------------------------------------------

    def test_outputs_read_by_kevconverter_loader_and_never_overwrite(self):
        for positions, files in ((1, self.p1), (3, self.p3)):
            result = self.ours(files[DataFormat.COMPACT], positions=positions)
            encal, side, status = (self.root / f"p{positions}.encal", self.root / f"p{positions}.encal.json",
                                   self.root / f"p{positions}_status.txt")
            digest = cal.write_calibration(result, encal, side, status)
            loaded = load_calibration(encal, self.fx.mapping, expected_regions=positions, metadata_path=side)
            self.assertEqual(loaded.sha256, digest)
            self.assertEqual(loaded.layout, "per_slab" if positions == 1 else "position")
            self.assertEqual(len(loaded.values), len(result.factors))           # 0\t0 rows are no factor
            self.assertEqual(loaded.unfitted, len(result.keys) - len(result.factors))
            converter = KevConverter(str(encal), "cornell" if positions == 1 else "cornell_position")
            for key, (mu, _) in result.factors.items():
                self.assertEqual(converter.kev_factors[key], round(mu, 3))
                self.assertEqual(loaded.values[key][0], round(mu, 3))
            metadata = json.loads(side.read_text(encoding="utf-8"))
            self.assertEqual((metadata["layout"], metadata["num_regions"]), (loaded.layout, positions))
            self.assertEqual(status.read_text(encoding="utf-8"), cal.status_text(result))
            for target in (encal, side, status):
                before = target.read_bytes()
                with self.assertRaises(FileExistsError):
                    cal.write_calibration(result, *(target if t == target else self.root / f"new_{t.name}"
                                                    for t in (encal, side, status)))
                self.assertEqual(target.read_bytes(), before)
            plot = self.root / f"p{positions}.png"
            cal.plot_summary(result, plot)
            self.assertGreater(plot.stat().st_size, 1000)

    def test_listmode_accepts_per_slab_calibration_as_one_factor_per_slab(self):
        """A per-slab .encal gives exactly the LM of the equivalent one-region position calibration."""
        from src.cornell.inputs import calibration_layout
        helper = ListmodeFixtures.at(self.root / "lm")
        descriptors, maps = helper.inputs(count=600, random_slabs=False)
        one_region = helper.calibration(regions=1, name="position_1regions.encal")
        rows = ["ID(t_ch, slab)\tmu\tsigma\n"]
        for (channel, slab, _), (mu, sigma) in sorted(one_region.values.items()):
            rows.append(f"{(channel, slab)}\t{mu}\t{sigma}\n")
        unused = next(k for k in helper.mapping.types if ChannelType.TIME in helper.mapping.types[k]
                      and all(key[0] != k for key in one_region.values))
        position = int(helper.mapping.local[unused][2])
        rows.append(f"{(unused, 2 * position)}\t0\t0\n")          # reference "no factor" row
        path = helper.root / "per_slab.encal"
        path.write_text("".join(rows), encoding="utf-8")
        per_slab = load_calibration(path, helper.mapping, expected_regions=1)
        self.assertEqual((per_slab.layout, per_slab.unfitted, len(per_slab.values)), ("per_slab", 1, len(one_region.values)))
        self.assertEqual(calibration_layout(path), ("per_slab", 1))
        self.assertEqual(calibration_layout(one_region.path), ("position", 1))
        a = helper.generate(descriptors, {**maps, "calibration": one_region}, helper.lm_parent / "one region")
        b = helper.generate(descriptors, {**maps, "calibration": per_slab}, helper.lm_parent / "per slab")
        self.assertGreater(a.records, 0)
        self.assertEqual(a.records, b.records)
        self.assertEqual(a.output.read_bytes(), b.output.read_bytes())
        from src.cornell import listmode as lm
        job = (helper.lm_parent / "per slab" / lm.JOB_FILE).read_text(encoding="utf-8")
        self.assertIn('"layout": "per_slab"', job)                 # provenance names the calibration layout
        five = helper.generate(descriptors, maps, helper.lm_parent / "five regions")   # position LM unchanged
        self.assertGreater(five.records, 0)

    def test_tracked_module_never_imports_local_scripts(self):
        tree = ast.parse((REPO / "src/cornell/calibration.py").read_text(encoding="utf-8"))
        names = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        names |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(any(name.startswith(("scripts", "scripts_cornell")) for name in names), names)


@pytest.mark.real_data
@pytest.mark.fr("003-FR-21")  # spec 003 T20
def test_owner_reference_encal_loads(real_cal_file):
    """The owner's reference-written per-slab file (zeros included) loads with the January map."""
    mapping = load_channel_map(REPO / "maps/cornell_map_full_system_old.yaml", processing_root=REPO)
    owner = load_calibration(real_cal_file(OWNER_ENCAL), mapping)
    assert (owner.layout, owner.num_regions) == ("per_slab", 1)
    assert len(owner.values) > 1000
