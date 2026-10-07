"""Listmode: binary contract, reference parity, rejections, debug, resume, compact input (spec 003 T9, T21, T25.5, T29.2; spec 007 T10).

Moved from scripts/petsys_manager_listmode_check.py. The reference outputs
(scripts_cornell/cornell_listmode_cog_fixed_position.py) are the golden files under tests/data/golden/listmode/
(T8); no test loads the reference.
"""

import ast
import ctypes
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct
import unittest

import numpy as np
import pytest

from helpers import DATA, REPO
from manager_helpers import METADATA, PAIRS, ListmodeFixtures, PrivateOutput, encode_compact, encode_fixed
from src.cornell import listmode as lm
from src.cornell.inputs import InputError, load_calibration, load_limits
from src.petsys_manager.contracts import InputDescriptor
from src.petsys_manager.settings import LMMetadata, ProfileError
from src.read_fixed import read_fixed_file_numpy

GOLDEN = DATA / "golden" / "listmode"
RECORD = struct.Struct("<fHHfbbbbbbHh2x")      # independent CoincidenceV5 wire layout
HEADER_OFFSETS = {                             # independent LMHeader wire layout (T1)
    "identifier": (0, "16s"), "rawCounts": (16, "d"), "acqTime": (24, "d"), "activity": (32, "d"),
    "isotope": (40, "16s"), "detectorSizeX": (56, "d"), "detectorSizeY": (64, "d"), "startTime": (72, "d"),
    "measurementTime": (80, "d"), "moduleNumber": (88, "i"), "ringNumber": (92, "i"), "ringDistance": (96, "d"),
    "detectorDistance": (104, "d"), "isotopeHalfLife": (112, "d"), "weight": (120, "f"), "maxTemp": (124, "f"),
    "percentLoss": (128, "f"), "detectorPixelSizeX": (132, "f"), "detectorPixelSizeY": (136, "f"),
    "reserved": (140, "3f"), "version": (152, "2b"), "breast": (154, "c"), "unused_1": (155, "c"),
    "gatePeriod": (156, "d"), "DOILayer": (164, "h"), "method": (166, "h"), "StudyId": (168, "h"),
    "detectorPixelsX": (170, "b"), "detectorPixelsY": (171, "b"), "unused_2": (172, "4s"),
}
RECORD_OFFSETS = {"time": 0, "energy1": 4, "energy2": 6, "amount": 8, "xPosition1": 12, "yPosition1": 13,
                  "zPosition1": 14, "xPosition2": 15, "yPosition2": 16, "zPosition2": 17, "pair": 18, "dt": 20}
LEGACY = LMMetadata("Na22", 10.0, 10.0, 51.61, 51.61, 120, 5, 820.0, 100, 100, "ps")


def decode_header(content):
    values = {}
    for name, (offset, fmt) in HEADER_OFFSETS.items():
        value = struct.unpack_from("<" + fmt, content, offset)
        values[name] = value if len(value) > 1 else value[0]
    return values


def golden(name):
    """A reference output frozen at T8 (FR-3): bytes, or parsed JSON."""
    content = (GOLDEN / name).read_bytes()
    return json.loads(content.decode("utf-8")) if name.endswith(".json") else content


def values(stored):
    """A golden value list ``{"dtype", "values"}`` as an array."""
    return np.array(stored["values"], dtype=stored["dtype"])


def dense(stored):
    """A golden sparse array ``{"shape", "dtype", "fill", "entries"}`` as a dense array."""
    array = np.full(stored["shape"], np.nan if stored["fill"] == "nan" else stored["fill"], dtype=stored["dtype"])
    for index, value in stored["entries"]:
        array[tuple(index)] = value
    return array


@pytest.mark.fr("003-FR-2", "003-FR-9", "003-FR-12", "003-FR-15", "003-FR-16")  # spec 003 T9
class ListmodeChecks(ListmodeFixtures, PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-listmode-"

    # Binary contract -----------------------------------------------------------

    def test_listmode_structures_and_every_offset(self):
        self.assertEqual((ctypes.sizeof(lm.LMHeader), lm.RECORD_DTYPE.itemsize, RECORD.size), (176, 24, 24))
        self.assertEqual({n: getattr(lm.LMHeader, n).offset for n, *_ in lm.LMHeader._fields_},
                         {n: offset for n, (offset, _) in HEADER_OFFSETS.items()})
        self.assertEqual({n: lm.RECORD_DTYPE.fields[n][1] for n in RECORD_OFFSETS}, RECORD_OFFSETS)

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_listmode_legacy_metadata_header_bytes_equal_reference(self):
        self.assertEqual(lm.header_bytes(LEGACY), golden("legacy_header.bin"))

    def test_listmode_supplied_header_fields_decode_independently(self):
        content = lm.header_bytes(METADATA)
        values = decode_header(content)
        legacy = decode_header(lm.header_bytes(LEGACY))
        expected = {"acqTime": 300.5, "measurementTime": 299.0, "detectorSizeX": 102.4, "detectorSizeY": 96.0,
                    "moduleNumber": 30, "ringNumber": 2, "ringDistance": 410.5, "detectorPixelsX": 90,
                    "detectorPixelsY": 96, "startTime": 0.0, "version": (9, 5)}
        for name, value in expected.items():
            self.assertEqual(values[name], value, name)
        self.assertEqual(values["identifier"].rstrip(b"\0"), b"Cornell")
        self.assertEqual(values["isotope"].rstrip(b"\0"), b"F18")
        self.assertEqual(values["detectorPixelSizeX"], np.float32(np.diff(np.linspace(0, 102.4, 91))[0]))
        self.assertEqual(values["detectorPixelSizeY"], np.float32(1.0))
        for name in lm.ZERO_HEADER_FIELDS + ("unused_1", "unused_2"):
            self.assertFalse(np.any(np.frombuffer(struct.pack("<" + HEADER_OFFSETS[name][1],
                             *(values[name] if isinstance(values[name], tuple) else (values[name],))), np.uint8)), name)
        changed = {name for name in HEADER_OFFSETS if values[name] != legacy[name]}
        self.assertEqual(changed, set(lm.SUPPLIED_HEADER_FIELDS))   # only supplied fields differ from legacy
        self.assertEqual(len(content), 176)

    def test_listmode_missing_or_overflowing_metadata_rejected(self):
        self.assertEqual(len(LMMetadata().missing()), 11)            # no 10 s / module-count defaults
        descriptors, maps = self.inputs(count=40)
        for name in LMMetadata.__dataclass_fields__:
            destination = self.lm_parent / f"missing-{name}"
            with self.assertRaises(InputError):
                self.generate(descriptors, {**maps, "metadata": replace(METADATA, **{name: None})}, destination)
            self.assertFalse(destination.exists(), name)
        for bad in ({"isotope": "x" * 17}, {"detector_pixels_x": 128}, {"module_number": 2 ** 31},
                    {"acquisition_time_s": float("nan")}, {"detector_size_x_mm": -1.0}, {"detector_pixels_y": 0}):
            with self.assertRaises(ProfileError):
                replace(METADATA, **bad)
        with self.assertRaises(InputError):
            lm.header_bytes(replace(METADATA, detector_size_x_mm=1e300))    # float32 pixel-size overflow
        destination = self.lm_parent / "too-high"
        with self.assertRaises(InputError):
            self.generate(descriptors, maps, destination, config=self.processing({"en_min_ch": 0.2,
                          "energy_range": [357, 70000]}, "high.yaml"))
        self.assertFalse(destination.exists())

    # Numerical parity ----------------------------------------------------------

    @pytest.mark.slow  # ~8 s
    @pytest.mark.fr("003-FR-22", "007-FR-3")  # T25.5; golden files
    def test_listmode_parallel_files_with_per_file_seeds(self):
        """FR-15/FR-22 (T25.5): with an LM seed each file draws from its own stream, so 1, 2 and 3 worker
        processes and repeated runs give the same .lm; it equals the reference loop seeded the same way per file;
        another seed changes only slab draws; the seed is in the job (resume) and the provenance."""
        descriptors, maps = self.inputs(names=("par_coinc_1.ldat", "par_coinc_2.ldat", "par_coinc_3.ldat"),
                                        count=1200)
        base = self.generate(descriptors, maps, self.lm_parent / "seed5 w1", lm_seed=5)
        content = base.output.read_bytes()
        for workers in (2, 3):
            result = self.generate(descriptors, maps, self.lm_parent / f"seed5 w{workers}", lm_seed=5, workers=workers)
            self.assertEqual(result.output.read_bytes(), content, workers)
            for name in ("rejected", "observations", "slab_flags"):
                self.assertEqual(result.totals(name), base.totals(name), (workers, name))
            self.assertEqual([f.records_written for f in result.files], [f.records_written for f in base.files])
        again = self.generate(descriptors, maps, self.lm_parent / "seed5 again", lm_seed=5, seed=99)   # global seed
        self.assertEqual(again.output.read_bytes(), content)                                        # does not matter
        seeds = [lm.file_seed(5, index) for index in range(len(descriptors))]
        self.assertEqual(content[176:], golden("per_file_seeds.records.bin"))
        self.assertEqual([f.records_written for f in base.files], golden("per_file_seeds.json")["records_written"])
        other = self.generate(descriptors, maps, self.lm_parent / "seed6", lm_seed=6)
        self.assertNotEqual(other.output.read_bytes(), content)
        self.assertEqual(other.totals("rejected")["min_channels"], base.totals("rejected")["min_channels"])
        self.assertEqual([f.records_read for f in other.files], [f.records_read for f in base.files])
        provenance = json.loads(base.sidecar.read_text(encoding="utf-8"))
        self.assertEqual(provenance["random_streams"], {"lm_seed": 5, "file_seeds": seeds})
        unseeded = json.loads(self.generate(descriptors, maps, self.lm_parent / "unseeded").sidecar.read_text(
            encoding="utf-8"))
        self.assertIsNone(unseeded["random_streams"]["lm_seed"])
        self.assertEqual(len(set(seeds)), len(seeds))
        for options in ({"workers": 2}, {"lm_seed": -1}, {"lm_seed": 1.5}, {"workers": 0, "lm_seed": 5}):
            with self.subTest(options=options), self.assertRaises(InputError):
                self.generate(descriptors, maps, self.lm_parent / f"refused {len(options)}{sorted(options)[0]}",
                              **options)
        with self.assertRaisesRegex(InputError, "Resume refused"):
            self.generate(descriptors, maps, self.lm_parent / "seed5 w1", lm_seed=6, resume=True)
        calls = []
        with self.assertRaises(lm.ListmodeCancelled):
            self.generate(descriptors, maps, self.lm_parent / "cancel w2", lm_seed=5, workers=2,
                          cancelled=lambda: calls.append(1) or len(calls) > 2)

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_listmode_records_counts_and_merge_order_match_reference(self):
        descriptors, maps = self.inputs()
        result = self.generate(descriptors, maps)
        expected, written = golden("inputs_2500.records.bin"), golden("inputs_2500.json")["records_written"]
        content = result.output.read_bytes()
        self.assertEqual(content[:176], lm.header_bytes(METADATA))
        self.assertEqual(content[176:], expected)                     # byte-identical records, merged in order
        self.assertEqual(result.records, sum(written))
        self.assertEqual([f.records_written for f in result.files], written)
        self.assertEqual(result.output.name, "acq_coinc_all.lm")
        self.assertEqual(hashlib.sha256(content).hexdigest(), result.sha256)
        self.assertEqual(len(content), 176 + 24 * result.records)
        self.assertGreater(result.records, 300)
        for item in result.files:
            self.assertEqual(item.records_read, item.records_written + sum(item.rejected.values()))
            self.assertEqual(item.records_read, 2500)
        rejected = result.totals("rejected")
        for reason in set(lm.REJECTIONS) - {"missing_timestamp"}:
            self.assertGreater(rejected[reason], 0, reason)
        self.assertGreater(result.totals("observations")["dt_wrapped_int16"], 0)
        flags = result.totals("slab_flags")
        self.assertTrue(set(flags) <= {0, 1, 2, 3} and sum(flags.values()) > 0)
        # Independent decode of every record.
        times = {float(np.float32(1_000_000 + 997 * i + d)) for i in range(2500) for d in range(-300, 300)}
        pair_ids = set(PAIRS.values())
        for offset in range(176, len(content), 24):
            t, e1, e2, amount, x1, y1, z1, x2, y2, z2, pair, dt = RECORD.unpack_from(content, offset)
            self.assertEqual(amount, 1.0)
            self.assertIn(pair, pair_ids)
            self.assertTrue(357 <= e1 <= 665 and 357 <= e2 <= 665, (e1, e2))
            self.assertTrue(0 <= z1 <= 20 and 0 <= z2 <= 20)
            self.assertTrue(-1 <= x1 <= 90 and -1 <= y1 <= 96)
            self.assertTrue(t in times or t >= 1_000_000)

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_listmode_non_positive_mu_is_no_factor_like_the_reference(self):
        """FR-12 (owner decision 2026-10-02): failed legacy fits (mu <= 0) reject their pairs, as the reference does."""
        descriptors, maps = self.inputs()
        baseline = self.generate(descriptors, maps, self.lm_parent / "baseline")
        path = maps["calibration"].path
        bad = [f"{(ch, slab, region)}\t" for ch, slab in self.slab_keys((7, 0), 2) for region in (1, 2)]
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines = [line.split("\t")[0] + ("\t-2878.171\t3818.350\n" if line.startswith(tuple(bad[:3])) else
                                        "\t0\t1\n" if line.startswith(bad[3]) else "\t" + line.split("\t", 1)[1])
                 if "\t" in line and not line.startswith("ID") else line for line in lines]
        path.write_text("".join(lines), encoding="utf-8")
        calibration = load_calibration(path, self.mapping, expected_regions=5)
        self.assertEqual(len(calibration.non_positive), 4)
        maps = {**maps, "calibration": calibration}
        result = self.generate(descriptors, maps, self.lm_parent / "non_positive")
        expected, written = golden("non_positive_mu.records.bin"), golden("non_positive_mu.json")["records_written"]
        self.assertEqual(result.output.read_bytes()[176:], expected)
        self.assertEqual([f.records_written for f in result.files], written)
        self.assertGreater(result.totals("rejected")["missing_calibration"],
                           baseline.totals("rejected")["missing_calibration"])
        self.assertLess(result.records, baseline.records)
        provenance = json.loads(result.sidecar.read_text(encoding="utf-8"))
        recorded = provenance["sources"]["calibration"]["non_positive_mu_as_no_factor"]
        self.assertEqual(recorded, {"count": 4, "keys": [list(k) for k in calibration.non_positive]})

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_listmode_zero_width_limits_fall_out_of_range_like_the_reference(self):
        """FR-12 (owner decision 2026-10-02): a left = right limits row is used with the reference formulas."""
        descriptors, maps = self.inputs()
        baseline = self.generate(descriptors, maps, self.lm_parent / "baseline")
        zero = self.slab_keys((7, 0), 2)[0]
        for name, kind in (("doi_limits", "doi"), ("cog_limits", "cog")):
            path = maps[name].path
            rows = path.read_text(encoding="utf-8").splitlines(keepends=True)
            rows = [f"{zero}\t3.2\t3.2\n" if row.startswith(f"{zero}\t") else row for row in rows]
            path.write_text("".join(rows), encoding="utf-8")
            limits = load_limits(path, self.mapping, kind=kind)
            self.assertEqual(limits.zero_width, (zero,))
            changed = {**maps, name: limits}
            result = self.generate(descriptors, changed, self.lm_parent / f"zero_{kind}")
            expected = golden(f"zero_width_{kind}.records.bin")
            written = golden(f"zero_width_{kind}.json")["records_written"]
            self.assertEqual(result.output.read_bytes()[176:], expected, kind)
            self.assertEqual([f.records_written for f in result.files], written)
            self.assertLess(result.records, baseline.records, kind)
            provenance = json.loads(result.sidecar.read_text(encoding="utf-8"))
            self.assertEqual(provenance["sources"][name]["zero_width_keys"], [list(zero)])
            path.write_text("".join(row for row in rows if not row.startswith(f"{zero}\t")) + f"{zero}\t3.3\t3.2\n",
                            encoding="utf-8")
            with self.assertRaises(InputError):          # reversed still refuses the file
                load_limits(path, self.mapping, kind=kind)
            maps = changed

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_listmode_controlled_pair_orientation_energy_and_timestamp(self):
        _, maps = self.inputs(count=40)
        g = self.geometry
        rng = np.random.default_rng(1)
        a = g.side(rng, (7, 0), 2, +1, 100.0, timestamp=5000)   # region 0, mu(region) = 100 * (1 + 0.01 r)
        b = g.side(rng, (21, 9), 4, +1, 85.0, timestamp=5123)   # region 12
        descriptor = self.ldat("ctrl_coinc_0.ldat", [(a, b), (b, a)])
        result = self.generate([descriptor], maps, self.lm_parent / "ctrl")
        content = result.output.read_bytes()
        expected = golden("controlled_pair.records.bin")
        self.assertEqual(content[176:], expected)
        rows = [RECORD.unpack_from(content, 176 + 24 * i) for i in range(result.records)]
        self.assertEqual(len(rows), 2)
        first, second = rows
        self.assertEqual((first[0], second[0]), (5000.0, 5000.0))          # min raw timestamp, float32
        self.assertEqual((first[10], second[10]), (2, 2))                    # (0, 12) -> pair 2 either way
        self.assertEqual((first[11], second[11]), (-123, -123))              # dt follows pair orientation
        self.assertEqual(first[1:3], second[1:3])                            # energies ordered by the pair
        self.assertEqual(first[4:10], second[4:10])

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_listmode_loaders_and_calibration_array_match_reference_parsers(self):
        descriptors, maps = self.inputs(count=40)
        reference = golden("loaders.json")
        self.assertEqual(dict(maps["pairs"].values), {tuple(k): (v,) for k, v in reference["pair_map"]})
        self.assertEqual({tuple(k): (r, tuple(map(float, o))) for k, (r, o) in reference["region_map"]},
                         dict(maps["regions"].values))
        ctx = lm.ListmodeContext.build(self.config, maps["calibration"], maps["cog_limits"], maps["doi_limits"],
                                       maps["pairs"], maps["regions"], METADATA)
        np.testing.assert_array_equal(ctx.cal_map, dense(reference["calibration_array"]))
        np.testing.assert_array_equal(ctx.cog_left, dense(reference["cog_left"]))
        np.testing.assert_array_equal(ctx.cog_right, dense(reference["cog_right"]))
        for bad in ("1 2", "1 2 100", "70000 1 2", "1 2 x", "1  2 3", "-1 2 3"):
            path = self.root / "bad_pairs.txt"
            path.write_text(bad + "\n", encoding="utf-8")
            with self.assertRaises(InputError, msg=bad):
                lm.load_pair_map(path)
        for bad in ("(7, 0)\t100\t(0, 0)", "(9, 0)\t1\t(0, 0)", "(7, 0)\t1\t(0, nan)", "(7, 0)\t1", "(7, 0)\t1\t0"):
            path = self.root / "bad_regions.tsv"
            path.write_text(bad + "\n", encoding="utf-8")
            with self.assertRaises(InputError, msg=bad):
                lm.load_region_map(path, self.mapping)
        path = self.root / "dup_pairs.txt"
        path.write_text("1 0 10\n2 0 10\n", encoding="utf-8")
        with self.assertRaises(InputError):
            lm.load_pair_map(path)

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_listmode_reference_failure_cases_become_counted_rejections(self):
        descriptors, maps = self.inputs(names=("acq_coinc_0.ldat",), count=1500)
        # 1. Pair-table row 99: the reference reads it for region -1 (unmapped minimodule).
        wrapped = self.pair_map({**PAIRS, (99, 10): 9, (99, 12): 9, (99, 0): 9}, name="pairs99.txt")
        ours = self.generate(descriptors, {**maps, "pairs": wrapped}, self.lm_parent / "row99")
        reference = golden("pair_row99.records.bin")
        written = {RECORD.unpack_from(reference, i)[10] for i in range(0, len(reference), 24)}
        self.assertIn(9, written)                                              # reference fabricates pair 9
        self.assertNotIn(9, {RECORD.unpack_from(ours.output.read_bytes(), 176 + 24 * i)[10]
                             for i in range(ours.records)})
        self.assertGreater(ours.totals("rejected")["unmapped_region"], 0)
        # 2. Unmapped minimodule beyond the reference region array: it raises, ours counts.
        regions = self.region_map(skip=((21, 15), (21, 14), (21, 13), (21, 12), (21, 11), (21, 10), (21, 9)),
                                  name="short_regions.tsv")
        ours = self.generate(descriptors, {**maps, "regions": regions}, self.lm_parent / "short")
        self.assertGreater(ours.totals("rejected")["unmapped_region"], 0)
        raises = golden("pair_row99.json")["reference_raises"]
        self.assertEqual(raises["short_region_map"]["exception"], "IndexError")
        # 3. Partial decompression limits: the reference masks with mismatched lengths and raises.
        self.assertEqual(raises["partial_decompression_limits"]["exception"], "IndexError")
        ctx = lm.ListmodeContext.build(self.config, maps["calibration"], maps["cog_limits"], maps["doi_limits"],
                                       maps["pairs"], maps["regions"], METADATA)
        y_min = ctx.y_min.copy()
        y_min[self.slab_keys((7, 0), 2)[1]] = np.nan
        ctx = replace(ctx, y_min=y_min)
        rejected, observations, flags = dict.fromkeys(lm.REJECTIONS, 0), dict.fromkeys(lm.OBSERVATIONS, 0), {}
        np.random.seed(3)
        written = sum(len(lm.process_batch(chunk, ctx, rejected, observations, flags))
                      for chunk in read_fixed_file_numpy(str(descriptors[0].path), 1000, group_events=False))
        self.assertEqual(written + sum(rejected.values()), 1500)
        self.assertGreater(rejected["no_position_region"], 0)

    @pytest.mark.slow  # ~14 s
    @pytest.mark.fr("007-FR-3")  # golden files
    def test_listmode_debug_summaries_match_reference_and_stay_bounded(self):
        descriptors, maps = self.inputs()
        result = self.generate(descriptors, maps, debug=True)
        reference = golden("debug.json")
        energies = values(reference["energies_kev"])
        np.testing.assert_array_equal(result.debug.energy, np.histogram(energies, 1500, (0, 1500))[0])
        self.assertEqual(result.debug.sm_hits, {sm: c for sm, c in reference["sm_hits"] if c})
        floods = {region: values(points) for region, points in reference["floodmap_points"]}
        for region in range(100):
            points = floods.get(region)
            expected = (np.histogram2d(points[:, 0], points[:, 1], bins=(64, 64),
                                       range=[[0, 51.2], [0, 51.2]])[0] if points is not None else np.zeros((64, 64)))
            np.testing.assert_array_equal(result.debug.flood[region], expected)
        self.assertEqual(len(result.debug_outputs), 3)
        self.assertTrue(all(path.stat().st_size > 1000 for path in result.debug_outputs))
        with self.assertRaises(FileExistsError):
            lm.debug_plots(result.debug, result.output.parent, result.output.name[:-3])
        sizes = []
        for count in (300, 3000):
            many, other = self.inputs(names=(f"size{count}_coinc_0.ldat",), count=count)
            sizes.append(self.generate(many, other, self.lm_parent / f"size{count}", debug=True).debug.nbytes)
        self.assertEqual(sizes[0], sizes[1])                                # histograms, not per-event lists

    @pytest.mark.slow  # ~5 s
    def test_listmode_debug_floodmap_with_counts(self):
        """Bug fix 2026-10-04 (T19 step 7): a nonempty flood reached colorbar(ax=nested list) and failed."""
        summary = lm.DebugSummary()
        summary.flood[3, 10, 10] = summary.flood[97, 40, 5] = 5
        summary.energy[511] = 50
        summary.sm_hits = {0: 10}
        directory = self.lm_parent / "flood_counts"
        directory.mkdir(parents=True)
        paths = lm.debug_plots(summary, directory, "flood")
        self.assertEqual([p.name for p in paths], ["flood_total_energy.png", "flood_sm_hits.png",
                                                  "flood_floodmap_all_regions.png"])
        self.assertTrue(all(path.stat().st_size > 1000 for path in paths))

    # Outputs, provenance, resume -------------------------------------------------

    def test_listmode_sidecar_provenance(self):
        descriptors, maps = self.inputs(count=300)
        result = self.generate(descriptors, maps)
        side = json.loads(result.sidecar.read_text(encoding="utf-8"))
        self.assertEqual(side["output"]["sha256"], result.sha256)
        self.assertEqual(side["output"]["records"], result.records)
        self.assertEqual(side["header"]["acqTime"], 300.5)
        self.assertEqual(side["header"]["moduleNumber"], 30)
        self.assertFalse(side["cuts"]["en_min_ch_applied"])
        self.assertEqual(side["cuts"]["en_min_ch_au"], 0.2)
        self.assertIsNone(side["timestamp"]["scaling"])
        self.assertEqual(side["timestamp"]["operator_declared_unit"], "ps")
        self.assertEqual(side["sources"]["calibration"]["sha256"], maps["calibration"].sha256)
        self.assertEqual(side["sources"]["pair_map"]["sha256"], maps["pairs"].sha256)
        self.assertEqual(side["position"]["num_regions"], 5)
        self.assertEqual(side["totals"]["records_read"],
                         side["totals"]["records_written"] + sum(side["totals"]["rejected"].values()))
        self.assertEqual([Path(i["path"]).name for i in side["inputs"]], ["acq_coinc_2.ldat", "acq_coinc_10.ldat"])
        with self.assertRaises(FileExistsError):                        # exclusive job directory
            self.generate(descriptors, maps)

    def test_listmode_guarded_resume(self):
        names = ("run_coinc_1.ldat", "run_coinc_2.ldat", "run_coinc_3.ldat")
        descriptors, maps = self.inputs(names=names, count=1200, random_slabs=False)
        fresh = self.generate(descriptors, maps, self.lm_parent / "fresh")
        job = self.lm_parent / "job"
        progress = []                                                    # cancel inside the second file
        with self.assertRaises(lm.ListmodeCancelled):
            self.generate(descriptors, maps, job, cancelled=lambda: any(p[0] == 1 for p in progress),
                          progress=lambda *a: progress.append(a))
        self.assertEqual(progress[-1][:2], (1, descriptors[1].path))
        orphans = sorted(p.name for p in (job / "segments").iterdir())
        self.assertEqual(sum(name.endswith(".json") for name in orphans), 1)
        self.assertFalse(any(p.suffix == ".lm" for p in job.iterdir()))
        snapshot = {p: p.read_bytes() for p in job.rglob("*") if p.is_file()}

        def refused(maps_=maps, config=None, descriptors_=descriptors, destination=job):
            with self.assertRaises(InputError):
                self.generate(descriptors_, maps_, destination, config=config, resume=True)
            self.assertEqual({p: p.read_bytes() for p in job.rglob("*") if p.is_file()}, snapshot)

        refused(config=self.processing({"en_min_ch": 0.2, "energy_range": [350, 665]}, "other.yaml"))
        refused(maps_={**maps, "metadata": replace(METADATA, acquisition_time_s=301.0)})
        refused(maps_={**maps, "pairs": self.pair_map({**PAIRS, (3, 3): 8}, name="pairs2.txt")})
        refused(descriptors_=descriptors[:2])
        refused(destination=self.lm_parent / "fresh")                   # no job record / completed job
        os.utime(descriptors[0].path, ns=(descriptors[0].path.stat().st_atime_ns,
                                          descriptors[0].path.stat().st_mtime_ns + 10 ** 9))
        refused()                                                        # input identity changed
        os.utime(descriptors[0].path, ns=(descriptors[0].path.stat().st_atime_ns,
                                          descriptors[0].path.stat().st_mtime_ns - 10 ** 9))
        stray = job / "segments" / "run_coinc_9.lm"                       # unrelated LM file
        stray.write_bytes(b"\0" * 24)
        snapshot[stray] = stray.read_bytes()
        refused()
        stray.unlink()                                                   # the check's own fixture file
        del snapshot[stray]
        record = next((job / "segments").glob("*.json"))
        segment = job / "segments" / json.loads(record.read_text())["segment"]["name"]
        original = segment.read_bytes()
        segment.write_bytes(bytes(reversed(original)))                   # same size, different bytes
        snapshot[segment] = segment.read_bytes()
        refused()
        segment.write_bytes(original)
        snapshot[segment] = original
        resumed = self.generate(descriptors, maps, job, resume=True)
        self.assertEqual([f.reused for f in resumed.files], [True, False, False])
        self.assertEqual(resumed.output.read_bytes(), fresh.output.read_bytes())   # no ambiguous slabs: RNG-neutral
        self.assertEqual(len(resumed.ignored), 1)                         # orphan partial of the cancelled file
        self.assertTrue(all(p.exists() for p in snapshot))                # nothing deleted
        side = json.loads(resumed.sidecar.read_text(encoding="utf-8"))
        self.assertEqual(side["resume"]["reused_segments"], 1)
        with self.assertRaises(InputError):
            self.generate(descriptors, maps, job, resume=True)            # already complete

    @pytest.mark.fr("003-FR-13")  # T29.2
    def test_listmode_in_place_and_resume_in_place(self):
        """T29.2: in place, LM writes into the caller's existing folder beside the caller's files: the same .lm
        as a new job directory; existing LM outputs refuse a new job without touching anything; an in-place
        resume ignores the caller's files and still refuses LM-named strays."""
        names = ("run_coinc_1.ldat", "run_coinc_2.ldat", "run_coinc_3.ldat")
        descriptors, maps = self.inputs(names=names, count=1200, random_slabs=False)
        fresh = self.generate(descriptors, maps, self.lm_parent / "fresh")
        folder = self.lm_parent / "run_coinc_lm-P5_2026-10-05_0936"
        (folder / ".history").mkdir(parents=True)
        caller = {folder / "run.json": b'{"status": "partial"}', folder / "request.json": b"{}"}
        for path, content in caller.items():
            path.write_bytes(content)
        progress = []                                                    # cancel inside the second file
        with self.assertRaises(lm.ListmodeCancelled):
            self.generate(descriptors, maps, folder, in_place=True,
                          cancelled=lambda: any(p[0] == 1 for p in progress), progress=lambda *a: progress.append(a))
        with self.assertRaises(InputError):                              # a new job over this job's files
            self.generate(descriptors, maps, folder, in_place=True)
        stray = folder / "run_coinc_all_notes.txt"                       # LM-named: still refused on resume
        stray.write_bytes(b"x")
        with self.assertRaises(InputError):
            self.generate(descriptors, maps, folder, in_place=True, resume=True)
        stray.unlink()
        with self.assertRaises(InputError):                              # not in place: foreign files refused
            self.generate(descriptors, maps, folder, resume=True)
        resumed = self.generate(descriptors, maps, folder, in_place=True, resume=True)
        self.assertEqual([f.reused for f in resumed.files], [True, False, False])
        self.assertEqual(resumed.output.parent, folder)
        self.assertEqual(resumed.output.read_bytes(), fresh.output.read_bytes())
        self.assertEqual({p: p.read_bytes() for p in caller}, caller)
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         [".history", "lm-job.json", "request.json", "run.json", "run_coinc_all.lm",
                          "run_coinc_all.lm.json", "segments"])
        other = self.lm_parent / "second"
        other.mkdir()
        once = self.generate(descriptors, maps, other, in_place=True)
        self.assertEqual(once.output.read_bytes(), fresh.output.read_bytes())
        before = {p: p.read_bytes() for p in other.rglob("*") if p.is_file()}
        with self.assertRaises(InputError):                              # finished job: nothing replaced
            self.generate(descriptors, maps, other, in_place=True)
        self.assertEqual({p: p.read_bytes() for p in other.rglob("*") if p.is_file()}, before)
        with self.assertRaises((InputError, FileNotFoundError)):         # in place needs an existing folder
            self.generate(descriptors, maps, self.lm_parent / "missing", in_place=True)

    @pytest.mark.slow  # ~11 s
    @pytest.mark.fr("003-FR-10", "003-FR-13", "003-FR-22", "007-FR-3")  # T21; golden files
    def test_listmode_compact_is_byte_identical_to_fixed_and_reference(self):
        """FR-22: compact decoded at the conversion hit limit = fixed, records, counts and debug."""
        from src.cornell import calibration as cal
        descriptors, maps = self.inputs()
        compact = self.compact_twins(descriptors)
        fixed = self.generate(descriptors, maps, self.lm_parent / "fixed", debug=True)
        block = cal.READ_BLOCK_BYTES
        cal.READ_BLOCK_BYTES = 3001        # many short reader batches: regrouped into 1000-record chunks
        try:
            result = self.generate(compact, maps, self.lm_parent / "compact", debug=True, hit_limit=16)
        finally:
            cal.READ_BLOCK_BYTES = block
        expected = golden("inputs_2500.records.bin")
        content = result.output.read_bytes()
        self.assertEqual(content, fixed.output.read_bytes())
        self.assertEqual(content[176:], expected)
        self.assertGreater(result.records, 300)
        self.assertGreater(result.totals("slab_flags").get(1, 0), 0)       # random slab draws exercised
        for name in ("rejected", "observations", "slab_flags"):
            self.assertEqual(result.totals(name), fixed.totals(name), name)
        np.testing.assert_array_equal(result.debug.energy, fixed.debug.energy)   # per-batch subsample
        np.testing.assert_array_equal(result.debug.flood, fixed.debug.flood)
        side = json.loads(result.sidecar.read_text(encoding="utf-8"))
        self.assertIn("compact coincidence", side["population"])
        self.assertIn("16 hits per side", side["population"])
        job = json.loads((self.lm_parent / "compact" / lm.JOB_FILE).read_text(encoding="utf-8"))["job"]
        self.assertEqual(job["compact_decoding"]["hit_limit"], 16)
        self.assertEqual({i["format"] for i in job["inputs"]}, {"compact"})
        fixed_job = json.loads((self.lm_parent / "fixed" / lm.JOB_FILE).read_text(encoding="utf-8"))["job"]
        self.assertNotIn("compact_decoding", fixed_job)                   # fixed job records unchanged
        # Another conversion hit limit: compact decoded at 20 = fixed written with 20 slots.
        wide = self.root / "wide"
        wide.mkdir()
        path = wide / "acq_coinc_2.ldat"
        encode_fixed(path, self.records(2500, seed=7), hit_limit=20)
        wide_fixed = self.generate([InputDescriptor(path, "fixed", "coincidence")], maps, self.lm_parent / "w20")
        wide_compact = self.generate(compact[:1], maps, self.lm_parent / "c20", hit_limit=20)
        self.assertEqual(wide_compact.output.read_bytes(), wide_fixed.output.read_bytes())

    @pytest.mark.fr("003-FR-10", "003-FR-13", "003-FR-22")  # T21
    def test_listmode_compact_contract_rejections(self):
        descriptors, maps = self.inputs(count=40)
        compact = self.compact_twins(descriptors, count=40)
        too_many = self.root / "compact" / "many_coinc_1.ldat"
        side = self.records(1)[0][0]
        encode_compact(too_many, [(side * 4, side)])                     # 4 x 5 hits > 16
        cases = {"no hit limit": (compact, {}), "zero hit limit": (compact, {"hit_limit": 0}),
                 "fixed with hit limit": (descriptors, {"hit_limit": 16}),
                 "mixed formats": ([descriptors[0], compact[1]], {"hit_limit": 16}),
                 "compact group": ([InputDescriptor(compact[0].path, "compact", "group")], {"hit_limit": 16}),
                 "side above the hit limit": ([InputDescriptor(too_many, "compact", "coincidence")],
                                              {"hit_limit": 16}),
                 "hit limit below the data": (compact, {"hit_limit": 4}),
                 "fixed bytes declared compact": ([InputDescriptor(descriptors[0].path, "compact", "coincidence")],
                                                  {"hit_limit": 16})}
        for i, (label, (case, options)) in enumerate(cases.items()):
            with self.assertRaises(InputError, msg=label):
                self.generate(case, maps, self.lm_parent / f"case{i}", **options)

    def test_listmode_input_contract_rejections(self):
        descriptors, maps = self.inputs(count=40)
        cases = [
            [InputDescriptor(descriptors[0].path, "fixed", "group")],
            list(reversed(descriptors)),                                   # not natural order
            [],
        ]
        other = self.root / "other"
        other.mkdir()
        cases.append([descriptors[0], self.ldat(descriptors[0].path.name, self.records(5), other)])
        unmapped = self.ldat("z_coinc_99.ldat", [([(1, 10.0, 99999)] * 5, [(1, 10.0, 99999)] * 5)])
        for i, case in enumerate(cases):
            with self.assertRaises(InputError, msg=i):
                self.generate(case, maps, self.lm_parent / f"case{i}")
            self.assertFalse((self.lm_parent / f"case{i}").exists(), i)
        with self.assertRaises(InputError):
            self.generate([unmapped], maps, self.lm_parent / "unmapped")
        custom = load_calibration(maps["calibration"].path, self.mapping, expected_regions=5,
                                  region_boundaries=[0, 0.2, 0.4, 0.6, 0.8, 1])
        with self.assertRaises(InputError):
            self.generate(descriptors, {**maps, "calibration": custom}, self.lm_parent / "boundaries")
        with self.assertRaises(InputError):
            self.generate(descriptors, {**maps, "cog_limits": maps["doi_limits"]}, self.lm_parent / "kind")

    def test_listmode_names_and_natural_order_match_reference(self):
        from natsort import natsorted
        names = ["acq_coinc_10.ldat", "acq_coinc_2.ldat", "acq_coinc_1.ldat", "acq_coinc_1b.ldat", "b_coinc_0.ldat"]
        self.assertEqual(sorted(names, key=lm.natural_key), natsorted(names))
        self.assertEqual(lm.segment_name("/x/acq_coinc_2.ldat"), "acq_coinc_2.lm")
        self.assertEqual(lm.merged_name("acq_coinc_2.lm"), "_".join("lmdir/acq_coinc_2.lm".split("_")[:-1])[6:] + "_all.lm")
        self.assertEqual(lm.merged_name("single.lm"), "single_all.lm")
        from datetime import datetime
        now = datetime(2026, 10, 1, 12, 30, 5)
        first = "/data/20261001_TwoNa22_120s_coincFixed_0.ldat"
        self.assertEqual(lm.default_job_name(first, 5, now),
                         f"lm-20261001-123005-{'_'.join(Path(first).name.split('_')[1:-2])}_COG_POSITION_5REG")

    def test_listmode_tracked_module_never_imports_local_scripts(self):
        tree = ast.parse((REPO / "src/cornell/listmode.py").read_text(encoding="utf-8"))
        modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse({m for m in modules if m and m.startswith(("scripts", "docopt", "multiprocessing",
                                                                     "tkinter", "pandas", "natsort"))})
