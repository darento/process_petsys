"""LDAT structural validation, input selection and map/limits/calibration parsing (spec 003 T5, T23; spec 007 T7).

Moved from scripts/petsys_manager_numeric_check.py --formats; the other modes are the calibration,
listmode, QC, bounded and scope checks (T9-T12).
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import struct
import sys
import tracemalloc
import unittest
from unittest.mock import patch

import pytest
import yaml

from helpers import REPO
from manager_helpers import HIT, PrivateOutput, compact_record, fixed_record, fixture_map
from src.cornell.inputs import (InputError, ValidationCancelled, describe_legacy, load_channel_map,
    load_processing_config, load_limits, load_calibration, select_inputs, validate_ldat)
from src.petsys_manager.contracts import InputDescriptor

# Independent wire oracle: 2-byte side counts; four slots/side, 16 bytes/slot.
SIDE_A = [(10, 2.5, 0), (11, 3.0, 128)]
SIDE_B = [(20, -0.25, 262144), (21, 4.0, 262272)]


def record_by_record(path, fmt, sides, channels, *, batch_records=5000, max_batch_bytes=16 * 1024 * 1024,
                     expected_hit_limit=None, max_records=None):
    """The T5 record-by-record validator before the 2026-10-02 compiled pass, as the oracle.

    Returns (records, hits, consumed, hit limit) or raises InputError with its message.
    """
    import math
    records = hits = consumed = 0
    limit = None

    def hit_check(payload, count, offset, record, side):
        for index in range(count):
            _, energy, channel = HIT.unpack_from(payload, offset + index * HIT.size)
            if channel not in channels:
                raise InputError(f"{path}: record {record}, side {side}: unmapped channel {channel}")
            if not math.isfinite(energy):
                raise InputError(f"{path}: record {record}, side {side}: nonfinite energy")
    size = path.stat().st_size
    with path.open("rb") as stream:
        if size == 0:
            raise InputError(f"Empty LDAT: {path}")
        if fmt == "fixed":
            header = stream.read(4)
            if len(header) != 4:
                raise InputError("Truncated fixed hit-limit header")
            limit = struct.unpack("<i", header)[0]
            if not 1 <= limit <= 255:
                raise InputError("fixed hit limit")
            if expected_hit_limit is not None and limit != expected_hit_limit:
                raise InputError("Fixed hit limit differs from explicit conversion settings")
            record_bytes = sides + sides * limit * HIT.size
            if size <= 4 or (size - 4) % record_bytes:
                raise InputError("Empty/truncated fixed records or inconsistent remainder/population")
            batch_bytes = min(batch_records, max_batch_bytes // record_bytes) * record_bytes
            consumed, stopped = 4, False
            while not stopped:
                payload = stream.read(batch_bytes)
                if not payload:
                    break
                for offset in range(0, len(payload), record_bytes):
                    if max_records is not None and records >= max_records:
                        stopped = True
                        break
                    for side in range(sides):
                        count = payload[offset + side]
                        if not 1 <= count <= limit:
                            raise InputError(f"Invalid fixed hit count at record {records}, side {side}")
                        hit_check(payload, count, offset + sides + side * limit * HIT.size, records, side)
                        hits += count
                    records += 1
                consumed += len(payload)
        else:
            while True:
                if max_records is not None and records >= max_records:
                    break
                header = stream.read(2)
                if not header:
                    break
                if len(header) != 2:
                    raise InputError("Truncated compact coincidence header")
                consumed += 2
                for side, count in enumerate(header):
                    if not 1 <= count <= (expected_hit_limit or 255):
                        raise InputError(f"Invalid compact hit count at record {records}, side {side}")
                    payload = stream.read(count * HIT.size)
                    if len(payload) != count * HIT.size:
                        raise InputError(f"Truncated compact hits at record {records}, side {side}")
                    hit_check(payload, count, 0, records, side)
                    hits += count
                    consumed += len(payload)
                records += 1
    return records, hits, consumed, limit


class FormatFixtures:
    """Per-test map, processing YAML and file builders (the script's ``FormatChecks`` setup)."""

    def setUp(self):
        super().setUp()
        self.root = self.output / self._testMethodName
        self.root.mkdir()
        (self.root / "maps").mkdir()
        (self.root / "configs").mkdir()
        self.map_path = self.root / "maps/selected.yaml"
        self.write_map(fixture_map())
        self.mapping = load_channel_map(self.map_path, processing_root=self.root)
        self.config_path = self.root / "configs/processing.yaml"
        self.config = {"map_file": "maps/selected.yaml", "min_ch": 4,
                       "en_min_ch": 0.2, "energy_range": [357, 665]}
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def write_map(self, value):
        self.map_path.write_text(yaml.safe_dump(value), encoding="utf-8")

    def data(self, content, format="fixed", population="coincidence", name="input.ldat"):
        path = self.root / name
        path.write_bytes(content)
        return InputDescriptor(path, format, population)

    def validate(self, descriptor, **options):
        return validate_ldat(descriptor, self.mapping.modules, **options)

    def calibration_file(self, lines=None, count=5):
        path = self.root / "position.encal"
        path.write_text(f"# Position-dependent energy calibration ({count} regions per slab)\n"
                        "ID(time_ch, slab, region)\tmu\tsigma\n" +
                        ("(0, 14, 2)\t100.000\t0.000\n" if lines is None else lines), encoding="utf-8")
        return path

    def limits_file(self, text="(0, 14)\t0\t25.6\n"):
        path = self.root / "limits.txt"
        path.write_text(text, encoding="utf-8")
        return path


@pytest.mark.fr("003-FR-3", "003-FR-10", "003-FR-11", "003-FR-12", "003-FR-15", "003-FR-16")  # spec 003 T5
class FormatChecks(FormatFixtures, PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-formats-"

    def test_manual_fixed_coincidence_counts_and_bytes(self):
        record = fixed_record([SIDE_A, SIDE_B])
        self.assertEqual(len(record), 130)
        descriptor = self.data(struct.pack("<i", 4) + record * 3)
        summary = self.validate(descriptor, batch_records=2, expected_hit_limit=4)
        self.assertEqual((summary.records, summary.detector_sides, summary.channel_hits), (3, 6, 12))
        self.assertEqual(summary.bytes_read, 394)
        self.assertEqual(summary.peak_buffer_bytes, 260)
        self.assertTrue(summary.descriptor.validated)
        self.assertFalse(descriptor.validated)

    def test_manual_fixed_group_counts_are_groups_not_singles(self):
        record = fixed_record([SIDE_A])
        self.assertEqual(len(record), 65)
        summary = self.validate(self.data(struct.pack("<i", 4) + record * 2, population="group"))
        self.assertEqual((summary.records, summary.detector_sides, summary.channel_hits), (2, 2, 4))
        self.assertEqual(summary.bytes_read, 134)

    def test_manual_compact_counts_and_wire_bytes(self):
        record = compact_record([SIDE_A, SIDE_B])
        self.assertEqual(len(record), 66)
        summary = self.validate(self.data(record * 3, "compact"), expected_hit_limit=4)
        self.assertEqual((summary.records, summary.detector_sides, summary.channel_hits), (3, 6, 12))
        self.assertEqual(summary.bytes_read, 198)
        self.assertIsNone(summary.hit_limit)
        self.assertEqual(summary.peak_buffer_bytes, 198)   # compact is read in bounded blocks (whole file here)

    def test_empty_header_only_and_truncated_fixed_header_rejected(self):
        for content in (b"", b"\x04", b"\x04\0\0", struct.pack("<i", 4)):
            with self.subTest(content=content), self.assertRaises(InputError):
                self.validate(self.data(content))

    def test_invalid_fixed_hit_limits_and_expected_limit_rejected(self):
        for limit in (-1, 0, 256, 2**31 - 1):
            with self.subTest(limit=limit), self.assertRaises(InputError):
                self.validate(self.data(struct.pack("<i", limit) + b"bytes"))
        valid = self.data(struct.pack("<i", 4) + fixed_record([SIDE_A, SIDE_B]))
        with self.assertRaises(InputError):
            self.validate(valid, expected_hit_limit=16)

    def test_fixed_truncated_hits_trailing_garbage_and_bad_counts_rejected(self):
        good = struct.pack("<i", 4) + fixed_record([SIDE_A, SIDE_B])
        for content in (good[:-1], good + b"\0", good[:4] + b"\x05" + good[5:],
                        good[:5] + b"\0" + good[6:]):
            with self.subTest(length=len(content)), self.assertRaises(InputError):
                self.validate(self.data(content))

    def test_group_zero_and_overlimit_counts_rejected(self):
        payload = struct.pack("<i", 4) + fixed_record([SIDE_A])
        for count in (0, 5):
            with self.assertRaises(InputError):
                self.validate(self.data(payload[:4] + bytes([count]) + payload[5:], population="group"))

    def test_compact_empty_truncated_headers_hits_and_counts_rejected(self):
        good = compact_record([SIDE_A, SIDE_B])
        for content in (b"", b"\x02", good[:-1], good + b"\x01", b"\0\x02" + good[2:]):
            with self.subTest(length=len(content)), self.assertRaises(InputError):
                self.validate(self.data(content, "compact"))
        with self.assertRaises(InputError):
            self.validate(self.data(good, "compact"), expected_hit_limit=1)

    def test_unmapped_channels_nonfinite_energy_and_late_invalid_record_rejected(self):
        for channel, energy in ((999999, 2.0), (-1, 2.0), (0, float("nan")), (0, float("inf"))):
            bad = [(1, energy, channel), SIDE_A[1]]
            for format in ("fixed", "compact"):
                payload = (struct.pack("<i", 4) + fixed_record([SIDE_A, SIDE_B]) + fixed_record([bad, SIDE_B])
                           if format == "fixed" else compact_record([SIDE_A, SIDE_B]) + compact_record([bad, SIDE_B]))
                with self.subTest(channel=channel, format=format), self.assertRaises(InputError):
                    self.validate(self.data(payload, format), batch_records=1)

    def test_unused_fixed_padding_is_not_a_channel_or_hit_population(self):
        payload = bytearray(struct.pack("<i", 4) + fixed_record([SIDE_A, SIDE_B]))
        payload[38:54] = HIT.pack(0, float("nan"), -1)  # First side inactive third slot.
        summary = self.validate(self.data(payload))
        self.assertEqual(summary.channel_hits, 4)

    def test_format_population_and_legacy_confirmation_are_explicit(self):
        descriptor = self.data(struct.pack("<i", 4) + fixed_record([SIDE_A, SIDE_B]))
        for options in ({}, {"format": "fixed", "population": "coincidence"},
                        {"format": "fixed", "confirmed": True}):
            with self.assertRaises(InputError):
                describe_legacy([descriptor.path], **options)
        legacy = describe_legacy([descriptor.path], format="fixed", population="coincidence", confirmed=True)
        self.assertEqual(legacy, (descriptor,))
        with self.assertRaises(InputError):
            self.validate(replace(descriptor, format="compact", population="group"))
        with self.assertRaises(InputError):
            self.validate(replace(descriptor, format="compact"))

    def test_indistinguishable_fixed_layout_preserves_explicit_population(self):
        # Two one-hit groups can also be one structurally valid coincidence.
        # Active shifted slots remain mapped/finite; padding is intentionally ignored.
        record = fixed_record([[(1, 0., 0)]])
        group = self.data(struct.pack("<i", 4) + record * 2, population="group")
        paired = replace(group, population="coincidence")
        self.assertEqual(self.validate(group).records, 2)
        self.assertEqual(self.validate(paired).records, 1)
        self.assertEqual(self.validate(group).descriptor.population.value, "group")
        self.assertEqual(self.validate(paired).descriptor.population.value, "coincidence")

    def test_selection_keeps_exact_order_no_wildcard_sibling_discovery(self):
        first = self.data(b"fixture", name="split_00000010.ldat")
        second = self.data(b"fixture", name="split_00000002.ldat")
        self.data(b"unrelated", name="split_other.ldat")
        selection = select_inputs([first, second], "calibrate", processing_root=self.root)
        self.assertEqual([item.path.name for item in selection], [first.path.name, second.path.name])
        self.assertEqual(len(selection), 2)
        self.assertFalse(any(item.validated for item in selection))

    def test_missing_input_list_reports_each_exact_path_without_similar_files(self):
        self.data(b"exists", name="missing1_other.ldat")
        descriptors = [InputDescriptor(Path(name), "fixed", "coincidence")
                       for name in ("missing1.ldat", "missing2.ldat")]
        with self.assertRaises(InputError) as raised:
            select_inputs(descriptors, "listmode", processing_root=self.root)
        self.assertEqual(len(raised.exception.issues), 2)
        self.assertTrue(all(str(self.root / name) in str(raised.exception) for name in
                            ("missing1.ldat", "missing2.ldat")))

    @pytest.mark.fr("003-FR-21")  # T20
    def test_wrong_routes_mixed_groups_duplicates_and_untyped_selection_rejected(self):
        fixed = self.data(b"fixture")
        compact = replace(fixed, format="compact")
        group = replace(fixed, population="group")
        compact_group = replace(compact, population="group")   # FR-21 (T20): compact coincidence calibrates
        for selection, action in (([compact_group], "calibrate"), ([compact, fixed], "calibrate"), ([fixed], "qc_analyze"),
                                  ([group], "listmode"), ([group, fixed], "calibrate"),
                                  ([fixed, fixed], "calibrate"), ([fixed.path], "calibrate"), ([], "calibrate")):
            with self.subTest(action=action), self.assertRaises(InputError):
                select_inputs(selection, action, processing_root=self.root)
        self.assertEqual(len(select_inputs([group], "calibrate", processing_root=self.root)), 1)
        self.assertEqual(len(select_inputs([compact], "calibrate", processing_root=self.root)), 1)

    def test_bounded_batch_arguments_and_missing_mapping_rejected(self):
        descriptor = self.data(struct.pack("<i", 4) + fixed_record([SIDE_A, SIDE_B]))
        for options in ({"batch_records": 0}, {"batch_records": True}, {"max_batch_bytes": 0},
                        {"expected_hit_limit": -1}, {"max_batch_bytes": 17 * 1024 * 1024}):
            with self.assertRaises(InputError):
                self.validate(descriptor, **options)
        with self.assertRaises(InputError):
            validate_ldat(descriptor, {})
        with self.assertRaises(InputError):
            self.validate(replace(descriptor, path=Path("relative.ldat")))

    def test_cancelled_scan_never_returns_validated_summary(self):
        descriptor = self.data(struct.pack("<i", 4) + fixed_record([SIDE_A, SIDE_B]) * 3)
        calls = [0]
        def cancellation():
            calls[0] += 1
            return calls[0] >= 3
        with self.assertRaises(ValidationCancelled):
            self.validate(descriptor, batch_records=1, cancelled=cancellation)
        self.assertFalse(descriptor.validated)

    def test_input_changed_during_scan_rejected(self):
        descriptor = self.data(struct.pack("<i", 4) + fixed_record([SIDE_A, SIDE_B]))
        calls = [0]
        def mutation():
            calls[0] += 1
            if calls[0] == 3:
                with descriptor.path.open("ab") as out:
                    out.write(fixed_record([SIDE_A, SIDE_B]))
            return False
        with self.assertRaises(InputError):
            self.validate(descriptor, batch_records=1, cancelled=mutation)

    def test_repeated_files_counts_and_memory_bound_independent_of_length(self):
        record = fixed_record([SIDE_A, SIDE_B])
        descriptor = self.data(struct.pack("<i", 4) + record * 100)
        peaks = []
        for count in (100, 12000):
            with descriptor.path.open("wb") as out:
                out.write(struct.pack("<i", 4))
                for _ in range(count):
                    out.write(record)
            tracemalloc.start()
            try:
                summary = self.validate(descriptor, batch_records=17, max_batch_bytes=8192)
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
            self.assertEqual((summary.records, summary.channel_hits), (count, 4 * count))
            self.assertEqual(summary.peak_buffer_bytes, 17 * 130)
        self.assertLess(peaks[1], peaks[0] + 100000)
        self.assertLess(peaks[1], 200000)
        (self.root / "memory-evidence.json").write_text(json.dumps({"records": [100, 12000],
            "tracemalloc_peak_bytes": peaks, "batch_records": 17, "batch_bytes": 2210}), encoding="utf-8")

    def test_compact_storage_bound_on_repeated_records(self):
        record = compact_record([SIDE_A, SIDE_B])
        for count in (5000, 50000):
            descriptor = self.data(record * count, "compact", name=f"compact{count}.ldat")
            summary = self.validate(descriptor, max_batch_bytes=8192)
            self.assertEqual((summary.records, summary.channel_hits), (count, 4 * count))
            self.assertLessEqual(summary.peak_buffer_bytes, 8192 + len(record))   # block + one carried record

    def test_processing_map_resolves_from_root_not_yaml_directory(self):
        config = load_processing_config(self.config_path, processing_root=self.root, action="listmode")
        self.assertEqual(config.mapping.path, self.map_path)
        self.assertEqual(config.mapping.modules[0], (7, 0))
        self.assertEqual(config.mapping.modules[262144], (41, 0))
        self.assertEqual(len(config.mapping.modules), 512)
        self.assertEqual(config.values["energy_range"], (357, 665))
        self.assertEqual(config.sha256, hashlib.sha256(self.config_path.read_bytes()).hexdigest())

    def test_processing_absolute_map_and_immutable_pickleable_tables(self):
        self.config["map_file"] = str(self.map_path)
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        config = load_processing_config(self.config_path, processing_root=self.root, action="calibrate")
        with self.assertRaises(TypeError):
            config.mapping.modules[0] = (99, 99)
        self.assertEqual(dict(pickle.loads(pickle.dumps(config.mapping.modules))), dict(config.mapping.modules))

    def test_yaml_duplicates_aliases_unsafe_tags_and_nonfinite_values_rejected(self):
        for text in ("map_file: x\nmap_file: y\n", "x: &x [1]\ny: *x\n",
                     "!!python/object/apply:os.system ['never executed']", "x: .nan", "[]"):
            self.config_path.write_text(text, encoding="utf-8")
            with self.assertRaises(InputError):
                load_processing_config(self.config_path, processing_root=self.root, action="calibrate")

    def test_bad_processing_cuts_unpopulated_map_keys_and_missing_maps_rejected(self):
        for changed in ({"min_ch": 0}, {"min_ch": True}, {"min_ch": 9}, {"en_min_ch": float("nan")},
                        {"energy_range": [665, 357]}, {"map_file": "maps/missing.yaml"},
                        {"unpopulated_minimodules": {99: [0]}}, {"unpopulated_minimodules": {99: []}},
                        {"unpopulated_minimodules": {7: [0, 0]}}):
            self.config_path.write_text(yaml.safe_dump({**self.config, **changed}), encoding="utf-8")
            with self.subTest(changed=changed), self.assertRaises((InputError, FileNotFoundError)):
                load_processing_config(self.config_path, processing_root=self.root, action="listmode")

    def test_invalid_map_schema_lengths_duplicates_addresses_and_types_rejected(self):
        mutations = ({"Time": [0]}, {"Energy": list(range(128))}, {"FEM": "unknown"}, {"FEM": []},
                     {"x_pitch": 0}, {"channels": True}, {"mM_disposition": []},
                     {"sum_rows_cols": "True"}, {"mod_feb_map": {7: [0, 0, 0], 41: [0, 0, 0]}},
                     {"mod_feb_map": {7: [-1, 0, 0]}}, {"channels_j1": [], "channels_j2": []})
        for changed in mutations:
            self.write_map({**fixture_map(), **changed})
            with self.subTest(changed=str(changed)[:60]), self.assertRaises(InputError):
                load_channel_map(self.map_path, processing_root=self.root)

    def test_non_summed_map_is_parseable_but_cornell_processing_route_rejected(self):
        self.write_map({**fixture_map(), "sum_rows_cols": False})
        mapping = load_channel_map(self.map_path, processing_root=self.root)
        self.assertEqual(len(mapping.local[0]), 2)
        with self.assertRaises(InputError):
            load_processing_config(self.config_path, processing_root=self.root, action="calibrate")

    def test_guarded_import_does_not_launch_gui_processes_or_workers(self):
        import importlib.util
        before = set(sys.modules)
        with patch("subprocess.Popen", side_effect=AssertionError("process at import")), \
             patch("threading.Thread.start", side_effect=AssertionError("worker at import")), patch.dict(sys.modules):
            spec = importlib.util.spec_from_file_location("cornell_inputs_import_check", REPO / "src/cornell/inputs.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.assertFalse(any(name.split(".")[0] in ("tkinter", "customtkinter")
                                 for name in set(sys.modules) - before))

    def test_selected_cornell_and_imas_mapping_matches_existing_factory(self):
        from src.mapping_generator import map_factory
        for name in ("cornell_map_full_system.yaml", "cornell_map_full_system_old.yaml", "imas1DAQ_map.yaml"):
            path = REPO / "maps" / name
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            mapping = load_channel_map(path, processing_root=REPO)
            local, modules, types, _ = map_factory(str(path))
            self.assertEqual(dict(mapping.local), {k: tuple(v) for k, v in local.items()})
            self.assertEqual(dict(mapping.modules), modules)
            self.assertEqual(dict(mapping.types), {k: tuple(v) for k, v in types.items()})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_valid_cog_doi_limits_keep_sparse_keys_and_missing_unavailable(self):
        path = self.limits_file()
        for kind in ("cog", "doi"):
            limits = load_limits(path, self.mapping, kind=kind)
            self.assertEqual(limits.values[(0, 14)], (0, 25.6))
            self.assertEqual(limits.values.missing([(0, 14), (0, 15)]), ((0, 15),))

    def test_limits_bad_rows_duplicate_nonfinite_reversed_and_invalid_map_keys_rejected(self):
        zero = load_limits(self.limits_file("(0, 14)\t3.2\t3.2\n"), self.mapping, kind="doi")   # FR-12 2026-10-02
        self.assertEqual((zero.zero_width, zero.values[(0, 14)]), (((0, 14),), (3.2, 3.2)))
        for text in ("", "(0, 14)\t1\t0", "(0, 14)\tNaN\t1", "(0, 14)\t0\tinf",
                     "(0, 14)\t0", "(128, 14)\t0\t1", "(99999, 14)\t0\t1",
                     "(0, -1)\t0\t1", "(0, 1)\t0\t1", "(0, 14)\t0\t1\n(0, 14)\t1\t2"):
            with self.subTest(text=text), self.assertRaises(InputError):
                load_limits(self.limits_file(text), self.mapping, kind="cog")

    def test_valid_position_calibration_retains_legacy_contract_missing_boundaries_and_zero_sigma(self):
        path = self.calibration_file()
        calibration = load_calibration(path, self.mapping, expected_regions=5)
        self.assertEqual(calibration.values[(0, 14, 2)], (100, 0))
        self.assertIsNone(calibration.region_boundaries)
        self.assertIn("unavailable", calibration.region_provenance)
        self.assertEqual(calibration.values.missing([(0, 14, 2), (0, 14, 3)]), ((0, 14, 3),))
        from src.utils import KevConverter
        converter = KevConverter(str(path), "cornell_position")
        self.assertEqual(converter.num_regions, calibration.num_regions)
        self.assertEqual(converter.kev_factors[(0, 14, 2)], calibration.values[(0, 14, 2)][0])

    def test_bad_calibration_factors_duplicate_tuple_shapes_regions_channels_and_slabs_rejected(self):
        for line in ("(0, 14, 2)\tnan\t1", "(0, 14, 2)\t-inf\t1", "(0, 14, 2)\t-1\t-1", "(0, 14, 2)\t-1\tnan",
                     "(0, 14, 2)\t100\tinf", "(0, 14, 2)\t100\t-1", "(0, 14, 5)\t100\t1",
                     "(0, 14)\t100\t1", "(True, 14, 2)\t100\t1", "(128, 14, 2)\t100\t1",
                     "(9999, 14, 2)\t100\t1", "(0, 16, 2)\t100\t1",
                     "(0, 14, 2)\t100\t1\n(0, 14, 2)\t101\t2"):
            with self.subTest(line=line), self.assertRaises(InputError):
                load_calibration(self.calibration_file(line + "\n"), self.mapping)

    def test_non_positive_mu_rows_are_keys_without_factor_like_the_reference(self):
        # FR-12 owner decision 2026-10-02: the reference LM uses only mu > 0.
        lines = ("(0, 14, 0)\t-2878.171\t3818.350\n(0, 14, 1)\t0\t1\n(0, 14, 2)\t100.000\t5.000\n"
                 "(0, 14, 3)\t0\t0\n")
        calibration = load_calibration(self.calibration_file(lines), self.mapping, expected_regions=5)
        self.assertEqual(dict(calibration.values), {(0, 14, 2): (100.0, 5.0)})
        self.assertEqual(calibration.non_positive, ((0, 14, 0), (0, 14, 1)))
        self.assertEqual(calibration.unfitted, 1)
        self.assertEqual(calibration.values.missing([(0, 14, 0), (0, 14, 1), (0, 14, 3)]),
                         ((0, 14, 0), (0, 14, 1), (0, 14, 3)))

    def test_calibration_headers_empty_and_selected_region_mismatch_rejected(self):
        path = self.calibration_file()
        with self.assertRaises(InputError):
            load_calibration(path, self.mapping, expected_regions=4)
        for text in ("", "ID(t_ch, slab)\tmu\n(0,14)\t100", "# unknown\nID\n(0,14,2)\t100\t1"):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(InputError):
                load_calibration(path, self.mapping)

    def test_region_sidecar_count_boundaries_and_fingerprint_checked(self):
        path = self.calibration_file()
        boundaries = [0, 3/11, 14/33, 19/33, 8/11, 1]
        metadata = {"schema_version": 1, "num_regions": 5, "region_boundaries": boundaries,
                    "calibration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "cuts": {"units": "a.u.", "min_ch": 4}}
        sidecar = self.root / "position.json"
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        calibrated = load_calibration(path, self.mapping, region_boundaries=boundaries, metadata_path=sidecar)
        self.assertEqual(calibrated.region_boundaries, tuple(boundaries))
        for changed in ({"num_regions": 4}, {"region_boundaries": [0, 0.2, 0.4, 0.6, 0.8, 1]},
                        {"calibration_sha256": "stale"}, {"region_boundaries": [0, 0.2, 0.2, 0.6, 0.8, 1]},
                        {"region_boundaries": [0, 1]}, {"schema_version": True}):
            sidecar.write_text(json.dumps({**metadata, **changed}), encoding="utf-8")
            with self.subTest(changed=changed), self.assertRaises(InputError):
                load_calibration(path, self.mapping, region_boundaries=boundaries, metadata_path=sidecar)

    def test_selected_region_boundaries_invalid_or_nonfinite_rejected(self):
        path = self.calibration_file()
        for values in ([0, 1], [0, .2, .2, .6, .8, 1], [0, .2, float("nan"), .6, .8, 1],
                       [-1, .2, .4, .6, .8, 1], [0, .2, .4, .6, .8, 2]):
            with self.assertRaises(InputError):
                load_calibration(path, self.mapping, region_boundaries=values)

    def test_metadata_file_and_row_bounds_rejected(self):
        path = self.limits_file("x" * 4100)
        with self.assertRaises(InputError):
            load_limits(path, self.mapping, kind="cog")
        with patch("src.cornell.inputs.MAX_METADATA_BYTES", 64), self.assertRaises(InputError):
            load_channel_map(self.map_path, processing_root=self.root)

    def test_yaml_node_bound_and_non_file_inputs_rejected(self):
        with patch("src.cornell.inputs.MAX_ENTRIES", 10), self.assertRaises(InputError):
            load_channel_map(self.map_path, processing_root=self.root)
        with self.assertRaises(InputError):
            self.validate(InputDescriptor(self.root, "fixed", "coincidence"))
        with self.assertRaises(InputError):
            load_limits(self.root, self.mapping, kind="cog")

    def test_all_external_fixture_inputs_remain_byte_identical(self):
        descriptor = self.data(struct.pack("<i", 4) + fixed_record([SIDE_A, SIDE_B]))
        limits = self.limits_file()
        calibration = self.calibration_file()
        paths = [descriptor.path, limits, calibration, self.map_path, self.config_path]
        before = {p: p.read_bytes() for p in paths}
        self.validate(descriptor)
        load_limits(limits, self.mapping, kind="cog")
        load_calibration(calibration, self.mapping)
        load_processing_config(self.config_path, processing_root=self.root, action="listmode")
        self.assertEqual({p: p.read_bytes() for p in paths}, before)


@pytest.mark.fr("003-FR-1", "003-FR-15", "003-FR-16")  # spec 003 T23
class ValidationSpeedChecks(FormatFixtures, PrivateOutput, unittest.TestCase):
    """2026-10-02 Cornell freeze: validation of a 23 GB compact conversion took ~12 min in a pure-Python
    loop on the workflow thread and starved the Tk thread (GIL). The compiled pass must give the same
    results and first error, be much faster and leave the GIL free."""

    fixture_prefix = "petsys-manager-speed-"

    def random_files(self, rng, count):
        keys = sorted(self.mapping.modules)
        cases = []
        for i in range(count):
            fmt = ("fixed", "compact")[i % 2]
            population = "group" if fmt == "fixed" and i % 7 == 3 else "coincidence"
            sides = 1 if population == "group" else 2
            limit = int(rng.integers(1, 6))
            body = [[[(int(rng.integers(0, 2**40)), float(rng.normal(20, 5)), int(rng.choice(keys)))
                      for _ in range(int(rng.integers(1, limit + 1)))] for _ in range(sides)]
                    for _ in range(int(rng.integers(1, 40)))]
            fault = int(rng.integers(0, 9))
            r = int(rng.integers(0, len(body)))
            s = int(rng.integers(0, sides))
            h = int(rng.integers(0, len(body[r][s])))
            t, e, c = body[r][s][h]
            if fault == 1:
                body[r][s][h] = (t, e, 999999)                         # unmapped
            elif fault == 2:
                body[r][s][h] = (t, float(("nan", "inf", "-inf")[i % 3]), c)
            elif fault == 3:
                body[r][s][h] = (t, e, -5)                              # negative channel
            elif fault == 4:
                body[r][s][h] = (t, float("nan"), 999999)               # unmapped wins over nonfinite
            if fmt == "fixed":
                content = struct.pack("<i", limit) + b"".join(fixed_record(rec, limit) for rec in body)
                if fault == 5:                                          # count 0 or above the limit
                    offset = 4 + r * (sides + sides * limit * 16) + s
                    content = content[:offset] + bytes([(0, limit + 1)[i % 2]]) + content[offset + 1:]
            else:
                content = b"".join(compact_record(rec) for rec in body)
                if fault == 5:
                    offset = sum(len(compact_record(rec)) for rec in body[:r]) + s
                    content = content[:offset] + b"\0" + content[offset + 1:]
            if fault in (6, 7):                                         # truncated anywhere
                content = content[:int(rng.integers(1, len(content)))]
            cases.append((self.data(content, fmt, population, name=f"case{i}.ldat"), fmt, sides,
                          dict(batch_records=int(rng.integers(1, 9)),
                               expected_hit_limit=None if i % 3 else limit,
                               max_records=None if i % 4 else int(rng.integers(1, 50)))))
        return cases

    def outcome(self, function):
        try:
            return function()
        except InputError as exc:
            return ("error", str(exc))

    def test_compiled_validation_equals_record_by_record_reading(self):
        import numpy as np
        from src.cornell.inputs import probe_ldat
        rng = np.random.default_rng(23)
        messages = []
        cases = self.random_files(rng, 600)
        for descriptor, fmt, sides, options in cases:
            limit, max_records = options["expected_hit_limit"], options["max_records"]
            expected = self.outcome(lambda: record_by_record(
                descriptor.path, fmt, sides, self.mapping.modules, batch_records=options["batch_records"],
                max_batch_bytes=8192, expected_hit_limit=limit, max_records=max_records))
            if max_records is None:
                got = self.outcome(lambda: (lambda s: (s.records, s.channel_hits, s.bytes_read, s.hit_limit))(
                    validate_ldat(descriptor, self.mapping.modules, batch_records=options["batch_records"],
                                  max_batch_bytes=8192, expected_hit_limit=limit)))
            else:
                got = self.outcome(lambda: (lambda s: (s.records_checked, s.channel_hits_checked))(
                    probe_ldat(descriptor, self.mapping.modules, max_records=max_records,
                               expected_hit_limit=limit)))
                expected = expected if expected[0] == "error" else expected[:2]
            if expected[0] == "error" and "fixed hit limit" in expected[1]:
                self.assertEqual(got[0], "error", descriptor.path)
                continue
            self.assertEqual(got, expected, (descriptor.path, fmt, options))
            if got[0] == "error":
                messages.append(got[1])
        joined = "\n".join(messages)
        for kind in ("unmapped channel 999999", "unmapped channel -5", "nonfinite energy", "Invalid fixed hit count",
                     "Invalid compact hit count", "Truncated compact hits", "Truncated compact coincidence header",
                     "Empty/truncated fixed records"):
            self.assertIn(kind, joined)
        self.assertGreater(len(messages), 200)
        self.assertGreater(len(cases) - len(messages), 100)                 # valid files compared too

    def large_compact(self, records):
        record = compact_record([SIDE_A * 8, SIDE_B * 8])               # 16 hits per side, 514 bytes
        path = self.root / "large.ldat"
        with path.open("wb") as out:
            for _ in range(records // 1000):
                out.write(record * 1000)
        return InputDescriptor(path, "compact", "coincidence"), record

    def test_compiled_validation_is_fast_bounded_and_releases_the_gil(self):
        import threading
        import time
        descriptor, record = self.large_compact(200_000)                # 103 MB
        started = time.perf_counter()
        summary = self.validate(descriptor)
        compiled = time.perf_counter() - started
        self.assertEqual((summary.records, summary.channel_hits), (200_000, 6_400_000))
        self.assertLessEqual(summary.peak_buffer_bytes, 16 * 1024 * 1024 + len(record))
        started = time.perf_counter()
        expected = record_by_record(descriptor.path, "compact", 2, self.mapping.modules, max_records=20_000)
        oracle = (time.perf_counter() - started) * 10
        self.assertEqual(expected[:2], (20_000, 640_000))
        self.assertGreater(oracle / compiled, 10, (oracle, compiled))
        # GIL hand-offs on the main thread while a worker validates: each time.sleep(0) releases and
        # reacquires the GIL, as each Tk -> Python callback does. A worker holding the GIL starves them.
        def handoffs(seconds):
            count, end = 0, time.perf_counter() + seconds
            while time.perf_counter() < end:
                time.sleep(0)
                count += 1
            return count
        def contended(target):
            stop = threading.Event()
            def loop():
                while not stop.is_set():
                    target()
            worker = threading.Thread(target=loop)
            worker.start()
            time.sleep(0.05)
            count = handoffs(0.3)
            stop.set()
            worker.join()
            return count
        idle = handoffs(0.3)
        rates = sorted(contended(lambda: self.validate(descriptor)) / idle for _ in range(3))
        old = contended(lambda: record_by_record(descriptor.path, "compact", 2, self.mapping.modules,
                                                 max_records=20_000)) / idle
        evidence = {"records": 200_000, "compiled_s": compiled, "record_by_record_s_est": oracle,
                    "main_thread_gil_handoffs_vs_idle": rates, "record_by_record_handoffs_vs_idle": old}
        (self.root / "speed-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(f"\n[validation speed] {json.dumps(evidence)}")
        self.assertGreater(rates[1], 0.5, evidence)
        self.assertLess(old, 0.2, evidence)                       # the measure does detect a GIL-bound worker
