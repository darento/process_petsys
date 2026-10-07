"""FR-24: each stage validates exactly the records it reads, in one pass (spec 003 T24; spec 007 T12).

Moved from scripts/petsys_manager_scope_check.py. Synthetic Cornell fixtures from the listmode, calibration and
QC builders; no hardware or GUI. The listmode reference records are the golden file
tests/data/golden/listmode/inputs_2500.records.bin (T8).
"""

import unittest
from unittest.mock import patch

import pytest

from helpers import DATA
from manager_helpers import (P1_SPECS, CalibrationFixtures, ListmodeFixtures, PrivateOutput, QCFixtures,
                             encode_compact, pairs, sides_for)
from src.cornell import calibration as cal
from src.cornell.inputs import InputError
from src.petsys_manager.contracts import DataFormat, InputDescriptor

GOLDEN = DATA / "golden" / "listmode"
UNMAPPED = 999999


def corrupt(records, index, side=1):
    """Record ``index`` with its side's first hit on an unmapped channel."""
    records = list(records)
    sides = [list(records[index][0]), list(records[index][1])]
    t, e, _ = sides[side][0]
    sides[side][0] = (t, e, UNMAPPED)
    records[index] = (sides[0], sides[1])
    return records


def no_whole_file_validation():
    """Any separate whole-file validation pass (validate_ldat/probe) fails the test."""
    return patch("src.cornell.inputs._scan", side_effect=AssertionError("separate validation pass"))


@pytest.mark.fr("003-FR-24")  # spec 003 T24
class ListmodeScopeChecks(ListmodeFixtures, PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-scope-lm-"

    @pytest.mark.fr("007-FR-3")  # golden files
    def test_scope_listmode_one_pass_validates_while_reading(self):
        descriptors, maps = self.inputs(count=2500)
        with no_whole_file_validation():
            result = self.generate(descriptors, maps, self.lm_parent / "clean")
        expected = (GOLDEN / "inputs_2500.records.bin").read_bytes()
        self.assertEqual(result.output.read_bytes()[176:], expected)          # unchanged by FR-24
        self.assertEqual([f.validated_records for f in result.files], [2500, 2500])
        folder = self.root / "bad"
        folder.mkdir()
        bad = self.ldat("acq_coinc_10.ldat", corrupt(self.records(2500, seed=8), 1700), folder)
        cases = {"fixed unmapped at record 1700": ([descriptors[0], bad], {}, "record 1700, side 1: unmapped "
                                                   f"channel {UNMAPPED}")}
        compact = folder / "compact"
        compact.mkdir()
        twin = compact / "acq_coinc_10.ldat"
        records = self.records(2500, seed=8)
        t, _, c = records[2100][0][0]
        records[2100] = ([(t, float("nan"), c)] + list(records[2100][0][1:]), records[2100][1])
        encode_compact(twin, records)
        cases["compact nonfinite at record 2100"] = ([InputDescriptor(twin, "compact", "coincidence")],
                                                     {"hit_limit": 16}, "nonfinite energy")
        remainder = folder / "acq_coinc_7.ldat"
        remainder.write_bytes(descriptors[0].path.read_bytes() + b"\x01\x02\x03")
        cases["fixed remainder"] = ([InputDescriptor(remainder, "fixed", "coincidence")], {}, "remainder")
        for i, (label, (case, options, message)) in enumerate(cases.items()):
            destination = self.lm_parent / f"bad{i}"
            with self.subTest(label), no_whole_file_validation():
                with self.assertRaises(InputError) as raised:
                    self.generate(case, maps, destination, **options)
                self.assertIn(message, str(raised.exception))
                self.assertFalse(list(destination.glob("*_all.lm")) if destination.exists() else [])


@pytest.mark.fr("003-FR-24")  # spec 003 T24
class CalibrationScopeChecks(PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-scope-"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fx = CalibrationFixtures(cls.output / "calibration")

    def test_scope_calibration_reads_and_validates_only_up_to_the_event_limit(self):
        records = pairs(sides_for(self.fx.geometry, P1_SPECS, seed=11))
        bad = corrupt(records, len(records) - 10, side=0)
        for fmt in (DataFormat.FIXED, DataFormat.COMPACT):
            with self.subTest(fmt.value):
                descriptor = self.fx.write(f"late_{fmt.value}.ldat", bad, fmt)
                with no_whole_file_validation():
                    result = cal.calibrate([descriptor], self.fx.config, None, positions=1, event_limit=2000,
                                           batch_records=64)
                sample = result.files[0]
                self.assertTrue(sample.stopped_at_limit)
                self.assertLess(sample.records_validated, len(records) - 10)     # the bad record is never read
                clean = self.fx.write(f"clean_{fmt.value}.ldat", records, fmt)
                same = cal.calibrate([clean], self.fx.config, None, positions=1, event_limit=2000, batch_records=64)
                self.assertEqual(same.factors, result.factors)                    # results from the same records
                with self.assertRaises(InputError) as raised:                     # read it: it fails
                    cal.calibrate([descriptor], self.fx.config, None, positions=1, event_limit=None)
                self.assertIn(f"unmapped channel {UNMAPPED}", str(raised.exception))


@pytest.mark.fr("003-FR-24")  # spec 003 T24
class QCScopeChecks(QCFixtures, PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-scope-qc-"

    def test_scope_qc_validates_its_sample_and_reports_unknown_file_totals(self):
        records = self.records(3000)
        descriptor = self.compact("qc_late.ldat", corrupt(records, 2900))
        with no_whole_file_validation():
            result = self.qc_run([descriptor], pair_limit=200)
        sample = result.files[0]
        self.assertTrue(sample.stopped_at_limit)
        self.assertEqual(sample.validated_records, sample.records_read)
        self.assertLess(sample.records_read, 2900)
        self.assertIsNone(sample.records_in_file)                             # compact: not read whole
        whole = self.qc_run([self.compact("qc_clean.ldat", records)])
        self.assertFalse(whole.files[0].stopped_at_limit)
        self.assertEqual(whole.files[0].records_in_file, 3000)
        for label, index, limit in (("in the sample", 50, 200), ("whole file", 2900, 1_000_001)):
            with self.subTest(label), self.assertRaises(InputError) as raised:
                self.qc_run([self.compact(f"qc_bad_{index}_{limit}.ldat", corrupt(records, index))], pair_limit=limit)
            self.assertIn(f"record {index}, side 1: unmapped channel {UNMAPPED}", str(raised.exception))
