"""Conversion controls, exact ordered LDAT selection and structure checks (spec 003 T15, T20;
spec 007 T15), and the next-step offers after a conversion (spec 005 T11).

Moved from scripts/petsys_manager_gui_check.py --conversion and --real. ``gui``: skipped without a
display. ``RealSelectionChecks`` is ``real_data``: Cornell acquisitions under PETSYS_DATA_DIR, read only.
"""

from pathlib import Path
import re
import time

import pytest

from exe_programs import petsys_manager_gui as gui
from helpers import REPO
from manager_gui_helpers import (COMPACT_COINC, ConversionBase, FIXED_COINC, SETTLE_S, encode, pump, synthetic_sides,
                                 texts, wait_threads)
from src.petsys_manager.artifacts import read_manifest


OFFERS = ("calibrate", "listmode", "qc_analyze")


@pytest.mark.gui
@pytest.mark.fr("003-FR-10", "003-FR-11", "003-FR-16")  # spec 003 T15
class ConversionChecks(ConversionBase):
    fixture_prefix = "pm-gui-conv-"

    @pytest.mark.fr("003-FR-9", "003-FR-13")
    def test_conversion_request_format_population_and_exact_outputs(self):
        root, app = self.converter_app()
        raw = self.raw(root, "cornell run.v2.rawf")  # unsuffixed (no _<n>s) and dotted name
        (root / "data" / "cornell run.v2_coincCompact_1.ldat").write_bytes(b"old look-alike")
        self.assertNotIn("convert_group", app.buttons)                       # FR-10: no group conversion
        self.assertFalse(hasattr(app, "coincidence_format"))                 # no format choice
        app.splits.set("2")
        app.convert_duration.set("30")
        self.settle_edit(app, app.raw_input, str(raw))
        prefix = raw.with_suffix("")
        self.assertIn(f"Converter input (-i): {prefix}  (reads cornell run.v2.rawf and its .idxf index)",
                      app.raw_plan.cget("text"))
        self.assertIn("a new run folder cornell-run-v2_conv_<date>_<time> in the Output Data Folder, with "
                      "cornell run.v2_coincCompact[_<n>].ldat", app.raw_plan.cget("text"))     # T29
        self.assertIn("--splitTime 15.1 s: duration 30 s / 2 splits + 0.1 s", app.split_plan.cget("text"))
        text = self.run_conversion(app, "convert_coincidence")
        argv = self.converter.launched[-1]
        self.assertEqual(Path(argv[0]).name, "convert_raw_to_coincidence")
        self.assertEqual(argv[argv.index("-i") + 1], str(prefix))
        self.assertEqual(argv[argv.index("--writeMultipleHits") + 1], "16")
        self.assertEqual(argv[argv.index("--splitTime") + 1], str(30.0 / 2 + 0.1))
        self.assertIn("--writeBinaryCompact", argv)
        self.assertNotIn("--writeBinaryFixed", argv)
        self.assertTrue(text.startswith("Conversion succeeded"), text)
        self.assertIn(f"RAW input: {raw}", text)
        self.assertIn("Outputs (2, compact coincidence, structure-checked against the selected map, in this order; each processing stage validates the records it reads):", text)
        outputs = [Path(line.split(". ", 1)[1].split("   (")[0]) for line in text.splitlines() if line.startswith("  ")]
        self.assertEqual([path.name for path in outputs], ["cornell run.v2_coincCompact_3.ldat",
                                                           "cornell run.v2_coincCompact_4.ldat"])  # no split 1
        self.assertTrue(all(re.fullmatch(r"cornell-run-v2_conv_\d{4}-\d\d-\d\d_\d{4}", path.parent.name)
                            for path in outputs))                                   # T29: flat run folder
        self.assertIn("(40 records)", text)
        # T34 (Cornell 2026-10-06: the summary still said "kept"): removed after success, said so.
        self.assertIn("Removed after conversion (not outputs): 3 .lidx index file(s); empty split(s): "
                      "cornell run.v2_coincCompact_5.ldat", text)
        self.assertNotIn("kept, not outputs", text)
        self.assertEqual(sorted(path.name for path in outputs[0].parent.iterdir() if path.suffix in (".ldat", ".lidx")),
                         ["cornell run.v2_coincCompact_3.ldat", "cornell run.v2_coincCompact_4.ldat"])
        manifest = read_manifest(outputs[0].parent)
        options = manifest["settings"]["options"]
        self.assertEqual((options["output_format"], options["population"], options["splits"], options["duration_s"],
                          options["hit_limit"]), ("compact", "coincidence", 2, 30.0, 16))
        self.assertEqual(Path(options["raw_input"]), raw)
        # Hand the exact outputs to processing: compact coincidence -> calibration, LM and offline QC (spec 005:
        # one offer per tab).
        for key in ("calibrate", "listmode", "qc_analyze"):
            self.assertEqual(self.state(app, f"offer_{key}"), "normal")
            app.buttons[f"offer_{key}"].invoke()
        self.wait(app, lambda: app._check_id is None and app.shown_generation == app._awaiting, SETTLE_S)
        for key in ("calibrate", "listmode", "qc_analyze"):
            selection = app.selections[key]
            self.assertEqual(selection.paths, outputs)
            self.assertEqual(selection.declared.get(), "compact_coincidence")
            self.assertTrue(selection.confirmed.get())
            self.assertIn("from conversion run", selection.summary.cget("text"))
            self.assertNotIn("Input LDAT files", self.reason(app, key))
        self.assertIn("Create Energy cal file: prerequisites met", self.reason(app, "calibrate"))
        # No splitting: no --splitTime, one unsplit output.
        self.settle_edit(app, app.splits, "1")
        text = self.run_conversion(app, "convert_coincidence")
        argv = self.converter.launched[-1]
        self.assertIn("--writeBinaryCompact", argv)
        self.assertNotIn("--splitTime", argv)
        self.assertIn("Outputs (1, compact coincidence", text)
        self.assertIn("cornell run.v2_coincCompact.ldat", text)
        # The file name's "_300s" is not read: the entered duration sets the split time.
        named = self.raw(root, "acq_300s.rawf")
        app.convert_duration.set("60")
        app.splits.set("4")
        self.settle_edit(app, app.raw_input, str(named))
        self.run_conversion(app, "convert_coincidence")
        argv = self.converter.launched[-1]
        self.assertEqual(argv[argv.index("--splitTime") + 1], str(60.0 / 4 + 0.1))
        self.assertTrue(all(Path(argv[0]).name == "convert_raw_to_coincidence" and "--writeBinaryFixed" not in argv
                            for argv in self.converter.launched))
        self.assertEqual((root / "data" / "cornell run.v2_coincCompact_1.ldat").read_bytes(), b"old look-alike")
        self.assertEqual(app.guard.violations, [])

    def test_close_during_conversion_closes_on_the_first_request(self):
        """Bug fix 2026-10-05 (T19 step 13): the shutdown thread read the finished workflow's handle before the
        request thread cleared it, reported "A workflow started during shutdown" and needed a second close."""
        root, app = self.converter_app()
        factory = app.session._coordinator_factory

        def late_clearing(**sinks):
            coordinator = factory(**sinks)
            start = coordinator.start

            def started(plan, **options):
                handle = start(plan, **options)
                wait = handle.wait

                def request_wait(timeout=None):
                    outcome = wait(timeout)
                    if timeout is None:     # the request thread: it clears the session's handle late
                        time.sleep(0.5)
                    return outcome
                handle.wait = request_wait
                return handle
            coordinator.start = started
            return coordinator
        app.session._coordinator, app.session._coordinator_factory = None, late_clearing
        self.settle_edit(app, app.raw_input, str(self.raw(root, "close me.rawf")))
        self.converter.behaviour = "block"
        app.buttons["convert_coincidence"].invoke()
        self.wait(app, lambda: "Writing to:" in app.convert_status.cget("text"), 10)
        logged, log = [], app.log
        app.log = lambda *messages: (logged.extend(messages), log(*messages))
        app.on_close()
        self.assertTrue(pump(app.root, lambda: app.closed or any("Close incomplete" in m for m in logged), 15), logged)
        self.assertTrue(app.closed, logged)
        self.assertFalse([m for m in logged if "Close incomplete" in m])
        for child in self.converter.children:
            self.assertIsNotNone(child.poll())
        self.assertTrue(wait_threads(("petsys-work", "petsys-shut")))

    @pytest.mark.slow  # ~14 s
    def test_split_duration_hits_and_raw_validation_without_name_parsing(self):
        root, app = self.converter_app()
        self.settle_edit(app, app.raw_input, str(self.raw(root, "plain.rawf")))
        self.assertEqual(self.state(app, "convert_coincidence"), "normal", self.reason(app, "convert_coincidence"))
        cases = [(app.splits, "0", "splits must be an integer in 1..unbounded", "2"),
                 (app.splits, "1.5", "Number of Split Files must be a positive integer", "2"),
                 (app.splits, "-2", "splits must be an integer in 1..unbounded", "2"),
                 (app.convert_duration, "0", "duration_s must be greater than 0", "10"),
                 (app.convert_duration, "-5", "duration_s must be greater than 0", "10"),
                 (app.convert_duration, "abc", "RAW Acquisition Duration (s) must be a positive number", "10"),
                 (app.hit_limit, "0", "hit_limit must be an integer in 1..255", "8"),
                 (app.hit_limit, "256", "hit_limit must be an integer in 1..255", "8"),
                 (app.hit_limit, "x", "Max Hits per Side must be a positive integer", "8")]
        for variable, value, expected, good in cases:
            self.settle_edit(app, variable, value)
            self.assertIn(expected, self.reason(app, "convert_coincidence"), value)
            self.assertEqual(self.state(app, "convert_coincidence"), "disabled", value)
            app.convert()
            self.assertIsNone(app._token)
            self.settle_edit(app, variable, good)
        self.assertEqual(self.converter.launched, [])
        self.assertIn("--splitTime 5.1 s", app.split_plan.cget("text"))
        text = self.run_conversion(app, "convert_coincidence")
        argv = self.converter.launched[-1]
        self.assertEqual(argv[argv.index("--writeMultipleHits") + 1], "8")  # explicit hit limit reaches argv
        self.assertIn("Conversion succeeded", text)
        # RAW selection: the converter reads <prefix>.rawf and its index; nothing else is accepted.
        ldat = root / "data" / "picked.ldat"
        ldat.write_bytes(b"x")
        self.settle_edit(app, app.raw_input, str(ldat))
        self.assertIn("Select the acquisition's .rawf file", self.reason(app, "convert_coincidence"))
        self.assertIn("WARNING: select the acquisition's .rawf file", app.raw_plan.cget("text"))
        lonely = root / "data" / "no index.rawf"
        lonely.write_bytes(b"\x01")
        self.settle_edit(app, app.raw_input, str(lonely))
        self.assertIn("RAW index not found", self.reason(app, "convert_coincidence"))
        self.settle_edit(app, app.raw_input, str(root / "data" / "absent.rawf"))
        self.assertIn("RAW Data File: File not found", self.reason(app, "convert_coincidence"))
        # Compact output needs no converter capability (FR-10: the stock converter).
        self.settle_edit(app, app.raw_input, str(self.raw(root, "plain.rawf")))
        self.assertIn("prerequisites met", self.reason(app, "convert_coincidence"))
        self.assertEqual(len(self.converter.launched), 1)
        self.assertEqual(app.guard.violations, [])

    def test_conversion_stop_and_failures_never_publish_outputs(self):
        root, app = self.converter_app()
        self.settle_edit(app, app.raw_input, str(self.raw(root, "stop me.rawf")))
        self.assertTrue(self.run_conversion(app, "convert_coincidence").startswith("Conversion succeeded"))
        self.assertIsNotNone(app.last_conversion)  # cleared below: a later run's failure never hands these on
        self.converter.behaviour = "block"
        app.buttons["convert_coincidence"].invoke()
        self.wait(app, lambda: bool(self.converter.launched), 10)
        self.wait(app, lambda: "Writing to:" in app.convert_status.cget("text"), 10)
        self.assertEqual({key: self.state(app, key) for key in ("convert_coincidence", "acquire",
                                                                 "offer_calibrate", "stop", "convert_stop")},
                         {"convert_coincidence": "disabled", "acquire": "disabled",
                          "offer_calibrate": "disabled", "stop": "normal", "convert_stop": "normal"})
        app.buttons["convert_stop"].invoke()
        self.assertEqual(self.state(app, "convert_stop"), "disabled")
        self.wait(app, lambda: app._token is None, 15)
        text = app.convert_status.cget("text")
        self.assertTrue(text.startswith("Conversion cancelled"), text)
        self.assertNotIn("Outputs (", text)
        self.assertIsNone(app.last_conversion)
        self.assertEqual(self.state(app, "offer_calibrate"), "disabled")
        for child in self.converter.children:
            self.assertIsNotNone(child.poll())
        for behaviour, expected in (("wrong", "Converter output invalid"), ("fail", "Conversion failed")):
            self.converter.behaviour = behaviour
            text = self.run_conversion(app, "convert_coincidence")
            self.assertTrue(text.startswith("Conversion failed"), text)
            self.assertIn(expected, text + self.log(app))
            self.assertNotIn("Outputs (", text)
            self.assertIsNone(app.last_conversion)
        self.converter.behaviour = "ok"
        self.assertTrue(self.run_conversion(app, "convert_coincidence").startswith("Conversion succeeded"))
        self.assertIsNotNone(app.last_conversion)
        self.assertEqual(app.guard.violations, [])

    def test_exact_ordered_selection_siblings_and_legacy_confirmation(self):
        root, app = self.converter_app()
        folder = root / "legacy files"
        folder.mkdir()
        names = ("acq_coincCompact_12.ldat", "acq_coincCompact_4.ldat", "acq_coincCompact_5.ldat",
                 "acq_coincCompact_extra_2.ldat", "acq_coincCompact2_1.ldat", "acq_coincCompact.ldat",
                 "other_coincCompact_1.ldat", "acq_coincCompact_7.lidx")
        for name in names:
            (folder / name).write_bytes(b"\0" * 8)
        selection = app.selections["calibrate"]
        app.ask_files = lambda **options: [str(folder / "acq_coincCompact_5.ldat")]
        selection.buttons["add"].invoke()
        self.assertEqual([path.name for path in selection.paths],
                         ["acq_coincCompact_4.ldat", "acq_coincCompact_5.ldat", "acq_coincCompact_12.ldat"])
        prompt = app.prompts[-1]
        self.assertIn("2 other split file(s) of exactly acq_coincCompact_<n>.ldat", prompt)
        for name in ("acq_coincCompact_4.ldat", "acq_coincCompact_12.ldat"):
            self.assertIn(name, prompt)
        for name in ("extra", "acq_coincCompact2_1", "other_", "acq_coincCompact.ldat", ".lidx"):
            self.assertNotIn(name, prompt)
        self.assertEqual([selection.listbox.get(i).split("   (")[0] for i in range(3)],
                         [f"{i}. {folder / name}" for i, name in
                          enumerate(("acq_coincCompact_4.ldat", "acq_coincCompact_5.ldat", "acq_coincCompact_12.ldat"), 1)])
        self.wait(app, lambda: app._check_id is None and app.shown_generation == app._awaiting, SETTLE_S)
        self.assertIn("NOT confirmed", selection.summary.cget("text"))
        self.assertIn("Confirm that the listed files are compact coincidence", self.reason(app, "calibrate"))
        self.settle_edit(app, selection.confirmed, True)
        self.assertIn("prerequisites met", self.reason(app, "calibrate"))
        request = app.requests()[0]["calibrate"]
        self.assertEqual([d.path.name for d in request[2]], [p.name for p in selection.paths])
        self.assertEqual({(d.format, d.population) for d in request[2]}, {COMPACT_COINC})
        # Declining the offer keeps exactly the chosen file; another stem's split is offered separately.
        other = app.selections["listmode"]
        app.answer = False
        app.ask_files = lambda **options: [str(folder / "acq_coincCompact_12.ldat")]
        other.buttons["add"].invoke()
        self.assertEqual([path.name for path in other.paths], ["acq_coincCompact_12.ldat"])
        # Reorder/remove: the request follows the shown order exactly.
        selection.listbox.selection_set(2)
        selection.buttons["up"].invoke()
        selection.buttons["up"].invoke()
        self.assertEqual([path.name for path in selection.paths],
                         ["acq_coincCompact_12.ldat", "acq_coincCompact_4.ldat", "acq_coincCompact_5.ldat"])
        selection.buttons["remove"].invoke()
        self.assertEqual([path.name for path in selection.paths], ["acq_coincCompact_4.ldat", "acq_coincCompact_5.ldat"])
        self.wait(app, lambda: app._check_id is None and app.shown_generation == app._awaiting, SETTLE_S)
        self.assertEqual([d.path.name for d in app.requests()[0]["calibrate"][2]],
                         ["acq_coincCompact_4.ldat", "acq_coincCompact_5.ldat"])
        self.assertTrue(selection.confirmed.get())  # removing/reordering keeps the declaration
        app.ask_files = lambda **options: [str(folder / "acq_coincCompact_4.ldat")]
        app.answer = False
        selection.buttons["add"].invoke()
        self.assertIn("Already listed, not added again: acq_coincCompact_4.ldat", self.log(app))
        # FR-10: compact coincidence is the only declared content; every list offers that confirmation only.
        self.assertEqual(set(gui.DECLARED), {"compact_coincidence"})
        for key in ("calibrate", "listmode", "qc_analyze"):
            self.assertEqual(app.selections[key].declared.get(), "compact_coincidence")
            self.assertIn("compact coincidence", app.selections[key].confirm_check.cget("text"))
        qc = app.selections["qc_analyze"]
        app.ask_files = lambda **options: [str(folder / "acq_coincCompact.ldat")]
        prompts = len(app.prompts)
        qc.buttons["add"].invoke()
        self.assertEqual(len(app.prompts), prompts)  # an unsplit name offers nothing
        self.assertEqual([path.name for path in qc.paths], ["acq_coincCompact.ldat"])
        self.settle_edit(app, qc.confirmed, True)
        self.assertIn("prerequisites met", self.reason(app, "qc_analyze"))
        (folder / "acq_coincCompact_5.ldat").unlink()
        selection.refresh()
        self.assertIn("(MISSING)", selection.listbox.get(1))
        self.settle_edit(app, selection.confirmed, False)
        self.settle_edit(app, selection.confirmed, True)
        self.assertIn("Input file not found", self.reason(app, "calibrate"))
        self.assertEqual(self.converter.launched, [])
        self.assertEqual(app.guard.violations, [])

    def test_structure_check_feedback_is_bounded_and_newest_only(self):
        root, app = self.converter_app()
        folder = root / "checked"
        folder.mkdir()
        small = folder / "run_coincCompact_1.ldat"
        records = encode(small, synthetic_sides(self.channels, 60), *COMPACT_COINC)
        big = folder / "big_coincCompact_1.ldat"
        encode(big, synthetic_sides(self.channels, 2 * 12000), *COMPACT_COINC)
        compact = folder / "run_coincCompact_2.ldat"
        compact_records = encode(compact, synthetic_sides(self.channels, 50), *COMPACT_COINC)
        fixed = folder / "legacy_coincFixed_1.ldat"          # a legacy fixed file (FR-10: not a manager input)
        encode(fixed, synthetic_sides(self.channels, 60), *FIXED_COINC)
        unmapped = folder / "unmapped_coincCompact_1.ldat"
        encode(unmapped, [[(1, 1.0, 999999)]] * 4, *COMPACT_COINC)
        selection = app.selections["calibrate"]
        app.answer = False
        app.ask_files = lambda **options: [str(small), str(big)]
        selection.buttons["add"].invoke()
        self.assertEqual([path.name for path in selection.paths], ["big_coincCompact_1.ldat", "run_coincCompact_1.ldat"])
        selection.confirmed.set(True)
        selection.buttons["check"].invoke()
        self.assertEqual(selection.buttons["check"].cget("state"), "disabled")
        self.wait(app, lambda: selection.probe_request is None, 30, "structure check did not finish")
        text = selection.feedback.cget("text")
        self.assertIn("OK big_coincCompact_1.ldat: first 10,000 records pass", text)   # compact: no size total
        self.assertIn(f"OK run_coincCompact_1.ldat: whole file checked, {records:,} records", text)
        self.assertIn("Partial check only: full validation runs before processing", text)
        self.assertIn("Create Energy cal file inputs checked: OK big_coincCompact_1.ldat", self.log(app))
        # A legacy fixed file confirmed as compact fails the structure check.
        qc = app.selections["qc_analyze"]
        app.ask_files = lambda **options: [str(compact), str(fixed)]
        qc.buttons["add"].invoke()
        qc.confirmed.set(True)
        qc.buttons["check"].invoke()
        self.wait(app, lambda: qc.probe_request is None, 30)
        text = qc.feedback.cget("text")
        self.assertIn(f"OK run_coincCompact_2.ldat: whole file checked, {compact_records:,} records", text)
        self.assertIn("FAILED legacy_coincFixed_1.ldat", text)  # fixed bytes declared compact
        self.assertNotIn("Partial check only", text)
        listmode = app.selections["listmode"]
        app.ask_files = lambda **options: [str(unmapped)]
        listmode.buttons["add"].invoke()
        listmode.confirmed.set(True)
        listmode.buttons["check"].invoke()
        self.wait(app, lambda: listmode.probe_request is None, 30)
        self.assertIn("FAILED unmapped_coincCompact_1.ldat", listmode.feedback.cget("text"))
        self.assertIn("record 0, side 0: unmapped channel 999999", listmode.feedback.cget("text"))
        # A result for an older list is never shown.
        selection.buttons["check"].invoke()
        stale = selection.probe_request
        selection.listbox.selection_set(0)
        selection.buttons["remove"].invoke()
        self.assertIsNone(selection.probe_request)
        pump(app.root, timeout=1.0)
        self.assertEqual(selection.feedback.cget("text"), "")
        self.assertIsNotNone(stale)
        # No processing YAML: a reason, never a pass.
        self.settle_edit(app, app.vars["yaml_file"], "")
        qc.buttons["check"].invoke()
        self.wait(app, lambda: qc.probe_request is None, 10)
        self.assertIn("Not checked: Select the processing YAML", qc.feedback.cget("text"))
        self.assertEqual(self.converter.launched, [])
        self.assertEqual(app.guard.violations, [])
        self.assertTrue(wait_threads(("petsys-inputs",)))


@pytest.mark.gui
@pytest.mark.fr("003-FR-10", "003-FR-12", "003-FR-15", "003-FR-16", "003-FR-21")  # spec 003 T20
class CalibrationRouteChecks(ConversionBase):
    """T20: compact input and the positions count reach calibration readiness; COG limits only for >= 2."""

    fixture_prefix = "pm-gui-cal-"

    def test_positions_count_compact_route_and_cog_limits_only_for_positions(self):
        root, app = self.converter_app()
        compact = root / "data" / "run_coincCompact_1.ldat"
        encode(compact, synthetic_sides(self.channels, 40), *COMPACT_COINC)
        selection = app.selections["calibrate"]
        app.answer = False
        app.ask_files = lambda **options: [str(compact)]
        selection.buttons["add"].invoke()
        selection.declared.set("compact_coincidence")
        self.settle_edit(app, selection.confirmed, True)
        self.assertIn("prerequisites met", self.reason(app, "calibrate"))
        self.assertEqual(app.positions.get(), "5")
        self.assertEqual(app.requests()[0]["calibrate"][1].regions, 5)
        self.settle_edit(app, app.vars["cog_limits_file"], "")
        self.assertIn("COG Limits File: Select an existing file", self.reason(app, "calibrate"))
        self.settle_edit(app, app.positions, "1")      # per-slab calibration: no COG limits needed
        self.assertIn("prerequisites met", self.reason(app, "calibrate"))
        self.assertEqual(app.requests()[0]["calibrate"][1].regions, 1)
        self.assertIn("COG Limits File", self.reason(app, "listmode"))   # LM still needs them
        for value, expected in (("0", "regions must be an integer in 1..127"),
                                ("128", "regions must be an integer in 1..127"),
                                ("x", "Positions per Slab must be a positive integer")):
            self.settle_edit(app, app.positions, value)
            self.assertIn(expected, self.reason(app, "calibrate"), value)
            self.assertEqual(self.state(app, "calibrate"), "disabled")
        self.assertIn("Positions per Slab:", texts(app.tabs[2]))
        self.assertEqual(app.guard.violations, [])


@pytest.mark.gui
@pytest.mark.real_data
@pytest.mark.fr("003-FR-10", "003-FR-11", "003-FR-16", "003-FR-22")  # spec 003 T15
class RealSelectionChecks(ConversionBase):
    """Representative Cornell acquisitions (read-only): exact prefixes, missing split 1 and structure.

    Data under PETSYS_DATA_DIR/Cornell/full_system: September Ge68 ``SEPT`` splits 4-5 (no split 1 on
    disk) and January ``JAN`` compact splits 3-8 plus the legacy fixed split 3. Maps: September and
    January through the tracked copies of the operator configs (only ``map_file`` differs); no
    calibration or energy cuts: the counts are the structure check's (first 10,000 records per file,
    unmapped channels).
    """

    fixture_prefix = "pm-gui-real-"
    YAML = REPO / "tests/data/configs/cornell_september.yaml"
    YAML_JAN = REPO / "tests/data/configs/cornell_january.yaml"
    SEPT = "allbrokenboardsRepeared_ge68_coincCompact"
    JAN = "20260119_2NaSourcesAxialSeparated_vBiasCompDiscCalibAdjusted2hits_300s"

    @pytest.fixture(autouse=True)
    def cornell_data(self, real_data_file):
        self.DATA = real_data_file(f"Cornell/full_system/{self.SEPT}_00000005.ldat").parent
        real_data_file(f"Cornell/full_system/{self.JAN}_coincCompact11s_00000005.ldat")

    @pytest.mark.slow  # ~6 s
    def test_real_cornell_selection_and_structure(self):
        before = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in self.DATA.iterdir()}
        root, app = self.converter_app()
        self.settle_edit(app, app.vars["processing_root"], str(REPO))
        self.settle_edit(app, app.vars["yaml_file"], str(self.YAML))
        report = {}

        def check(selection, label, timeout=300):
            started = time.monotonic()
            selection.buttons["check"].invoke()
            self.wait(app, lambda: selection.probe_request is None, timeout)
            text = selection.feedback.cget("text")
            report[label] = (round(time.monotonic() - started, 1), text)
            return text

        # September acquisition, no split 1 on disk: picking _5 offers exactly _4.
        qc = app.selections["qc_analyze"]
        app.ask_files = lambda **options: [str(self.DATA / f"{self.SEPT}_00000005.ldat")]
        qc.buttons["add"].invoke()
        self.assertEqual([path.name for path in qc.paths], [f"{self.SEPT}_0000000{i}.ldat" for i in (4, 5)])
        self.assertIn(f"1 other split file(s) of exactly {self.SEPT}_<n>.ldat", app.prompts[-1])
        self.settle_edit(app, qc.confirmed, True)
        self.assertNotIn("Input LDAT files", self.reason(app, "qc_analyze"))
        text = check(qc, "sept compact / September map")
        self.assertEqual(text.count("first 10,000 records pass"), 2, text)
        # January acquisition: _coincCompact11s siblings only, never _coincFixed11s or other runs.
        app.ask_files = lambda **options: [str(self.DATA / f"{self.JAN}_coincCompact11s_00000005.ldat")]
        qc.buttons["add"].invoke()
        prompt = app.prompts[-1]
        self.assertIn(f"5 other split file(s) of exactly {self.JAN}_coincCompact11s_<n>.ldat", prompt)
        for foreign in ("coincFixed11s", "background", "Source_", "allbroken"):
            self.assertNotIn(foreign, prompt)
        self.assertEqual([path.name for path in qc.paths][2:],
                         [f"{self.JAN}_coincCompact11s_0000000{i}.ldat" for i in range(3, 9)])
        qc.confirmed.set(True)
        text = check(qc, "mixed list / September map")
        self.assertEqual(text.count("first 10,000 records pass"), 2, text)
        self.assertEqual(text.count("unmapped channel"), 6, text)  # January data were taken with another map
        self.settle_edit(app, app.vars["yaml_file"], str(self.YAML_JAN))
        text = check(qc, "mixed list / January map")
        self.assertEqual(text.count("first 10,000 records pass"), 6, text)
        self.assertEqual(text.count("unmapped channel"), 2, text)
        # January compact coincidence for calibration (FR-10: compact only; a legacy fixed file fails as compact).
        calibrate = app.selections["calibrate"]
        app.ask_files = lambda **options: [str(self.DATA / f"{self.JAN}_coincCompact11s_00000008.ldat")]
        calibrate.buttons["add"].invoke()
        self.assertEqual([path.name for path in calibrate.paths],
                         [f"{self.JAN}_coincCompact11s_0000000{i}.ldat" for i in range(3, 9)])
        self.settle_edit(app, calibrate.confirmed, True)
        self.assertNotIn("Input LDAT files", self.reason(app, "calibrate"))
        text = check(calibrate, "jan compact / January map")
        self.assertEqual(text.count("first 10,000 records pass"), 6, text)
        legacy = app.selections["listmode"]
        app.answer = False
        app.ask_files = lambda **options: [str(self.DATA / f"{self.JAN}_coincFixed11s_00000003.ldat")]
        legacy.buttons["add"].invoke()
        self.settle_edit(app, legacy.confirmed, True)
        text = check(legacy, "jan legacy fixed confirmed as compact")
        self.assertEqual(text.count("FAILED"), 1, text)
        legacy.buttons["clear"].invoke()
        lm = app.selections["listmode"]
        app.ask_files = lambda **options: [str(self.DATA / f"{self.JAN}_coincCompact11s_00000003.ldat")]
        app.answer = False
        lm.buttons["add"].invoke()
        lm.declared.set("compact_coincidence")
        self.settle_edit(app, lm.confirmed, True)
        self.assertNotIn("LM generation takes", self.reason(app, "listmode"))      # FR-22: compact LM route
        _, options, inputs = app.requests()[0]["listmode"]
        self.assertEqual(([d.format.value for d in inputs], options.hit_limit), (["compact"], 16))
        for label, (seconds, text) in report.items():
            print(f"\n[{label}] {seconds} s\n{text}")
        after = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in self.DATA.iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(app.guard.violations, [])


@pytest.mark.gui
@pytest.mark.fr("005-FR-4")  # spec 005 T11
class OfferChecks(ConversionBase):
    fixture_prefix = "pm-gui-offer-"

    def converted(self):
        root, app = self.converter_app()
        app.splits.set("2")
        app.convert_duration.set("30")
        self.settle_edit(app, app.raw_input, str(self.raw(root, "acq.rawf")))
        text = self.run_conversion(app, "convert_coincidence")
        self.assertTrue(text.startswith("Conversion succeeded"), text)
        outputs = [Path(line.split(". ", 1)[1].split("   (")[0]) for line in text.splitlines() if line.startswith("  ")]
        self.assertEqual(len(outputs), 2)
        self.started = []
        app.session.start_workflow = lambda *args, **kwargs: self.started.append(args) or 99
        return root, app, outputs

    def lists(self, app):
        return {key: list(selection.paths) for key, selection in app.selections.items()}

    def test_offer_fills_only_its_tab_switches_to_it_and_starts_nothing(self):
        root, app, outputs = self.converted()
        self.assertEqual([app.buttons[f"offer_{key}"].cget("text") for key in OFFERS],
                         ["Calibrate", "Generate LM", "Run QC"])
        self.assertNotIn("use_outputs", app.buttons)
        for key in OFFERS:
            before = self.lists(app)
            app.buttons[f"offer_{key}"].invoke()
            after = self.lists(app)
            self.assertEqual(after[key], outputs)
            self.assertEqual({k: v for k, v in after.items() if k != key}, {k: v for k, v in before.items() if k != key})
            self.assertEqual(app.selections[key].declared.get(), "compact_coincidence")
            self.assertTrue(app.selections[key].confirmed.get())
            self.assertIn("from conversion run", app.selections[key].summary.cget("text"))
            self.assertEqual(app.tabview.get(), gui.TABS[gui.CHECKS[key][2]])
        pump(app.root, timeout=0.3)
        self.assertEqual(self.started, [])
        self.assertIsNone(app._token)
        self.assertEqual(app.guard.violations, [])

    def test_offer_of_a_changed_output_is_refused_and_changes_nothing(self):
        root, app, outputs = self.converted()
        outputs[1].write_bytes(b"truncated")
        before, tab = self.lists(app), app.tabview.get()
        app.buttons["offer_listmode"].invoke()
        self.assertEqual((self.lists(app), app.tabview.get()), (before, tab))
        self.assertIn("Generate LM not offered: Recorded output changed size", self.log(app))
        self.assertIn(outputs[1].name, self.log(app))

    def test_offers_disabled_after_a_failed_conversion(self):
        root, app, outputs = self.converted()
        app.session.start_workflow = type(app.session).start_workflow.__get__(app.session)
        self.assertEqual({self.state(app, f"offer_{key}") for key in OFFERS}, {"normal"})
        self.converter.behaviour = "fail"
        self.assertTrue(self.run_conversion(app, "convert_coincidence").startswith("Conversion failed"))
        self.assertEqual({self.state(app, f"offer_{key}") for key in OFFERS}, {"disabled"})

    def test_offer_keeps_the_confirmation_and_validation_gates(self):
        root, app, outputs = self.converted()
        app.buttons["offer_calibrate"].invoke()
        self.wait(app, lambda: app._check_id is None and app.shown_generation == app._awaiting, SETTLE_S)
        self.assertIn("prerequisites met", self.reason(app, "calibrate"))
        self.settle_edit(app, app.selections["calibrate"].confirmed, False)
        self.assertIn("Confirm that the listed files are compact coincidence", self.reason(app, "calibrate"))
        self.assertEqual(self.state(app, "calibrate"), "disabled")
        self.settle_edit(app, app.selections["calibrate"].confirmed, True)
        self.settle_edit(app, app.positions, "zero")
        self.assertEqual(self.state(app, "calibrate"), "disabled")
        app.calibrate()
        self.assertEqual(self.started, [])
