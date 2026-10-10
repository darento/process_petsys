"""Calibration, LM, offline/live QC and the complete pipeline through real windows (spec 003 T16;
spec 007 T15).

Moved from scripts/petsys_manager_gui_check.py --processing. The real session, coordinator,
runner and acquisition service run against the ``ToolWorld`` backend (fake tools, fake or real
``src.cornell.cli`` children on seeded LM/QC fixtures). ``gui``: skipped without a display.
"""

import hashlib
import json
from pathlib import Path
import re
from dataclasses import replace

import pytest
import yaml

from exe_programs import petsys_manager_gui as gui
from helpers import REPO
from manager_gui_helpers import (ConversionBase, FAST, GUIBase, Hardware, RunPanelBase, SAFETY, SETTLE_S, SHM,
                                 TOOLS, pump, world)
from manager_helpers import (CALIBRATION, METADATA, SPLITS, FakeChild, ListmodeFixtures, QCFixtures, ToolWorld,
                             recorded_run)
from src.petsys_manager.acquisition import DaqdState
from src.petsys_manager.artifacts import RunStore
from src.petsys_manager.contracts import Action, Artifact, CommandResult, ResultStatus, to_plain
from src.petsys_manager.session import WorkflowResult
from src.petsys_manager.settings import MachineProfile, ToolCapabilities, load_profile, save_profile
from src.petsys_manager.workflow import StageOutcome, WorkflowCoordinator, WorkflowOutcome


class ProcessingWorld(ToolWorld):
    """The T12 check backend (fake or real ``src.cornell.cli`` children) plus ``block``: runs until STOP."""

    def launch(self, command):
        argv = command.argv
        if self.faults.get(command.identity.stage_id) == "block" and Path(argv[0]).name != "set_bias":
            cli = argv[1:4] == ("-u", "-m", "src.cornell.cli")
            self.launched.append((command.identity.stage_id, f"cli:{argv[4]}" if cli else Path(argv[0]).name, argv))
            return FakeChild(None, stdout=b"")
        return super().launch(command)


@pytest.mark.gui
@pytest.mark.fr("003-FR-1", "003-FR-5", "003-FR-11", "003-FR-12", "003-FR-13", "003-FR-14", "003-FR-16")  # spec 003 T16
class ProcessingChecks(ConversionBase):
    """Manual calibration/LM/offline QC, live QC and the complete pipeline through real windows.

    The REAL session, WorkflowCoordinator, CommandRunner and AcquisitionService run against
    the T12 tool backend: fake acquisition/converter children, and fake or real CLI children
    on the seeded T9/T10 LM and QC fixtures. DAQD/initialization use the T14 fake hardware.
    """

    fixture_prefix = "pm-gui-proc-"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        assets = cls.output / "assets"
        lm = ListmodeFixtures.at(assets / "lm")
        descriptors, maps = lm.inputs(names=("acq_coinc_1.ldat", "acq_coinc_2.ldat"), count=1500, random_slabs=False)
        # The manager processes compact coincidence only (FR-10): compact twins of the seeded records.
        descriptors = lm.compact_twins(descriptors, count=1500, random_slabs=False)
        cls.lm_root, cls.lm_files = lm.root, tuple(d.path for d in descriptors)
        cls.files = {"cog_limits_file": str(maps["cog_limits"].path), "doi_limits_file": str(maps["doi_limits"].path),
                     "calibration_file": str(maps["calibration"].path), "pair_map_file": str(maps["pairs"].path),
                     "region_map_file": str(maps["regions"].path)}
        qc = QCFixtures.at(assets / "qc")
        cls.qc_root = qc.root
        cls.compact = tuple(d.path for d in qc.inputs(count=600, names=("qc_with_source_coincCompact_1.ldat",
                                                                         "qc_with_source_coincCompact_2.ldat")))

    def processing_app(self, kind="lm", *, real_cli=False, **changes):
        root = self.root = self.base
        for name in ("tools", "data", "encal", "rep", "lm", "dev"):
            (root / name).mkdir()
        for tool in TOOLS:
            (root / "tools" / tool).write_text("marker", encoding="utf-8")
        for name in ("card0", "card1"):
            (root / "dev" / name).write_text("marker", encoding="utf-8")
        (root / "daq.ini").write_text("[marker]\n", encoding="utf-8")
        self.processing_root = self.lm_root if kind == "lm" else self.qc_root
        values = dict(petsys_folder=str(root / "tools"), processing_root=str(self.processing_root),
                      ini_file=str(root / "daq.ini"), yaml_file=str(self.processing_root / "configs/processing.yaml"),
                      data_dir=str(root / "data"), calibration_dir=str(root / "encal"), report_dir=str(root / "rep"),
                      lm_dir=str(root / "lm"), cards=(str(root / "dev/card0"), str(root / "dev/card1")),
                      shared_memory_path=SHM, safety=SAFETY, lm_metadata=METADATA,
                      capabilities=ToolCapabilities(),
                      **(self.files if kind == "lm" else {}))
        values.update(changes)
        self.profile_path = save_profile(MachineProfile(**values), root / "manager profile.yaml")
        self.world = ProcessingWorld(self, sources=self.lm_files if kind == "lm" else self.compact, real_cli=real_cli)
        self.hardware, self.coordinator = Hardware(), None

        def coordinator(**sinks):
            self.coordinator = WorkflowCoordinator(backend=self.world, policy=FAST, **sinks)
            return self.coordinator
        app = self.open(self.profile_path, repo_root=REPO, daqd_factory=self.hardware.daqd_factory,
                        coordinator_factory=coordinator)
        app.ask_yes_no = lambda title, message: False   # the listed files are exactly the chosen ones
        app.statuses = []
        for label in (app.cal_status, app.lm_status, app.qc_status, app.pipeline_status):
            label.configure = self.recorder(app, label.configure)
        return app

    @staticmethod
    def recorder(app, configure):
        def record(**options):
            if "text" in options:
                app.statuses.append(options["text"])
            return configure(**options)
        return record

    def settled(self, app):
        self.wait(app, lambda: app._check_id is None and app._awaiting is not None
                  and app.shown_generation == app._awaiting, SETTLE_S, "readiness never settled")

    def select(self, app, key, paths, declared):
        selection = app.selections[key]
        app.ask_files = lambda **options: [str(path) for path in paths]
        selection.buttons["add"].invoke()
        selection.declared.set(declared)
        self.settle_edit(app, selection.confirmed, True)
        return selection

    def initialized(self, app):
        app.buttons["daqd"].toggle()
        self.wait(app, lambda: app.daqd_status is not None and app.daqd_status.state == DaqdState.READY)
        app.buttons["initialize"].invoke()
        self.wait(app, lambda: not app._init_pending and app.daqd_status.initialized, message="not initialized")

    def finish(self, app, key, label, timeout=120.0):
        self.assertEqual(self.state(app, key), "normal", self.reason(app, key))
        app.buttons[key].invoke()
        self.assertIsNotNone(app._token)
        self.wait(app, lambda: app._token is None, timeout, f"{key} did not finish")
        return label.cget("text")

    def run_root(self, text):
        return Path(re.search(r"^Run directory: (.+)$", text, re.M).group(1))

    def requests(self, run_root):
        found = {}
        for path in run_root.rglob("request.json"):
            request = json.loads(path.read_text(encoding="utf-8"))
            found[request["action"]] = request
        return found

    def digests(self):
        """Profile settings, processing YAML, map, limits, calibration and input LDAT files."""
        paths = [path for path in self.processing_root.rglob("*") if path.is_file() and path != self.profile_path]
        found = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        found[str(self.profile_path)] = self.settings()
        return found

    def settings(self):
        """The profile file's settings: spec 005 FR-6 writes ``last_folders`` alone as soon as a dialog returns."""
        return replace(load_profile(self.profile_path), last_folders={})

    def ui_state(self, app):
        values = {f"profile.{name}": var.get() for name, var in app.vars.items()}
        values.update({f"lm.{name}": var.get() for name, var in app.lm_vars.items()})
        for name in ("acq_time", "splits", "positions", "hit_limit", "convert_duration",
                     "raw_input", "lm_debug", "qc_source", "qc_plots", "qc_slabs"):
            values[name] = getattr(app, name).get()
        values["lists"] = {key: [str(path) for path in selection.paths] for key, selection in app.selections.items()}
        return values, replace(app.session.profile, last_folders={})   # dialog folders: spec 005 FR-6

    def launched_since(self, count):
        return [(stage, tool) for stage, tool, _ in self.world.launched[count:]]

    def argv(self, tool, since=0):
        return next(argv for _, name, argv in self.world.launched[since:] if name == tool)

    @pytest.mark.slow  # ~11 s
    @pytest.mark.fr("003-FR-21")
    def test_manual_calibration_and_lm_show_and_use_recorded_artifacts(self):
        app = self.processing_app("lm", real_cli=True)
        before, saved = self.digests(), self.settings()
        selection = self.select(app, "calibrate", self.lm_files, "compact_coincidence")
        selection.listbox.selection_set(0)
        selection.buttons["down"].invoke()                         # the operator's order: 2, then 1
        self.settled(app)
        self.assertEqual(self.reason(app, "calibrate"), "Create Energy cal file: prerequisites met")
        text = self.finish(app, "calibrate", app.cal_status, 600)
        self.assertTrue(text.startswith("Energy calibration succeeded"), text)
        run_root = self.run_root(text)
        self.assertEqual(run_root.parent, self.base / "encal")
        request = self.requests(run_root)["calibrate"]
        self.assertEqual([item["path"] for item in request["inputs"]], [str(path) for path in reversed(self.lm_files)])
        self.assertEqual((request["options"]["positions"], request["files"]["cog_limits"]),
                         (5, self.files["cog_limits_file"]))
        encal = Path(request["outputs"]["encal"])
        self.assertTrue(encal.is_file() and encal.is_relative_to(run_root), encal)
        self.assertEqual(encal.name, "acq_coinc_position_5regions.encal")   # reference base: split number dropped (T32)
        for line in (f"Energy cal file: {encal}", f"Calibration provenance: {request['outputs']['sidecar']}",
                     f"Fit status per key: {request['outputs']['status']}", f"Summary plot: {request['outputs']['plot']}",
                     "5 position regions per slab;", "mapped keys have a factor (",
                     "Event limit (target): K ", " x T 3,000 = ", " kept sides, ", "used per file: ",
                     "Sides per key (", "below T 3,000", "-event fit minimum",
                     "Each file read once: ", "(budget 8,192 MB)"):
            self.assertIn(line, text)
        self.assertNotIn("request.json", text)
        self.assertTrue(any(": fits " in status and " keys" in status for status in app.statuses))   # T25.4 progress
        self.assertTrue(any("(read)" in status for status in app.statuses))
        self.assertEqual(app.vars["calibration_file"].get(), self.files["calibration_file"])   # never applied by itself
        self.assertEqual(self.state(app, "offer_lm_calibration"), "normal")
        self.assertEqual(app.buttons["offer_lm_calibration"].cget("text"), "Generate LM with this calibration")
        app.buttons["offer_lm_calibration"].invoke()
        self.settled(app)
        self.assertEqual(app.tabview.get(), gui.TABS[3])                     # spec 005: switched to LM, not started
        self.assertIsNone(app._token)
        self.assertEqual(app.vars["calibration_file"].get(), str(encal))
        self.assertIn("unsaved edits", app.profile_status.cget("text"))
        self.assertIn(f"System Energy cal file set to {encal} from run {run_root.name}", self.log(app))
        # LM from the operator-chosen recorded calibration; its regions come from that file.
        self.select(app, "listmode", self.lm_files, "compact_coincidence")
        self.settle_edit(app, app.lm_debug, False)
        text = self.finish(app, "listmode", app.lm_status, 600)
        self.assertTrue(text.startswith("LM generation succeeded"), text)
        run_root = self.run_root(text)
        self.assertEqual(run_root.parent, self.base / "lm")
        request = self.requests(run_root)["listmode"]
        self.assertEqual(request["files"]["calibration"], str(encal))
        self.assertEqual((request["options"]["num_regions"], request["options"]["debug"]), (5, False))
        self.assertEqual(request["options"]["metadata"], to_plain(METADATA))
        lm_file = next(path for path in run_root.rglob("*.lm"))
        self.assertIn(f"LM file: {lm_file}", text)
        self.assertGreater(int(re.search(r"([\d,]+) LM records written", text).group(1).replace(",", "")), 0)
        self.assertTrue(any("LM records written" in status and "file " in status for status in app.statuses))
        # Positions 1: per-slab calibration without COG limits, named as the reference.
        self.world.real_cli = False
        self.settle_edit(app, app.positions, "1")
        text = self.finish(app, "calibrate", app.cal_status)
        request = self.requests(self.run_root(text))["calibrate"]
        self.assertEqual((request["options"]["positions"], request["files"]["cog_limits"]), (1, None))
        self.assertTrue(request["outputs"]["encal"].endswith("acq_coinc_resolved.encal"))
        self.assertEqual(self.digests(), before)
        self.assertEqual(self.settings(), saved)
        self.assertEqual(app.guard.violations, [])

    @pytest.mark.slow  # ~6 s
    @pytest.mark.fr("003-FR-15", "003-FR-21")
    def test_calibration_event_limit_shown_before_the_run_and_reference_selectable(self):
        """FR-21 (T25.2): target mode by default, N and the share shown from the selected map before the run;
        T is a profile value; reference mode is one choice away; the pipeline plan names its limit."""
        from src.cornell import calibration as cal
        from src.cornell.inputs import load_processing_config
        app = self.processing_app("lm")
        mapping = load_processing_config(self.processing_root / "configs/processing.yaml",
                                         processing_root=self.processing_root, action="calibrate").mapping
        self.select(app, "calibrate", self.lm_files, "compact_coincidence")
        self.settled(app)

        from src.cornell.parallel import resolve_workers

        def plan(workers=0, **changes):
            values = dict(limit_mode="target", event_limit=10_000_000, target_per_key=3000)
            values.update(changes)
            return gui.limit_plan_text(cal.event_limit_plan(mapping, 5, len(self.lm_files), **values)
                                       | {"workers": resolve_workers(workers)})
        self.assertEqual(app.limit_plan.cget("text"), plan())
        self.assertIn(f"K {len(cal._mapped_keys(mapping, 1)):,} keys x P 5 x T 3,000 = ", plan())
        self.assertEqual(app.target_per_key.get(), "3000")                       # the profile's T
        self.assertEqual(app.requests()[0]["calibrate"][1].calibration_limit_mode, "target")
        self.assertIn("Calibration event limit (target): ", app.pipeline_plan.cget("text"))
        self.assertIn("over 1 split(s)", app.pipeline_plan.cget("text"))         # the pipeline's own file count
        app.target_per_key.set("x")                                              # a profile error: no check runs
        self.wait(app, lambda: app._check_id is None and app._awaiting is None, SETTLE_S)
        self.assertIn("Profile: Target sides per histogram (T) must be a positive integer",
                      self.reason(app, "calibrate"))
        self.assertEqual(app.limit_plan.cget("text"), "Event limit: shown when the calibration prerequisites are met")
        self.settle_edit(app, app.target_per_key, "500")
        self.assertEqual(app.limit_plan.cget("text"), plan(target_per_key=500))
        self.assertEqual(app.profile_from_ui().limits.calibration_target_per_key, 500)
        self.assertIn("unsaved edits", app.profile_status.cget("text"))
        self.settle_edit(app, app.limit_mode, "reference")
        self.assertEqual(app.limit_plan.cget("text"), plan(limit_mode="reference", target_per_key=500))
        self.assertTrue(app.limit_plan.cget("text").startswith(
            "Event limit (reference): 10,000,000 passing coincidences per file, 2 file(s)"))
        self.assertEqual(app.requests()[0]["calibrate"][1].calibration_limit_mode, "reference")
        self.assertIn("Calibration event limit (reference): ", app.pipeline_plan.cget("text"))
        # FR-15: the worker count is a profile field and the plan names what the run will use.
        self.assertEqual(app.workers.get(), "0")
        self.settle_edit(app, app.workers, "1")
        self.assertIn("1 worker (no parallel processing)", app.limit_plan.cget("text"))
        self.assertEqual(app.profile_from_ui().limits.workers, 1)
        self.settle_edit(app, app.workers, "3")
        self.assertEqual(app.limit_plan.cget("text"), plan(workers=3, limit_mode="reference", target_per_key=500))
        app.workers.set("x")
        self.wait(app, lambda: app._check_id is None and app._awaiting is None, SETTLE_S)
        self.assertIn("Profile: Workers must be a whole number (0 = automatic)", self.reason(app, "calibrate"))
        app.workers.set("-1")
        self.wait(app, lambda: app._check_id is None and app._awaiting is None, SETTLE_S)
        self.assertIn("workers must be an integer in 0..", self.reason(app, "calibrate"))
        self.settle_edit(app, app.workers, "0")
        self.settle_edit(app, app.positions, "0")                                # not ready: no plan shown
        self.assertEqual(app.limit_plan.cget("text"), "Event limit: shown when the calibration prerequisites are met")
        self.assertEqual(app.guard.violations, [])

    @pytest.mark.slow  # ~5 s
    def test_offline_and_live_qc_presets_slabs_need_plots_and_findings_are_not_verdicts(self):
        app = self.processing_app("qc")
        self.select(app, "qc_analyze", self.compact, "compact_coincidence")
        self.settle_edit(app, app.qc_slabs, True)        # set without plots: never requested
        text = self.finish(app, "qc_analyze", app.qc_status)
        self.assertTrue(text.startswith("Offline QC succeeded"), text)
        run_root = self.run_root(text)
        self.assertEqual(run_root.parent, self.base / "rep")
        options = self.requests(run_root)["qc"]["options"]
        self.assertEqual((options["plots"], options["slabs"], options["source_mode"], options["acquisition_time_s"]),
                         (False, False, None, None))
        results = Path(re.search(r"Results directory: (.+)$", text, re.M).group(1))
        self.assertTrue(results.is_dir() and results.is_relative_to(run_root), results)
        self.assertEqual(results, Path(self.requests(run_root)["qc"]["outputs"]["directory"]))
        self.assertIn("not a detector verdict", text)
        self.assertIn("minimodules without hits: 3", text)
        self.assertNotIn("PASS", text)
        self.assertEqual(self.state(app, "qc"), "disabled")
        self.assertIn("prerequisites met (start DAQD and initialize the system to enable)", self.reason(app, "qc"))
        self.initialized(app)
        self.assertEqual(self.reason(app, "qc"), "Run Quality Control: prerequisites met")
        self.settle_edit(app, app.splits, "2")
        for source, seconds, plots, slabs in (("with", 60.0, True, True), ("without", 180.0, True, False)):
            with self.subTest(source):
                app.qc_source.set(source)
                app.qc_plots.set(plots)
                app._plots_toggled()
                self.settle_edit(app, app.qc_slabs, slabs)
                self.assertIn(f"acquire {seconds:g} s {source} source -> compact coincidence conversion, 2 split(s)",
                              app.qc_plan.cget("text"))
                count = len(self.world.launched)
                text = self.finish(app, "qc", app.qc_status)
                self.assertTrue(text.startswith("Quality control succeeded"), text)
                self.assertEqual(self.launched_since(count), [("acquisition", "acquire_sipm_data"),
                                                              ("conversion", "convert_raw_to_coincidence"),
                                                              ("qc", "cli:qc")])
                acquire, convert = self.argv("acquire_sipm_data", count), self.argv("convert_raw_to_coincidence", count)
                self.assertEqual(float(acquire[acquire.index("--time") + 1]), seconds)
                self.assertTrue(Path(acquire[acquire.index("-o") + 1]).name.startswith(f"acquisition_qc_{source}_source"))
                self.assertIn("--writeBinaryCompact", convert)
                run_root = self.run_root(text)
                self.assertEqual(run_root.parent, self.base / "data")
                from src.cornell.parallel import resolve_workers
                self.assertEqual(self.requests(run_root)["qc"]["options"],
                                 {"plots": plots, "slabs": slabs, "source_mode": source, "acquisition_time_s": seconds,
                                  "pair_limit": 1_000_001, "in_place": True, "report_title": run_root.name,
                                  "qc_seed": 0, "workers": resolve_workers(0)})
                for line in ("Acquisition: succeeded in ", "(1 attempt(s))", "Conversion: succeeded in ",
                             "(2 structure-checked compact coincidence file(s))",
                             "QC analysis: succeeded", "Results directory: ", "not a detector verdict"):
                    self.assertIn(line, text)
                self.assertNotIn("PASS", text)
        # A failed or stopped QC is never completed QC.
        self.world.faults = {"qc": "nonzero"}
        text = self.finish(app, "qc", app.qc_status)
        self.assertTrue(text.startswith("Quality control failed"), text)
        for absent in ("Findings", "Results directory", "PASS", "succeeded:"):
            self.assertNotIn(absent, text)
        self.world.faults = {"qc": "block"}
        app.buttons["qc_analyze"].invoke()
        self.wait(app, lambda: any(stage == "qc" for stage, _, _ in self.world.launched[-1:]) and
                  self.state(app, "qc_stop") == "normal")
        app.buttons["qc_stop"].invoke()
        self.wait(app, lambda: app._token is None, 20)
        text = app.qc_status.cget("text")
        self.assertTrue(text.startswith("Offline QC cancelled"), text)
        self.assertNotIn("Findings", text)
        self.wait(app, lambda: self.state(app, "qc") == "normal")    # controls return
        self.assertEqual(app.guard.violations, [])

    @pytest.mark.fr("003-FR-9")
    def test_pipeline_consumes_recorded_artifacts_and_leaves_settings_unchanged(self):
        app = self.processing_app("lm")
        self.assertFalse(hasattr(app, "pipeline_format"))   # FR-10: the pipeline always converts to compact
        app.splits.set("2")
        app.positions.set("3")
        self.settle_edit(app, app.lm_debug, False)
        app.acq_name.set("F18 Run1")                       # T29: the acquisition name is checked per request
        requests, issues = app.requests()
        self.assertEqual({key for key in ("acquire", "pipeline", "qc") if key in requests}, set())
        self.assertTrue(all("Acquisition name" in issues[key][0].message for key in ("acquire", "pipeline", "qc")))
        app.acq_name.set("F18_Run1")
        _, options, _ = app.requests()[0]["pipeline"]
        self.assertEqual((options.output_format.value, options.population.value, options.hit_limit),
                         ("compact", "coincidence", 16))
        self.assertEqual([app.requests()[0][key][1].acquisition_name for key in ("acquire", "pipeline", "qc")],
                         ["F18_Run1"] * 3)
        self.assertIn("One new run folder F18_Run1_pipeline-P3_<date>_<time>", app.pipeline_plan.cget("text"))
        self.assertIn("F18_Run1_qc-with-source_<date>_<time>", app.qc_plan.cget("text"))
        self.assertEqual(self.state(app, "pipeline"), "disabled")
        self.assertIn("(start DAQD and initialize the system to enable)", self.reason(app, "pipeline"))
        self.assertIn("compact coincidence conversion, 2 split(s), max 16 hits per side", app.pipeline_plan.cget("text"))
        self.assertIn("3 position(s) per slab", app.pipeline_plan.cget("text"))
        self.assertIn("header acquisition/measurement time 10 s (Acq. Time)", app.pipeline_plan.cget("text"))
        self.initialized(app)
        before, saved, ui = self.digests(), self.settings(), self.ui_state(app)
        text = self.finish(app, "pipeline", app.pipeline_status)
        self.assertTrue(text.startswith("Complete pipeline succeeded"), text)
        self.assertEqual(self.launched_since(0), [("acquisition", "acquire_sipm_data"),
                                                  ("conversion", "convert_raw_to_coincidence"),
                                                  ("calibration", "cli:calibrate"), ("listmode", "cli:listmode")])
        acquire, convert = self.argv("acquire_sipm_data"), self.argv("convert_raw_to_coincidence")
        self.assertEqual(convert[convert.index("-i") + 1], acquire[acquire.index("-o") + 1])
        self.assertIn("--writeBinaryCompact", convert)
        self.assertNotIn("--writeBinaryFixed", convert)
        self.assertEqual(convert[convert.index("--splitTime") + 1], str(10.0 / 2 + 0.1))
        run_root = self.run_root(text)
        self.assertEqual(run_root.parent, self.base / "data")
        self.assertRegex(run_root.name, r"\AF18_Run1_pipeline-P3_\d{4}-\d\d-\d\d_\d{4}(_\d+)?\Z")
        self.assertEqual(Path(acquire[acquire.index("-o") + 1]), run_root / "1_acquisition/attempt-1/F18_Run1")
        requests = self.requests(run_root)
        outputs = sorted(path for path in (run_root / "2_conversion").glob("*.ldat")
                         if path.stat().st_size)
        self.assertEqual([item["path"] for item in requests["calibrate"]["inputs"]], [str(path) for path in outputs])
        self.assertEqual([item["path"] for item in requests["listmode"]["inputs"]], [str(path) for path in outputs])
        self.assertEqual((requests["calibrate"]["options"]["positions"], requests["calibrate"]["files"]["cog_limits"]),
                         (3, self.files["cog_limits_file"]))
        encal = requests["calibrate"]["outputs"]["encal"]
        self.assertEqual(requests["listmode"]["files"]["calibration"], encal)
        self.assertNotEqual(encal, self.files["calibration_file"])
        self.assertEqual((requests["listmode"]["options"]["num_regions"], requests["listmode"]["options"]["debug"]),
                         (3, False))
        # Header times are this run's Acq. Time (10 s), not the profile's 300.5/299.0 s (owner, 2026-10-02).
        self.assertEqual(requests["listmode"]["options"]["metadata"],
                         to_plain(replace(METADATA, acquisition_time_s=10.0, measurement_time_s=10.0)))
        for line in [f"  {index}. {path}" for index, path in enumerate(outputs, 1)] + [
                f"Energy cal file: {encal}", "LM file: ", "Acquisition: succeeded in ", "(1 attempt(s))",
                "(2 structure-checked compact coincidence file(s))", "Energy calibration: succeeded in ",
                "LM generation: succeeded in "]:
            self.assertIn(line, text)
        self.assertEqual(len(re.findall(r"\n  Elapsed: \d", text)), 4)      # FR-1: one per stage (T25.1)
        joined = "\n".join(app.statuses)
        for step in ("Step 2/4 (Conversion): ", "Step 3/4 (Energy calibration): ", "Step 4/4 (LM generation): "):
            self.assertIn(step, joined)
        # Nothing persistent or selected changed; the new calibration is offered, not applied.
        self.assertEqual(self.ui_state(app), ui)
        self.assertEqual(self.digests(), before)
        self.assertEqual(self.settings(), saved)
        self.assertEqual(load_profile(self.profile_path), app.session.profile)
        self.assertEqual(app.last_calibration[1], Path(encal))
        self.assertEqual(self.state(app, "offer_lm_calibration"), "normal")
        self.wait(app, lambda: self.state(app, "pipeline") == "normal")
        self.assertEqual(app.guard.violations, [])

    def test_injected_failures_and_stop_never_become_success(self):
        app = self.processing_app("lm")
        self.settle_edit(app, app.splits, "2")
        self.initialized(app)
        before, saved, ui = self.digests(), self.settings(), self.ui_state(app)
        stages = ("acquisition", "conversion", "calibration", "listmode")
        for stage, fault, status in (("conversion", "nonzero", "failed"), ("calibration", "invalid", "failed"),
                                     ("calibration", "stale", "failed"), ("listmode", "nonzero", "failed"),
                                     ("acquisition", "block", "cancelled"), ("calibration", "block", "cancelled")):
            with self.subTest(stage=stage, fault=fault):
                self.world.faults = {stage: fault}
                count = len(self.world.launched)
                if fault == "block":
                    app.buttons["pipeline"].invoke()
                    self.wait(app, lambda: any(item == stage for item, _, _ in self.world.launched[count:]), 20)
                    app.buttons["stop"].invoke()
                    self.wait(app, lambda: app._token is None, 30, "pipeline did not stop")
                    text = app.pipeline_status.cget("text")
                else:
                    text = self.finish(app, "pipeline", app.pipeline_status)
                self.assertTrue(text.startswith(f"Complete pipeline {status}"), text)
                self.assertIn(f"{STAGE_LABELS[stage]}: {status}", text)
                later = stages[stages.index(stage) + 1:]
                self.assertFalse([item for item, tool in self.launched_since(count) if item in later and
                                  tool != "set_bias"])
                calibrated = stages.index(stage) > stages.index("calibration")
                self.assertEqual(app.last_calibration is not None, calibrated)
                if stage == "acquisition":
                    self.assertEqual(self.launched_since(count)[-1], ("acquisition", "set_bias"))   # bias off
                self.wait(app, lambda: self.state(app, "pipeline") == "normal", 10, "controls did not return")
        self.world.faults = {"calibration": "nonzero"}      # manual calibration failure
        self.select(app, "calibrate", self.lm_files, "compact_coincidence")
        text = self.finish(app, "calibrate", app.cal_status)
        self.assertTrue(text.startswith("Energy calibration failed"), text)
        self.assertNotIn("Energy cal file:", text)
        self.assertEqual(self.state(app, "offer_lm_calibration"), "disabled")
        values, profile = self.ui_state(app)
        values["lists"]["calibrate"] = []
        self.assertEqual((values, profile), ui)                         # only the calibration list was edited
        self.assertEqual(self.digests(), before)
        self.assertEqual(self.settings(), saved)
        self.assertEqual(app.guard.violations, [])

    @pytest.mark.slow  # ~13 s
    def test_incomplete_metadata_options_and_prerequisites_cannot_start(self):
        app = self.processing_app("lm")
        self.select(app, "listmode", self.lm_files, "compact_coincidence")
        self.initialized(app)
        self.assertEqual(app.profile_from_ui().lm_metadata, METADATA)            # shown values round-trip
        self.assertEqual(app.lm_vars["acquisition_time_s"].get(), "300.5")
        for key in ("listmode", "pipeline"):
            self.assertEqual(self.state(app, key), "normal", self.reason(app, key))
        self.settle_edit(app, app.lm_vars["isotope"], "")
        for key in ("listmode", "pipeline"):
            self.assertIn("LM metadata missing from the profile: isotope (LM File Generation tab)",
                          self.reason(app, key))
            self.assertEqual(self.state(app, key), "disabled")
        self.assertNotIn("LM metadata", self.reason(app, "qc"))
        count = len(self.world.launched)
        app.generate_lm()                         # the action itself refuses, not only the button
        app.run_pipeline()
        self.assertIsNone(app._token)
        self.assertIn("Generate LM File not started: prerequisites not met", self.log(app))
        self.settle_edit(app, app.lm_vars["isotope"], "F18")
        for name in ("acquisition_time_s", "measurement_time_s"):    # the pipeline supplies both header times
            self.settle_edit(app, app.lm_vars[name], "")
        self.assertIn("LM metadata missing from the profile: acquisition_time_s, measurement_time_s "
                      "(LM File Generation tab)", self.reason(app, "listmode"))
        self.assertEqual(self.state(app, "listmode"), "disabled")
        self.assertNotIn("LM metadata", self.reason(app, "pipeline"))
        self.assertEqual(self.state(app, "pipeline"), "normal", self.reason(app, "pipeline"))
        for name in ("acquisition_time_s", "measurement_time_s"):
            self.settle_edit(app, app.lm_vars[name], format(getattr(METADATA, name), ".15g"))
        for name, value, reason in (("module_number", "x", "LM Module number must be an integer"),
                                    ("detector_pixels_x", "200", "detector_pixels_x must be an integer in 1..127"),
                                    ("ring_distance_mm", "-1", "ring_distance_mm must be greater than 0")):
            app.lm_vars[name].set(value)
            self.wait(app, lambda: app._check_id is None and app._awaiting is None, SETTLE_S)
            for key, text in self.reasons(app).items():
                self.assertIn(f"Profile: {reason}", text, key)
            self.assertEqual(self.state(app, "listmode"), "disabled")
            self.settle_edit(app, app.lm_vars[name], format(getattr(METADATA, name), ".15g"))
        self.assertEqual(app.profile_from_ui().lm_metadata, METADATA)
        for variable, value, keys, reason in ((app.positions, "0", ("pipeline", "calibrate"), "regions must be"),
                                              (app.splits, "x", ("pipeline", "qc"),
                                               "Number of Split Files must be a positive integer"),
                                              (app.hit_limit, "300", ("pipeline", "qc"), "hit_limit must be"),
                                              (app.acq_time, "0", ("pipeline", "acquire"), "duration_s must be")):
            self.settle_edit(app, variable, value)
            for key in keys:
                self.assertIn(reason, self.reason(app, key), key)
                self.assertEqual(self.state(app, key), "disabled", key)
            self.settle_edit(app, variable, {"0": "5", "x": "1", "300": "16"}[value] if variable is not app.acq_time
                             else "10")
        app.selections["listmode"].confirmed.set(False)           # unconfirmed inputs block LM
        self.settled(app)
        self.assertIn("Confirm that the listed files are compact coincidence", self.reason(app, "listmode"))
        app.generate_lm()
        self.assertIsNone(app._token)
        self.assertEqual(self.world.launched[count:], [])
        for name in ("data", "lm", "encal", "rep"):
            self.assertEqual(list((self.base / name).iterdir()), [], name)   # no run was created
        self.assertEqual(app.guard.violations, [])


STAGE_LABELS = getattr(gui, "STAGE_TEXT", {})


@pytest.mark.gui
@pytest.mark.fr("005-FR-4")  # spec 005 T11
class CalibrationOfferChecks(RunPanelBase):
    fixture_prefix = "pm-gui-cal-offer-"

    def calibration_run(self, status=ResultStatus.SUCCEEDED):
        """A recorded calibration run and the outcome the window receives for it."""
        destination = self.base / "cal"
        destination.mkdir()
        store = RunStore.reserve(destination, {"action": "calibrate"}, name="cal", stages=["calibration"])
        attempt = store.reserve_attempt("calibration", attempt_id="attempt-1")
        artifacts = []
        for name, kind in (("x.encal", "encal"), ("x_plot.png", "calibration_plot")):
            (attempt.directory / name).write_bytes(name.encode())
            artifacts.append(Artifact(attempt.directory / name, kind))
        ok = status == ResultStatus.SUCCEEDED
        store.finish_attempt(attempt, CommandResult(attempt.identity, status, 0 if ok else 1, "done", artifacts,
                                                    outputs_validated=ok))
        store.finish(status, "" if ok else "fits failed")
        stage = StageOutcome("calibration", status, "done", "attempt-1", store.root, tuple(artifacts))
        return store.root, WorkflowOutcome(Action.CALIBRATE, status, "done", store.root, (stage,))

    def test_offer_lm_with_this_calibration_is_an_unsaved_edit_and_switches_to_lm(self):
        app = self.begin()
        root, outcome = self.calibration_run()
        app._workflow_done(WorkflowResult(1, outcome, ""), [])
        saved = (self.base / "profile.yaml").read_bytes()
        started = []
        app.session.start_workflow = lambda *args, **kwargs: started.append(args) or 2
        app._refresh_controls()
        self.assertEqual(app.buttons["offer_lm_calibration"].cget("state"), "normal")
        app.buttons["offer_lm_calibration"].invoke()
        self.assertEqual(app.vars["calibration_file"].get(), str(root / "x.encal"))
        self.assertEqual(app.tabview.get(), gui.TABS[3])
        self.assertTrue(pump(app.root, lambda: "unsaved edits" in app.profile_status.cget("text")),
                        app.profile_status.cget("text"))
        self.assertIn(f"System Energy cal file set to {root / 'x.encal'} from run run-1 (unsaved profile edit)", self.log(app))
        self.assertEqual(started, [])
        self.assertEqual((self.base / "profile.yaml").read_bytes(), saved)

    def test_offer_lm_calibration_disabled_after_a_failed_calibration(self):
        app = self.begin()
        root, outcome = self.calibration_run(ResultStatus.FAILED)
        app._workflow_done(WorkflowResult(1, outcome, ""), [])
        app._refresh_controls()
        self.assertEqual(app.buttons["offer_lm_calibration"].cget("state"), "disabled")


@pytest.mark.gui
@pytest.mark.fr("005-FR-6")  # spec 005 T13-T14
class LastFolderChecks(GUIBase):
    fixture_prefix = "pm-gui-last-folder-"
    STORED = ("raw_input", "ldat_inputs.calibrate", "conversion_run", "cog_limits_file", "data_dir")

    def window(self, last_folders=None, name="w"):
        self.world = self.base / name
        profile = world(self.world)
        self.path = save_profile(replace(profile, last_folders=last_folders or {}), self.base / f"{name}.yaml")
        app = self.open(self.path, repo_root=self.world)
        self.asked = []
        return app

    def answer(self, kind, value):
        def ask(**options):
            self.asked.append((kind, options.get("initialdir")))
            return value
        return ask

    def stored(self):
        folders = {key: self.base / "stored" / key.replace(".", "_") for key in self.STORED}
        for folder in folders.values():
            folder.mkdir(parents=True)
        return {key: str(folder) for key, folder in folders.items()}

    def open_every_dialog(self, app):
        """Cancel each remembered dialog once; the start folders in order."""
        self.asked = []
        app.ask_file, app.ask_directory, app.ask_files = (self.answer("file", ""), self.answer("directory", ""),
                                                          self.answer("files", ()))
        app._browse("cog_limits_file", "file")
        app._browse("data_dir", "dir")
        app._browse_raw()
        app.selections["calibrate"].add()
        app.selections["calibrate"].buttons["run"].invoke()
        return [folder for _, folder in self.asked]

    def test_last_folder_dialogs_open_in_the_stored_folder_or_fall_back(self):
        folders = self.stored()
        app = self.window(folders)
        saved = self.path.read_bytes()
        self.assertEqual(self.open_every_dialog(app), [folders[key] for key in (
            "cog_limits_file", "data_dir", "raw_input", "ldat_inputs.calibrate", "conversion_run")])
        self.assertEqual(self.path.read_bytes(), saved)                 # cancelled: nothing remembered
        self.assertEqual(dict(app.session.profile.last_folders), folders)
        self.assertNotIn("Conversion run not added", self.log(app))
        # A stored folder that no longer exists is ignored: today's start folders.
        app = self.window({key: str(self.base / "gone" / key) for key in self.STORED}, name="w2")
        self.assertEqual(self.open_every_dialog(app), [str(self.world / "lim"), str(self.world / "data"),
                                                       str(self.world / "data"), None, str(self.world / "data")])

    def test_last_folder_choice_saved_without_marking_the_profile_edited(self):
        app = self.window()
        before = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        ldats = self.base / "ldats"
        ldats.mkdir()
        (ldats / "a.ldat").write_bytes(b"x")
        app.ask_files = self.answer("files", (str(ldats / "a.ldat"),))
        app.selections["listmode"].add()
        self.assertEqual(dict(app.session.profile.last_folders), {"ldat_inputs.listmode": str(ldats)})
        after = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.assertEqual(after.pop("last_folders"), {"ldat_inputs.listmode": str(ldats)})
        self.assertEqual(after, before)
        self.assertEqual(app.profile_from_ui(), app.session.profile)
        pump(app.root, timeout=0.3)
        self.assertNotIn("unsaved edits", app.profile_status.cget("text"))
        # With an unsaved edit pending: the folder is written, the edit is neither saved nor lost.
        app.vars["data_dir"].set(str(self.base / "elsewhere"))
        self.assertTrue(pump(app.root, lambda: "unsaved edits" in app.profile_status.cget("text")))
        app.ask_file = self.answer("file", str(self.world / "lim" / "doi.txt"))
        app._browse("cog_limits_file", "file")
        self.assertEqual(app.vars["cog_limits_file"].get(), str(self.world / "lim" / "doi.txt"))
        after = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.assertEqual(after.pop("last_folders"), {"ldat_inputs.listmode": str(ldats),
                                                     "cog_limits_file": str(self.world / "lim")})
        self.assertEqual(after, before)
        self.assertEqual(app.session.profile.data_dir, before["data_dir"])
        self.assertEqual(app.vars["data_dir"].get(), str(self.base / "elsewhere"))
        self.assertNotEqual(app.profile_from_ui(), app.session.profile)
        pump(app.root, timeout=0.3)
        self.assertIn("unsaved edits", app.profile_status.cget("text"))
        app.ask_file = self.answer("file", str(self.world / "data" / "run ; & $x.rawf"))
        app._browse_raw()
        app.ask_directory = self.answer("directory", str(self.world / "encal"))
        app._browse("report_dir", "dir")
        self.assertEqual({key: load_profile(self.path).last_folders[key] for key in ("raw_input", "report_dir")},
                         {"raw_input": str(self.world / "data"), "report_dir": str(self.world / "encal")})

    def test_conversion_run_folder_lists_exactly_its_recorded_ldats(self):
        app = self.window()
        run = recorded_run(self.world / "data", ["conversion"], {"conversion": SPLITS})
        recorded = sorted(run.rglob("x_coincCompact_*.ldat"))
        (recorded[0].parent / "x_coincCompact_3.ldat").write_bytes(b"not an output")
        selection = app.selections["calibrate"]
        self.assertEqual(selection.buttons["run"].cget("text"), "Add conversion run...")
        app.ask_directory = self.answer("directory", str(run))
        selection.buttons["run"].invoke()
        self.assertEqual(selection.paths, recorded)
        self.assertEqual([path.name for path in selection.paths], [name for name, _, _ in SPLITS])
        self.assertTrue(selection.confirmed.get())
        self.assertIn(f"from conversion run {run.name}", selection.summary.cget("text"))
        self.assertIn(f"3 output(s) of conversion run {run.name}", self.log(app))
        self.assertEqual(dict(app.session.profile.last_folders), {"conversion_run": str(run.parent)})
        self.assertEqual([folder for _, folder in self.asked], [str(self.world / "data")])

    def test_refused_conversion_run_folders_leave_the_list_unchanged(self):
        app = self.window()
        destination = self.world / "data"
        plain = self.base / "plain"
        plain.mkdir()
        refused = (plain, recorded_run(destination, ["calibration"], {"calibration": CALIBRATION}, name="cal"),
                   recorded_run(destination, ["conversion"], {"conversion": (SPLITS, ResultStatus.FAILED)}, name="bad"))
        listed = self.base / "listed.ldat"
        listed.write_bytes(b"x")
        selection = app.selections["qc_analyze"]
        selection.add([listed])
        for folder in refused:
            app.ask_directory = self.answer("directory", str(folder))
            selection.buttons["run"].invoke()
            self.assertEqual((selection.paths, selection.origin), ([listed], None))
            last = self.log(app).splitlines()[-1]
            self.assertIn("Conversion run not added: ", last)
            self.assertIn(folder.name, last)
        lines = len(self.log(app).splitlines())
        app.ask_directory = self.answer("directory", "")
        selection.buttons["run"].invoke()
        self.assertEqual(len(self.log(app).splitlines()), lines)
        self.assertEqual(selection.paths, [listed])
