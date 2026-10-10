"""Manager shell: tabs, log, profiles, INI/YAML controls, readiness, main-thread rule, relocated
assets, Inspector independence (spec 003 T13; spec 007 T15).

Moved from scripts/petsys_manager_gui_check.py --shell. Withdrawn windows on fixture profiles
(``gui``: skipped without a display).
"""

import ast
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import subprocess
import textwrap
import threading

import pytest

from exe_programs import petsys_manager_gui as gui
from helpers import REPO
from manager_gui_helpers import FixtureProbe, GUIBase, PY, RunPanelBase, STAMP, child, pump, texts, world
from manager_helpers import SPLITS, recorded_run
from src.petsys_manager.acquisition import DaqdState, DaqdStatus
from src.petsys_manager.contracts import Action, ResultStatus
from src.petsys_manager.session import ShellEvent, WorkflowResult
from src.petsys_manager.settings import MachineProfile, SystemProbe, load_profile, save_profile
from src.petsys_manager import recent
from src.petsys_manager.recent import Recent, RecentRun
from src.petsys_manager.workflow import RUNS, WorkflowOutcome, append_overview


PRIVATE = ("/home/sie", "/data/nvmDisk", "nvmDisk")
INSPECTOR = ("exe_programs/LDATInspector.py", "exe_programs/ldat_inspector_gui.py",
             "exe_programs/LDATInspector_legacy.py", "src/ldat_inspector")
INI_CHECKS = {"initialize", "acquire", "pipeline", "convert_coincidence", "qc"}
YAML_CHECKS = {"pipeline", "calibrate", "listmode", "qc", "qc_analyze"}


@pytest.mark.gui
@pytest.mark.fr("003-FR-1", "003-FR-3", "003-FR-7", "003-FR-16")  # spec 003 T13
class ShellChecks(GUIBase):
    fixture_prefix = "pm-gui-sh-"

    @pytest.mark.fr("003-FR-10")
    def test_tabs_log_and_startup(self):
        profile = world(self.base / "w")
        path = save_profile(profile, self.base / "profile.yaml")
        before = {thread.name for thread in threading.enumerate()}
        app = self.open(path, repo_root=self.base / "w")
        self.assertEqual(tuple(app.tabview._name_list), gui.TABS)
        self.assertEqual(gui.TABS, ("System Setup & Acquisition", "RAWF to LDAT Conversion", "LDAT Processing",
                                    "LM File Generation", "System Quality Control", "Recent Runs"))   # spec 005 T12
        self.assertIn("Output Log:", texts(app.root))
        log = self.log(app)
        self.assertIn(f"PETsys Manager {gui.__version__}; checkout", log)
        self.assertIn(f"Loaded profile {path}", log)
        new = {thread.name for thread in threading.enumerate()} - before
        self.assertEqual(new, {"petsys-preflight"})
        self.assertTrue(all(name == "petsys-preflight" for name in app.session.probe.threads))
        for key, button in app.buttons.items():  # T14: only DAQD start is possible before a daemon runs
            # spec 005 T12: reading Recent Runs is always possible
            self.assertEqual(button.cget("state"), "normal" if key in ("daqd", "recent_refresh") else "disabled", key)
        expected_met = {"daqd", "initialize", "acquire", "pipeline", "qc"}
        for key, text in self.reasons(app).items():
            if key in expected_met:
                self.assertIn("prerequisites met", text, key)
            elif key.startswith("convert"):
                self.assertEqual(text.splitlines()[1:], ["  • RAW Data File: Select an existing file"])
            else:
                self.assertEqual(text.splitlines()[1:], ["  • Input LDAT files: Select the exact ordered input files"])
        self.edit(app, app.raw_input, str(self.base / "w" / "data" / "run ; & $x.rawf"))
        self.assertIn("prerequisites met", self.reasons(app)["convert_coincidence"])
        self.assertNotIn("convert_group", self.reasons(app))          # FR-10: no group conversion
        for index in range(1500):
            app.session.log(f"bulk {index}")
        self.assertTrue(pump(app.root, lambda: self.log(app).rstrip().endswith("bulk 1499")))
        lines = self.log(app).splitlines()
        self.assertEqual(len(lines), 1000)
        self.assertRegex(lines[0], r"\A\d\d:\d\d:\d\d bulk 500\Z")          # FR-1: local time on every line
        self.assertTrue(all(STAMP.match(line) for line in lines))
        self.assertEqual(app.guard.violations, [])
        app.on_close()
        self.assertFalse(any(thread.name == "petsys-preflight" for thread in threading.enumerate()))

    def test_profile_reload_and_save(self):
        profile = world(self.base / "w")
        path = save_profile(profile, self.base / "profile.yaml")
        saved = path.read_bytes()
        app = self.open(path, repo_root=self.base / "w")
        for name in gui.PROFILE_FIELDS:
            self.assertEqual(app.vars[name].get(), getattr(profile, name) or "", name)
        self.assertEqual(app.vars["cards"].get(), ", ".join(profile.cards))
        app.vars["ini_file"].set(str(self.base / "w" / "cfg" / "other.ini"))
        app.vars["cards"].set(str(self.base / "w" / "dev" / "card1"))
        self.settle(app)
        self.assertIn("unsaved edits", app.profile_status.cget("text"))
        self.assertEqual(path.read_bytes(), saved, "edits must not autosave")
        self.assertEqual(app.session.profile, profile)
        self.assertTrue(app.reload_profile())
        self.settle(app)
        self.assertEqual(app.vars["ini_file"].get(), profile.ini_file)
        self.assertEqual(app.vars["cards"].get(), ", ".join(profile.cards))
        self.assertNotIn("unsaved", app.profile_status.cget("text"))
        # Save: shown fields change, hidden safety/limits/capabilities/LM metadata/shared memory are kept.
        app.vars["data_dir"].set(str(self.base / "w" / "data2"))
        app.vars["processing_root"].set(str(self.base / "w"))
        self.assertTrue(app.save_profile())
        self.settle(app)
        reread = load_profile(path)
        self.assertEqual(reread, MachineProfile(**{**to_dict(profile), "data_dir": str(self.base / "w" / "data2"),
                                                   "processing_root": str(self.base / "w")}))
        for name in ("safety", "limits", "capabilities", "lm_metadata", "shared_memory_path"):
            self.assertEqual(getattr(reread, name), getattr(profile, name), name)
        # A broken file is reported; the window keeps its values and the file is not rewritten.
        broken = b"schema_version: 1\ndaq_type: [unterminated\n"
        path.write_bytes(broken)
        app.vars["lm_dir"].set(str(self.base / "w" / "rep"))
        self.assertFalse(app.reload_profile())
        self.assertIn("Profile not reloaded; current values kept", self.log(app))
        self.assertEqual(app.vars["lm_dir"].get(), str(self.base / "w" / "rep"))
        self.assertEqual(path.read_bytes(), broken)
        # Save refuses to replace a non-profile file (the processing YAML) and leaves it byte-identical.
        target = self.base / "w" / "cfg" / "proc.yaml"
        before = target.read_bytes()
        app.profile_path.set(str(target))
        self.assertFalse(app.save_profile())
        self.assertEqual(target.read_bytes(), before)
        self.assertIn("Profile not saved", self.log(app))
        # A new path is created exclusively; a missing explicit profile falls back without writing.
        fresh = self.base / "new" / "p.yaml"
        app.profile_path.set(str(fresh))
        self.assertTrue(app.save_profile())
        self.assertEqual(load_profile(fresh).lm_dir, str(self.base / "w" / "rep"))
        missing = self.base / "missing.yaml"
        other = self.open(missing)
        self.assertEqual(other.session.profile, MachineProfile())
        self.assertIn("defaults in use, nothing written", self.log(other))
        self.assertFalse(missing.exists())
        bad = self.base / "bad.yaml"
        bad.write_bytes(broken)
        third = self.open(bad)
        self.assertIn("Profile not loaded, defaults in use and file left unchanged", self.log(third))
        self.assertIn("not loaded (see log)", third.profile_status.cget("text"))
        self.assertEqual(bad.read_bytes(), broken)

    def test_separate_ini_yaml_controls(self):
        profile = world(self.base / "w")
        app = self.open(save_profile(profile, self.base / "p.yaml"), repo_root=self.base / "w")
        self.assertIsNot(app.vars["ini_file"], app.vars["yaml_file"])
        ini_entry, = app.entries["ini_file"]
        yaml_entry, = app.entries["yaml_file"]
        self.assertIsNot(ini_entry, yaml_entry)
        labels = texts(app.tabs[0])
        self.assertIn("PETsys INI (DAQ/conversion):", labels)
        self.assertIn("Processing YAML (cal/LM/QC):", labels)
        missing_ini = str(self.base / "w" / "cfg" / "gone.ini")
        self.edit(app, app.vars["ini_file"], missing_ini)
        self.assertEqual(app.vars["yaml_file"].get(), profile.yaml_file)
        for key, text in self.reasons(app).items():
            self.assertEqual(f"PETsys INI (DAQ/conversion): File not found: {Path(missing_ini).resolve()}" in text,
                             key in INI_CHECKS, key)
            self.assertNotIn("Processing YAML", text, key)
        self.edit(app, app.vars["ini_file"], profile.ini_file)
        missing_yaml = str(self.base / "w" / "cfg" / "gone.yaml")
        self.edit(app, app.vars["yaml_file"], missing_yaml)
        self.assertEqual(app.vars["ini_file"].get(), profile.ini_file)
        for key, text in self.reasons(app).items():
            self.assertEqual("Processing YAML (cal/LM/QC): File not found" in text, key in YAML_CHECKS, key)
            self.assertNotIn("PETsys INI", text, key)
        # Relative map paths: the default processing root is the checkout, an explicit one replaces it.
        self.edit(app, app.vars["yaml_file"], profile.yaml_file)
        self.edit(app, app.vars["processing_root"], str(self.base / "w" / "data"))
        self.assertIn(f"Map file named by the processing YAML: File not found: "
                      f"{(self.base / 'w' / 'data' / 'maps' / 'map.yaml').resolve()}", self.reasons(app)["calibrate"])
        self.assertEqual(app.guard.violations, [])

    @pytest.mark.slow  # ~7 s
    def test_prerequisite_reasons_without_private_paths(self):
        app = self.open(self.base / "none.yaml", probe=SystemProbe(), repo_root=REPO)
        reasons = self.reasons(app)
        for key, text in reasons.items():
            self.assertIn("unavailable:", text, key)
            self.assertTrue(text.splitlines()[1].startswith("  • "), key)
        joined = "\n".join(texts(app.root) + [self.log(app)])
        # The fixture folder and the checkout are shown on purpose; on the Cornell PC both are under /home/sie (T24).
        for shown in (str(self.output), str(REPO)):
            joined = joined.replace(shown, "<shown>")
        for private in PRIVATE:
            self.assertNotIn(private, joined)
        source = "\n".join(Path(gui.__file__).read_text(encoding="utf-8").splitlines() +
                           (REPO / "src/petsys_manager/session.py").read_text(encoding="utf-8").splitlines())
        for private in PRIVATE:
            self.assertNotIn(private, source)
        self.assertIn("PETsys Tools Folder: select it (needs daqd)", reasons["daqd"])
        self.assertIn("LM metadata missing from the profile: isotope,", reasons["listmode"])
        # Run options and profile edits give specific reasons only where they apply.
        profile = world(self.base / "w")
        app = self.open(save_profile(profile, self.base / "p.yaml"), repo_root=self.base / "w")
        self.edit(app, app.acq_time, "ten")
        for key, text in self.reasons(app).items():
            self.assertEqual("Run options: Acq. Time (s) must be a positive number" in text,
                             key in ("acquire", "pipeline"), key)
        self.edit(app, app.acq_time, "-5")
        self.assertIn("Run options: duration_s must be greater than 0", self.reasons(app)["acquire"])
        self.edit(app, app.acq_time, "10")
        self.edit(app, app.splits, "0")
        self.assertIn("Run options: splits must be an integer", self.reasons(app)["convert_coincidence"])
        self.assertIn("prerequisites met", self.reasons(app)["acquire"])
        self.edit(app, app.splits, "2")
        self.edit(app, app.raw_input, str(self.base / "w" / "data" / "absent.rawf"))
        self.assertIn("RAW Data File: File not found", self.reasons(app)["convert_coincidence"])
        self.edit(app, app.raw_input, "")
        app.vars["cards"].set("relative/card")
        self.assertTrue(pump(app.root, lambda: app._check_id is None and app._awaiting is None))
        for key, text in self.reasons(app).items():
            self.assertIn("Profile: DAQ card paths must be absolute", text, key)
        self.assertIn("edits invalid", app.profile_status.cget("text"))
        self.edit(app, app.vars["cards"], ", ".join(profile.cards))
        self.assertIn("prerequisites met", self.reasons(app)["acquire"])
        # Slab analysis requires plots.
        self.assertEqual(app.qc_slabs_check.cget("state"), "disabled")
        app.qc_plots.set(True)
        app._plots_toggled()
        app.qc_slabs.set(True)
        self.assertEqual(app.qc_slabs_check.cget("state"), "normal")
        app.qc_plots.set(False)
        app._plots_toggled()
        self.assertFalse(app.qc_slabs.get())
        self.assertEqual(app.qc_slabs_check.cget("state"), "disabled")
        self.settle(app)
        self.assertEqual(app.guard.violations, [])

    def test_main_thread_only_and_stale_readiness(self):
        profile = world(self.base / "w")
        probe = FixtureProbe()
        app = self.open(save_profile(profile, self.base / "p.yaml"), probe=probe, repo_root=self.base / "w")
        shown = []
        original = app._show_readiness
        app._show_readiness = lambda issues: shown.append(dict(issues)) or original(issues)
        posted = []
        events = app.session.events
        put = events.put
        app.session.events = type("Q", (), {"put": lambda _, e: posted.append(e) or put(e),
                                            "get_nowait": lambda _: events.get_nowait()})()
        probe.gate = threading.Event()
        app.vars["ini_file"].set(str(self.base / "w" / "cfg" / "gone.ini"))  # first check: INI reason
        self.assertTrue(pump(app.root, probe.blocked.is_set))
        first = app._awaiting
        app.vars["ini_file"].set(profile.ini_file)
        app.vars["data_dir"].set(str(self.base / "w" / "nodata"))  # newer check: data folder reason
        self.assertTrue(pump(app.root, lambda: app._check_id is None and app._awaiting != first))
        app.session.log("still responsive while a check is blocked")
        self.assertTrue(pump(app.root, lambda: "still responsive" in self.log(app)))
        self.assertEqual(shown, [])
        probe.gate.set()
        probe.gate = None
        self.settle(app)
        readiness = [event.payload.generation for event in posted if event.kind == "readiness"]
        self.assertEqual(readiness, [first, app._awaiting])
        self.assertEqual(len(shown), 1)
        self.assertFalse(any("PETsys INI" in "\n".join(gui.issue_lines(found)) for found in shown[0].values()))
        self.assertIn("Output Data Folder: Destination not writable", self.reasons(app)["acquire"])
        self.assertNotIn("PETsys INI", self.reasons(app)["acquire"])
        self.assertEqual(app.guard.violations, [])
        self.assertGreater(app.guard.calls, 100)
        self.assertEqual(probe.threads, {"petsys-preflight"})
        # Negative control: the guard does catch a worker touching a widget.
        failure = []
        worker = threading.Thread(target=lambda: _capture(failure, lambda: app.log("from a worker")), name="rogue")
        worker.start()
        worker.join(5)
        self.assertEqual(len(app.guard.violations), 1)
        self.assertEqual(app.guard.violations[0][0], "rogue")
        self.assertIsInstance(failure[0], RuntimeError)

    def test_relocated_assets(self):
        relocated = self.base / "re lo ; [x]"
        (relocated / "exe_programs").mkdir(parents=True)
        for name in ("PETsysManager.py", "petsys_manager_gui.py"):
            shutil.copy2(REPO / "exe_programs" / name, relocated / "exe_programs" / name)
        shutil.copytree(REPO / "exe_programs" / "assets", relocated / "exe_programs" / "assets")
        (relocated / "src").mkdir()
        shutil.copy2(REPO / "src" / "__init__.py", relocated / "src" / "__init__.py")
        shutil.copytree(REPO / "src" / "petsys_manager", relocated / "src" / "petsys_manager",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (relocated / "maps").mkdir()
        (relocated / "maps" / "map.yaml").write_text("marker: 1\n", encoding="utf-8")
        elsewhere = self.base / "cwd"
        elsewhere.mkdir()
        profile = MachineProfile(yaml_file=str(self.base / "proc.yaml"), report_dir=str(elsewhere))
        (self.base / "proc.yaml").write_text("map_file: maps/map.yaml\nmin_ch: 1\nen_min_ch: 5\n", encoding="utf-8")
        path = save_profile(profile, self.base / "p.yaml")
        code = textwrap.dedent(f"""
            import json, sys, time
            sys.path.insert(0, {str(relocated)!r})
            import customtkinter as ctk
            from exe_programs import petsys_manager_gui as gui
            from src.petsys_manager.session import ManagerSession
            root = ctk.CTk(); root.withdraw()
            app = gui.PETsysManager(root, ManagerSession({str(path)!r}))
            end = time.monotonic() + 10
            while time.monotonic() < end and app.shown_generation != app._awaiting:
                root.update(); time.sleep(0.01)
            result = dict(module=gui.__file__, logo=str(app.logo_path) if app.logo_path else None,
                          checkout=str(app.session.repo_root), tabs=list(app.tabview._name_list),
                          calibrate=app.readiness["calibrate"].cget("text"),
                          log=app.log_text.get("1.0", "end-1c"),
                          imported=sorted(m.__file__ for m in list(sys.modules.values())
                                          if getattr(m, "__file__", None) and "petsys_manager" in m.__file__))
            app.on_close()
            print(json.dumps(result))
        """)
        result = child(code, cwd=elsewhere)
        self.assertEqual(Path(result["module"]).resolve(), (relocated / "exe_programs" / "petsys_manager_gui.py").resolve())
        self.assertEqual(Path(result["logo"]), (relocated / "exe_programs" / "assets" / "onco_logo.jpeg").resolve())
        self.assertEqual(Path(result["checkout"]), relocated.resolve())
        self.assertTrue(all(Path(item).resolve().is_relative_to(relocated.resolve()) for item in result["imported"]))
        self.assertEqual(result["tabs"], list(gui.TABS))
        self.assertNotIn("Map file", result["calibrate"])  # maps/map.yaml resolves inside the relocated checkout
        self.assertNotIn(str(REPO), result["log"] + result["calibrate"])
        shutil.rmtree(relocated / "exe_programs" / "assets")
        result = child(code, cwd=elsewhere)
        self.assertIsNone(result["logo"])
        self.assertIn("Optional logo not shown", result["log"])
        self.assertEqual(result["tabs"], list(gui.TABS))
        run = subprocess.run([PY, "-X", "utf8", str(relocated / "exe_programs" / "PETsysManager.py"), "--help"],
                             cwd=elsewhere, capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--profile", run.stdout)

    def test_inspector_independent(self):
        manager = child(textwrap.dedent("""
            import json, sys
            sys.path.insert(0, %r)
            import exe_programs.petsys_manager_gui
            import src.petsys_manager.session
            print(json.dumps(sorted(sys.modules)))
        """ % str(REPO)), cwd=self.base)
        self.assertFalse([name for name in manager if "ldat_inspector" in name or name == "matplotlib"])
        session = child(textwrap.dedent("""
            import json, sys
            sys.path.insert(0, %r)
            import src.petsys_manager.session
            print(json.dumps(sorted(sys.modules)))
        """ % str(REPO)), cwd=self.base)
        self.assertFalse([name for name in session if name.split(".")[0] in ("tkinter", "_tkinter", "customtkinter")])
        inspector = child(textwrap.dedent("""
            import json, sys
            sys.path.insert(0, %r)
            import exe_programs.ldat_inspector_gui
            print(json.dumps(sorted(sys.modules)))
        """ % str(REPO)), cwd=self.base)
        self.assertFalse([name for name in inspector if "petsys_manager" in name])
        for path in (REPO / "exe_programs" / "PETsysManager.py", Path(gui.__file__), REPO / "src/petsys_manager/session.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                         [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                self.assertFalse([name for name in names if "ldat_inspector" in name or "LDATInspector" in name], path)
        git = lambda *args: subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(git("diff", "--quiet", "HEAD", "--", *INSPECTOR).returncode, 0)
        self.assertEqual(git("ls-files", "--others", "--exclude-standard", "--", *INSPECTOR).stdout, "")
        run = subprocess.run([PY, "-X", "utf8", "-m", "compileall", "-q", "exe_programs/PETsysManager.py",
                              "exe_programs/petsys_manager_gui.py", "src/petsys_manager", "src/cornell"],
                             cwd=REPO, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


def to_dict(profile):
    return {name: getattr(profile, name) for name in MachineProfile.__dataclass_fields__}


def _capture(sink, function):
    try:
        function()
    except Exception as exc:
        sink.append(exc)


@pytest.mark.gui
@pytest.mark.fr("005-FR-1")  # spec 005 T5
class RunPanelChecks(RunPanelBase):
    fixture_prefix = "pm-gui-rp-"

    def test_run_panel_unknown_total_is_indeterminate_with_elapsed_only(self):
        app = self.begin()
        self.feed(app, "stage_started", "calibration", directory=str(self.base / "run"))
        self.feed(app, "stage_progress", "calibration", phase="read", phases=["read", "fits"],
                  bytes_read=1_000_000, bytes_total=None)
        mode, counter, timing = self.at(app, 112.0)
        self.assertEqual(mode, "indeterminate")
        self.assertNotIn("remaining", timing)
        self.assertIn("Energy calibration", app.run_panel.title.cget("text"))

    def test_run_panel_bytes_give_a_bar_and_estimate_per_phase_after_5_s_and_1_percent(self):
        app = self.begin()
        phases = ["read", "pass 2", "fits"]
        self.feed(app, "stage_started", "calibration", directory=str(self.base / "run"))
        self.now = 103.0      # 3 s into the phase, 5 %: no estimate yet
        self.feed(app, "stage_progress", "calibration", phase="read", phases=phases,
                  bytes_read=200_000_000, bytes_total=4_000_000_000)
        mode, counter, timing = self.at(app, 103.0)
        self.assertEqual((mode, counter), ("indeterminate", "0.20 / 4.00 GB input read (read, phase 1 of 3)"))
        self.assertNotIn("remaining", timing)
        self.now = 110.0      # 10 s, 25 %: 30 s left in this phase
        self.feed(app, "stage_progress", "calibration", phase="read", phases=phases,
                  bytes_read=1_000_000_000, bytes_total=4_000_000_000)
        mode, counter, timing = self.at(app, 110.0)
        self.assertEqual((mode, counter), ("determinate", "1.00 / 4.00 GB input read (read, phase 1 of 3)"))
        self.assertAlmostEqual(app.run_panel.bar.get(), 0.25, places=3)
        self.assertIn("about 30.0 s remaining", timing)
        self.now = 111.0      # a new phase restarts the bar and the estimate
        self.feed(app, "stage_progress", "calibration", phase="pass 2", phases=phases,
                  bytes_read=400_000_000, bytes_total=4_000_000_000)
        mode, counter, timing = self.at(app, 111.0)
        self.assertEqual((mode, counter), ("indeterminate", "0.40 / 4.00 GB input read (pass 2, phase 2 of 3)"))
        self.assertNotIn("remaining", timing)
        self.now = 120.0      # 10 s since the read phase was last seen, 50 %: 10 s left
        self.feed(app, "stage_progress", "calibration", phase="pass 2", phases=phases,
                  bytes_read=2_000_000_000, bytes_total=4_000_000_000)
        mode, counter, timing = self.at(app, 120.0)
        self.assertEqual(mode, "determinate")
        self.assertAlmostEqual(app.run_panel.bar.get(), 0.5, places=3)
        self.assertIn("about 10.0 s remaining", timing)

    def test_run_panel_fits_count_keys(self):
        app = self.begin()
        self.feed(app, "stage_started", "calibration", directory=str(self.base / "run"))
        self.now = 110.0
        self.feed(app, "stage_progress", "calibration", phase="read", phases=["read", "fits"],
                  bytes_read=4_000_000_000, bytes_total=4_000_000_000)
        self.now = 112.0
        self.feed(app, "stage_progress", "calibration", phase="fits", phases=["read", "fits"],
                  keys_done=10, keys_total=400)
        mode, counter, timing = self.at(app, 112.0)
        self.assertEqual((mode, counter), ("indeterminate", "fits 10 / 400 keys (phase 2 of 2)"))
        self.now = 120.0      # 10 s into the fits, half the keys: 10 s left
        self.feed(app, "stage_progress", "calibration", phase="fits", phases=["read", "fits"],
                  keys_done=200, keys_total=400)
        mode, counter, timing = self.at(app, 120.0)
        self.assertEqual((mode, counter), ("determinate", "fits 200 / 400 keys (phase 2 of 2)"))
        self.assertIn("about 10.0 s remaining", timing)

    def convert(self, app, now, **payload):
        self.now = now
        self.feed(app, "stage_progress", "conversion", **{"phase": "convert", "rawf_bytes": 2_000_000_000,
                                                          "converter_exited": False, **payload})
        return self.at(app, now)

    def test_run_panel_one_split_conversion_is_indeterminate(self):
        app = self.begin(Action.CONVERT, ("conversion",))
        self.feed(app, "stage_started", "conversion", raw="x.rawf", directory=str(self.base / "run"))
        mode, counter, timing = self.convert(app, 130.0, ldat_bytes=500_000_000, splits=1, split_durations_s=[])
        self.assertEqual((mode, counter), ("indeterminate", "LDAT 0.5 GB written, RAW 2.0 GB; splits 0 / 1 closed"))
        mode, counter, timing = self.convert(app, 160.0, ldat_bytes=900_000_000, splits=1, split_durations_s=[58.0],
                                             converter_exited=True)
        self.assertEqual((mode, counter), ("indeterminate", "LDAT 0.9 GB written, RAW 2.0 GB; splits 1 / 1 closed"))
        self.assertNotIn("remaining", timing)

    def test_run_panel_three_split_conversion_estimates_after_the_first_closed_split(self):
        app = self.begin(Action.CONVERT, ("conversion",))
        self.feed(app, "stage_started", "conversion", raw="x.rawf", directory=str(self.base / "run"))
        mode, counter, timing = self.convert(app, 110.0, ldat_bytes=300_000_000, splits=3, split_durations_s=[])
        self.assertEqual((mode, counter), ("indeterminate", "LDAT 0.3 GB written, RAW 2.0 GB; splits 0 / 3 closed"))
        self.assertNotIn("remaining", timing)
        mode, counter, timing = self.convert(app, 113.0, ldat_bytes=700_000_000, splits=3, split_durations_s=[12.0])
        self.assertEqual((mode, counter), ("determinate", "LDAT 0.7 GB written, RAW 2.0 GB; splits 1 / 3 closed"))
        self.assertAlmostEqual(app.run_panel.bar.get(), 1 / 3, places=3)
        self.assertIn("about 24.0 s remaining", timing)       # 2 splits left x 12 s

    def test_run_panel_final_state_stays_until_the_next_run(self):
        app = self.begin()
        self.feed(app, "stage_started", "calibration", directory=str(self.base / "run"))
        self.now = 110.0
        self.feed(app, "stage_progress", "calibration", phase="read", phases=["read"],
                  bytes_read=1_000_000_000, bytes_total=4_000_000_000)
        self.now = 130.0
        self.feed(app, "stage_finished", "calibration", status="succeeded", elapsed_s=30.0)
        self.feed(app, "workflow_finished", "workflow", status="succeeded", stages=["calibration"])
        self.now = 500.0      # the clock keeps going; the panel keeps the final state
        panel = app.run_panel
        self.assertTrue(pump(app.root, lambda: "succeeded" in panel.title.cget("text")), panel.title.cget("text"))
        self.assertEqual(panel.timing.cget("text"), "Elapsed 30.0 s")
        self.assertEqual(panel.bar.cget("mode"), "determinate")
        self.assertAlmostEqual(panel.bar.get(), 1.0, places=3)
        self.assertEqual(panel.counter.cget("text"), "1.00 / 4.00 GB input read")
        app._token = app._run_id = None       # the result has arrived
        app._start(app.session.profile, Action.LISTMODE, None, app.lm_status, "Starting...")
        self.assertTrue(pump(app.root, lambda: panel.title.cget("text") == "LM generation"), panel.title.cget("text"))
        self.assertEqual((panel.counter.cget("text"), panel.timing.cget("text")), ("", "Elapsed 0.0 s"))

    def test_run_panel_failed_run_keeps_its_bar(self):
        app = self.begin()
        self.feed(app, "stage_started", "calibration", directory=str(self.base / "run"))
        self.now = 110.0
        self.feed(app, "stage_progress", "calibration", phase="read", phases=["read"],
                  bytes_read=1_000_000_000, bytes_total=4_000_000_000)
        self.now = 115.0
        self.feed(app, "stage_finished", "calibration", status="failed", elapsed_s=15.0)
        self.feed(app, "workflow_finished", "workflow", status="failed", stages=["calibration"])
        self.now = 500.0
        panel = app.run_panel
        self.assertTrue(pump(app.root, lambda: "failed" in panel.title.cget("text")), panel.title.cget("text"))
        self.assertEqual(panel.timing.cget("text"), "Elapsed 15.0 s")
        self.assertEqual(panel.bar.cget("mode"), "determinate")
        self.assertAlmostEqual(panel.bar.get(), 0.25, places=3)

    def test_run_panel_run_that_never_started_ends_the_panel(self):
        app = self.open(save_profile(world(self.base / "w"), self.base / "profile.yaml"), repo_root=self.base / "w")
        self.now = 100.0
        app.clock = lambda: self.now
        app.session.start_workflow = lambda *args, **kwargs: 1
        app._start(app.session.profile, Action.CALIBRATE, None, app.cal_status, "Starting...")
        app._workflow_done(WorkflowResult(1, None, "not started: busy"), [])
        self.now = 104.0
        panel = app.run_panel
        self.assertTrue(pump(app.root, lambda: panel.title.cget("text") == "Energy calibration not started"),
                        panel.title.cget("text"))
        self.assertEqual((panel.bar.cget("mode"), panel.timing.cget("text")), ("determinate", "Elapsed 0.0 s"))

    def test_run_panel_idle_before_the_first_run_breathes_without_spinning(self):
        app = self.open(save_profile(world(self.base / "w"), self.base / "profile.yaml"), repo_root=self.base / "w")
        panel = app.run_panel
        trough = tuple(gui.ctk.ThemeManager.theme["CTkProgressBar"]["fg_color"])
        self.assertTrue(pump(app.root, lambda: panel.title.cget("text") == "● Ready"), panel.title.cget("text"))
        self.assertEqual((panel.counter.cget("text"), panel.timing.cget("text")),
                         ("Pick a tab, set inputs, press START.", ""))
        self.assertEqual(panel.bar.cget("mode"), "determinate")
        self.assertEqual(panel.bar.get(), 0.0)
        self.assertTrue(pump(app.root, lambda: tuple(panel.bar.cget("fg_color")) != trough), "the empty bar never breathed")
        self.now = 100.0
        app.clock = lambda: self.now
        app.session.start_workflow = lambda *args, **kwargs: 1
        app._start(app.session.profile, Action.CALIBRATE, None, app.cal_status, "Starting...")
        self.assertTrue(pump(app.root, lambda: panel.title.cget("text") == "Energy calibration"), panel.title.cget("text"))
        self.assertFalse(pump(app.root, lambda: tuple(panel.bar.cget("fg_color")) != trough, timeout=0.5),
                         "the bar still breathes during a run")

    def test_run_panel_acquisition_step_shows_the_raw_size(self):
        app = self.begin(Action.QC, ("acquisition", "conversion", "qc"))
        self.now = 104.0
        self.feed(app, "acquisition_attempt_started", "acquisition", attempt=1, max_attempts=2, directory="d")
        self.feed(app, "acquisition_rawf_progress", "acquisition", size=12_500_000, bytes_per_s=2.5e6, growing=True)
        mode, counter, timing = self.at(app, 105.0)
        self.assertEqual((mode, counter), ("indeterminate", "RAW 12.5 MB written"))
        self.assertEqual(app.run_panel.title.cget("text"), "Quality control - step 1/3: Acquisition")


@pytest.mark.gui
@pytest.mark.fr("005-FR-2")  # spec 005 T6
class StageRowChecks(RunPanelBase):
    fixture_prefix = "pm-gui-sr-"

    def stage_texts(self, app):
        return [label.cget("text") for label in app.run_panel.stage_labels]

    def test_stage_row_lists_pipeline_stages_with_states(self):
        app = self.begin(Action.PIPELINE, ("acquisition", "conversion", "calibration", "listmode"))
        self.now = 101.0
        self.feed(app, "acquisition_attempt_started", "acquisition", attempt=1, max_attempts=2, directory="d")
        self.now = 161.0
        self.feed(app, "stage_started", "conversion", raw="x.rawf", directory="d")
        self.at(app, 165.0)
        self.assertEqual(self.stage_texts(app), ["Acquisition: succeeded 1 min 00 s", "Conversion: running 4.0 s",
                                                 "Energy calibration: pending", "LM generation: pending"])
        self.assertEqual(app.run_panel.stage_row.winfo_manager(), "grid")
        self.now = 171.0
        self.feed(app, "stage_finished", "conversion", status="failed", elapsed_s=10.0)
        self.feed(app, "workflow_finished", "workflow", status="failed", stages=["acquisition", "conversion"])
        self.now = 400.0
        self.assertTrue(pump(app.root, lambda: "failed" in app.run_panel.title.cget("text")))
        self.assertEqual(self.stage_texts(app), ["Acquisition: succeeded 1 min 00 s", "Conversion: failed 10.0 s",
                                                 "Energy calibration: not run", "LM generation: not run"])

    def test_stage_row_absent_for_a_single_stage_run(self):
        app = self.begin()
        self.feed(app, "stage_started", "calibration", directory="d")
        self.at(app, 103.0)
        self.assertEqual(app.run_panel.stage_row.winfo_manager(), "")


@pytest.mark.gui
@pytest.mark.fr("005-FR-7")  # spec 005 T7
class BannerChecks(RunPanelBase):
    fixture_prefix = "pm-gui-bn-"

    def shown(self, app):
        """(kind, text, button text) of each banner row shown, top to bottom."""
        area = app.banner_area
        if area.winfo_manager() != "pack":
            return []
        rows = sorted((row for row in area.rows.values() if row.winfo_manager() == "pack"),
                      key=lambda row: area.pack_slaves().index(row))
        return [(row.kind, row.label.cget("text"), row.button.cget("text") if row.button else "") for row in rows]

    def done(self, app, status, message):
        app._workflow_done(WorkflowResult(app._token, WorkflowOutcome(app._action, status, message, None), ""), [])

    def restart(self, app, action=Action.LISTMODE):
        app._start(app.session.profile, action, None, app.lm_status, "Starting...")

    def test_banner_failed_run_shows_a_dismissible_red_banner_on_every_tab_until_the_next_run(self):
        app = self.begin()
        self.done(app, ResultStatus.FAILED, "fits failed")
        self.assertEqual(self.shown(app), [("failure", "Energy calibration failed: fits failed", "Dismiss")])
        self.assertEqual(app.banner_area.rows["failure"].cget("border_color"), gui.RED)
        self.assertFalse(str(app.banner_area).startswith(str(app.tabview)), "the banner sits inside a tab")
        for label in gui.TABS:
            app.tabview.set(label)
            pump(app.root, timeout=0.05)
            self.assertEqual(len(self.shown(app)), 1, label)
        self.restart(app)
        self.assertEqual(app.banner_area.winfo_manager(), "", "an empty banner area still takes space")
        self.done(app, ResultStatus.LAUNCH_ERROR, "converter missing")
        self.assertEqual(self.shown(app), [("failure", "LM generation launch_error: converter missing", "Dismiss")])
        app.banner_area.rows["failure"].button.invoke()
        self.assertEqual(self.shown(app), [])

    def test_banner_stopped_run_until_dismissed_and_success_adds_none(self):
        app = self.begin()
        self.done(app, ResultStatus.CANCELLED, "stopped by the operator")
        self.assertEqual(self.shown(app), [("stopped", "Energy calibration stopped: stopped by the operator", "Dismiss")])
        app.banner_area.rows["stopped"].button.invoke()
        self.assertEqual(self.shown(app), [])
        self.restart(app)
        self.done(app, ResultStatus.SUCCEEDED, "done")
        self.assertEqual(self.shown(app), [])

    def test_banner_failed_initialization_and_daqd_failed(self):
        app = self.open(save_profile(world(self.base / "w"), self.base / "profile.yaml"), repo_root=self.base / "w")
        outcome = type("O", (), {"status": None, "initialized": False, "message": "no answer from the cards"})()
        app.session.events.put(ShellEvent("init_done", outcome))
        self.assertTrue(pump(app.root, lambda: self.shown(app)), "no banner")
        self.assertEqual(self.shown(app), [("failure", "Initialization failed: no answer from the cards", "Dismiss")])
        app.session.events.put(ShellEvent("daqd", DaqdStatus(10, DaqdState.FAILED, 1, message="daemon exited (1)")))
        self.assertTrue(pump(app.root, lambda: "DAQD" in self.shown(app)[0][1]), self.shown(app))
        self.assertEqual(self.shown(app), [("failure", "DAQD FAILED: daemon exited (1)", "Dismiss")])
        app.session.events.put(ShellEvent("daqd", DaqdStatus(11, DaqdState.FAILED, 1, message="daemon exited (1)")))
        app.banner_area.rows["failure"].button.invoke()
        pump(app.root, timeout=0.3)
        self.assertEqual(self.shown(app), [], "an unchanged FAILED status brought the banner back")

    def test_banner_bias_unknown_stays_across_a_new_run_until_the_bias_is_confirmed(self):
        app = self.begin()
        app.show_bias_unknown("Bias-off was not confirmed.")
        self.done(app, ResultStatus.FAILED, "acquisition failed")
        bias = ("bias", "SiPM bias state unknown. Bias-off was not confirmed.", "I checked the bias")
        self.assertEqual(self.shown(app), [bias, ("failure", "Energy calibration failed: acquisition failed", "Dismiss")])
        self.assertIs(app.bias_ack, app.banner_area.rows["bias"].button)
        self.restart(app)
        self.assertEqual(self.shown(app), [bias])
        self.assertTrue(app.bias_unknown)
        self.assertEqual(app._live_hint("acquire"), " (confirm the SiPM bias state to enable)")
        app.bias_ack.invoke()
        self.assertEqual(self.shown(app), [])
        self.assertFalse(app.bias_unknown)
        self.assertIn("Operator confirmed the SiPM bias state", app.log_text.get("1.0", "end"))


def tree_digest(root):
    """Every file under ``root`` with its content hash and modification time."""
    return {str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.gui
@pytest.mark.fr("005-FR-3")  # spec 005 T12
class RecentRunsChecks(GUIBase):
    fixture_prefix = "pm-gui-recent-"

    def recent_app(self):
        """A window whose data and report destinations each record one run in runs.tsv."""
        root = self.base / "w"
        app = self.open(save_profile(world(root), self.base / "profile.yaml"), repo_root=root)
        self.opened, self.asked, self.started = [], [], []
        app.open_path = self.opened.append
        app.session.start_workflow = lambda *args, **kwargs: self.started.append(args) or 99
        self.conversion = recorded_run(root / "data", ["conversion"], {"conversion": SPLITS}, name="conv_2026-10-10_1000")
        self.qc = recorded_run(root / "rep", ["qc"], {"qc": [("report.pdf", "qc_report", b"pdf"),
                                                            ("summary.json", "qc_summary", b"{}")]},
                               name="qc_2026-10-10_1100")
        append_overview(root / "data", ("2026-10-10 10:00:00", self.conversion.name, "convert", "acq.rawf",
                                        "succeeded", "x_coincCompact_0.ldat (+2 more)"))
        append_overview(root / "rep", ("2026-10-10 11:00:00", self.qc.name, "qc_analyze", "3 file(s): x.ldat",
                                       "succeeded", "report.pdf"))
        with open(root / "rep" / RUNS, "ab") as out:
            out.write(b"not a run line\n")
        return root, app

    def rows(self, app):
        return [tuple(app.recent_tree.item(item, "values")) for item in app.recent_tree.get_children()]

    def show_tab(self, app, count):
        app.tabview.set("Recent Runs")
        self.assertTrue(pump(app.root, lambda: len(self.rows(app)) == count), self.rows(app))

    def select(self, app, folder):
        item = next(item for item in app.recent_tree.get_children() if app.recent_tree.item(item, "values")[1] == folder)
        app.recent_tree.selection_set(item)
        pump(app.root, timeout=0.1)

    def state(self, app, key):
        return app.buttons[key].cget("state")

    def test_recent_runs_fake_event_fills_rows_newest_first_with_skipped_count(self):
        app = self.open(save_profile(world(self.base / "w"), self.base / "profile.yaml"), repo_root=self.base / "w")
        self.assertIn("Recent Runs", gui.TABS)
        rows = (RecentRun(datetime(2026, 10, 10, 12, 30, 5), "calibration_dir", self.base / "cal_run", "calibrate",
                          "2 file(s): a.ldat", "failed", "-"),
                RecentRun(datetime(2026, 10, 9, 8, 0, 0), "data_dir", self.base / "conv_run", "convert", "acq.rawf",
                          "succeeded", "acq_coincCompact.ldat"))
        app.session.events.put(ShellEvent("recent", Recent(rows, 3, ("lm_dir",))))
        self.assertTrue(pump(app.root, lambda: len(self.rows(app)) == 2), self.rows(app))
        self.assertEqual(self.rows(app), [
            ("2026-10-10 12:30:05", "cal_run", "calibrate", "2 file(s): a.ldat", "failed", "-"),
            ("2026-10-09 08:00:00", "conv_run", "convert", "acq.rawf", "succeeded", "acq_coincCompact.ldat")])
        status = app.recent_status.cget("text")
        self.assertIn("2 run(s)", status)
        self.assertIn("3 malformed runs.tsv line(s) skipped", status)
        self.assertIn("no runs recorded in LM Destination", status)
        self.assertEqual([self.state(app, key) for key in ("recent_open_folder", "recent_open_report", "recent_use")],
                         ["disabled"] * 3)

    def test_recent_runs_tab_reads_off_the_tk_thread_opens_folders_and_reports_and_writes_nothing(self):
        root, app = self.recent_app()
        before = tree_digest(root)
        threads = []
        original = recent.list_recent

        def recorded(destinations):
            threads.append(__import__("threading").current_thread().name)
            return original(destinations)
        recent.list_recent = recorded
        try:
            self.show_tab(app, 2)
            self.assertEqual([row[1] for row in self.rows(app)], [self.qc.name, self.conversion.name])  # newest first
            self.assertTrue(threads and "MainThread" not in threads, threads)
            self.assertIn("1 malformed runs.tsv line(s) skipped", app.recent_status.cget("text"))
            self.select(app, self.conversion.name)
            self.assertEqual([self.state(app, key) for key in ("recent_open_folder", "recent_open_report", "recent_use")],
                             ["normal", "disabled", "normal"])
            app.buttons["recent_open_folder"].invoke()
            self.select(app, self.qc.name)
            self.assertEqual(self.state(app, "recent_open_report"), "normal")
            app.buttons["recent_open_report"].invoke()
            self.assertEqual(self.opened, [self.conversion, self.qc / "report.pdf"])
            append_overview(root / "lm", ("2026-10-10 12:00:00", "lm_2026-10-10_1200", "listmode", "-", "failed", "-"))
            calls = len(threads)
            app.buttons["recent_refresh"].invoke()
            self.assertTrue(pump(app.root, lambda: len(self.rows(app)) == 3), self.rows(app))
            self.assertEqual(self.rows(app)[0][1], "lm_2026-10-10_1200")
            self.assertGreater(len(threads), calls)
        finally:
            recent.list_recent = original
        before[str((root / "lm" / RUNS).relative_to(root))] = tree_digest(root)[str((root / "lm" / RUNS).relative_to(root))]
        self.assertEqual(tree_digest(root), before)
        self.assertEqual(self.started, [])

    def test_recent_runs_refresh_after_a_run_finishes(self):
        root, app = self.recent_app()
        self.show_tab(app, 2)
        append_overview(root / "lm", ("2026-10-10 12:00:00", "lm_2026-10-10_1200", "listmode", "-", "failed", "-"))
        app._start(app.session.profile, Action.LISTMODE, None, app.lm_status, "Starting...")
        app._workflow_done(WorkflowResult(99, WorkflowOutcome(Action.LISTMODE, ResultStatus.FAILED, "x", None), ""), [])
        self.assertTrue(pump(app.root, lambda: len(self.rows(app)) == 3), self.rows(app))

    @pytest.mark.fr("005-FR-4")
    def test_recent_runs_use_as_input_offers_the_three_targets_and_applies_one(self):
        root, app = self.recent_app()
        self.show_tab(app, 2)
        self.select(app, self.conversion.name)
        app.ask_offer = lambda offers: self.asked.append([offer.label for offer in offers])
        app.buttons["recent_use"].invoke()                       # cancelled: nothing changes
        self.assertEqual([list(selection.paths) for selection in app.selections.values()], [[], [], []])
        app.ask_offer = lambda offers: self.asked.append([offer.label for offer in offers]) or offers[1]
        app.buttons["recent_use"].invoke()
        self.assertEqual(self.asked, [["Calibrate", "Generate LM", "Run QC"]] * 2)
        self.assertEqual(app.selections["listmode"].paths, [self.conversion / name for name, _, _ in SPLITS])
        self.assertEqual([app.selections[key].paths for key in ("calibrate", "qc_analyze")], [[], []])
        self.assertEqual(app.tabview.get(), gui.TABS[3])
        self.assertIn("from conversion run conv_2026-10-10_1000", app.selections["listmode"].summary.cget("text"))
        self.select(app, self.qc.name)                           # a QC run records no next step
        app.tabview.set("Recent Runs")
        app.buttons["recent_use"].invoke()
        self.assertIn(f"Use as input: run {self.qc.name} records no next-step inputs", self.log(app))
        self.assertEqual(len(self.asked), 2)
        self.assertEqual(self.started, [])

    @pytest.mark.fr("005-FR-4")
    def test_recent_runs_offer_dialog_returns_the_clicked_offer(self):
        root, app = self.recent_app()
        offers = recent.run_offers(self.conversion)
        stuck = []

        def dialogs():
            return [widget for widget in app.root.winfo_children()
                    if isinstance(widget, gui.ctk.CTkToplevel) and widget.winfo_exists()]

        def click(label):
            buttons = [widget for dialog in dialogs() for widget in dialog.winfo_children()
                       if isinstance(widget, gui.ctk.CTkButton) and widget.cget("text") == label]
            if buttons:
                buttons[-1].invoke()
            else:
                app.root.after(50, lambda: click(label))

        def watchdog():                 # never leave a modal dialog waiting: close it and fail
            for dialog in dialogs():
                stuck.append(dialog.title())
                dialog.destroy()
        for label, expected in (("Run QC", offers[2]), ("Cancel", None)):
            guard = app.root.after(10000, watchdog)
            app.root.after(100, lambda label=label: click(label))
            self.assertEqual(app.ask_offer(offers), expected)
            app.root.after_cancel(guard)
        self.assertEqual(stuck, [])
