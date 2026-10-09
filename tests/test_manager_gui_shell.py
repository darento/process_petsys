"""Manager shell: tabs, log, profiles, INI/YAML controls, readiness, main-thread rule, relocated
assets, Inspector independence (spec 003 T13; spec 007 T15).

Moved from scripts/petsys_manager_gui_check.py --shell. Withdrawn windows on fixture profiles
(``gui``: skipped without a display).
"""

import ast
from pathlib import Path
import shutil
import subprocess
import textwrap
import threading

import pytest

from exe_programs import petsys_manager_gui as gui
from helpers import REPO
from manager_gui_helpers import FixtureProbe, GUIBase, PY, STAMP, child, pump, texts, world
from src.petsys_manager.settings import MachineProfile, SystemProbe, load_profile, save_profile


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
                                    "LM File Generation", "System Quality Control"))
        self.assertIn("Output Log:", texts(app.root))
        log = self.log(app)
        self.assertIn(f"PETsys Manager {gui.__version__}; checkout", log)
        self.assertIn(f"Loaded profile {path}", log)
        new = {thread.name for thread in threading.enumerate()} - before
        self.assertEqual(new, {"petsys-preflight"})
        self.assertTrue(all(name == "petsys-preflight" for name in app.session.probe.threads))
        for key, button in app.buttons.items():  # T14: only DAQD start is possible before a daemon runs
            self.assertEqual(button.cget("state"), "normal" if key == "daqd" else "disabled", key)
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
