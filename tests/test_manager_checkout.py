"""Clean runtime-checkout audit: tracked closure, declared imports, an isolated HEAD copy (spec 003 T18; spec 007 T14).

Moved from scripts/petsys_manager_checkout_check.py (--tracked-runtime). Runs on Windows and Linux.

- The runtime import closure of the manager (launcher, GUI, src/petsys_manager,
  src/cornell and the shared src modules they import) is computed from the source.
  Every file must be tracked (or listed in INTENDED), unmodified against HEAD, not
  ignored, and must not import or name the ignored scripts or the sibling GUI. While a
  runtime file is uncommitted the class skips, naming the files.
- Every third-party import must be declared in process_petsys.yml or be required
  by a declared distribution (recorded as transitive).
- ``git archive HEAD`` of exactly that closure is extracted to a private folder. Child
  interpreters (-E -s, no PYTHONPATH, an empty HOME/APPDATA/XDG, no MPLBACKEND or
  DISPLAY) run from another cwd. They import every module headlessly, show
  ``PETsysManager.py --help``, load a default profile without private settings, run the
  three processing CLI actions launched as the manager launches them, and run a
  manual LM workflow from an explicit external profile through the copy's
  WorkflowCoordinator. Outputs are compared with the working checkout.
Synthetic fixtures only; no hardware, no Tk window, no build/release.
"""

import ast
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import unittest

import pytest

from helpers import REPO
from manager_helpers import METADATA, CLIFixtures, PrivateOutput

ENTRIES = ["exe_programs/PETsysManager.py", "exe_programs/petsys_manager_gui.py"]
PACKAGES = ["src/petsys_manager", "src/cornell"]
ASSETS = ["exe_programs/assets/onco_logo.jpeg"]       # optional logo, loaded module-relative
INTENDED: list[str] = []      # new runtime files not yet committed (none at T18)
FORBIDDEN_IMPORTS = {"scripts", "scripts_cornell", "scripts_imas", "gui_cornell", "docopt", "natsort", "colorama",
                     "runpy", "PyQt5", "pyqtgraph", "tensorflow", "keras"}
FORBIDDEN_TEXT = ("scripts_cornell", "scripts_imas", "gui_cornell", "scripts/", "scripts\\")
PRIVATE_TEXT = ("C:\\Users\\", "C:/Users/", "/home/", "Desktop\\data", "Desktop/data")

PROBE = r'''
import json, os, sys
from pathlib import Path
copy, mode, params = Path(sys.argv[1]), sys.argv[2], json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
sys.path.insert(0, str(copy))
sys.path.insert(0, str(copy / "exe_programs"))
out = {"cwd": os.getcwd()}
if mode == "imports":
    import importlib
    for name in params["modules"]:
        importlib.import_module(name)
    import PETsysManager                                  # the launcher module itself, without main()
    from src.petsys_manager.settings import MachineProfile, default_profile_path, load_profile
    out["default_profile_path"] = str(default_profile_path())
    out["default_profile_exists"] = default_profile_path().exists()
    out["default_profile_is_empty"] = load_profile() == MachineProfile()
    out["tk_root"] = getattr(__import__("tkinter"), "_default_root", None) is not None
    out["matplotlib_backend"] = sys.modules["matplotlib"].get_backend() if "matplotlib" in sys.modules else None
elif mode == "workflow":
    from src.petsys_manager import workflow as wf
    from src.petsys_manager.contracts import Action, InputDescriptor
    from src.petsys_manager.settings import load_profile, preflight
    profile = load_profile(params["profile"])
    inputs = tuple(InputDescriptor(Path(p), "compact", "coincidence") for p in params["inputs"])   # FR-10
    report = preflight(profile, Action.LISTMODE, None, inputs)
    out["issues"] = [f"{i.field}: {i.message}" for i in report.issues]
    backend, launched = None, []
    if not sys.platform.startswith("linux"):
        # Check-only direct children: the production backend refuses non-Linux by design.
        import subprocess

        class Child:
            selectable_pipes = False

            def __init__(self, process):
                self.process, self.pid = process, process.pid
                self.stdout, self.stderr = process.stdout, process.stderr

            def poll(self):
                return self.process.poll()

            def group_alive(self):
                return self.poll() is None

            def terminate(self):
                self.process.terminate()

            def kill(self):
                self.process.kill()

            def wait(self, timeout):
                return self.process.wait(timeout=timeout)

        class Backend:
            def launch(self, command):
                launched.append([list(command.argv), str(command.cwd)])
                return Child(subprocess.Popen(command.argv, shell=False, cwd=command.cwd,
                                              env=dict(command.environment), stdin=subprocess.DEVNULL,
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        backend = Backend()
    out["backend"] = "production" if backend is None else "check-only direct children"
    out["launched"] = launched
    if report.ready:
        plan = wf.prepare(report.settings)
        handle = wf.WorkflowCoordinator(backend=backend).start(plan)
        outcome = handle.wait(600)
        out["status"] = None if outcome is None else outcome.status.value
        out["message"] = None if outcome is None else outcome.message
        out["run_root"] = None if outcome is None else str(outcome.run_root)
        out["outputs"] = [] if outcome is None else [[a.kind, str(a.path)] for a in outcome.outputs("listmode")]
files = {}
for module in list(sys.modules.values()):
    name = getattr(module, "__file__", None)
    if name:
        files[module.__name__] = str(Path(name).resolve())
out["module_files"] = files
print("PROBE-JSON " + json.dumps(out))
'''


def git(*args, binary=False):
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True)
    return proc.stdout if binary else proc.stdout.decode("utf-8")


def _module_path(name):
    parts = name.split(".")
    for candidate in (REPO.joinpath(*parts).with_suffix(".py"), REPO.joinpath(*parts, "__init__.py")):
        if candidate.is_file():
            return candidate
    return None


def import_roots(tree, package):
    """(dotted in-repo candidates, third-party/stdlib roots) of one module."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[:len(package.split(".")) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module
            names += [module] + [f"{module}.{alias.name}" for alias in node.names]
    return names


def runtime_closure():
    """Every in-repo .py file the manager can import, and third-party roots per file."""
    todo = [REPO / entry for entry in ENTRIES]
    for package in PACKAGES:
        todo += sorted((REPO / package).glob("*.py"))
    seen, third = set(), {}
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        relative = path.relative_to(REPO)
        package = ".".join(relative.with_suffix("").parts[:-1])
        for name in import_roots(ast.parse(path.read_text(encoding="utf-8")), package):
            if not name:
                continue
            if name.split(".")[0] in ("petsys_manager_gui", "PETsysManager"):   # launcher-relative imports
                todo.append(REPO / "exe_programs" / f"{name.split('.')[0]}.py")
                continue
            target = _module_path(name)
            if target is not None:
                todo.append(target)
                parent = target.parent
                while parent != REPO and (parent / "__init__.py").is_file():
                    todo.append(parent / "__init__.py")
                    parent = parent.parent
            elif name.split(".")[0] not in sys.stdlib_module_names and not (REPO / name.split(".")[0]).exists():
                third.setdefault(name.split(".")[0], set()).add(relative.as_posix())
    return sorted(p.relative_to(REPO).as_posix() for p in seen), third


def declared_dependencies():
    names = {}
    for line in (REPO / "process_petsys.yml").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s+- ([A-Za-z0-9_.\-]+)(?:==([^\s#]+))?\s*(#.*)?", line)
        if match:
            names[normal(match[1])] = match[2]
    return names


def normal(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_closure(declared):
    """Distributions required by the declared ones (installed metadata, unconditional requirements)."""
    from importlib import metadata
    found, todo = {}, list(declared)
    while todo:
        name = todo.pop()
        try:
            requires = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        for requirement in requires:
            if "extra ==" in requirement:
                continue
            match = re.match(r"[A-Za-z0-9_.\-]+", requirement)
            child = normal(match[0])
            if child not in found and child not in declared:
                found[child] = name
                todo.append(child)
    return found


def clean_environment(home):
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "MPLBACKEND", "DISPLAY", "WAYLAND_DISPLAY")}
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME"):
        env[key] = str(home)
    env["PYTHONNOUSERSITE"] = "1"
    return env


class CLIHelper(CLIFixtures, unittest.TestCase):
    """The CLI request builders and in-process runner for the copy tests, in their own private folder."""

    __test__ = False

    @classmethod
    def build(cls, test, name):
        cls.setUpClass()
        test.addCleanup(cls.tearDownClass)
        helper = cls()
        helper._testMethodName = name
        helper.setUp()
        return helper


@pytest.mark.fr("003-FR-2", "003-FR-16", "003-FR-18")  # spec 003 T18
class CheckoutChecks(PrivateOutput, unittest.TestCase):
    fixture_prefix = "pm-checkout-"

    @classmethod
    def setUpClass(cls):
        cls.closure, cls.third = runtime_closure()
        # The copy is HEAD: it audits the checkout only once the runtime files are committed (owner 2026-10-07).
        tracked = set(git("ls-files").split("\n"))
        modified = set(git("diff", "--name-only", "HEAD", "--", *cls.closure, *ASSETS).split())
        uncommitted = sorted((modified | {p for p in cls.closure + ASSETS if p not in tracked}) - set(INTENDED))
        if uncommitted:
            raise unittest.SkipTest(f"checkout: runtime files differ from HEAD ({', '.join(uncommitted)}); "
                                    "commit them to audit the checkout")
        super().setUpClass()
        cls.copy = cls.output / "runtime copy ; & [x]"
        cls.elsewhere = cls.output / "operator cwd"
        cls.home = cls.output / "empty home"
        cls.tools = cls.output / "probe"
        for folder in (cls.copy, cls.elsewhere, cls.home, cls.tools):
            folder.mkdir()
        tracked = [p for p in cls.closure + ASSETS if p not in INTENDED]
        archive = git("archive", "--format=tar", "HEAD", "--", *tracked, binary=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(cls.copy)
        for relative in INTENDED:
            target = cls.copy / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / relative, target)
        (cls.tools / "probe.py").write_text(PROBE, encoding="utf-8")
        cls.env = clean_environment(cls.home)

    def child(self, argv, *, cwd=None, timeout=900):
        return subprocess.run(argv, cwd=cwd or self.elsewhere, env=self.env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)

    def probe(self, mode, params):
        path = self.tools / f"{mode}.json"
        path.write_text(json.dumps(params), encoding="utf-8")
        proc = self.child([sys.executable, "-E", "-s", str(self.tools / "probe.py"), str(self.copy), mode, str(path)])
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        line = [text for text in proc.stdout.splitlines() if text.startswith("PROBE-JSON ")][-1]
        return json.loads(line[len("PROBE-JSON "):])

    def assertNothingFromCheckout(self, module_files):
        original = str(REPO.resolve()).lower()
        for name, path in module_files.items():
            self.assertFalse(path.lower().startswith(original + os.sep), (name, path))

    # Audit ------------------------------------------------------------------------------------

    def test_checkout_runtime_closure_is_tracked_unmodified_and_clean(self):
        tracked = set(git("ls-files").split("\n"))
        modified = set(git("diff", "--name-only", "HEAD", "--", *self.closure, *ASSETS).split())
        ignored = subprocess.run(["git", "check-ignore", "--no-index", *self.closure, *ASSETS], cwd=REPO,
                                 capture_output=True, text=True).stdout.split()
        print(f"\n  runtime closure: {len(self.closure)} modules + {len(ASSETS)} asset", file=sys.stderr)
        for relative in self.closure + ASSETS:
            self.assertTrue(relative in tracked or relative in INTENDED, f"untracked runtime file {relative}")
        self.assertEqual(sorted(modified - set(INTENDED)), [], "runtime files differ from HEAD (commit them)")
        self.assertEqual(ignored, [])
        for package in PACKAGES:
            self.assertIn(f"{package}/__init__.py", self.closure)
        for relative in self.closure:
            self.assertFalse(relative.startswith(("scripts", "gui_cornell")), relative)
            tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
            docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                          if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                          and node.body and isinstance(node.body[0], ast.Expr)
                          and isinstance(node.body[0].value, ast.Constant)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    roots = {node.module.split(".")[0]}
                else:
                    roots = set()
                self.assertFalse(roots & FORBIDDEN_IMPORTS, (relative, roots))
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                    for text in FORBIDDEN_TEXT + PRIVATE_TEXT:
                        self.assertNotIn(text, node.value, relative)

    def test_checkout_third_party_imports_are_declared(self):
        from importlib import metadata
        declared = declared_dependencies()
        transitive = requirement_closure(declared)
        providers = metadata.packages_distributions()
        report = []
        for name, users in sorted(self.third.items()):
            distributions = {normal(d) for d in providers.get(name, [])}
            if not distributions:   # Python 3.10 maps only wheels with top_level.txt (not pandas)
                try:
                    distributions = {normal(metadata.distribution(name).metadata["Name"])}
                except metadata.PackageNotFoundError:
                    pass
            self.assertTrue(distributions, f"{name} (imported by {sorted(users)}) is not installed")
            direct = distributions & set(declared)
            via = {d: transitive[d] for d in distributions if d in transitive}
            self.assertTrue(direct or via, f"{name} imported by {sorted(users)} is not declared")
            report.append(f"{name}: {', '.join(sorted(direct))}" if direct else
                          f"{name}: via {', '.join(f'{d} <- {p}' for d, p in sorted(via.items()))}")
        print("\n  " + "; ".join(report), file=sys.stderr)
        self.assertIn("reportlab", self.third)
        self.assertIn("customtkinter", self.third)

    # Isolated copy ----------------------------------------------------------------------------

    def test_checkout_copy_imports_headless_without_private_settings(self):
        modules = sorted(".".join(Path(p).with_suffix("").parts) for p in self.closure if p.startswith("src/"))
        modules = [m[:-len(".__init__")] if m.endswith(".__init__") else m for m in modules]
        result = self.probe("imports", {"modules": modules})
        self.assertEqual(Path(result["cwd"]).resolve(), self.elsewhere.resolve())
        self.assertTrue(Path(result["default_profile_path"]).resolve().is_relative_to(self.home.resolve()),
                        result["default_profile_path"])
        self.assertFalse(result["default_profile_exists"])
        self.assertTrue(result["default_profile_is_empty"])
        self.assertFalse(result["tk_root"])
        self.assertNothingFromCheckout(result["module_files"])
        for name in modules + ["petsys_manager_gui", "PETsysManager"]:
            self.assertTrue(Path(result["module_files"][name]).is_relative_to(self.copy.resolve()), name)
        self.assertFalse(set(result["module_files"]) & FORBIDDEN_IMPORTS - {"colorama"})
        proc = self.child([sys.executable, "-E", "-s", str(self.copy / "exe_programs/PETsysManager.py"), "--help"])
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("--profile", proc.stdout)
        # argparse wraps the default path: compare without whitespace.
        self.assertIn("".join(str(self.home).split()), "".join(proc.stdout.split()))
        self.assertEqual(list(self.home.rglob("petsys_manager.yaml")), [])          # nothing saved

    @pytest.mark.slow  # ~18 s
    def test_checkout_copy_runs_processing_cli_like_the_manager(self):
        from src.cornell import calibration as cal
        helper = CLIHelper.build(self, "test_cli_copy_cli")      # its own work folder
        fx, descriptors, limits, calibrate = helper.calibration_request(positions=5)
        _, _, _, listmode = helper.listmode_request(count=600)
        _, _, qc = helper.qc_request(count=600)
        for action, request in (("calibrate", calibrate), ("listmode", listmode), ("qc", qc)):
            with self.subTest(action):
                request_path, result_path = helper.write_request(request, f"copy {action}")
                # commands.build_internal: sys.executable -u -m src.cornell.cli, cwd = the checkout root.
                proc = self.child([sys.executable, "-E", "-s", "-u", "-m", "src.cornell.cli", action, "--request",
                                   str(request_path), "--result", str(result_path)], cwd=self.copy)
                self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
                from src.cornell import cli
                result = cli.read_result(result_path)
                self.assertEqual(result["status"], "succeeded")
                self.assertEqual(Path(result["executable"]), Path(sys.executable))
        expected = cal.calibrate(descriptors, fx.config, limits, positions=5, event_limit=10_000_000,
                                 batch_records=5000)
        self.assertEqual(Path(calibrate["outputs"]["encal"]).read_text(encoding="utf-8"), cal.encal_text(expected))
        self.assertEqual(Path(calibrate["outputs"]["status"]).read_text(encoding="utf-8"), cal.status_text(expected))

    @pytest.mark.slow  # ~6 s
    def test_checkout_copy_manual_lm_workflow_from_explicit_external_profile(self):
        from dataclasses import replace
        from src.petsys_manager.settings import MachineProfile, save_profile
        helper = CLIHelper.build(self, "test_cli_copy_lm")
        h, descriptors, maps, request = helper.listmode_request(count=600, debug=False)
        descriptors = h.compact_twins(descriptors, count=600, random_slabs=False)   # the manager reads compact only
        external = self.output / "operator settings"
        lm_out = self.output / "operator lm"
        for folder in (external, lm_out):
            folder.mkdir()
        profile = replace(MachineProfile(), processing_root=str(h.root), yaml_file=str(h.root / "configs/processing.yaml"),
                          lm_dir=str(lm_out), calibration_file=str(maps["calibration"].path),
                          cog_limits_file=str(maps["cog_limits"].path), doi_limits_file=str(maps["doi_limits"].path),
                          pair_map_file=str(maps["pairs"].path), region_map_file=str(maps["regions"].path),
                          lm_metadata=METADATA)
        profile_path = external / "petsys_manager.yaml"
        save_profile(profile, profile_path)
        before = profile_path.read_bytes()
        result = self.probe("workflow", {"profile": str(profile_path), "inputs": [str(d.path) for d in descriptors]})
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["status"], "succeeded", result.get("message"))
        self.assertEqual(Path(result["run_root"]).parent, lm_out)
        print(f"\n  manual LM backend: {result['backend']}", file=sys.stderr)
        if result["launched"]:      # the stage ran from the copy, exactly as build_internal builds it
            (argv, cwd), = result["launched"]
            self.assertEqual(argv[1:5], ["-u", "-m", "src.cornell.cli", "listmode"])
            self.assertEqual(Path(cwd).resolve(), self.copy.resolve())
        kinds = [kind for kind, _ in result["outputs"]]
        self.assertIn("listmode", kinds)
        self.assertNothingFromCheckout(result["module_files"])
        self.assertEqual(profile_path.read_bytes(), before)                  # the explicit profile is unchanged
        self.assertEqual([p for p in self.home.rglob("*") if p.name == "petsys_manager.yaml"], [])
        # Same LM bytes as the working checkout's in-process run of the same request.
        lm_file = Path(next(path for kind, path in result["outputs"] if kind == "listmode"))
        same = dict(request, outputs={"directory": str(self.output / "checkout lm")})
        code, reference, _, stderr = helper.in_process("listmode", same)
        self.assertEqual(code, 0, stderr[-2000:])
        ours = next(o for o in reference["outputs"] if o["kind"] == "listmode")
        self.assertEqual(Path(ours["path"]).read_bytes(), lm_file.read_bytes())
