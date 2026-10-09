"""Headless processing CLI: child processes, failures, cancellation, contracts, audits (spec 003 T11; spec 007 T13).

Moved from scripts/petsys_manager_cli_check.py. Each action is launched as a real child process from
commands.build_internal (sys.executable -u -m src.cornell.cli ...), without DISPLAY/MPLBACKEND, on
synthetic fixtures whose every path contains spaces and shell metacharacters (``CLIFixtures``); no
hardware or GUI. The audits keep checking that the Manager never imports gui_cornell or scripts*.
"""

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from threading import Event
import unittest
from unittest.mock import patch

import pytest

from helpers import REPO
from manager_helpers import CLIFixtures, compact_record
from src.cornell import calibration as cal, cli, listmode as lm, qc, qc_report
from src.petsys_manager.contracts import SourceMode

FORBIDDEN_IMPORTS = {"scripts", "scripts_cornell", "scripts_imas", "gui_cornell", "docopt", "natsort", "colorama",
                     "tkinter", "customtkinter", "runpy"}
FORBIDDEN_TEXT = ("scripts_cornell", "scripts_imas", "gui_cornell", "scripts/", "scripts\\")
DISTRIBUTIONS = {"yaml": "pyyaml"}


def declared_dependencies():
    names = {}
    for line in (REPO / "process_petsys.yml").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s+- ([A-Za-z0-9_.\-]+)(?:==([^\s#]+))?\s*(#.*)?", line)
        if match:
            names[match[1].lower()] = match[2]
    return names


@pytest.mark.fr("003-FR-2", "003-FR-4", "003-FR-9", "003-FR-12", "003-FR-14", "003-FR-16")  # spec 003 T11
class CLIChecks(CLIFixtures, unittest.TestCase):
    fixture_prefix = "petsys-manager-cli-"

    # Actions --------------------------------------------------------------

    def test_cli_calibrate_child_process_literal_paths_match_in_process(self):
        for positions, data_format in ((5, "fixed"), (1, "compact")):
            h, descriptors, limits, request = self.calibration_request(positions=positions, data_format=data_format)
            proc, result, result_path = self.run_cli("calibrate", request)
            self.assertSucceeded(proc, result, result_path)
            outputs = request["outputs"]
            self.assertEqual([(o["kind"], o["path"]) for o in result["outputs"]],
                             [("encal", outputs["encal"]), ("calibration_sidecar", outputs["sidecar"]),
                              ("calibration_status", outputs["status"]), ("calibration_plot", outputs["plot"])])
            expected = cal.calibrate([d for d in descriptors], h.config, limits if positions > 1 else None,
                                     positions=positions, event_limit=10_000_000, batch_records=5000)
            self.assertEqual(Path(outputs["encal"]).read_text(encoding="utf-8"), cal.encal_text(expected))
            self.assertEqual(Path(outputs["status"]).read_text(encoding="utf-8"), cal.status_text(expected))
            self.assertEqual((result["summary"]["layout"], result["summary"]["format"]),
                             ("per_slab" if positions == 1 else "position", data_format))
        self.assertGreater(len(expected.factors), 10)
        self.assertEqual(result["summary"]["factors"], len(expected.factors))
        self.assertEqual(result["summary"]["status_counts"], expected.status_counts())
        self.assertEqual([i["accepted_sides"] for i in result["summary"]["inputs"]],
                         [f.accepted_sides for f in expected.files])
        sidecar = json.loads(Path(outputs["sidecar"]).read_text(encoding="utf-8"))
        self.assertEqual(sidecar["calibration_sha256"], result["outputs"][0]["sha256"])
        self.assertEqual(Path(outputs["plot"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    @pytest.mark.fr("003-FR-21")  # spec 003 T25.2
    def test_cli_calibrate_target_event_limit_and_coverage(self):
        """FR-21 (T25.2): the request's limit mode and T reach the library; the summary reports them."""
        h, descriptors, limits, request = self.calibration_request(positions=1, data_format="compact")
        request["options"].update(limit_mode="target", target_per_key=2000, memory_budget_mb=64, workers=2)
        code, result, _, stderr = self.in_process("calibrate", request)
        self.assertEqual(code, 0, stderr[-2000:])
        expected = cal.calibrate(descriptors, h.config, None, positions=1, limit_mode="target", target_per_key=2000,
                                 memory_budget=64 * 2 ** 20, batch_records=5000)
        self.assertEqual(Path(request["outputs"]["encal"]).read_text(encoding="utf-8"), cal.encal_text(expected))
        summary = result["summary"]
        self.assertEqual(summary["limit_plan"], expected.limit_plan)
        self.assertEqual(summary["coverage"], expected.coverage)
        self.assertEqual((summary["decoding"], summary["decoding"]["decodings"]), (expected.decoding, 1))   # T25.3
        self.assertIsNone(summary["passing_event_limit_per_file"])          # target mode counts kept sides
        self.assertEqual(summary["workers"], 2)                                   # T25.4: same files as in-process
        for bad in ({"limit_mode": "guess"}, {"limit_mode": "target", "target_per_key": None}):
            h, descriptors, limits, request = self.calibration_request(positions=1, data_format="compact")
            request["options"].update(bad)
            code, result, _, _ = self.in_process("calibrate", request)
            self.assertIn(code, (1, 2), bad)                 # 2: request refused; 1: processing refused
            self.assertTrue(result is None or result["status"] != "succeeded", bad)
            self.assertFalse(Path(request["outputs"]["encal"]).exists())

    @pytest.mark.slow  # ~11 s
    def test_cli_listmode_child_process_bytes_match_in_process(self):
        h, descriptors, maps, request = self.listmode_request()
        proc, result, result_path = self.run_cli("listmode", request)
        events = self.assertSucceeded(proc, result, result_path)
        kinds = [o["kind"] for o in result["outputs"]]
        self.assertEqual(kinds[:3], ["listmode", "listmode_provenance", "listmode_job"])
        self.assertEqual(set(kinds[3:]), {"listmode_debug_plot"})
        directory = Path(request["outputs"]["directory"])
        self.assertEqual(Path(result["outputs"][0]["path"]).parent, directory)
        expected = h.generate(descriptors, maps, h.lm_parent / "in process", debug=True)
        self.assertEqual(result["outputs"][0]["sha256"], expected.sha256)
        self.assertEqual(result["summary"]["records_written"], expected.records)
        self.assertGreater(expected.records, 100)
        self.assertEqual(result["summary"]["rejected"], expected.totals("rejected"))
        self.assertEqual(Path(result["outputs"][0]["path"]).name, expected.output.name)
        self.assertTrue(any("records_written" in e for e in events if e["kind"] == "progress"))

    @pytest.mark.fr("003-FR-15", "003-FR-22")  # spec 003 T25.5
    def test_cli_listmode_parallel_child_with_seed(self):
        """FR-22 (T25.5): a CLI child with lm_seed and 2 workers writes the in-process seeded .lm; a seedless
        parallel request and a negative seed are refused."""
        h, descriptors, maps, request = self.listmode_request(debug=False)
        request["options"].update(lm_seed=11, workers=2)
        proc, result, result_path = self.run_cli("listmode", request)
        self.assertSucceeded(proc, result, result_path)
        expected = h.generate(descriptors, maps, h.lm_parent / "in process seeded", lm_seed=11)
        self.assertEqual(result["outputs"][0]["sha256"], expected.sha256)
        for bad in ({"lm_seed": None, "workers": 2}, {"lm_seed": -1, "workers": 1}):
            h, descriptors, maps, request = self.listmode_request(debug=False)
            request["options"].update(bad)
            code, result, _, _ = self.in_process("listmode", request)
            self.assertIn(code, (1, 2), bad)
            self.assertTrue(result is None or result["status"] != "succeeded", bad)

    @pytest.mark.slow  # ~13 s
    @pytest.mark.fr("003-FR-15")  # spec 003 T33
    def test_cli_qc_parallel_child_with_seed(self):
        """FR-15 (T33): a CLI child with qc_seed and 2 workers gives the in-process seeded QC summary (one
        worker); the summary records seeds and workers; a seedless parallel request is refused."""
        h, descriptors, request = self.qc_request(count=900)
        request["options"].update(qc_seed=11, workers=2)
        proc, result, result_path = self.run_cli("qc", request)
        self.assertSucceeded(proc, result, result_path)
        ours = json.loads((Path(request["outputs"]["directory"]) / qc_report.SUMMARY).read_text(encoding="utf-8"))
        expected = qc.run_qc(descriptors, h.config, plots=True, slabs=True, source_mode=SourceMode.WITH,
                             acquisition_time_s=60, qc_seed=11)
        reference_dir = h.results / "in process seeded"
        qc_report.write_report(expected, reference_dir)
        theirs = json.loads((reference_dir / qc_report.SUMMARY).read_text(encoding="utf-8"))
        for key in ("totals", "findings", "minimodule_fits", "slab_fits", "inputs"):
            self.assertEqual(ours[key], theirs[key], key)
        self.assertEqual(ours["sampling"]["random_streams"], theirs["sampling"]["random_streams"])
        self.assertEqual((ours["sampling"]["workers"], theirs["sampling"]["workers"]), (2, 1))
        h, descriptors, request = self.qc_request(count=50, plots=False, slabs=False)
        request["options"].update(qc_seed=None, workers=2)
        code, result, _, _ = self.in_process("qc", request)
        self.assertEqual(code, 1)
        self.assertNotEqual(result["status"], "succeeded")

    @pytest.mark.slow  # ~12 s
    def test_cli_qc_child_process_report_matches_in_process(self):
        h, descriptors, request = self.qc_request()
        proc, result, result_path = self.run_cli("qc", request)
        self.assertSucceeded(proc, result, result_path)
        directory = Path(request["outputs"]["directory"])
        self.assertEqual(result["outputs"][-1], {**result["outputs"][-1], "kind": "qc_summary",
                                                  "path": str(directory / qc_report.SUMMARY)})
        random.seed(5)
        expected = qc.run_qc(descriptors, h.config, plots=True, slabs=True, source_mode=SourceMode.WITH,
                             acquisition_time_s=60)
        reference_dir = h.results / "in process"
        written = qc_report.write_report(expected, reference_dir)
        self.assertEqual(sorted(Path(o["path"]).name for o in result["outputs"]), sorted(p.name for p in written))
        ours = json.loads((directory / qc_report.SUMMARY).read_text(encoding="utf-8"))
        theirs = json.loads((reference_dir / qc_report.SUMMARY).read_text(encoding="utf-8"))
        # The single-time-channel slab draw is Python random (unseeded in the child): slab-dependent
        # values are excluded; populations, findings and minimodule fits do not depend on it.
        for key in ("totals", "findings", "minimodule_fits", "expectation", "source_mode", "acquisition_time_s",
                    "options", "cuts", "sampling"):
            self.assertEqual(ours[key], theirs[key], key)
        self.assertEqual(result["summary"]["totals"], theirs["totals"])
        self.assertEqual(result["summary"]["source_mode"], "with")
        self.assertGreater(theirs["totals"]["accepted_pairs"], 1000)

    @pytest.mark.fr("003-FR-13")  # spec 003 T29.2
    def test_cli_listmode_and_qc_in_place(self):
        """T29.2: with options.in_place, a CLI child writes LM and QC outputs into the existing folder beside the
        caller's files (same .lm as a new directory; QC files exclusive); a missing or symlinked folder, a bad
        report_title, or in_place false on an existing folder exit 2; a repeated in-place QC fails without
        replacing; write_report passes report_title to the PDF."""
        h, descriptors, maps, request = self.listmode_request(debug=False)
        folder = h.lm_parent / "run_lm-P5 in place ; & $x"
        folder.mkdir()
        (folder / "run.json").write_bytes(b"{}")
        request["options"]["in_place"] = True
        request["outputs"]["directory"] = str(folder)
        proc, result, result_path = self.run_cli("listmode", request)
        self.assertSucceeded(proc, result, result_path)
        self.assertTrue(all(Path(o["path"]).parent == folder for o in result["outputs"]))
        expected = h.generate(descriptors, maps, h.lm_parent / "in process new dir")
        self.assertEqual(result["outputs"][0]["sha256"], expected.sha256)
        self.assertEqual((folder / "run.json").read_bytes(), b"{}")

        h, descriptors, request = self.qc_request(count=600, plots=False, slabs=False)
        folder = h.results / "F18_qc-with-source in place"
        folder.mkdir()
        (folder / "run.json").write_bytes(b"{}")
        request["options"].update(in_place=True, report_title="F18_qc-with-source_2026-10-05_0533")
        request["outputs"]["directory"] = str(folder)
        proc, result, result_path = self.run_cli("qc", request)
        self.assertSucceeded(proc, result, result_path)
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         sorted([qc_report.PDF, qc_report.SUMMARY, "run.json"]))
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        code, result, _, _ = self.in_process("qc", request)              # same names again: exclusive, fails
        self.assertEqual(code, 1)
        self.assertNotEqual(result["status"], "succeeded")
        self.assertEqual({p.name: p.read_bytes() for p in folder.iterdir()}, before)
        for change in ({"options": {"report_title": ""}}, {"options": {"report_title": "a\nb"}},
                       {"options": {"in_place": False}}, {"outputs": {"directory": str(folder / "missing")}}):
            bad = json.loads(json.dumps(request))
            for section, values in change.items():
                bad[section].update(values)
            code, result, _, _ = self.in_process("qc", bad)
            self.assertEqual(code, 2, change)
        result_qc = qc.run_qc(descriptors, h.config)
        target = h.results / "title check"
        target.mkdir()
        with patch.object(qc_report, "write_pdf", wraps=qc_report.write_pdf) as pdf:
            qc_report.write_report(result_qc, target, in_place=True, title="Run title")
        self.assertEqual(pdf.call_args.args[2], "Run title")

    # Failure paths --------------------------------------------------------

    def test_cli_invalid_requests_exit_two_without_success(self):
        h, descriptors, base = self.qc_request(count=50, plots=False, slabs=False)
        directory = base["outputs"]["directory"]

        def mutate(**changes):
            value = json.loads(json.dumps(base))
            for dotted, new in changes.items():
                target, *path = dotted.split(".")
                node = value
                for part in [target, *path][:-1]:
                    node = node[part]
                if new is KeyError:
                    del node[[target, *path][-1]]
                else:
                    node[[target, *path][-1]] = new
            return value

        cases = {
            "missing key": mutate(files=KeyError),
            "unknown key": {**base, "extra": 1},
            "schema version": mutate(schema_version=2),
            "bool as version": mutate(schema_version=True),
            "action mismatch": mutate(action="listmode"),
            "relative input": mutate(**{"inputs": [{**base["inputs"][0], "path": "relative.ldat"}]}),
            "untyped input": mutate(**{"inputs": [{"path": base["inputs"][0]["path"]}]}),
            "unknown format": mutate(**{"inputs": [{**base["inputs"][0], "format": "ldat"}]}),
            "empty inputs": mutate(inputs=[]),
            "slabs without plots": mutate(**{"options.slabs": True}),
            "unknown option": mutate(**{"options.threads": 4}),
            "missing option": mutate(**{"options.pair_limit": KeyError}),
            "bool limit": mutate(**{"options.pair_limit": True}),
            "zero limit": mutate(**{"options.pair_limit": 0}),
            "bad source": mutate(**{"options.source_mode": "maybe"}),
            "negative duration": mutate(**{"options.acquisition_time_s": -60}),
            "relative output": mutate(**{"outputs.directory": "QC"}),
            "output parent missing": mutate(**{"outputs.directory": str(h.results / "missing" / "QC")}),
        }
        for name, request in cases.items():
            with self.subTest(name):
                code, result, _, _ = self.in_process("qc", request)
                self.assertFailedClosed(code, result, code=2, kind="invalid_request", absent=(directory,))
        existing = Path(directory)
        existing.mkdir()
        code, result, _, _ = self.in_process("qc", base)
        self.assertFailedClosed(code, result, code=2, kind="invalid_request")
        self.assertEqual(list(existing.iterdir()), [])
        existing.rmdir()
        # Byte-level problems, launched as real children.
        raw = {"duplicate key": '{"schema_version": 1, "schema_version": 1}', "NaN": '{"schema_version": NaN}',
               "not JSON": "schema_version: 1", "oversized": " " * (cli.MAX_REQUEST_BYTES + 1)}
        for name, text in raw.items():
            with self.subTest(name):
                request_path, result_path = self.write_request({}, name.replace(" ", "_"))
                request_path.write_text(text, encoding="utf-8")
                proc = self.launch("qc", request_path, result_path)
                result = json.loads(result_path.read_text(encoding="utf-8"))
                self.assertFailedClosed(proc, result, code=2, kind="invalid_request", absent=(directory,))
        # Incomplete LM metadata is rejected before any listmode work.
        _, _, _, lm_request = self.listmode_request(count=20, debug=False)
        lm_request["options"]["metadata"]["isotope"] = None
        proc, result, _ = self.run_cli("listmode", lm_request)
        self.assertFailedClosed(proc, result, code=2, kind="invalid_request",
                                absent=(lm_request["outputs"]["directory"],))
        self.assertIn("isotope", " ".join(result["errors"]))

    def test_cli_unusable_result_path_writes_nothing(self):
        _, _, request = self.qc_request(count=50, plots=False, slabs=False)
        request_path, result_path = self.write_request(request)
        result_path.write_text("operator file", encoding="utf-8")
        proc = self.launch("qc", request_path, result_path)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(result_path.read_text(encoding="utf-8"), "operator file")
        self.assertFalse(os.path.lexists(request["outputs"]["directory"]))
        for bad in ("relative result.json", str(self.work / "missing dir" / "result.json")):
            code = cli.main(["qc", "--request", str(request_path), "--result", bad], stdout=io.StringIO(),
                            stderr=io.StringIO())
            self.assertEqual(code, 2)
            self.assertFalse(os.path.lexists(bad))
        usage = subprocess.run([sys.executable, "-m", "src.cornell.cli", "acquire", "--request", "x", "--result",
                                "y"], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(usage.returncode, 2)
        self.assertFalse((REPO / "y").exists())
        # The output path may not repeat the result path.
        clash = json.loads(json.dumps(request))
        target = self.work / "clash ; & $x.json"
        clash["outputs"]["directory"] = str(target)
        clash_path = self.work / "clash request.json"
        clash_path.write_text(json.dumps(clash), encoding="utf-8")
        code = cli.main(["qc", "--request", str(clash_path), "--result", str(target)], stdout=io.StringIO(),
                        stderr=io.StringIO())
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["error_kind"], "invalid_request")

    @pytest.mark.slow  # ~5 s
    def test_cli_corrupted_inputs_exit_one_without_success(self):
        # Truncated compact QC input (last hit cut short).
        h, descriptors, request = self.qc_request(count=200, plots=True, slabs=False)
        with open(descriptors[1].path, "r+b") as stream:
            stream.truncate(descriptors[1].path.stat().st_size - 5)
        proc, result, _ = self.run_cli("qc", request)
        self.assertFailedClosed(proc, result, code=1, kind="input_or_processing",
                                absent=(request["outputs"]["directory"],))
        # Unmapped channel in a compact input.
        bad = h.root / "unmapped ; & $x.ldat"
        side = h.manual_side((1, 0), 2, +1, [20.0] * 5)
        bad.write_bytes(compact_record((side, [(1, 10.0, 999_999)] + side[1:])))
        request["inputs"] = [{"path": str(bad), "format": "compact", "population": "coincidence"}]
        proc, result, _ = self.run_cli("qc", request)
        self.assertFailedClosed(proc, result, code=1, kind="input_or_processing",
                                absent=(request["outputs"]["directory"],))
        # A compact file declared fixed: the fixed layout does not validate.
        _, cal_descriptors, _, cal_request = self.calibration_request(records=200, plot=False)
        cal_request["inputs"][1]["path"] = str(descriptors[0].path)
        proc, result, _ = self.run_cli("calibrate", cal_request)
        self.assertFailedClosed(proc, result, code=1, kind="input_or_processing",
                                absent=(cal_request["outputs"]["encal"], cal_request["outputs"]["sidecar"]))
        # Missing selected input.
        cal_request["inputs"][1]["path"] = str(self.work / "missing ; & $x.ldat")
        code, result, _, _ = self.in_process("calibrate", cal_request)
        self.assertFailedClosed(code, result, code=1, kind="input_or_processing")
        # Listmode: second input truncated after the first segment was written.
        _, lm_descriptors, _, lm_request = self.listmode_request(count=200, debug=False)
        with open(lm_descriptors[1].path, "r+b") as stream:
            stream.truncate(lm_descriptors[1].path.stat().st_size - 7)
        proc, result, _ = self.run_cli("listmode", lm_request)
        self.assertFailedClosed(proc, result, code=1, kind="input_or_processing")
        directory = Path(lm_request["outputs"]["directory"])
        self.assertFalse(any(p.suffix == ".lm" for p in directory.iterdir()))   # no merged LM
        self.assertTrue((directory / lm.JOB_FILE).is_file())                    # evidence kept

    def test_cli_numerical_failures_exit_one_without_success(self):
        # Every key below the 200-event minimum: no factor, nothing written.
        _, _, _, request = self.calibration_request(records=20, plot=True)
        proc, result, _ = self.run_cli("calibrate", request)
        self.assertFailedClosed(proc, result, code=1, kind="input_or_processing",
                                absent=tuple(request["outputs"].values()))
        self.assertIn("No key received a calibration factor", " ".join(result["errors"]))
        # An unexpected numerical exception before and after the report starts.
        _, _, qc_request = self.qc_request(count=300, plots=True, slabs=False)
        with patch("src.cornell.qc.fit_photopeaks", side_effect=FloatingPointError("forced fit failure")):
            code, result, _, stderr = self.in_process("qc", qc_request)
        self.assertFailedClosed(code, result, code=1, kind="unexpected", absent=(qc_request["outputs"]["directory"],))
        self.assertIn("FloatingPointError", stderr)
        with patch("src.cornell.qc_report.plot_floods", side_effect=ValueError("forced plot failure")):
            code, result, _, _ = self.in_process("qc", qc_request)
        self.assertFailedClosed(code, result, code=1, kind="unexpected")
        self.assertTrue(Path(qc_request["outputs"]["directory"]).is_dir())       # partial evidence kept, unlisted
        self.assertFalse((Path(qc_request["outputs"]["directory"]) / qc_report.SUMMARY).exists())
        # A calibration that does not read back is not a success.
        _, _, _, request = self.calibration_request(plot=False)
        with patch("src.cornell.cli.load_calibration", side_effect=cli.InputError("forced read-back failure")):
            code, result, _, _ = self.in_process("calibrate", request)
        self.assertFailedClosed(code, result, code=1, kind="input_or_processing")

    @pytest.mark.fr("003-FR-24")  # spec 003 T24
    def test_cli_cancellation_is_never_success(self):
        _, _, request = self.qc_request(count=300, plots=False, slabs=False)
        event = Event()
        event.set()
        code, result, _, _ = self.in_process("qc", request, cancel_event=event)
        self.assertFailedClosed(code, result, code=3, kind="cancelled", absent=(request["outputs"]["directory"],))
        self.assertEqual(result["status"], "cancelled")

        class After(Event):
            def __init__(self, calls):
                super().__init__()
                self.calls = calls

            def is_set(self):
                self.calls -= 1
                return self.calls < 0

        for action, build, name in (("qc", lambda: self.qc_request(count=3000, plots=False, slabs=False)[2], "QC"),
                                    ("listmode", lambda: self.listmode_request(count=300, debug=False)[3], "LM")):
            with self.subTest(action):
                request = build()
                # No separate validation pass polls cancellation now (FR-24): a smaller call budget.
                code, result, _, _ = self.in_process(action, request, cancel_event=After(5 if action == "qc" else 1))
                self.assertFailedClosed(code, result, code=3, kind="cancelled")
                directory = Path(request["outputs"]["directory"])
                self.assertFalse(directory.exists() and any(p.suffix == ".lm" for p in directory.iterdir()))
        # Cancellation that arrives after processing still publishes no success.
        request = self.calibration_request(plot=False)[3]
        late = Event()
        original = cli.run_calibrate

        def run_then_cancel(*args):
            value = original(*args)
            late.set()
            return value
        with patch.dict(cli.RUNNERS, {"calibrate": run_then_cancel}):
            code, result, _, _ = self.in_process("calibrate", request, cancel_event=late)
        self.assertFailedClosed(code, result, code=3, kind="cancelled")

    # Contracts ------------------------------------------------------------

    def test_cli_result_and_event_contracts(self):
        _, _, request = self.qc_request(count=200, plots=False, slabs=False)
        code, result, stdout, _ = self.in_process("qc", request)
        self.assertEqual(code, 0)
        result_path = self.work / f"result {self.count} ; & $x.json"
        self.assertEqual(cli.read_result(result_path)["status"], "succeeded")
        events = [cli.parse_event("[stdout] " + line) for line in stdout.splitlines()]
        self.assertTrue(all(events))
        self.assertIsNone(cli.parse_event("[stderr] " + stdout.splitlines()[0]))
        self.assertIsNone(cli.parse_event(cli.EVENT_PREFIX + "{not json"))
        self.assertIsNone(cli.parse_event("plain log line"))
        self.assertTrue(all(len(line) < 4096 for line in stdout.splitlines()))     # runner line bound
        # A changed output invalidates the success result.
        pdf = Path(result["outputs"][0]["path"])
        content = pdf.read_bytes()
        pdf.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        with self.assertRaises(cli.InputError):
            cli.read_result(result_path)
        self.assertEqual(cli.read_result(result_path, verify_hashes=False)["status"], "succeeded")
        pdf.write_bytes(content[:-1])
        with self.assertRaises(cli.InputError):
            cli.read_result(result_path, verify_hashes=False)
        for forged in ({**result, "status": "failed"}, {**result, "outputs": []}, {**result, "exit_code": 1},
                       {**result, "schema_version": 2}):
            path = self.work / f"forged {len(os.listdir(self.work))}.json"
            path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaises(cli.InputError):
                cli.read_result(path)
        # The result is never replaced; the request is echoed by digest.
        request_path = self.work / f"request {self.count} ; & $x.json"
        self.assertEqual(result["request"], {"path": str(request_path),
                                             "sha256": hashlib.sha256(request_path.read_bytes()).hexdigest()})
        before = result_path.read_bytes()
        code = cli.main(["qc", "--request", str(request_path), "--result", str(result_path)], stdout=io.StringIO(),
                        stderr=io.StringIO())
        self.assertEqual(code, 2)
        self.assertEqual(result_path.read_bytes(), before)
        self.assertFalse([p for p in self.work.iterdir() if p.name.endswith(".partial")])

    # Audits ---------------------------------------------------------------

    def runtime_probe(self, request_path, result_path):
        program = (
            "import json, sys, matplotlib\n"
            "from src.cornell import cli, calibration, listmode, qc, qc_report\n"
            f"code = cli.main(['qc', '--request', {str(request_path)!r}, '--result', {str(result_path)!r}])\n"
            "files = sorted({m.__file__ for m in list(sys.modules.values()) if getattr(m, '__file__', None)})\n"
            "print(json.dumps({'code': code, 'files': files, 'modules': sorted(sys.modules),\n"
            "                  'backend': matplotlib.get_backend().lower()}), file=sys.stderr)\n")
        env = {k: v for k, v in os.environ.items() if k not in ("MPLBACKEND", "DISPLAY", "WAYLAND_DISPLAY")}
        proc = subprocess.run([sys.executable, "-c", program], cwd=REPO, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=900)
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        return json.loads(proc.stderr.strip().splitlines()[-1])

    @pytest.mark.slow  # ~7 s
    def test_cli_source_audit_and_headless_runtime_imports(self):
        tracked = set(subprocess.run(["git", "ls-files", "src"], cwd=REPO, capture_output=True, text=True,
                                     check=True).stdout.split())
        # New intended runtime files that are not yet committed (audited like tracked ones).
        intended = set(subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "src/cornell",
                                       "src/petsys_manager"], cwd=REPO, capture_output=True, text=True,
                                      check=True).stdout.split())
        for path in sorted((REPO / "src/cornell").glob("*.py")) + sorted((REPO / "src/petsys_manager").glob("*.py")):
            relative = path.relative_to(REPO).as_posix()
            self.assertIn(relative, tracked | intended)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                          if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                          and node.body and isinstance(node.body[0], ast.Expr)
                          and isinstance(node.body[0].value, ast.Constant)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom):
                    roots = {node.module.split(".")[0]} if node.level == 0 and node.module else set()
                else:
                    roots = set()
                self.assertFalse(roots & FORBIDDEN_IMPORTS, (relative, roots))
                if relative.startswith("src/cornell/"):
                    # FR-15 (T25.4): worker processes only through the ordered pool helper; nothing else spawns.
                    allowed = {"multiprocessing"} if relative == "src/cornell/parallel.py" else set()
                    self.assertFalse(roots & ({"subprocess", "multiprocessing", "importlib"} - allowed),
                                     (relative, roots))
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
                        self.assertNotIn(node.attr, ("system", "popen", "execv", "execvp", "spawnv"), relative)
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                    for text in FORBIDDEN_TEXT:
                        self.assertNotIn(text, node.value, relative)
                if isinstance(node, ast.keyword) and node.arg == "shell":
                    self.assertTrue(isinstance(node.value, ast.Constant) and node.value.value is False, relative)
        _, _, request = self.qc_request(count=300, plots=True, slabs=True)
        request_path, result_path = self.write_request(request)
        probe = self.runtime_probe(request_path, result_path)
        self.assertEqual(probe["code"], 0)
        self.assertEqual(probe["backend"], "agg")
        # colorama is loaded by the shared reader's tqdm on Windows only; tracked code never imports it.
        self.assertFalse(set(probe["modules"]) & (FORBIDDEN_IMPORTS - {"colorama"} | {"PyQt5", "pyqtgraph"}))
        root = str(REPO.resolve()).lower()
        for name in probe["files"]:
            resolved = str(Path(name).resolve())
            self.assertNotIn("gui_cornell", resolved.lower())
            if resolved.lower().startswith(root):
                relative = Path(resolved).relative_to(REPO.resolve()).as_posix()
                self.assertIn(relative, tracked | intended)
                self.assertTrue(relative.startswith("src/"), relative)
        self.assertEqual(cli.read_result(result_path)["status"], "succeeded")
        self.probe_files = probe["files"]

    def test_cli_declared_dependencies_cover_actual_imports(self):
        declared = declared_dependencies()
        # This module's imports, loaded in a fresh interpreter as the script's own process did: in-process
        # sys.modules also holds what earlier tests imported (src/ldat_inspector in a serial full run).
        code = ("import sys, manager_helpers\n"
                "from src.cornell import calibration, cli, listmode, qc, qc_report\n"
                "from src.petsys_manager.contracts import SourceMode\n"
                "print('\\n'.join(getattr(m, '__file__', None) or '' for m in list(sys.modules.values())))")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO), str(REPO / "tests")]))
        loaded = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True,
                                check=True).stdout.splitlines()
        files = set()
        for name in loaded:
            path = Path(name).resolve() if name else None
            if path and path.suffix == ".py" and path.is_relative_to(REPO.resolve() / "src"):
                files.add(path)
        for module_name in ("cli", "calibration", "listmode", "qc", "qc_report", "inputs"):
            self.assertIn((REPO / "src/cornell" / f"{module_name}.py").resolve(), files)
        third_party = {}
        for path in sorted(files):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    roots = [node.module.split(".")[0]]
                else:
                    continue
                for name in roots:
                    if name not in sys.stdlib_module_names and name != "src":
                        third_party.setdefault(name, set()).add(path.relative_to(REPO).as_posix())
        self.assertIn("reportlab", third_party)                 # lazily imported by qc_report
        self.assertIn("openpyxl", third_party)
        for name, users in sorted(third_party.items()):
            self.assertIn(DISTRIBUTIONS.get(name, name).lower(), declared, f"{name} imported by {sorted(users)}")
        import reportlab
        self.assertEqual(declared["reportlab"], reportlab.Version)
        # The revision that declared reportlab added only that line (T11). Later specs add their own
        # dependencies (spec 006: pytest), so the comparison is that commit with its parent, not the working file.
        added_in = subprocess.run(["git", "log", "--format=%H", "-S", "reportlab==", "--", "process_petsys.yml"],
                                  cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
        self.assertTrue(added_in)
        before, after = (subprocess.run(["git", "show", f"{revision}:process_petsys.yml"], cwd=REPO,
                                        capture_output=True, text=True, check=True).stdout.splitlines()
                         for revision in (f"{added_in[-1]}~1", added_in[-1]))
        added = [line for line in after if line not in before]
        self.assertEqual([line for line in before if line not in after], [])
        self.assertEqual([re.sub(r"\s+#.*", "", line).strip() for line in added], [f"- reportlab=={reportlab.Version}"])
