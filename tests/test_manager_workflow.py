"""Workflow coordinator: pipelines, manual actions, faults, run folders (spec 003 T12; spec 007 T13).

Moved from scripts/petsys_manager_workflow_check.py. The real CommandRunner and AcquisitionService run
against a check-only tool backend (``ToolWorld``): fake acquisition/converter/bias children write files
(converters copy seeded synthetic LDAT from ``ListmodeFixtures``/``QCFixtures``), and src.cornell.cli
stages are either faked (result manifests with real digests) or launched as real child processes.
Faults are injected at every stage. No hardware.
"""

from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import pytest

from helpers import REPO
from manager_helpers import (METADATA, FakeChild, FixtureProbe, ListmodeFixtures, PrivateOutput, QCFixtures,
                             ToolWorld, encode_compact, sha256)
from src.petsys_manager import workflow as wf
from src.petsys_manager.artifacts import read_manifest
from src.petsys_manager.contracts import (Action, DataFormat, InputDescriptor, Population, ResultStatus,
                                          SourceMode, to_plain)
from src.petsys_manager.runner import RunnerPolicy
from src.petsys_manager.settings import (AcquisitionSafety, MachineProfile, ProcessingLimits, ProfileError,
                                         RunOptions, ToolCapabilities, load_profile, preflight, save_profile)

TOOLS = ("daqd", "init_system", "acquire_sipm_data", "set_bias", "convert_raw_to_coincidence")
SAFETY = AcquisitionSafety(startup_timeout_s=30.0, growth_window_s=30.0, poll_interval_s=0.05, min_growth_bytes=1,
                           max_attempts=2, retry_delay_s=0.0, terminate_grace_s=0.2)
POLICY = RunnerPolicy(poll_interval_s=0.01, terminate_grace_s=0.2, reap_timeout_s=2.0, drain_timeout_s=1.0)
FAULTS = ("spawn", "nonzero", "invalid", "missing", "stop", "stop_after")
EXPECTED = {"spawn": {ResultStatus.LAUNCH_ERROR, ResultStatus.FAILED}, "nonzero": {ResultStatus.FAILED},
            "invalid": {ResultStatus.FAILED}, "missing": {ResultStatus.FAILED}, "stop": {ResultStatus.CANCELLED},
            "stop_after": {ResultStatus.CANCELLED}, "stale": {ResultStatus.FAILED},
            "outside": {ResultStatus.FAILED}}


def forged(options, **changes):
    """Run options that bypass RunOptions validation: a request the GUI can no longer build (FR-10)."""
    options = replace(options)
    for name, value in changes.items():
        object.__setattr__(options, name, value)
    return options


@pytest.mark.fr("003-FR-5", "003-FR-7", "003-FR-9", "003-FR-10", "003-FR-11", "003-FR-13", "003-FR-14",
                "003-FR-16")  # spec 003 T12
class WorkflowChecks(PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-wf-"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fixtures = {}

    def setUp(self):
        self.root = self.output / self._testMethodName.replace("test_workflow_", "")[:14]
        self.root.mkdir()
        self.events = []
        self.coordinator = None

    # Fixtures ---------------------------------------------------------------

    def assets(self, kind):
        """Shared per kind: processing YAML/map/limits and seeded synthetic LDAT sources."""
        if kind not in self.fixtures:
            base = self.output / f"assets-{kind}"
            if kind == "lm":
                h = ListmodeFixtures.at(base / "h")
                descriptors, maps = h.inputs(names=("acq_coinc_1.ldat", "acq_coinc_2.ldat"), count=1500,
                                             random_slabs=False)
                # The manager converts and processes compact coincidence only (FR-10): compact sources.
                descriptors = h.compact_twins(descriptors, count=1500, random_slabs=False)
                files = {"cog_limits_file": maps["cog_limits"].path, "doi_limits_file": maps["doi_limits"].path,
                         "calibration_file": maps["calibration"].path, "pair_map_file": maps["pairs"].path,
                         "region_map_file": maps["regions"].path}
            else:
                h = QCFixtures.at(base / "h")
                descriptors = h.inputs(count=600, names=("qc_with_source_coincCompact_1.ldat",
                                                          "qc_with_source_coincCompact_2.ldat"))
                files = {}
            self.fixtures[kind] = (h.root, h.root / "configs/processing.yaml", tuple(d.path for d in descriptors),
                                   files, h)
        return self.fixtures[kind][:4]

    def profile(self, kind="lm", **changes):
        processing_root, config, sources, files = self.assets(kind)
        tools = self.root / "tools"
        tools.mkdir(exist_ok=True)
        for tool in TOOLS:
            (tools / tool).write_text("fixture marker", encoding="utf-8")
        for name in ("data", "encal", "reports", "lm", "private"):
            (self.root / name).mkdir(exist_ok=True)
        for name in ("card0", "card1", "selected.ini"):
            (self.root / "private" / name).write_text("fixture", encoding="utf-8")
        values = dict(petsys_folder=str(tools), processing_root=str(processing_root),
                      ini_file=str(self.root / "private/selected.ini"), yaml_file=str(config),
                      data_dir=str(self.root / "data"), calibration_dir=str(self.root / "encal"),
                      report_dir=str(self.root / "reports"), lm_dir=str(self.root / "lm"),
                      cards=(str(self.root / "private/card0"), str(self.root / "private/card1")),
                      safety=SAFETY, capabilities=ToolCapabilities(),
                      limits=ProcessingLimits(), lm_metadata=METADATA,
                      **{name: str(path) for name, path in files.items()})
        values.update(changes)
        profile = MachineProfile(**values)
        path = self.root / "manager profile.yaml"
        if path.exists():
            path.unlink()
        save_profile(profile, path)
        self.profile_path = path
        self.sources = sources
        return load_profile(path)

    def raw_fixture(self, name):
        """A RAW acquisition the converter can read: <prefix>.rawf plus its .idxf index."""
        raw = self.root / "private" / f"{name}.rawf"
        raw.write_bytes(b"\x01" * 64)
        raw.with_suffix(".idxf").write_bytes(b"0\t64\t0\t10\t0.0\t0.0\n")
        return raw

    def settings(self, action, options=None, inputs=(), *, kind="lm", profile=None, **changes):
        profile = profile or self.profile(kind, **changes)
        report = preflight(profile, action, options, inputs, repo_root=REPO, probe=FixtureProbe())
        self.assertTrue(report.ready, report.issues)
        return report.settings

    def external_digests(self, settings):
        paths = [self.profile_path] + [Path(p) for name, p in settings.paths.items()
                                        if p is not None and name not in ("processing_root", "petsys_folder")
                                        and Path(p).is_file()]
        paths += [Path(d.path) for d in settings.inputs] + [Path(p) for p in self.sources]
        return {str(p): sha256(p) for p in paths}

    def execute(self, settings, *, faults=None, real_cli=False, stop_after=None, plant=None, prerequisite=None,
            timeout=600):
        world = ToolWorld(self, faults=faults, sources=self.sources, real_cli=real_cli)
        self.events = []

        def sink(event):
            self.events.append(event)
            if plant and event.kind == "stage_started" and event.identity.stage_id == "conversion":
                Path(event.payload["directory"], plant).write_bytes(b"old converter output")
            if stop_after and event.kind == "stage_finished" and event.identity.stage_id == stop_after:
                world.stop_active()
            if stop_after == "acquisition" and event.kind == "acquisition_finished":
                world.stop_active()
        self.log_lines = []
        self.coordinator = wf.WorkflowCoordinator(backend=world, policy=POLICY, update_sink=sink,
                                                  log_sink=self.log_lines.append)
        before = self.external_digests(settings)
        snapshot = to_plain(settings)
        plan = wf.prepare(settings)
        handle = self.coordinator.start(plan, prerequisite=prerequisite or (lambda: (True, "initialized (fixture)")))
        outcome = handle.wait(timeout)
        self.assertIsNotNone(outcome, "workflow did not finish")
        self.assertFalse(self.coordinator.active)
        self.assertEqual(self.external_digests(settings), before)          # profile/YAML/maps/inputs untouched
        self.assertEqual(to_plain(settings), snapshot)
        if outcome.run_root is not None:
            manifest = read_manifest(outcome.run_root)
            self.assertEqual(manifest["status"], outcome.status.value)
            self.assertFalse([a for a in manifest["attempts"] if a["status"] == "partial"])
            self.assertEqual(json.loads(json.dumps(to_plain(manifest["settings"]))),
                             json.loads(json.dumps(snapshot)))
        sequences = [e.sequence for e in self.events]
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertEqual((self.events[0].kind, self.events[-1].kind), ("workflow_started", "workflow_finished"))
        return outcome, world

    def requests(self, outcome):
        found = {}
        for stage in outcome.stages:
            if stage.directory is not None and (stage.directory / wf.REQUEST).is_file():
                found[stage.stage_id] = json.loads((stage.directory / wf.REQUEST).read_text(encoding="utf-8"))
        return found

    # Successful graphs -------------------------------------------------------

    @pytest.mark.fr("003-FR-1")  # spec 003 T25.1
    def test_workflow_pipeline_uses_exact_recorded_artifacts(self):
        options = RunOptions(duration_s=10.0, splits=2)
        settings = self.settings(Action.PIPELINE, options)
        data = Path(settings.paths["data_dir"])
        (data / "acquisition_coincCompact_1.ldat").write_bytes(self.sources[0].read_bytes())   # old look-alike
        outcome, world = self.execute(settings)
        self.assertTrue(outcome.succeeded, outcome.message)
        self.assertEqual([s.stage_id for s in outcome.stages], list(wf.STAGES[Action.PIPELINE]))
        self.assertEqual(outcome.message, "All stages completed; conversion outputs structure-checked, calibration "
                                          "and LM validated the records they read")
        self.assertEqual(outcome.run_root.parent, data)
        tools = [(stage, tool) for stage, tool, _ in world.launched]
        self.assertEqual(tools, [("acquisition", "acquire_sipm_data"), ("conversion", "convert_raw_to_coincidence"),
                                 ("calibration", "cli:calibrate"), ("listmode", "cli:listmode")])
        acquire, convert = world.launched[0][2], world.launched[1][2]
        self.assertEqual(float(acquire[acquire.index("--time") + 1]), 10.0)
        self.assertIn("--writeBinaryCompact", convert)
        self.assertNotIn("--writeBinaryFixed", convert)
        raw_prefix = acquire[acquire.index("-o") + 1]
        self.assertEqual(convert[convert.index("-i") + 1], raw_prefix)            # this run's RAW, exactly
        conversion = outcome.outputs("conversion")
        self.assertEqual([a.path.name for a in conversion], ["acquisition_coincCompact_1.ldat",
                                                             "acquisition_coincCompact_2.ldat"])
        self.assertTrue(all(a.input_descriptor.validated and a.input_descriptor.format == DataFormat.COMPACT
                            and a.input_descriptor.population == Population.COINCIDENCE for a in conversion))
        manifest = read_manifest(outcome.run_root)
        record = next(a for a in manifest["attempts"] if a["stage_id"] == "conversion")
        # T34: the empty split and the converter's .lidx files are recorded, then removed after success.
        removed = {Path(a["path"]).name: a for a in record["artifacts"] if a["disposable"]}
        self.assertEqual(sorted(removed), ["acquisition_coincCompact_1.lidx", "acquisition_coincCompact_2.lidx",
                                           "acquisition_coincCompact_9.ldat", "acquisition_coincCompact_9.lidx"])
        for name, item in removed.items():
            self.assertEqual((item["kind"], item["cleanup"]), ("ldat" if name.endswith(".ldat") else "index",
                                                               "removed"), name)
            self.assertNotIn(item["path"], record["outputs"])
        stage_dir = outcome.stages[1].directory
        self.assertEqual(sorted(p.name for p in stage_dir.iterdir()), ["acquisition_coincCompact_1.ldat",
                                                                        "acquisition_coincCompact_2.ldat"])
        self.assertTrue(any("Removed 4 of 4 unused converter file(s) (3 .lidx index, 1 empty .ldat)" in line
                            for line in self.log_lines), self.log_lines[-5:])
        requests = self.requests(outcome)
        expected_inputs = [{"path": str(a.path), "format": "compact", "population": "coincidence"} for a in conversion]
        self.assertEqual(requests["calibration"]["inputs"], expected_inputs)
        self.assertEqual(requests["listmode"]["inputs"], expected_inputs)
        encal = {a.kind: a.path for a in outcome.outputs("calibration")}
        self.assertEqual(requests["listmode"]["files"]["calibration"], str(encal["encal"]))
        self.assertEqual(requests["listmode"]["files"]["calibration_sidecar"], str(encal["calibration_sidecar"]))
        # Owner 2026-10-02: the pipeline writes its own Acq. Time (10 s) as both header times, not the profile's
        # 300.5/299.0 s; the other nine fields stay the profile's.
        timed = replace(METADATA, acquisition_time_s=10.0, measurement_time_s=10.0)
        self.assertEqual(requests["listmode"]["options"]["metadata"], to_plain(timed))
        header_times = {"acquisition_time_s": 10.0, "measurement_time_s": 10.0, "source": "pipeline Acq. Time"}
        self.assertEqual(dict(outcome.stages[-1].details["lm_header_times"]), header_times)
        record = next(a for a in read_manifest(outcome.run_root)["attempts"] if a["stage_id"] == "listmode")
        self.assertEqual(dict(record["details"]["lm_header_times"]), header_times)
        self.assertEqual(requests["listmode"]["options"]["batch_records"], 1000)
        for stage in outcome.stages:
            for artifact in stage.artifacts:
                self.assertTrue(artifact.path.is_relative_to(stage.directory), artifact.path)
        self.assertIsNone(outcome.qc_findings)
        # FR-1 (T25.1): every stage records its elapsed time; recorded attempts and finished messages carry it.
        self.assertTrue(all(isinstance(stage.details["elapsed_s"], float) and stage.details["elapsed_s"] >= 0
                            for stage in outcome.stages))
        attempts = {a["stage_id"]: a for a in read_manifest(outcome.run_root)["attempts"]}
        for stage in ("conversion", "calibration", "listmode"):
            self.assertIsInstance(attempts[stage]["details"]["elapsed_s"], float)
            finished = [e for e in self.events if e.kind == "stage_finished" and e.identity.stage_id == stage]
            self.assertEqual(len(finished), 1)
            self.assertRegex(finished[0].message, r" \(in \d+\.\d s\)\Z")
            self.assertEqual(finished[0].payload["elapsed_s"], attempts[stage]["details"]["elapsed_s"])

    @pytest.mark.fr("003-FR-15", "003-FR-21")  # spec 003 T25.2
    def test_workflow_calibration_request_limit_mode(self):
        """FR-21 (T25.2): the calibration request carries the run's limit mode and the profile's T."""
        for mode in ("target", "reference"):
            with self.subTest(mode):
                settings = self.settings(Action.CALIBRATE, RunOptions(calibration_limit_mode=mode), self.lm_inputs())
                outcome, _ = self.execute(settings)
                self.assertTrue(outcome.succeeded, outcome.message)
                options = self.requests(outcome)["calibration"]["options"]
                self.assertEqual((options["limit_mode"], options["target_per_key"], options["event_limit"],
                                  options["memory_budget_mb"]), (mode, 3000, 10_000_000, 8192))
                from src.cornell.parallel import resolve_workers
                self.assertEqual(options["workers"], resolve_workers(0))          # FR-15: 0 = automatic

    @pytest.mark.fr("003-FR-1")  # spec 003 T25.1
    def test_workflow_elapsed_text(self):
        self.assertEqual([wf.format_elapsed(v) for v in (-1, 0, 42.34, 59.94, 60, 724, 3599.4, 3600, 3725)],
                         ["0.0 s", "0.0 s", "42.3 s", "59.9 s", "1 min 00 s", "12 min 04 s", "59 min 59 s",
                          "1 h 00 min", "1 h 02 min"])

    @pytest.mark.slow  # ~6 s
    @pytest.mark.fr("003-FR-22")  # spec 003 T21, T25.5
    def test_workflow_compact_pipeline_real_cli_at_the_conversion_hit_limit(self):
        """FR-10/FR-22: stock converter: compact conversion -> calibration -> LM with real CLI children; LM
        decodes at the conversion hit limit. (Compact = fixed bytes is a library check: listmode/calibration.)"""
        settings = self.settings(Action.PIPELINE, RunOptions(duration_s=10.0, splits=2))
        outcome, world = self.execute(settings, real_cli=True)
        self.assertTrue(outcome.succeeded, outcome.message)
        convert = world.launched[1][2]
        self.assertIn("--writeBinaryCompact", convert)
        self.assertNotIn("--writeBinaryFixed", convert)
        conversion = outcome.outputs("conversion")
        self.assertEqual([a.path.name for a in conversion], ["acquisition_coincCompact_1.ldat",
                                                             "acquisition_coincCompact_2.ldat"])
        requests = self.requests(outcome)
        expected_inputs = [{"path": str(a.path), "format": "compact", "population": "coincidence"} for a in conversion]
        self.assertEqual(requests["calibration"]["inputs"], expected_inputs)
        self.assertEqual(requests["listmode"]["inputs"], expected_inputs)
        self.assertEqual(requests["listmode"]["options"]["hit_limit"], 16)
        from src.cornell.parallel import resolve_workers
        self.assertEqual((requests["listmode"]["options"]["lm_seed"], requests["listmode"]["options"]["workers"]),
                         (0, resolve_workers(0)))                               # T25.5: profile seed, automatic
        summary = outcome.stages[-1].details["summary"]
        self.assertEqual((summary["input_format"], summary["compact_hit_limit"]), ("compact", 16))
        self.assertGreater(summary["records_written"], 0)

    @pytest.mark.fr("003-FR-24")  # spec 003 T24
    def test_workflow_conversion_structure_check_then_the_reading_stage_validates(self):
        """FR-24: conversion checks the first 10,000 records; a defect after them passes conversion and fails
        the calibration stage that reads it, before calibration or LM publish anything."""
        settings = self.settings(Action.PIPELINE, RunOptions(duration_s=10.0, splits=1))
        h = self.fixtures["lm"][4]
        records = h.records(10500, seed=9, random_slabs=False)
        sides = [list(records[10200][0]), list(records[10200][1])]
        time_, energy, _ = sides[1][0]
        sides[1][0] = (time_, energy, 999999)
        records[10200] = (sides[0], sides[1])
        folder = self.root / "big"
        folder.mkdir()
        encode_compact(folder / "big_coinc_1.ldat", records)
        self.sources = (folder / "big_coinc_1.ldat",)
        outcome, world = self.execute(settings, real_cli=True)
        self.assertFalse(outcome.succeeded)
        self.assertEqual([s.stage_id for s in outcome.stages], ["acquisition", "conversion", "calibration"])
        conversion, calibration = outcome.stages[1], outcome.stages[2]
        self.assertEqual(conversion.status, ResultStatus.SUCCEEDED)
        self.assertFalse(conversion.artifacts[0].input_descriptor.validated)        # structure-checked only
        item = dict(conversion.details["ldat"][0])
        self.assertEqual((item["records_checked"], item["records"], item["whole_file_checked"]),
                         (10000, None, False))     # compact: count known only when read whole (FR-24)
        self.assertIn("first 10,000 records", conversion.details["output_check"])
        self.assertNotEqual(calibration.status, ResultStatus.SUCCEEDED)
        self.assertIn("unmapped channel 999999", calibration.message)
        self.assertEqual(calibration.artifacts, ())
        self.assertNotIn("cli:listmode", [tool for _, tool, _ in world.launched])
        # A small output is read whole by the check: its descriptor is validated.
        small, _ = self.execute(self.settings(Action.PIPELINE, RunOptions(duration_s=10.0, splits=2)))
        self.assertTrue(small.succeeded, small.message)
        self.assertTrue(all(a.input_descriptor.validated for a in small.outputs("conversion")))

    def test_workflow_qc_presets_compact_route_and_findings_not_verdict(self):
        for source, seconds in ((SourceMode.WITH, 60.0), (SourceMode.WITHOUT, 180.0)):
            with self.subTest(source.value):
                settings = self.settings(Action.QC, RunOptions(source_mode=source, plots=True, slabs=True),
                                         kind="qc")
                outcome, world = self.execute(settings)
                self.assertTrue(outcome.succeeded, outcome.message)
                acquire, convert = world.launched[0][2], world.launched[1][2]
                self.assertEqual(float(acquire[acquire.index("--time") + 1]), seconds)
                self.assertEqual(Path(convert[0]).name, "convert_raw_to_coincidence")
                self.assertIn("--writeBinaryCompact", convert)
                self.assertTrue(Path(acquire[acquire.index("-o") + 1]).name.startswith(f"acquisition_qc_{source.value}_source"))
                from src.cornell.parallel import resolve_workers
                request = self.requests(outcome)["qc"]
                self.assertTrue(all(i["format"] == "compact" and i["population"] == "coincidence"
                                    for i in request["inputs"]))
                self.assertEqual(request["options"], {"plots": True, "slabs": True, "source_mode": source.value,
                                                      "acquisition_time_s": seconds, "pair_limit": 1_000_001,
                                                      "in_place": True, "report_title": outcome.run_root.name,
                                                      "qc_seed": 0, "workers": resolve_workers(0)})
                qc_stage = outcome.stages[-1]
                self.assertEqual(qc_stage.details["process"], "completed")
                self.assertEqual(dict(qc_stage.details["findings"]), {"minimodules_without_hits": 3,
                                                                      "missing_time_channels": 1})
                self.assertEqual(dict(outcome.qc_findings), dict(qc_stage.details["findings"]))
                self.assertIn("not a detector verdict", outcome.message)
                text = json.dumps(to_plain(read_manifest(outcome.run_root))) + " ".join(e.message for e in self.events)
                self.assertNotIn("PASS", text)

    @pytest.mark.slow  # ~11 s
    def test_workflow_real_cli_children_pipeline_and_qc(self):
        settings = self.settings(Action.PIPELINE, RunOptions(duration_s=10.0, splits=2))
        outcome, world = self.execute(settings, real_cli=True)
        self.assertTrue(outcome.succeeded, outcome.message)
        lm = {a.kind: a.path for a in outcome.outputs("listmode")}
        encal = {a.kind: a.path for a in outcome.outputs("calibration")}
        provenance = json.loads(lm["listmode_provenance"].read_text(encoding="utf-8"))
        self.assertIn(str(encal["encal"]), set(strings(provenance)))
        self.assertGreater(outcome.stages[-1].details["summary"]["records_written"], 0)
        self.assertEqual([i["path"] for i in outcome.stages[-1].details["summary"]["inputs"]],
                         [str(a.path) for a in outcome.outputs("conversion")])
        self.assertTrue(any(e.kind == "stage_progress" for e in self.events))
        settings = self.settings(Action.QC, RunOptions(source_mode=SourceMode.WITHOUT, plots=True, splits=2),
                                 kind="qc")
        outcome, _ = self.execute(settings, real_cli=True)
        self.assertTrue(outcome.succeeded, outcome.message)
        summary = next(a.path for a in outcome.outputs("qc") if a.kind == "qc_summary")
        content = json.loads(summary.read_text(encoding="utf-8"))
        self.assertEqual((content["source_mode"], content["acquisition_time_s"]), ("without", 180.0))
        self.assertEqual(content["totals"]["records_read"], 1200)
        self.assertEqual(dict(outcome.qc_findings)["expected_minimodules"], content["findings"]["expected_minimodules"])
        self.assertEqual(summary.parent, outcome.stages[-1].directory)            # T29: QC writes in its stage folder

    # Failures ----------------------------------------------------------------

    def check_fault(self, action, stage, fault, settings):
        stages = wf.STAGES[action]
        if fault == "stop_after":
            outcome, world = self.execute(settings, stop_after=stage)
        else:
            outcome, world = self.execute(settings, faults={stage: fault})
        self.assertIn(outcome.status, EXPECTED[fault], outcome.message)
        index = stages.index(stage)
        started = [s for s, tool, _ in world.launched if tool != "set_bias"]
        self.assertFalse(set(started) & set(stages[index + 1:]), started)       # no successor ever launched
        self.assertEqual([s.stage_id for s in outcome.stages], list(stages[:index + 1]))
        last = outcome.stages[-1]
        if fault != "stop_after":
            self.assertNotEqual(last.status, ResultStatus.SUCCEEDED)
            self.assertEqual(last.artifacts, ())
        self.assertIsNone(outcome.qc_findings)
        manifest = read_manifest(outcome.run_root)
        latest = {}
        for record in manifest["attempts"]:
            latest[record["stage_id"]] = record
        if fault != "stop_after":
            self.assertNotEqual(latest[stage]["status"], "succeeded")
        self.assertFalse(set(latest) & set(stages[index + 1:]))
        if stage == "conversion" and fault == "invalid":     # T34: a failed conversion keeps every file
            names = {p.name for p in last.directory.iterdir()}
            indexes = {n for n in names if n.endswith(".lidx")}
            self.assertTrue(indexes and {n[:-5] + ".ldat" for n in indexes} <= names, names)
            if action == Action.PIPELINE:
                self.assertIn("acquisition_coincCompact_9.ldat", names)               # the empty split too
        return outcome

    @pytest.mark.slow  # ~6 s
    def test_workflow_faults_at_every_stage_stop_the_graph(self):
        graphs = ((Action.PIPELINE, "lm", RunOptions(duration_s=10.0, splits=2)),
                  (Action.QC, "qc", RunOptions(source_mode=SourceMode.WITH)))
        count = 0
        for action, kind, options in graphs:
            settings = self.settings(action, options, kind=kind)
            for stage in wf.STAGES[action]:
                for fault in FAULTS:
                    with self.subTest(action=action.value, stage=stage, fault=fault):
                        outcome = self.check_fault(action, stage, fault, settings)
                        count += 1
                        if fault == "spawn" and stage != "acquisition":
                            self.assertEqual(outcome.stages[-1].status, ResultStatus.LAUNCH_ERROR)
                        if stage == "acquisition" and fault in ("invalid", "missing"):
                            self.assertEqual(outcome.stages[-1].details["attempts"], 2)   # retried, then stopped
        self.assertEqual(count, (4 + 3) * len(FAULTS))
        # After every injected end, the coordinator accepts a new workflow and it can succeed.
        outcome, _ = self.execute(self.settings(Action.QC, RunOptions(), kind="qc"))
        self.assertTrue(outcome.succeeded, outcome.message)

    def test_workflow_old_similar_files_never_substitute_for_new_output(self):
        settings = self.settings(Action.PIPELINE, RunOptions(duration_s=10.0))
        # A file with the converter's exact output name, present before the converter ran.
        outcome, _ = self.execute(settings, faults={"conversion": "stale"}, plant="acquisition_coincCompact.ldat")
        self.assertEqual(outcome.status, ResultStatus.FAILED)
        self.assertIn("Pre-existing converter output", outcome.message)
        for stage, fault in (("calibration", "stale"), ("listmode", "stale"), ("calibration", "outside"),
                             ("listmode", "invalid")):
            with self.subTest(stage=stage, fault=fault):
                outcome, world = self.execute(settings, faults={stage: fault})
                self.assertIn(outcome.status, EXPECTED[fault], outcome.message)
                self.assertEqual(outcome.stages[-1].stage_id, stage)
                self.assertEqual(outcome.stages[-1].artifacts, ())

    def test_workflow_single_foreground_workflow_and_close(self):
        settings = self.settings(Action.QC, RunOptions(), kind="qc")
        world = ToolWorld(self, faults={"acquisition": "block"}, sources=self.sources)

        def block(command):
            world.launched.append((command.identity.stage_id, Path(command.argv[0]).name, command.argv))
            return FakeChild(0 if Path(command.argv[0]).name == "set_bias" else None, stdout=b"")
        world.launch = block
        self.coordinator = wf.WorkflowCoordinator(backend=world, policy=POLICY)
        plan = wf.prepare(settings)
        handle = self.coordinator.start(plan, prerequisite=lambda: (True, "ok"))
        self.assertTrue(self.coordinator.active)
        for _ in range(500):
            if world.launched:
                break
            time.sleep(0.01)
        with self.assertRaises(wf.WorkflowBusy):
            self.coordinator.start(plan, prerequisite=lambda: (True, "ok"))
        self.assertTrue(self.coordinator.close(30))
        outcome = handle.wait(1)
        self.assertEqual(outcome.status, ResultStatus.CANCELLED)
        self.assertFalse(self.coordinator.active)
        self.assertEqual([tool for _, tool, _ in world.launched], ["acquire_sipm_data", "set_bias"])  # bias off
        self.assertEqual(read_manifest(outcome.run_root)["status"], "cancelled")
        self.assertTrue(self.coordinator.close(1))
        world.launch = ToolWorld(self, sources=self.sources).launch
        handle = self.coordinator.start(plan, prerequisite=lambda: (True, "ok"))
        self.assertTrue(handle.wait(120).succeeded)

    def test_workflow_preflight_checks_entire_graph_before_creating_anything(self):
        settings = self.settings(Action.PIPELINE, RunOptions(duration_s=10.0))
        data = Path(settings.paths["data_dir"])
        bad_pairs = self.root / "bad pairs.txt"
        bad_pairs.write_text("1 2\n", encoding="utf-8")
        cases = {
            "corrupted pair map (last stage input)": self.settings(Action.PIPELINE, RunOptions(),
                                                                    pair_map_file=str(bad_pairs)),
            "missing destination": replace(settings, paths=replace_paths(settings, data_dir=self.root / "absent")),
            "pipeline group": replace(settings, options=forged(settings.options, population=Population.GROUP)),
            "pipeline fixed": replace(settings, options=forged(settings.options, output_format=DataFormat.FIXED)),
            "qc wrong preset": replace(self.settings(Action.QC, RunOptions(), kind="qc"),
                                       options=RunOptions(duration_s=30.0, output_format="compact")),
            "no processing YAML for convert validation": without_yaml(
                self.settings(Action.CONVERT, RunOptions(raw_input=str(self.raw_fixture("convert input"))))),
            "LM with group inputs": replace(self.settings(Action.LISTMODE, inputs=self.lm_inputs()),
                                            inputs=(InputDescriptor(self.sources[0], "fixed", "group"),)),
            "LM with fixed inputs": replace(self.settings(Action.LISTMODE, inputs=self.lm_inputs()),
                                            inputs=(InputDescriptor(self.sources[0], "fixed", "coincidence"),)),
            "calibration with fixed inputs": replace(self.settings(Action.CALIBRATE, inputs=self.lm_inputs()),
                                                     inputs=(InputDescriptor(self.sources[0], "fixed",
                                                                             "coincidence"),)),
            "LM metadata incomplete": replace(settings, profile=replace(settings.profile,
                                                                        lm_metadata=replace(METADATA, isotope=None))),
            "manual LM without profile header times": replace(
                self.settings(Action.LISTMODE, inputs=self.lm_inputs()),
                profile=replace(settings.profile, lm_metadata=replace(METADATA, acquisition_time_s=None,
                                                                      measurement_time_s=None))),
        }
        for name, case in cases.items():
            with self.subTest(name):
                listing = sorted(os.listdir(data))
                with self.assertRaises(wf.WorkflowError):
                    wf.prepare(case)
                self.assertEqual(sorted(os.listdir(data)), listing)
        # The pipeline supplies both header times itself, so empty profile times do not block it.
        untimed = replace(settings.profile, lm_metadata=replace(METADATA, acquisition_time_s=None,
                                                                measurement_time_s=None))
        wf.prepare(replace(settings, profile=untimed))
        times = {"lm_metadata.acquisition_time_s", "lm_metadata.measurement_time_s"}
        for action, blocked in ((Action.PIPELINE, set()), (Action.LISTMODE, times)):
            report = preflight(untimed, action, RunOptions(duration_s=10.0), self.lm_inputs(), repo_root=REPO,
                               probe=FixtureProbe())
            self.assertEqual({issue.field for issue in report.issues} & times, blocked, action.value)
        for value in (None, "settings", to_plain(settings)):
            with self.assertRaises(wf.WorkflowError):
                wf.prepare(value)
        daqd = preflight(load_profile(self.profile_path), Action.DAQD, repo_root=REPO, probe=FixtureProbe()).settings
        with self.assertRaises(wf.WorkflowError):
            wf.prepare(daqd)
        world = ToolWorld(self, sources=self.sources)
        coordinator = wf.WorkflowCoordinator(backend=world, policy=POLICY)
        listing = sorted(os.listdir(data))
        with self.assertRaises(wf.WorkflowError):
            coordinator.start(wf.prepare(settings), prerequisite=lambda: (False, "system not initialized"))
        self.assertEqual((sorted(os.listdir(data)), world.launched, coordinator.active), (listing, [], False))

    # Manual actions ------------------------------------------------------------

    def lm_inputs(self):
        return tuple(InputDescriptor(path, "compact", "coincidence") for path in self.assets("lm")[2])

    def test_workflow_manual_actions_are_single_stage_and_compact_only(self):
        lm_inputs = self.lm_inputs()
        cases = [
            (Action.ACQUIRE, RunOptions(duration_s=5.0), (), "lm", "data_dir", ["acquire_sipm_data"]),
            (Action.CALIBRATE, None, lm_inputs, "lm", "calibration_dir", ["cli:calibrate"]),
            (Action.LISTMODE, None, lm_inputs, "lm", "lm_dir", ["cli:listmode"]),
        ]
        for action, options, inputs, kind, destination, tools in cases:
            with self.subTest(action.value):
                settings = self.settings(action, options, inputs, kind=kind)
                outcome, world = self.execute(settings)
                self.assertTrue(outcome.succeeded, outcome.message)
                self.assertEqual([t for _, t, _ in world.launched], tools)
                self.assertEqual(outcome.run_root.parent, Path(settings.paths[destination]))
                if action == Action.LISTMODE:
                    request = self.requests(outcome)["listmode"]
                    self.assertEqual(request["files"]["calibration"], str(settings.paths["calibration_file"]))
                    self.assertIsNone(request["files"]["calibration_sidecar"])
                    self.assertEqual([i["path"] for i in request["inputs"]], [str(p) for p in self.sources])
                    self.assertEqual(request["options"]["metadata"], to_plain(METADATA))   # profile times as given
                    self.assertEqual(dict(outcome.stages[-1].details["lm_header_times"]),
                                     {"acquisition_time_s": 300.5, "measurement_time_s": 299.0, "source": "profile"})
        settings = self.settings(Action.QC_ANALYZE, inputs=tuple(InputDescriptor(p, "compact", "coincidence")
                                                                 for p in self.assets("qc")[2]), kind="qc")
        outcome, world = self.execute(settings)
        self.assertTrue(outcome.succeeded, outcome.message)
        options = self.requests(outcome)["qc"]["options"]
        self.assertEqual((options["source_mode"], options["acquisition_time_s"]), (None, None))   # not recorded
        self.assertEqual(outcome.run_root.parent, Path(settings.paths["report_dir"]))
        # Manual conversion: one stage, compact coincidence outputs (FR-10: no fixed or group conversion).
        raw = self.raw_fixture("run 7 ; & [x]")
        settings = self.settings(Action.CONVERT, RunOptions(raw_input=str(raw)))
        self.sources = self.assets("lm")[2][:1]
        outcome, world = self.execute(settings)
        self.assertTrue(outcome.succeeded, outcome.message)
        self.assertEqual([t for _, t, _ in world.launched], ["convert_raw_to_coincidence"])
        self.assertEqual(outcome.message, "Outputs structure-checked (first 10,000 records of each file); each "
                                          "processing stage validates the records it reads")   # not "validated"
        convert = world.launched[0][2]
        self.assertEqual(convert[convert.index("-i") + 1], str(raw.with_suffix("")))
        self.assertIn("--writeBinaryCompact", convert)
        artifacts = outcome.outputs("conversion")
        self.assertEqual([a.path.name for a in artifacts], ["run 7 ; & [x]_coincCompact.ldat"])
        self.assertEqual((artifacts[0].input_descriptor.format, artifacts[0].input_descriptor.population),
                         (DataFormat.COMPACT, Population.COINCIDENCE))
        for changes in ({"population": Population.GROUP}, {"output_format": DataFormat.FIXED}):
            with self.assertRaises(wf.WorkflowError):
                wf.prepare(replace(settings, options=forged(settings.options, **changes)))
        group = (InputDescriptor(self.sources[0], "fixed", "group"),)
        report = preflight(load_profile(self.profile_path), Action.LISTMODE, None, group, repo_root=REPO,
                           probe=FixtureProbe())
        self.assertFalse(report.ready)

    # T29: readable run folders ----------------------------------------------------

    @pytest.mark.fr("003-FR-21")  # spec 003 T29, T32
    def test_workflow_common_base_and_run_names(self):
        cases = {("run_0024.ldat",): "run_0024",
                 tuple(f"20260930_F18_950uCi_Run1_60s_coincCompact_{i}.ldat" for i in (0, 1, 10, 33)):
                     "20260930_F18_950uCi_Run1_60s",
                 ("X_coincCompact_1.ldat", "X_coincCompact_10.ldat"): "X",
                 ("X_coincCompact.ldat",): "X", ("acq_coinc_1.ldat", "acq_coinc_2.ldat"): "acq_coinc",
                 ("run 7 ; & [x]_coincCompact.ldat",): "run-7-x", ("_coincCompact.ldat",): "coincCompact",
                 ("a_1.ldat", "b_1.ldat"): "data", ("é.ldat",): "data", ("x" * 90 + ".ldat",): "x" * 56}
        for names, expected in cases.items():
            self.assertEqual(wf.common_base(Path("/d") / n for n in names), expected, names)
        # T32: .encal base as cornell_slab_en_cal.py: "_".join(basename.split("_")[0:-1]).
        for name, expected in (("20260930_F18_3mCi_Run1_10s_coincCompact_00000055.ldat",
                                "20260930_F18_3mCi_Run1_10s_coincCompact"), ("acq_coinc_2.ldat", "acq_coinc"),
                               ("X_coincCompact.ldat", "X"), ("run_0024.ldat", "run"), ("single.ldat", "single")):
            self.assertEqual(wf.reference_base(Path("/d") / name), expected, name)
        inputs = (InputDescriptor(Path("/d/F18_coincCompact_00000055.ldat"), "compact", "coincidence"),)
        for positions, encal in ((1, "F18_coincCompact_resolved.encal"),
                                 (5, "F18_coincCompact_position_5regions.encal")):
            settings = self.settings(Action.CALIBRATE, RunOptions(regions=positions), self.lm_inputs())
            request = wf.processing_request(settings, "calibration", inputs, Path("/r"), {})
            self.assertEqual({key: Path(value).name for key, value in request["outputs"].items()},
                             {"encal": encal, "sidecar": f"{encal}.json", "status": encal[:-6] + "_status.txt",
                              "plot": encal[:-6] + ".png"})
        now = datetime(2026, 10, 5, 5, 33)
        lm_inputs = self.lm_inputs()
        self.profile()
        raw = self.raw_fixture("run_0002")
        named = [(Action.CALIBRATE, RunOptions(regions=5), lm_inputs, "acq_coinc_cal-P5-target_2026-10-05_0533"),
                 (Action.CALIBRATE, RunOptions(regions=1, calibration_limit_mode="reference"), lm_inputs,
                  "acq_coinc_cal-P1-reference_2026-10-05_0533"),
                 (Action.LISTMODE, None, lm_inputs, "acq_coinc_lm-P5_2026-10-05_0533"),
                 (Action.CONVERT, RunOptions(raw_input=str(raw)), (), "run_0002_conv_2026-10-05_0533"),
                 (Action.ACQUIRE, RunOptions(acquisition_name="F18_Run1"), (), "F18_Run1_acq_2026-10-05_0533"),
                 (Action.PIPELINE, RunOptions(acquisition_name="F18_Run1", regions=3), (),
                  "F18_Run1_pipeline-P3_2026-10-05_0533"),
                 (Action.QC, RunOptions(source_mode=SourceMode.WITHOUT, duration_s=180.0), (),
                  "acquisition_qc-without-source_2026-10-05_0533")]
        for action, options, inputs, expected in named:
            settings = self.settings(action, options, inputs, kind="lm")
            self.assertEqual(wf.run_name(settings, now), expected)
        for bad in ("", "a b", "a.b", "_a", "x" * 49, "é"):
            with self.subTest(name=bad), self.assertRaises(ProfileError):
                RunOptions(acquisition_name=bad)

    def test_workflow_run_folders_layout_collision_and_overview(self):
        """FR-9/FR-13 (T29): readable names; single stages write in the run folder, the pipeline in numbered
        stage folders; a same-minute rerun takes _2; runs.tsv lists every finished run, failed ones too; an
        unwritable runs.tsv never changes the verdict."""
        class Fixed(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 5, 9, 36, 12)
        lm_inputs = self.lm_inputs()
        with patch.object(wf, "datetime", Fixed):
            outcomes = [self.execute(self.settings(Action.CALIBRATE, RunOptions(regions=5), lm_inputs))[0]
                        for _ in range(2)]
            listmode = self.execute(self.settings(Action.LISTMODE, None, lm_inputs))[0]
            pipeline = self.execute(self.settings(Action.PIPELINE, RunOptions(duration_s=10.0, splits=2,
                                                                              acquisition_name="F18_Run1")))[0]
            raw = self.raw_fixture("run_0002")
            failed = self.execute(self.settings(Action.CONVERT, RunOptions(raw_input=str(raw))),
                                  faults={"conversion": "nonzero"})[0]
        self.assertTrue(all(o.succeeded for o in (*outcomes, listmode, pipeline)))
        self.assertEqual([o.run_root.name for o in outcomes], ["acq_coinc_cal-P5-target_2026-10-05_0936",
                                                               "acq_coinc_cal-P5-target_2026-10-05_0936_2"])
        root = outcomes[0].run_root
        self.assertEqual(outcomes[0].stages[0].directory, root)
        self.assertEqual(sorted(p.name for p in root.iterdir()),
                         sorted([".history", "run.json", wf.REQUEST, wf.RESULT,
                                 *(Path(a.path).name for a in outcomes[0].outputs("calibration")
                                   if a.kind not in ("processing_request", "processing_result"))]))
        self.assertEqual(read_manifest(root)["run_id"], root.name)
        self.assertEqual(json.loads((root / "run.json").read_text(encoding="utf-8"))["status"], "succeeded")
        lm_root = listmode.run_root
        self.assertEqual(lm_root.name, "acq_coinc_lm-P5_2026-10-05_0936")
        self.assertEqual(Path(self.requests(listmode)["listmode"]["outputs"]["directory"]), lm_root)
        self.assertTrue(all(Path(a.path).parent == lm_root for a in listmode.outputs("listmode")))
        p_root = pipeline.run_root
        self.assertEqual(p_root.name, "F18_Run1_pipeline-P5_2026-10-05_0936")
        self.assertEqual(sorted(p.name for p in p_root.iterdir()),
                         [".history", "1_acquisition", "2_conversion", "3_calibration", "4_listmode", "run.json"])
        self.assertEqual([s.directory for s in pipeline.stages],
                         [p_root / "1_acquisition/attempt-1", p_root / "2_conversion", p_root / "3_calibration",
                          p_root / "4_listmode"])
        self.assertTrue((p_root / "1_acquisition/attempt-1/F18_Run1.rawf").is_file())
        self.assertEqual([a.path.name for a in pipeline.outputs("conversion")],
                         ["F18_Run1_coincCompact_1.ldat", "F18_Run1_coincCompact_2.ldat"])
        self.assertEqual(failed.status, ResultStatus.FAILED)
        self.assertEqual(failed.run_root.name, "run_0002_conv_2026-10-05_0936")

        def table(destination):
            return [line.split("\t") for line in
                    (Path(destination) / wf.RUNS).read_text(encoding="utf-8").splitlines()]
        cal = table(root.parent)
        self.assertEqual(cal[0], list(wf.RUNS_HEADER))
        encal = next(Path(a.path).name for a in outcomes[0].outputs("calibration") if a.kind == "encal")
        self.assertEqual(cal[1:], [["2026-10-05 09:36:12", name, "calibrate", "2 file(s): acq_coinc_1.ldat",
                                    "succeeded", encal] for name in (root.name, f"{root.name}_2")])
        self.assertEqual(table(lm_root.parent)[1][5], next(Path(a.path).name for a in listmode.outputs("listmode")
                                                          if a.kind == "listmode"))
        data = table(p_root.parent)
        self.assertEqual(data[0], list(wf.RUNS_HEADER))
        self.assertEqual(data[1][1:], [p_root.name, "pipeline", "-", "succeeded",
                                       f"4_listmode/{next(a.path.name for a in pipeline.outputs('listmode') if a.kind == 'listmode')}"])
        self.assertEqual(data[2][1:], [failed.run_root.name, "convert", "run_0002.rawf", "failed", "-"])
        self.assertEqual(len(data), 3)
        blocked = Path(self.settings(Action.CALIBRATE, RunOptions(regions=1), lm_inputs).paths["calibration_dir"])
        (blocked / wf.RUNS).unlink()
        (blocked / wf.RUNS).mkdir()                        # unwritable overview: logged, verdict unchanged
        outcome, _ = self.execute(self.settings(Action.CALIBRATE, RunOptions(regions=1), lm_inputs))
        self.assertTrue(outcome.succeeded, outcome.message)
        self.assertEqual(list((blocked / wf.RUNS).iterdir()), [])

    # Audits --------------------------------------------------------------------

    def test_workflow_events_identity_and_headless_imports(self):
        settings = self.settings(Action.QC, RunOptions(), kind="qc")
        outcome, _ = self.execute(settings)
        run_id = outcome.run_root.name
        self.assertTrue(all(e.identity.run_id == run_id for e in self.events))
        kinds = [e.kind for e in self.events]
        for stage in ("conversion", "qc"):
            started = [e for e in self.events if e.kind == "stage_started" and e.identity.stage_id == stage]
            finished = [e for e in self.events if e.kind == "stage_finished" and e.identity.stage_id == stage]
            self.assertEqual((len(started), len(finished)), (1, 1))
            self.assertLess(self.events.index(started[0]), self.events.index(finished[0]))
        self.assertIn("acquisition_attempt_started", kinds)
        self.assertEqual(self.events[-1].payload["status"], "succeeded")
        program = ("import sys; import src.petsys_manager.workflow; "
                   "bad = [m for m in ('tkinter', 'customtkinter', 'numba', 'matplotlib', 'scripts') if m in sys.modules]; "
                   "print(bad)")
        proc = subprocess.run([sys.executable, "-c", program], cwd=REPO, capture_output=True, text=True, check=True)
        self.assertEqual(proc.stdout.strip(), "[]")
        import src.cornell.listmode as lm
        self.assertEqual(wf.LM_BATCH_RECORDS, lm.DEFAULT_BATCH_RECORDS)


def strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def without_yaml(settings):
    return replace(settings, paths=replace_paths(settings, yaml_file=None))


def replace_paths(settings, **changes):
    values = dict(settings.paths)
    values.update(changes)
    return values
