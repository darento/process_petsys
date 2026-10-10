"""Profile settings, typed contracts and preflight (spec 003 T2; spec 007 T6).

Moved from scripts/petsys_manager_check.py --settings; fixture files stay in a private temp folder.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import queue
import sys
import unittest
from unittest.mock import patch

import pytest
import yaml

from helpers import REPO
from manager_helpers import PrivateOutput, SettingsFixtures, profile_fixture
from src.petsys_manager.contracts import (Action, Artifact, CommandResult, CommandSpec, DataFormat,
    FrozenMapping, Identity, InputDescriptor, Population, ResultStatus, RunEvent, SourceMode, freeze,
    to_plain)
from src.petsys_manager.settings import (AcquisitionSafety, LMMetadata, MachineProfile, ProcessingLimits,
    ProfileError, RunOptions, RunSettings, ToolCapabilities, default_profile_path, load_profile, preflight,
    profile_from_mapping, save_last_folders, save_profile, validate_calibration_factor)
from src.petsys_manager.session import ManagerSession


@pytest.mark.fr("003-FR-2", "003-FR-3", "003-FR-16")  # spec 003 T2
class SettingsChecks(SettingsFixtures, PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-settings-"

    def test_yaml_round_trip(self):
        path = self.root / "manager.yaml"
        save_profile(self.profile, path)
        self.assertEqual(load_profile(path), self.profile)
        self.assertEqual(load_profile(path).cards, self.profile.cards)
        self.assertEqual(load_profile(path).schema_version, 1)

    def test_explicit_profile_update_only(self):
        path = self.root / "manager.yaml"
        save_profile(self.profile, path)
        changed = replace(self.profile, limits=ProcessingLimits(workers=2))
        with self.assertRaises(FileExistsError):
            save_profile(changed, path)
        self.assertEqual(load_profile(path), self.profile)
        save_profile(changed, path, overwrite=True)
        self.assertEqual(load_profile(path), changed)

    def test_processing_yaml_not_overwritten(self):
        path = self.root / "configs/selected.yaml"
        before = path.read_bytes()
        with self.assertRaises(ProfileError):
            save_profile(self.profile, path, overwrite=True)
        self.assertEqual(path.read_bytes(), before)

    def test_empty_startup_profile(self):
        with patch("src.petsys_manager.settings.default_profile_path", return_value=self.root / "missing.yaml"):
            self.assertEqual(load_profile(), MachineProfile())
        with self.assertRaises(FileNotFoundError):
            load_profile(self.root / "missing.yaml")

    def test_linux_profile_location(self):
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(self.root / "settings")}, clear=False):
            self.assertEqual(default_profile_path(), self.root / "settings/process_petsys/petsys_manager.yaml")

    def test_versions_and_unknown_fields_rejected(self):
        for value in ({}, {"schema_version": 2}, {"schema_version": True},
                      {"schema_version": 1, "mystery": 1}, {"schema_version": 1, "limits": {"workres": 2}},
                      {"schema_version": 1, "safety": "wrong"}):
            with self.subTest(value=value), self.assertRaises(ProfileError):
                profile_from_mapping(value)

    def test_malformed_duplicate_and_unsafe_yaml_rejected(self):
        path = self.root / "bad.yaml"
        for text in ("[", "[]", "schema_version: 1\nschema_version: 1\n",
                     "!!python/object/apply:os.system ['not executed']"):
            path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ProfileError):
                load_profile(path)

    def test_profiles_accept_stale_paths_not_stale_actions(self):
        profile = replace(self.profile, ini_file="private/missing.ini")
        path = self.root / "stale.yaml"
        save_profile(profile, path)
        self.assertEqual(load_profile(path), profile)
        self.assertReady(self.check_action(Action.CALIBRATE, profile=profile))
        failed = self.check_action(Action.ACQUIRE, profile=profile)
        self.assertFalse(failed.ready)
        self.assertIn("ini_file", [issue.field for issue in failed.issues])

    def test_missing_lm_paths_do_not_block_acquisition(self):
        profile = replace(self.profile, pair_map_file="missing.txt", lm_metadata=LMMetadata())
        self.assertReady(self.check_action(Action.ACQUIRE, profile=profile))
        report = self.check_action(Action.LISTMODE, profile=profile)
        self.assertFalse(report.ready)
        self.assertIn("pair_map_file", [issue.field for issue in report.issues])
        self.assertIn("lm_metadata.timestamp_unit", [issue.field for issue in report.issues])

    def test_missing_tools_do_not_block_internal_analysis(self):
        profile = replace(self.profile, petsys_folder="missing-tools", ini_file="missing.ini")
        self.assertReady(self.check_action(Action.CALIBRATE, profile=profile))
        report = self.check_action(Action.ACQUIRE, profile=profile)
        self.assertEqual({i.field for i in report.issues},
                         {"ini_file", "tool:daqd", "tool:init_system", "tool:acquire_sipm_data", "tool:set_bias"})

    def test_separate_ini_yaml_map_paths(self):
        report = self.check_action(Action.PIPELINE)
        self.assertReady(report)
        self.assertEqual(report.paths["ini_file"], self.root / "private/selected.ini")
        self.assertEqual(report.paths["yaml_file"], self.root / "configs/selected.yaml")
        self.assertEqual(report.paths["map_file"], self.root / "maps/selected.yaml")

    def test_relocated_checkout_and_explicit_processing_root(self):
        moved = self.root / "relocated"
        profile_fixture(moved)
        profile = replace(self.profile, cards=())  # Offline action does not need devices.
        report = preflight(profile, Action.CALIBRATE, inputs=self.inputs, repo_root=moved, probe=self.probe)
        self.assertReady(report)
        self.assertEqual(report.paths["map_file"], moved / "maps/selected.yaml")
        report = self.check_action(Action.CALIBRATE, profile=replace(profile, processing_root=str(moved)))
        self.assertReady(report)
        self.assertEqual(report.paths["yaml_file"], moved / "configs/selected.yaml")

    def test_metacharacter_paths_remain_literal(self):
        folder = self.root / "literal ; & [x] $HOME"
        folder.mkdir()
        ini = folder / "selected.ini"
        ini.write_text("literal", encoding="utf-8")
        report = self.check_action(Action.CONVERT,
            profile=replace(self.profile, ini_file=str(ini)),
            options=RunOptions(raw_input="private/raw data ; & [literal].rawf"))
        self.assertReady(report)
        self.assertEqual(report.paths["ini_file"], ini)
        self.assertEqual(report.paths["raw_input"].name, "raw data ; & [literal].rawf")

    def test_new_output_directory_validated_not_created(self):
        target = self.root / "data/new-run"
        report = self.check_action(Action.ACQUIRE, profile=replace(self.profile, data_dir=str(target)))
        self.assertReady(report)
        self.assertFalse(target.exists())

    def test_unwritable_destination_rejected(self):
        with patch.object(self.probe, "writable_directory", return_value=False):
            report = self.check_action(Action.ACQUIRE)
        self.assertFalse(report.ready)
        self.assertIn("data_dir", [i.field for i in report.issues])

    def test_negative_nonfinite_and_boolean_numeric_values(self):
        for value in (0, -1, float("nan"), float("inf"), True, "10", 10**1000):
            with self.subTest(value=value), self.assertRaises(ProfileError):
                RunOptions(duration_s=value)
        for value in (0, -1, float("nan"), float("inf"), True, 10**1000):
            with self.subTest(value=value), self.assertRaises(ProfileError):
                validate_calibration_factor(value)

    def test_integer_worker_batch_split_hit_bounds(self):
        self.assertEqual(ProcessingLimits().workers, 0)                        # FR-15: 0 = automatic
        self.assertEqual(ProcessingLimits().lm_seed, 0)                        # FR-22: fixed default LM seed
        with self.assertRaises(ProfileError):
            ProcessingLimits(lm_seed=-1)
        for ctor, kwargs in ((ProcessingLimits, {"workers": -1}), (ProcessingLimits, {"batch_records": 0}),
                             (RunOptions, {"splits": 0}), (RunOptions, {"hit_limit": 256}),
                             (RunOptions, {"hit_limit": True}), (RunOptions, {"regions": 0}),
                             (AcquisitionSafety, {"max_attempts": 0}),
                             (AcquisitionSafety, {"max_loss_percent": 101}),
                             (AcquisitionSafety, {"poll_interval_s": float("nan")})):
            with self.subTest(kwargs=kwargs), self.assertRaises(ProfileError):
                ctor(**kwargs)

    def test_format_and_option_validation(self):
        for kwargs in ({"output_format": "guess"}, {"population": "singles"},
                       {"source_mode": "guess"}, {"plots": "true"}, {"slabs": True},
                       {"population": "group", "output_format": "compact"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ProfileError):
                RunOptions(**kwargs)
        self.assertTrue(RunOptions(plots=True, slabs=True).slabs)

    def test_metadata_unknown_not_fabricated_and_bounds(self):
        self.assertEqual(set(LMMetadata().missing()), {f for f in to_plain(LMMetadata())})
        for kwargs in ({"acquisition_time_s": float("nan")}, {"module_number": 0},
                       {"detector_pixels_x": 128}, {"isotope": "x" * 17}, {"measurement_time_s": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ProfileError):
                LMMetadata(**kwargs)

    def test_cards_resources_and_custom_socket_contract(self):
        for kwargs in ({"cards": "not-list"}, {"cards": ("relative",)},
                       {"cards": ("/dev/a",) * 2}, {"socket_path": "relative"}, {"cards": (["bad"],)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ProfileError):
                MachineProfile(**kwargs)
        report = self.check_action(Action.DAQD, profile=replace(self.profile, socket_path="/tmp/other.sock"))
        self.assertFalse(report.ready)
        self.assertIn("socket_path", [i.field for i in report.issues])
        self.assertReady(self.check_action(Action.DAQD, profile=replace(self.profile,
            socket_path="/tmp/other.sock", capabilities=ToolCapabilities(custom_socket_confirmed=True))))

    def test_linux_platform_required_for_hardware_only(self):
        with patch.object(self.probe, "platform_name", "Windows"):
            self.assertIn("platform", [i.field for i in self.check_action(Action.ACQUIRE).issues])
            self.assertReady(self.check_action(Action.CALIBRATE))

    def test_conversion_is_compact_coincidence_only(self):
        # FR-10 (2026-10-05): fixed and group are refused before any check; compact coincidence is the default.
        for changes in ({"output_format": "fixed"}, {"population": "group"},
                        {"output_format": "fixed", "population": "group"}):
            with self.assertRaisesRegex(ProfileError, "compact coincidence only"):
                RunOptions(**changes)
        options = RunOptions(raw_input="private/raw data ; & [literal].rawf")
        self.assertEqual((options.output_format, options.population), (DataFormat.COMPACT, Population.COINCIDENCE))
        report = self.check_action(Action.CONVERT, options=options)
        self.assertReady(report)
        self.assertIn("tool:convert_raw_to_coincidence", report.paths)
        self.assertNotIn("tool:convert_raw_to_group", report.paths)

    def test_calibration_limit_settings(self):
        # FR-21 (T25.2): target (default) or reference per run; T in the profile limits, default 3,000.
        self.assertEqual(RunOptions().calibration_limit_mode, "target")
        self.assertEqual(RunOptions(calibration_limit_mode="reference").calibration_limit_mode, "reference")
        with self.assertRaises(ProfileError):
            RunOptions(calibration_limit_mode="per_file")
        self.assertEqual((ProcessingLimits().calibration_target_per_key, ProcessingLimits().calibration_memory_mb),
                         (3000, 8192))
        for value in (0, -1, 1.5, True):
            for name in ("calibration_target_per_key", "calibration_memory_mb"):
                with self.subTest(name=name, value=value), self.assertRaises(ProfileError):
                    ProcessingLimits(**{name: value})
        old = to_plain(self.profile)
        del old["limits"]["calibration_target_per_key"], old["limits"]["calibration_memory_mb"]
        limits = profile_from_mapping(old).limits
        self.assertEqual((limits.calibration_target_per_key, limits.calibration_memory_mb), (3000, 8192))

    def test_old_profile_fixed_capability_is_read_and_not_written(self):
        # FR-10: a profile saved before 2026-10-05 still loads; the retired setting is dropped on save.
        old = to_plain(self.profile)
        old["capabilities"]["fixed_output_confirmed"] = True
        path = self.root / "old-profile.yaml"
        path.write_text(yaml.safe_dump(old, sort_keys=False), encoding="utf-8")
        loaded = load_profile(path)
        self.assertEqual(loaded, self.profile)
        save_profile(loaded, path, overwrite=True)
        self.assertNotIn("fixed_output_confirmed", path.read_text(encoding="utf-8"))
        old["capabilities"]["unknown_flag"] = True
        path.write_text(yaml.safe_dump(old, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ProfileError, "unknown_flag"):
            load_profile(path)

    def test_qc_presets_are_snapshot_only(self):
        options = RunOptions(duration_s=7., source_mode=SourceMode.WITH)
        report = self.check_action(Action.QC, options=options)
        self.assertReady(report)
        self.assertEqual(report.settings.options.duration_s, 60.)
        self.assertEqual(report.settings.options.output_format, DataFormat.COMPACT)
        self.assertEqual(options.duration_s, 7.)
        report = self.check_action(Action.QC, options=replace(options, source_mode=SourceMode.WITHOUT))
        self.assertEqual(report.settings.options.duration_s, 180.)

    def test_pipeline_conversion_route(self):
        # FR-10/FR-22: the pipeline converts to compact coincidence with the stock converter.
        report = self.check_action(Action.PIPELINE, options=RunOptions())
        self.assertNotIn("output_format", [i.field for i in report.issues])
        self.assertIn("tool:convert_raw_to_coincidence", report.paths)
        self.assertNotIn("tool:convert_raw_to_group", report.paths)

    def test_input_descriptors_exact_routes_and_duplicates(self):
        # FR-10 (2026-10-05): every manager action takes compact coincidence only; the library keeps fixed.
        compact = self.inputs[0]
        for action in (Action.CALIBRATE, Action.LISTMODE, Action.QC_ANALYZE):
            self.assertReady(self.check_action(action, inputs=(compact,)))
            for data_format, population in (("fixed", "coincidence"), ("fixed", "group"), ("compact", "group")):
                other = replace(compact, format=DataFormat(data_format), population=Population(population))
                report = self.check_action(action, inputs=(other,))
                self.assertEqual([i.message for i in report.issues if i.field == "inputs"],
                                 [f"Unsupported format/population for {action.value}: {self.root / compact.path}"])
        self.assertFalse(self.check_action(Action.CALIBRATE, inputs=self.inputs * 2).ready)
        self.assertFalse(self.check_action(Action.CALIBRATE, inputs=()).ready)

    def test_processing_yaml_errors_are_action_specific(self):
        path = self.root / "configs/selected.yaml"
        for text in ("[]", "map_file: maps/missing.yaml\nmin_ch: 4\n",
                     "map_file: maps/selected.yaml\nmin_ch: .nan\n",
                     "map_file: maps/selected.yaml\nmin_ch: 4\nen_min_ch: .nan\n"):
            path.write_text(text, encoding="utf-8")
            self.assertFalse(self.check_action(Action.CALIBRATE).ready)
            self.assertReady(self.check_action(Action.ACQUIRE))

    def test_immutable_nested_snapshot_and_pickle(self):
        profile = profile_from_mapping(to_plain(self.profile))
        report = self.check_action(Action.LISTMODE, profile=profile)
        self.assertReady(report)
        snapshot = report.settings
        with self.assertRaises(FrozenInstanceError):
            snapshot.options.duration_s = 999
        with self.assertRaises(TypeError):
            snapshot.processing_config["energy_range"] = [0, 999]
        with self.assertRaises(TypeError):
            snapshot.processing_config["unpopulated_minimodules"][2][0] = 999
        changed = replace(profile, lm_metadata=replace(profile.lm_metadata, module_number=9))
        self.assertEqual(snapshot.profile.lm_metadata.module_number, 2)
        self.assertEqual(changed.lm_metadata.module_number, 9)
        plain = to_plain(snapshot)
        plain["processing_config"]["energy_range"][0] = 0
        self.assertEqual(snapshot.processing_config["energy_range"], (357, 665))
        self.assertEqual(pickle.loads(pickle.dumps(snapshot)), snapshot)
        self.assertIsInstance(json.loads(json.dumps(to_plain(snapshot))), dict)

    def test_snapshot_detaches_direct_constructor_containers(self):
        paths = {"map_file": self.root / "maps/selected.yaml"}
        config = {"window": [357, 665]}
        inputs = list(self.inputs)
        cards = list(self.profile.cards)
        profile = replace(self.profile, cards=cards)
        snapshot = RunSettings(Action.LISTMODE, profile, RunOptions(), self.root, paths, inputs, config)
        paths.clear()
        config["window"][0] = 0
        inputs.clear()
        cards.clear()
        self.assertEqual(snapshot.paths["map_file"], self.root / "maps/selected.yaml")
        self.assertEqual(snapshot.processing_config["window"], (357, 665))
        self.assertEqual(snapshot.inputs, self.inputs)
        self.assertEqual(snapshot.profile.cards, self.profile.cards)

    def test_snapshot_survives_processing_config_edits(self):
        snapshot = self.check_action(Action.LISTMODE).settings
        path = self.root / "configs/selected.yaml"
        path.write_text("map_file: maps/selected.yaml\nmin_ch: 1\nen_min_ch: 0.9\nenergy_range: [100, 200]\n",
                        encoding="utf-8")
        self.assertEqual(snapshot.processing_config["min_ch"], 4)
        self.assertEqual(snapshot.processing_config["energy_range"], (357, 665))
        next_snapshot = self.check_action(Action.LISTMODE).settings
        self.assertEqual(next_snapshot.processing_config["min_ch"], 1)

    def test_oversized_configuration_rejected(self):
        path = self.root / "oversized.yaml"
        path.write_bytes(b"#" * 129)
        with patch("src.petsys_manager.settings.MAX_CONFIG_BYTES", 128):
            with self.assertRaises(ProfileError):
                load_profile(path)

    def test_imports_do_not_launch_processes_threads_or_gui(self):
        before = set(sys.modules)
        files = (REPO / "src/petsys_manager/contracts.py", REPO / "src/petsys_manager/settings.py",
                 REPO / "src/petsys_manager/commands.py", REPO / "src/petsys_manager/runner.py",
                 REPO / "src/petsys_manager/artifacts.py",
                 REPO / "tests/manager_helpers.py", Path(__file__))
        with patch.dict(sys.modules), patch("subprocess.Popen", side_effect=AssertionError("process at import")), \
                patch("threading.Thread.start", side_effect=AssertionError("worker at import")):
            for index, path in enumerate(files):
                name = f"src.petsys_manager._import_check_{index}"
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            added = set(sys.modules) - before
            self.assertFalse(any(name.split(".")[0] in {"tkinter", "customtkinter"} for name in added))

    def test_existing_file_is_not_an_output_directory(self):
        report = self.check_action(Action.ACQUIRE,
            profile=replace(self.profile, data_dir="private/selected.ini"))
        self.assertFalse(report.ready)
        self.assertIn("data_dir", [i.field for i in report.issues])

    def test_input_configuration_and_private_data_preserved(self):
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob("*") if p.is_file()}
        for action in Action:
            options = RunOptions(raw_input="private/raw data ; & [literal].rawf")
            inputs = (replace(self.inputs[0], format=DataFormat.COMPACT),) if action == Action.QC_ANALYZE else self.inputs
            self.assertReady(self.check_action(action, options=options, inputs=inputs))
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_frozen_command_events_and_result_states(self):
        identity = Identity("run", "stage", "attempt")
        argv = ["python", "literal ; & path"]
        environment = {"SAMPLE": "literal ; value"}
        command = CommandSpec(argv, self.root, identity, environment)
        argv[1] = "changed"
        environment["SAMPLE"] = "changed"
        self.assertEqual(command.argv[1], "literal ; & path")
        self.assertEqual(command.environment["SAMPLE"], "literal ; value")
        payload = {"files": ["one", "two"]}
        event = RunEvent(identity, 1, "progress", payload=payload)
        payload["files"].append("three")
        self.assertEqual(event.payload["files"], ("one", "two"))
        artifact = Artifact(self.root / "output.ldat", "converted", self.inputs[0])
        for status, code in (("succeeded", 0), ("failed", 2), ("cancelled", None), ("launch_error", None)):
            result = CommandResult(identity, status, code, artifacts=(artifact,))
            self.assertEqual(result.status, ResultStatus(status))
        # Zero exit alone is insufficient when a required output is invalid.
        self.assertEqual(CommandResult(identity, "failed", 0, "missing required output").status, ResultStatus.FAILED)

    def test_invalid_contract_payloads(self):
        identity = Identity("run", "stage", "attempt")
        for operation in (lambda: CommandSpec("shell string", self.root, identity),
                          lambda: CommandSpec(("python",), Path("relative"), identity),
                          lambda: CommandResult(identity, "succeeded", 2),
                          lambda: CommandResult(identity, "launch_error", 0),
                          lambda: CommandResult(identity, "failed", 0, []),
                          lambda: RunEvent(identity, -1, "progress"),
                          lambda: RunEvent(identity, 1, "progress", message=[]),
                          lambda: FrozenMapping(((True, "bad key"),)),
                          lambda: InputDescriptor("", "fixed", "coincidence"),
                          lambda: Artifact("", "converted"),
                          lambda: InputDescriptor("input.ldat", "guessed", "coincidence")):
            with self.assertRaises(ValueError):
                operation()
        recursive = []
        recursive.append(recursive)
        with self.assertRaises(ValueError):
            freeze({"recursive": recursive})


@pytest.mark.fr("005-FR-6")  # spec 005 T13
class LastFoldersChecks(SettingsFixtures, PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-last-folders-"

    def folders(self, *names):
        return {name: str(self.root / "picked" / name) for name in names}

    def test_last_folders_round_trip(self):
        folders = self.folders("raw_input", "ldat_inputs.listmode", "conversion_run", "cog_limits_file")
        profile = replace(self.profile, last_folders=folders)
        self.assertEqual(dict(profile.last_folders), folders)
        path = save_profile(profile, self.root / "manager.yaml")
        self.assertEqual(yaml.safe_load(path.read_text(encoding="utf-8"))["last_folders"], folders)
        self.assertEqual(load_profile(path), profile)
        self.assertEqual(replace(self.profile, last_folders=dict(reversed(folders.items()))), profile)

    def test_last_folders_unknown_key_or_relative_path_rejected(self):
        for folders in ({"unknown_dialog": str(self.root)}, {"profile_path": str(self.root)},
                        {"raw_input": "relative/folder"}, {"raw_input": ""}, {"raw_input": 3}, ["raw_input"]):
            with self.subTest(folders=folders), self.assertRaises(ProfileError):
                replace(self.profile, last_folders=folders)
        plain = to_plain(self.profile)
        plain["last_folders"] = {"unknown_dialog": str(self.root)}
        path = self.root / "bad.yaml"
        path.write_text(yaml.safe_dump(plain, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ProfileError, "unknown_dialog"):
            load_profile(path)

    def test_last_folders_absent_from_1_0_0_profiles_and_from_saves_without_folders(self):
        old = to_plain(self.profile)
        old.pop("last_folders", None)
        path = self.root / "old.yaml"
        path.write_text(yaml.safe_dump(old, sort_keys=False), encoding="utf-8")
        self.assertEqual(dict(load_profile(path).last_folders), {})
        self.assertEqual(load_profile(path), self.profile)
        save_profile(load_profile(path), path, overwrite=True)   # 1.0.0 rejects unknown fields: stay readable
        self.assertNotIn("last_folders", yaml.safe_load(path.read_text(encoding="utf-8")))

    def test_save_last_folders_replaces_only_last_folders(self):
        on_disk = replace(self.profile, data_dir="disk/data", last_folders=self.folders("raw_input"))
        path = save_profile(on_disk, self.root / "manager.yaml")
        before = yaml.safe_load(path.read_text(encoding="utf-8"))
        folders = self.folders("raw_input", "ldat_inputs.calibrate")
        self.assertEqual(save_last_folders(path, folders), path)
        after = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(after.pop("last_folders"), folders)
        before.pop("last_folders")
        self.assertEqual(after, before)
        self.assertEqual(load_profile(path), replace(on_disk, last_folders=folders))
        old = to_plain(self.profile)           # a 1.0.0 file: the section is added, the rest kept as parsed
        old.pop("last_folders", None)
        old["capabilities"]["fixed_output_confirmed"] = True
        path.write_text(yaml.safe_dump(old, sort_keys=False), encoding="utf-8")
        save_last_folders(path, folders)
        after = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(after.pop("last_folders"), folders)
        self.assertEqual(after, old)

    def test_save_last_folders_refuses_bad_folders_and_bad_files_unchanged(self):
        path = save_profile(self.profile, self.root / "manager.yaml")
        before = path.read_bytes()
        with self.assertRaises(ProfileError):
            save_last_folders(path, {"raw_input": "relative"})
        selected = self.root / "configs/selected.yaml"         # not a manager profile: never rewritten
        processing = selected.read_bytes()
        with self.assertRaises(ProfileError):
            save_last_folders(selected, self.folders("raw_input"))
        self.assertEqual((path.read_bytes(), selected.read_bytes()), (before, processing))
        with patch("src.petsys_manager.settings.os.replace", side_effect=OSError("disk full")), \
                self.assertRaises(OSError):
            save_last_folders(path, self.folders("raw_input"))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(sorted(item.name for item in path.parent.iterdir() if item.name.startswith(".petsys")), [])


@pytest.mark.fr("005-FR-6")  # spec 005 T13
class SessionLastFoldersChecks(SettingsFixtures, PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-session-folders-"

    def session(self, path):
        session = ManagerSession(path, probe=self.probe, repo_root=self.root)
        session.open()
        self.logs(session)
        return session

    def logs(self, session):
        found = []
        while True:
            try:
                event = session.events.get_nowait()
            except queue.Empty:
                return found
            if event.kind == "log":
                found.append(event.payload)

    def test_session_save_last_folders_writes_the_file_and_memory(self):
        path = save_profile(self.profile, self.root / "manager.yaml")
        session = self.session(path)
        session.save_last_folders("raw_input", self.root / "data")
        session.save_last_folders("conversion_run", str(self.root / "runs"))
        expected = {"raw_input": str(self.root / "data"), "conversion_run": str(self.root / "runs")}
        self.assertEqual(dict(session.profile.last_folders), expected)
        self.assertEqual(load_profile(path), replace(self.profile, last_folders=expected))
        self.assertEqual(self.logs(session), [])
        with self.assertRaises(ProfileError):
            session.save_last_folders("unknown_dialog", self.root)
        self.assertEqual(dict(session.profile.last_folders), expected)

    def test_session_save_last_folders_keeps_the_files_other_fields(self):
        path = save_profile(self.profile, self.root / "manager.yaml")
        session = self.session(path)
        changed = replace(self.profile, data_dir="elsewhere")     # saved by another window meanwhile
        save_profile(changed, path, overwrite=True)
        session.save_last_folders("raw_input", self.root / "data")
        self.assertEqual(load_profile(path), replace(changed, last_folders={"raw_input": str(self.root / "data")}))
        self.assertEqual(session.profile.data_dir, self.profile.data_dir)

    def test_session_save_last_folders_without_a_profile_file_writes_nothing(self):
        path = self.root / "absent" / "manager.yaml"
        session = self.session(path)
        session.save_last_folders("raw_input", self.root / "data")
        session.save_last_folders("ldat_inputs.qc_analyze", self.root / "data")
        self.assertEqual(dict(session.profile.last_folders), {"raw_input": str(self.root / "data"),
                                                              "ldat_inputs.qc_analyze": str(self.root / "data")})
        self.assertFalse(path.parent.exists())
        logs = self.logs(session)
        self.assertEqual(len(logs), 1, logs)
        self.assertIn(str(path), logs[0])

    def test_session_save_last_folders_write_failure_is_logged_and_kept_in_memory(self):
        path = save_profile(self.profile, self.root / "manager.yaml")
        before = path.read_bytes()
        session = self.session(path)
        with patch("src.petsys_manager.settings.os.replace", side_effect=OSError("disk full")):
            session.save_last_folders("raw_input", self.root / "data")
        self.assertEqual(dict(session.profile.last_folders), {"raw_input": str(self.root / "data")})
        self.assertEqual(path.read_bytes(), before)
        logs = self.logs(session)
        self.assertEqual(len(logs), 1, logs)
        self.assertIn("disk full", logs[0])
