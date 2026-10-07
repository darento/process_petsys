"""Literal PETsys command builders (spec 003 T3, FR-23; spec 007 T6).

Moved from scripts/petsys_manager_check.py --commands.
"""

from __future__ import annotations

from dataclasses import replace
import os
import sys
import unittest
from unittest.mock import patch

import pytest

from helpers import REPO
from manager_helpers import PrivateOutput, SettingsFixtures
from src.petsys_manager.contracts import Action, DataFormat, Identity, Population
from src.petsys_manager.commands import (build_acquisition, build_bias_off, build_conversion, build_daqd,
    build_initialize, build_internal)
from src.petsys_manager.settings import ProfileError, RunOptions, ToolCapabilities, load_profile, save_profile


@pytest.mark.fr("003-FR-4", "003-FR-5", "003-FR-7", "003-FR-10", "003-FR-16")  # spec 003 T3
class CommandChecks(SettingsFixtures, PrivateOutput, unittest.TestCase):
    fixture_prefix = "petsys-manager-commands-"

    def setUp(self):
        super().setUp()
        self.identity = Identity("literal run", "stage", "attempt-2")
        self.output_prefix = self.root / "data/new ; & [literal] prefix"

    def test_command_daqd_card_order_and_resources(self):
        settings = self.check_action(Action.DAQD).settings
        command = build_daqd(settings, self.identity)
        self.assertEqual(command.argv, (str(self.root / "tools/daqd"), "--socket-name", "/tmp/d.sock",
            "--daq-type", "PFP_KX7", "--card", self.profile.cards[0], "--card", self.profile.cards[1]))
        self.assertEqual(command.identity, self.identity)
        self.assertEqual(command.cwd, self.root / "tools")
        self.assertFalse(self.output_prefix.exists())

    def test_command_init_does_not_invent_config_or_socket_flags(self):
        settings = self.check_action(Action.INITIALIZE).settings
        self.assertEqual(build_initialize(settings, self.identity).argv, (str(self.root / "tools/init_system"),))
        changed = replace(settings, profile=replace(self.profile, socket_path="/tmp/other.sock",
            capabilities=ToolCapabilities(custom_socket_confirmed=True)))
        with self.assertRaises(ValueError):
            build_initialize(changed, self.identity)
        with self.assertRaises(ValueError):
            build_acquisition(changed, self.identity, self.output_prefix)

    @pytest.mark.fr("003-FR-23")
    def test_command_petsys_python_runs_python_tools_without_manager_activation(self):
        """FR-23: init/acquire/bias run as <petsys_python> <tool>; compiled tools and empty setting unchanged."""
        python = self.root / "private" / "sys python" / "python3"
        python.parent.mkdir()
        python.write_text("fixture interpreter marker", encoding="utf-8")
        profile = replace(self.profile, petsys_python=str(python))
        activated = {"CONDA_PREFIX": "/opt/conda/envs/process_petsys", "CONDA_DEFAULT_ENV": "process_petsys",
                     "CONDA_SHLVL": "2", "_CE_CONDA": "", "PYTHONPATH": "/x", "PYTHONHOME": "/opt/conda",
                     "VIRTUAL_ENV": "/venv", "PYTHONUNBUFFERED": "1", "HOME": "/home/sie",
                     "PATH": os.pathsep.join(["/opt/conda/envs/process_petsys/bin", str(python.parent), "/usr/bin"])}
        with patch.dict(os.environ, activated, clear=True):
            settings = self.check_action(Action.ACQUIRE, profile=profile, options=RunOptions(duration_s=5.0)).settings
            commands = {"init_system": build_initialize(settings, self.identity),
                        "acquire_sipm_data": build_acquisition(settings, self.identity, self.output_prefix),
                        "set_bias": build_bias_off(settings, self.identity)}
            for tool, command in commands.items():
                with self.subTest(tool):
                    self.assertEqual(command.argv[:2], (str(python), str(self.root / "tools" / tool)))
                    env = dict(command.environment)
                    self.assertFalse({k for k in env if k.startswith(("CONDA_", "_CE_"))})
                    self.assertFalse({"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & set(env))
                    self.assertEqual(env["PATH"].split(os.pathsep),
                                     [str(python.parent), "/opt/conda/envs/process_petsys/bin", "/usr/bin"])
                    self.assertEqual((env["PYTHONUNBUFFERED"], env["HOME"]), ("1", "/home/sie"))
            self.assertEqual(commands["acquire_sipm_data"].argv[2:4], ("--config", str(self.root / "private/selected.ini")))
            daqd = build_daqd(settings, self.identity)                  # compiled: unchanged
            self.assertEqual(daqd.argv[0], str(self.root / "tools/daqd"))
            self.assertEqual(dict(daqd.environment), activated)
            plain = self.check_action(Action.ACQUIRE, options=RunOptions(duration_s=5.0)).settings
            self.assertEqual(build_initialize(plain, self.identity).argv, (str(self.root / "tools/init_system"),))
            self.assertEqual(dict(build_initialize(plain, self.identity).environment), activated)
        # Preflight: a named interpreter must exist for the actions that run the Python tools only.
        missing = replace(self.profile, petsys_python=str(self.root / "private/absent/python3"))
        for action in (Action.INITIALIZE, Action.ACQUIRE, Action.QC, Action.PIPELINE):
            self.assertIn("petsys_python", [i.field for i in self.check_action(action, profile=missing).issues], action)
        for action in (Action.DAQD, Action.CONVERT, Action.CALIBRATE):
            self.assertNotIn("petsys_python", [i.field for i in self.check_action(action, profile=missing).issues], action)
        with self.assertRaises(ProfileError):
            replace(self.profile, petsys_python="python3")               # relative: refused
        path = self.root / "profile with python.yaml"
        save_profile(profile, path)
        self.assertEqual(load_profile(path).petsys_python, str(python))

    def test_command_acquisition_ini_duration_mode_trigger(self):
        settings = self.check_action(Action.ACQUIRE,
            options=RunOptions(duration_s=31.5, acquisition_mode="qdc", hardware_trigger=True)).settings
        command = build_acquisition(settings, self.identity, self.output_prefix)
        self.assertEqual(command.argv, (str(self.root / "tools/acquire_sipm_data"), "--config",
            str(self.root / "private/selected.ini"), "-o", str(self.output_prefix), "--time", "31.5",
            "--mode", "qdc", "--enable-hw-trigger"))
        options = replace(settings.options, hardware_trigger=False, acquisition_mode="tot")
        command = build_acquisition(replace(settings, options=options), self.identity, self.output_prefix)
        self.assertNotIn("--enable-hw-trigger", command.argv)
        self.assertEqual(command.argv[-1], "tot")
        for kwargs in ({"acquisition_mode": "guess"}, {"hardware_trigger": "true"}):
            with self.assertRaises(ProfileError):
                RunOptions(**kwargs)

    def test_command_conversion_compact_and_split(self):
        options = RunOptions(duration_s=40., splits=4, hit_limit=12, raw_input="private/raw data ; & [literal].rawf")
        settings = self.check_action(Action.CONVERT, options=options).settings
        command = build_conversion(settings, self.identity, self.output_prefix)
        self.assertEqual(command.argv, (str(self.root / "tools" / "convert_raw_to_coincidence"), "--config",
            str(self.root / "private/selected.ini"), "-i", str(self.root / "private/raw data ; & [literal]"),
            "-o", str(self.output_prefix), "--writeBinaryCompact", "--writeMultipleHits", "12", "--splitTime", "10.1"))
        self.assertNotIn("--writeBinaryFixed", command.argv)
        unsplit = build_conversion(replace(settings, options=replace(options, splits=1)), self.identity, self.output_prefix)
        self.assertNotIn("--splitTime", unsplit.argv)

    def test_command_raw_prefix_strips_only_rawf_not_arbitrary_suffix(self):
        settings = self.check_action(Action.CONVERT,
            options=RunOptions(raw_input="private/raw data ; & [literal].rawf")).settings
        raw = self.root / "private/not_named_with_time.v2"
        command = build_conversion(settings, self.identity, self.output_prefix, raw_input=raw)
        self.assertEqual(command.argv[command.argv.index("-i") + 1], str(raw))

    def test_command_qc_presets_feed_compact_conversion(self):
        settings = self.check_action(Action.QC, options=RunOptions(source_mode="without")).settings
        acquire = build_acquisition(settings, self.identity, self.output_prefix)
        self.assertEqual(float(acquire.argv[acquire.argv.index("--time") + 1]), 180.)
        convert = build_conversion(settings, self.identity, self.output_prefix,
            raw_input=self.root / "private/QC.rawf")
        self.assertIn("--writeBinaryCompact", convert.argv)

    def test_command_internal_uses_current_interpreter_and_checkout(self):
        settings = self.check_action(Action.LISTMODE).settings
        request = self.root / "private/request ; & [x].json"
        result = self.root / "private/result ; & [x].json"
        for action, name in ((Action.CALIBRATE, "calibrate"), (Action.LISTMODE, "listmode"), (Action.QC_ANALYZE, "qc")):
            command = build_internal(settings, self.identity, action, request, result)
            self.assertEqual(command.argv, (sys.executable, "-u", "-m", "src.cornell.cli", name,
                "--request", str(request), "--result", str(result)))
            self.assertEqual(command.cwd, REPO)
        relocated = self.root / "relocated checkout"
        self.assertEqual(build_internal(settings, self.identity, Action.LISTMODE, request, result,
            checkout_root=relocated).cwd, relocated)
        with self.assertRaises(ValueError):
            build_internal(settings, self.identity, Action.ACQUIRE, request, result)

    def test_command_explicit_environment_snapshot_no_shell_activation(self):
        settings = self.check_action(Action.ACQUIRE).settings
        env = {"EXPLICIT": "value ; & [literal]"}
        with patch.dict(os.environ, {"SNAPSHOT": "before"}):
            command = build_acquisition(settings, self.identity, self.output_prefix, environment=env)
        env["EXPLICIT"] = "changed"
        self.assertEqual(command.environment["EXPLICIT"], "value ; & [literal]")
        self.assertEqual(command.environment["SNAPSHOT"], "before")
        self.assertNotIn("source", command.argv)
        self.assertNotIn("conda", command.argv)

    def test_command_rejects_relative_outputs_fixed_group_and_unknown_daq(self):
        settings = self.check_action(Action.ACQUIRE).settings
        with self.assertRaises(ValueError):
            build_acquisition(settings, self.identity, "relative")
        for name, value in (("output_format", DataFormat.FIXED), ("population", Population.GROUP)):
            forged = RunOptions()
            object.__setattr__(forged, name, value)          # bypasses RunOptions validation on purpose
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "compact coincidence only"):
                build_conversion(replace(settings, options=forged), self.identity, self.output_prefix,
                                 raw_input=self.root / "private/raw.rawf")
        for profile in (replace(self.profile, daq_type="DTFLY"),
                        replace(self.profile, shared_memory_path="/dev/shm/other")):
            with self.assertRaises(ValueError):
                build_daqd(replace(settings, profile=profile), self.identity)
