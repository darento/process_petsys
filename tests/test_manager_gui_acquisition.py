"""DAQD, initialization, acquisition, STOP, bias-off and close through the real services with fake
children (spec 003 T14, T35; spec 007 T15).

Moved from scripts/petsys_manager_gui_check.py --acquisition. ``gui``: skipped without a display.
"""

import os
from pathlib import Path
import shutil
import time
from dataclasses import replace
from unittest.mock import patch

import pytest

from exe_programs import petsys_manager_gui as gui
from manager_gui_helpers import GUIBase, Hardware, SAFETY, SHM, SOCKET, STAMP, pump, wait_threads, world
from src.petsys_manager.acquisition import DaqdState
from src.petsys_manager.artifacts import read_manifest
from src.petsys_manager.contracts import to_plain
from src.petsys_manager.session import ShellEvent, WorkflowResult
from src.petsys_manager.settings import save_profile


@pytest.mark.gui
@pytest.mark.fr("003-FR-5", "003-FR-6", "003-FR-7", "003-FR-8", "003-FR-16")  # spec 003 T14
class AcquisitionChecks(GUIBase):
    fixture_prefix = "pm-gui-acq-"

    def hardware_app(self, hardware, **safety):
        profile = world(self.base / "w", shm=SHM, safety=replace(SAFETY, **safety))
        app = self.open(save_profile(profile, self.base / "p.yaml"), repo_root=self.base / "w",
                        daqd_factory=hardware.daqd_factory, coordinator_factory=hardware.coordinator_factory)
        app.statuses = []
        configure = app.acq_status.configure

        def record(**options):
            if "text" in options:
                app.statuses.append(options["text"])
            return configure(**options)
        app.acq_status.configure = record
        return app

    def state(self, app, key):
        return app.buttons[key].cget("state")

    def click(self, app, key):
        self.assertEqual(self.state(app, key), "normal", f"{key} is not enabled")
        button = app.buttons[key]
        button.toggle() if key == "daqd" else button.invoke()

    def daqd(self, app):
        return app.daqd_status.state if app.daqd_status is not None else DaqdState.OFF

    def wait(self, app, predicate, timeout=10.0, message="condition not reached"):
        self.assertTrue(pump(app.root, predicate, timeout), message)

    def ready_initialized(self, app):
        self.click(app, "daqd")
        self.wait(app, lambda: self.daqd(app) == DaqdState.READY, message="DAQD not ready")
        self.click(app, "initialize")
        self.wait(app, lambda: not app._init_pending and app.daqd_status.initialized, message="not initialized")
        self.wait(app, lambda: self.state(app, "acquire") == "normal", message="acquire not enabled")

    def finished(self, app, timeout=20.0):
        self.wait(app, lambda: app._token is None, timeout, "workflow did not finish")
        lines = [STAMP.sub("", line, count=1) for line in self.log(app).splitlines()]
        return [line for line in lines if line.startswith("Acquisition ")][-1]

    def assertOwnedChildrenGone(self, hardware):
        for child in hardware.children:
            self.assertIsNotNone(child.poll(), child)

    def test_daqd_readiness_and_initialization_are_backend_driven(self):
        hardware = Hardware()
        app = self.hardware_app(hardware)
        self.assertEqual([self.state(app, key) for key in ("daqd", "initialize", "acquire", "stop")],
                         ["normal", "disabled", "disabled", "disabled"])
        hardware.resources.serve.clear()  # socket exists but DAQD does not answer yet
        self.click(app, "daqd")
        self.wait(app, lambda: self.daqd(app) == DaqdState.STARTING)
        pump(app.root, timeout=0.5)
        self.assertEqual(self.daqd(app), DaqdState.STARTING)
        self.assertEqual(app.buttons["daqd"].cget("text"), "DAQD STARTING")
        for key in ("initialize", "acquire"):
            self.assertEqual(self.state(app, key), "disabled", key)
        hardware.resources.serve.set()
        self.wait(app, lambda: self.daqd(app) == DaqdState.READY)
        self.assertTrue(app.daqd_state.get())
        self.assertEqual((self.state(app, "initialize"), self.state(app, "acquire")), ("normal", "disabled"))
        hardware.init_code = 1  # failed initialization never unlocks acquisition
        self.click(app, "initialize")
        self.wait(app, lambda: not app._init_pending)
        pump(app.root, timeout=0.2)
        self.assertIn("Initialization failed; acquisition stays locked", self.log(app))
        self.assertFalse(app.daqd_status.initialized)
        self.assertEqual(self.state(app, "acquire"), "disabled")
        hardware.init_code = 0
        self.click(app, "initialize")
        self.wait(app, lambda: not app._init_pending and app.daqd_status.initialized)
        self.wait(app, lambda: self.state(app, "acquire") == "normal")
        ready = app.daqd_status
        hardware.daemon.die(1)  # dead daemon: initialization and acquisition are lost
        self.wait(app, lambda: self.daqd(app) == DaqdState.FAILED)
        self.assertEqual([self.state(app, key) for key in ("daqd", "initialize", "acquire")],
                         ["normal", "disabled", "disabled"])
        app.session.events.put(ShellEvent("daqd", ready))  # stale READY+initialized status
        app.session.events.put(ShellEvent("init_done", type("O", (), {"status": ready, "initialized": True,
                                                                       "message": "stale"})()))
        pump(app.root, timeout=0.4)
        self.assertEqual(self.daqd(app), DaqdState.FAILED)
        self.assertEqual(self.state(app, "acquire"), "disabled")
        self.assertEqual(hardware.acquisitions(), [])
        self.assertEqual(app.guard.violations, [])

    @pytest.mark.fr("003-FR-1")
    def test_daqd_output_apart_from_the_log_counters_not_kept_and_log_saved(self):
        """T35: the daemon's lines go to the DAQD Output box, its CNT counters only to the counter
        label (one per card); init_system output stays in the log; Save Log never overwrites."""
        hardware = Hardware()
        hardware.daqd_stderr = (b"Got a new client: 8\n" + b"CNT  1  1  2  0  0  0  0 79.000000\n" * 500
                                + b"CNT  195312  7  390624  0  1953120  2148432  195312 999.000000\n"
                                + b"INFO: Client hung up or error\n")
        hardware.init_stdout = b"INFO: active units on ports:  0, 1\n"
        app = self.hardware_app(hardware)
        self.ready_initialized(app)
        self.wait(app, lambda: app.daqd_counter_lines == 501, message="counters not received")
        self.wait(app, lambda: "Client hung up" in app.daqd_text.get("1.0", "end-1c"))
        daemon, log = app.daqd_text.get("1.0", "end-1c"), self.log(app)
        self.assertIn("[stderr] Got a new client: 8", daemon)
        self.assertNotIn("CNT", daemon)
        self.assertNotIn("CNT", log)
        self.assertNotIn("Got a new client", log)
        self.assertIn("[stdout] INFO: active units on ports:  0, 1", log)    # init_system output
        self.assertIn("DAQD ON", log)                                       # state lines stay
        self.assertEqual(len(app.daqd_counters), 2)                          # two cards in the profile
        counters = app.daqd_counter_label.cget("text")
        self.assertIn("501 lines, not logged", counters)
        self.assertTrue(counters.endswith("CNT  195312  7  390624  0  1953120  2148432  195312 999.000000"),
                        counters)
        first = app.save_log()
        second = app.save_log()
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, Path(app.vars["report_dir"].get()))
        text = first.read_text(encoding="utf-8")
        for part in ("== Output Log ==", "== DAQD Output ==", "== DAQD counters ==", "Got a new client",
                     "active units on ports", "501 lines, not logged"):
            self.assertIn(part, text)
        before = first.read_bytes()
        with patch.object(gui, "datetime", type("D", (), {"now": staticmethod(
                lambda: __import__("datetime").datetime(2026, 10, 6, 12, 0, 0))})):
            third, fourth = app.save_log(), app.save_log()
        self.assertEqual((third.name, fourth.name), ("petsys_manager_log_2026-10-06_120000.txt",
                                                     "petsys_manager_log_2026-10-06_120000_2.txt"))
        self.assertEqual(first.read_bytes(), before)
        app.vars["report_dir"].set(str(self.base / "missing"))
        self.assertIsNone(app.save_log())
        self.assertIn("Log not saved: Report Destination is not an existing folder", self.log(app))
        self.assertFalse((self.base / "missing").exists())
        self.assertEqual(app.guard.violations, [])

    @pytest.mark.fr("003-FR-19", "003-FR-20")
    def test_acquisition_progress_stop_and_bias_off(self):
        hardware = Hardware(acquire=("grow",))
        app = self.hardware_app(hardware)
        for value, reason in (("abc", "Profile: Max frame loss (%) must be a number"),  # editable, loss included
                              ("101", "Profile: max_loss_percent must not exceed 100")):
            app.safety_vars["max_loss_percent"].set(value)
            self.wait(app, lambda: app._check_id is None and app._awaiting is None, 5)
            for text in self.reasons(app).values():
                self.assertIn(reason, text)
        self.edit(app, app.safety_vars["max_loss_percent"], "2.5")
        self.edit(app, app.safety_vars["min_growth_mb"], "0.000001")
        self.assertEqual(app.profile_from_ui().safety, replace(SAFETY, max_loss_percent=2.5))
        self.ready_initialized(app)
        self.click(app, "acquire")
        self.wait(app, lambda: any("stopped growing" in text for text in app.statuses), 15, "no stall warning")
        self.assertEqual([self.state(app, key) for key in ("daqd", "initialize", "acquire", "stop")],
                         ["disabled", "disabled", "disabled", "normal"])
        joined = "\n".join(app.statuses)
        for expected in ("Attempt 1/3: waiting for the RAW file", "RAW file started writing",
                         "File growing: growth check passed", "MB/s", "WARNING: RAW file stopped growing"):
            self.assertIn(expected, joined)
        self.click(app, "stop")
        self.assertEqual(self.state(app, "stop"), "disabled")
        final = self.finished(app)
        self.assertTrue(final.startswith("Acquisition cancelled"), final)
        self.assertEqual(hardware.order[-2:], ["acquire_sipm_data", "set_bias"])  # bias off, no new attempt
        self.assertEqual(len(hardware.acquisitions()), 1)
        self.assertOwnedChildrenGone_except_daemon(hardware)
        self.wait(app, lambda: self.state(app, "acquire") == "normal")  # controls return; DAQD still owned
        runs = [path for path in (self.base / "w" / "data").iterdir() if path.is_dir()]
        self.assertEqual(len(runs), 1)
        manifest = read_manifest(runs[0])
        self.assertEqual(manifest["status"], "cancelled")
        self.assertEqual(to_plain(manifest["settings"]["profile"]["safety"]), to_plain(replace(SAFETY, max_loss_percent=2.5)))
        self.assertEqual(app.guard.violations, [])

    def assertOwnedChildrenGone_except_daemon(self, hardware):
        for child in hardware.children:
            if child is not hardware.daemon:
                self.assertIsNotNone(child.poll(), child)

    def test_retry_wait_responsive_and_stop_prevents_new_attempts(self):
        hardware = Hardware(acquire=("nodata", "ok"))
        app = self.hardware_app(hardware, retry_delay_s=30.0, growth_window_s=30.0)
        self.assertEqual(app.safety_vars["retry_delay_s"].get(), "30")
        self.ready_initialized(app)
        self.click(app, "acquire")
        self.wait(app, lambda: any(text.startswith("Retrying") for text in app.statuses), 10, "no retry wait")
        app.session.log("marker during the retry wait")
        self.wait(app, lambda: "marker during the retry wait" in self.log(app), 1.0, "UI not responsive")
        started = time.monotonic()
        self.click(app, "stop")
        final = self.finished(app, 10)
        self.assertLess(time.monotonic() - started, 10)
        self.assertTrue(final.startswith("Acquisition cancelled"), final)
        self.assertIn("Stopped during the retry delay; no further attempts", final)
        self.assertEqual(len(hardware.acquisitions()), 1)
        # STOP during a running attempt: TERM, bias-off, no second attempt.
        hardware.acquire = ["block", "ok"]
        self.wait(app, lambda: self.state(app, "acquire") == "normal")
        seen = len(app.statuses)
        self.click(app, "acquire")
        self.wait(app, lambda: any("RAW file started writing" in text for text in app.statuses[seen:]), 10)
        self.click(app, "stop")
        final = self.finished(app)
        self.assertTrue(final.startswith("Acquisition cancelled"), final)
        self.assertEqual(hardware.order[-2:], ["acquire_sipm_data", "set_bias"])
        self.assertEqual(len(hardware.acquisitions()), 2)
        self.assertOwnedChildrenGone_except_daemon(hardware)
        self.assertEqual(app.guard.violations, [])

    @pytest.mark.fr("003-FR-19")
    def test_stale_events_failures_and_unknown_bias_never_restore_controls(self):
        hardware = Hardware(acquire=("block",))
        app = self.hardware_app(hardware, growth_window_s=30.0)  # a stalled 64-byte file must not retry here
        self.ready_initialized(app)
        ready = app.daqd_status
        self.click(app, "acquire")
        self.wait(app, lambda: any("RAW file started writing" in text for text in app.statuses), 10)
        stale = app._token - 1
        app.session.events.put(ShellEvent("workflow_done", WorkflowResult(stale, None, "old result")))
        app.session.events.put(ShellEvent("daqd", replace(ready, revision=ready.revision - 1)))
        old = type("E", (), {"kind": "workflow_finished", "message": "old run succeeded", "payload": {"status": "succeeded"},
                             "identity": type("I", (), {"run_id": "acquire-old"})()})()
        app.session.events.put(ShellEvent("workflow", old))
        pump(app.root, timeout=0.4)
        self.assertIn("Ignored a stale workflow result", self.log(app))
        self.assertNotIn("old run succeeded", "\n".join(app.statuses))
        self.assertEqual([self.state(app, key) for key in ("daqd", "initialize", "acquire", "stop")],
                         ["disabled", "disabled", "disabled", "normal"])
        hardware.bias = "fail"  # FR-19: unknown bias after STOP blocks the next acquisition until confirmed
        self.click(app, "stop")
        final = self.finished(app)
        self.assertTrue(final.startswith("Acquisition cancelled"), final)
        self.assertTrue(app.bias_unknown)
        self.assertTrue(app.bias_frame.winfo_ismapped() or app.bias_frame.winfo_manager() == "pack")
        self.assertIn("SiPM bias state unknown", app.bias_label.cget("text"))
        pump(app.root, timeout=0.2)
        self.assertEqual(self.state(app, "acquire"), "disabled")
        app.bias_ack.invoke()
        self.assertFalse(app.bias_unknown)
        self.assertEqual(self.state(app, "acquire"), "normal")
        hardware.bias, hardware.acquire = "ok", ["fail"]  # nonzero exit: failure, never success
        self.click(app, "acquire")
        final = self.finished(app)
        self.assertTrue(final.startswith("Acquisition failed"), final)
        self.assertEqual(len(hardware.acquisitions()), 2, (hardware.order, self.log(app)))  # a nonzero exit is not retried
        self.assertEqual(app.guard.violations, [])

    def test_existing_daqd_resources_block_start_without_cleanup(self):
        hardware = Hardware(existing=(SOCKET, SHM))
        app = self.hardware_app(hardware)
        self.click(app, "daqd")
        self.wait(app, lambda: "Existing DAQD resources block start" in self.log(app))
        self.assertEqual(self.daqd(app), DaqdState.OFF)
        self.assertEqual(hardware.resources.paths, {SOCKET, SHM})
        self.assertEqual(hardware.order, [])
        self.wait(app, lambda: self.state(app, "daqd") == "normal")
        self.assertEqual(self.state(app, "initialize"), "disabled")
        self.assertEqual(app.guard.violations, [])

    def test_close_waits_for_bias_off_then_stops_daqd(self):
        hardware = Hardware(acquire=("block",))
        app = self.hardware_app(hardware, growth_window_s=30.0)
        removed = []
        originals = {name: getattr(os, name) for name in ("unlink", "remove", "rmdir")}

        def recorder(name):
            def call(path, *args, **kwargs):
                if any(marker in str(path) for marker in ("d.sock", "daqd_shm")):
                    removed.append((name, str(path)))
                return originals[name](path, *args, **kwargs)
            return call
        with patch.multiple(os, **{name: recorder(name) for name in originals}), \
                patch.object(shutil, "rmtree", lambda *a, **k: removed.append(("rmtree", a))):
            self.ready_initialized(app)
            self.click(app, "acquire")
            self.wait(app, lambda: any("RAW file started writing" in text for text in app.statuses), 10)
            hardware.bias_gate.clear()  # bias-off is slow
            app.shutdown_timeout_s = 0.6
            app.on_close()
            self.assertTrue(app._shutting_down)
            self.assertEqual([self.state(app, key) for key in ("daqd", "initialize", "acquire", "stop")],
                             ["disabled"] * 4)
            self.wait(app, lambda: "set_bias" in hardware.order, 5)
            app.session.log("marker while closing")
            self.wait(app, lambda: "marker while closing" in self.log(app), 1.0, "polling stopped during close")
            self.wait(app, lambda: "Close incomplete" in self.log(app), 5, "no shutdown failure shown")
            self.assertFalse(app.closed)
            self.assertNotIn("daqd TERM", hardware.order)  # DAQD kept for the unfinished bias-off
            app.session.log("marker after failure")
            self.wait(app, lambda: "marker after failure" in self.log(app), 1.0, "polling stopped after failure")
            hardware.bias_gate.set()
            final = self.finished(app)
            self.assertTrue(final.startswith("Acquisition cancelled"), final)
            app.shutdown_timeout_s = 10.0
            app.on_close()
            self.wait(app, lambda: app.closed, 15, "window did not close")
        order = hardware.order
        self.assertLess(order.index("set_bias"), order.index("daqd TERM"))
        self.assertOwnedChildrenGone(hardware)
        self.assertEqual(removed, [])
        self.assertEqual(hardware.resources.paths, {SOCKET, SHM})  # fake daemon leftovers: reported, not removed
        self.assertEqual(app.guard.violations, [])
        self.assertTrue(wait_threads(("petsys-daqd", "petsys-work", "petsys-acq", "petsys-shut")))
