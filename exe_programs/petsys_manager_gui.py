"""PETsys Manager window: the five Cornell workflow tabs, command log and machine profile.

Views and main-thread event polling only. Profiles, prerequisite checks, DAQD,
initialization and workflows live in the toolkit-free `src.petsys_manager.session`;
workers reach this window only through its queue, drained here with `after` on
the Tk thread. Controls follow the backend: DAQD readiness/initialization come
from the service's revisioned statuses, a workflow's buttons from its token, and
older statuses or results never re-enable a control. DAQD, Initialize, Acquire
and STOP are connected (T14); conversion and processing controls are connected
by T15-T16 and stay disabled beside their prerequisite reasons until then.
Nothing launches at startup. Closing stops the workflow, waits for its bias-off,
then stops the owned DAQD, while the window keeps polling.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

from src.petsys_manager.acquisition import DaqdState
from src.petsys_manager.contracts import Action, Population, ResultStatus, SourceMode
from src.petsys_manager.session import ManagerSession
from src.petsys_manager.settings import (AcquisitionSafety, PrerequisiteIssue, ProfileError, RunOptions,
                                         default_profile_path)


__version__ = "0.1.0"
TITLE = "PETsys Manager - Cornell"
TABS = ("System Setup & Acquisition", "RAWF to LDAT Conversion", "LDAT Processing",
        "LM File Generation", "System Quality Control")
LOGO = Path(__file__).resolve().parent / "assets" / "onco_logo.jpeg"  # optional, module-relative
POLL_MS = 100
MAX_EVENTS_PER_POLL = 200
CHECK_DELAY_MS = 400
SHUTDOWN_TIMEOUT_S = 60.0
# Profile fields: (name, label, browse kind). INI and processing YAML are separate selections.
PROFILE_FIELDS = {
    "petsys_folder": ("PETsys Tools Folder:", "dir"),
    "ini_file": ("PETsys INI (DAQ/conversion):", "file"),
    "yaml_file": ("Processing YAML (cal/LM/QC):", "file"),
    "processing_root": ("Root for relative map paths:", "dir"),
    "data_dir": ("Output Data Folder:", "dir"),
    "calibration_dir": ("Energy Cal Destination:", "dir"),
    "report_dir": ("Report Destination:", "dir"),
    "cog_limits_file": ("COG Limits File:", "file"),
    "calibration_file": ("System Energy cal file:", "file"),
    "doi_limits_file": ("DOI Limits File:", "file"),
    "pair_map_file": ("LM Pair Map File:", "file"),
    "region_map_file": ("LM Region Map File:", "file"),
    "lm_dir": ("LM Destination:", "dir"),
}
DAQ_FIELDS = {"daq_type": "DAQ Type:", "cards": "DAQ Cards (comma-separated):", "socket_path": "DAQD Socket:"}
# Editable acquisition safety limits (FR-20): name -> (label, type). Growth is shown in MB (10^6 bytes).
SAFETY_FIELDS = {
    "startup_timeout_s": ("Startup timeout (s):", float), "growth_window_s": ("Growth window (s):", float),
    "min_growth_mb": ("Min growth in window (MB):", float), "poll_interval_s": ("Monitor poll (s):", float),
    "max_loss_percent": ("Max frame loss (%):", float), "max_attempts": ("Max attempts:", int),
    "retry_delay_s": ("Retry delay (s):", float), "terminate_grace_s": ("Stop grace (s):", float),
}
ISSUE_LABELS = {name: label.rstrip(":") for name, (label, _) in PROFILE_FIELDS.items()}
ISSUE_LABELS.update({name: label.rstrip(":") for name, label in DAQ_FIELDS.items()})
ISSUE_LABELS.update({
    "platform": "Platform", "raw_input": "RAW Data File", "inputs": "Input LDAT files",
    "map_file": "Map file named by the processing YAML", "output_format": "Conversion format",
    "capabilities.fixed_output_confirmed": "Converter fixed output", "options": "Run options",
    "settings": "Settings", "profile": "Profile",
})
CONNECTED = {"daqd", "initialize", "acquire"}  # checks whose controls this window drives
DAQD_TEXT = {DaqdState.OFF: "DAQD OFF", DaqdState.STARTING: "DAQD STARTING", DaqdState.READY: "DAQD ON",
             DaqdState.STOPPING: "DAQD STOPPING", DaqdState.FAILED: "DAQD FAILED"}
STATUS_COLOURS = {"info": ("gray10", "gray90"), "ok": ("#1e7d3a", "#6fcf8a"), "warn": ("#a04000", "#f0a050"),
                  "error": ("#b00020", "#ff6b6b")}
# Readiness checks: key -> (action, display name, tab index).
CHECKS = {
    "daqd": (Action.DAQD, "DAQD", 0),
    "initialize": (Action.INITIALIZE, "Initialize System", 0),
    "acquire": (Action.ACQUIRE, "Acquire Data", 0),
    "pipeline": (Action.PIPELINE, "Complete Pipeline", 0),
    "convert_coincidence": (Action.CONVERT, "Convert Raw to Coincidence", 1),
    "convert_group": (Action.CONVERT, "Convert Raw to Group", 1),
    "calibrate": (Action.CALIBRATE, "Create Energy cal file", 2),
    "listmode": (Action.LISTMODE, "Generate LM File", 3),
    "qc": (Action.QC, "Run Quality Control", 4),
    "qc_analyze": (Action.QC_ANALYZE, "Analyze existing compact LDAT", 4),
}
GREEN, GREEN_HOVER, RED, RED_HOVER = "#2ecc71", "#27ae60", "#e74c3c", "#c0392b"
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def issue_lines(issues):
    """Specific reasons; unselected tools and missing LM metadata collapse to one line each."""
    lines, tools, metadata = [], [], []
    for issue in issues:
        if issue.field.startswith("tool:") and issue.message == "Select the PETsys tools folder":
            tools.append(issue.field[5:])
        elif issue.field.startswith("lm_metadata."):
            metadata.append(issue.field[12:])
        elif issue.field.startswith("tool:"):
            lines.append(f"PETsys tool {issue.field[5:]}: {issue.message}")
        else:
            lines.append(f"{ISSUE_LABELS.get(issue.field, issue.field)}: {issue.message}")
    if tools:
        lines.append(f"{ISSUE_LABELS['petsys_folder']}: select it (needs {', '.join(tools)})")
    if metadata:
        lines.append(f"LM metadata missing from the profile: {', '.join(metadata)}")
    return lines


class PETsysManager:
    """Manager shell on an existing root; the caller owns the single Tk root."""

    def __init__(self, root, session):
        self.root = root
        self.session = session
        self._loading = False
        self._closing = False
        self._poll_id = None
        self._check_id = None
        self._awaiting = None  # only this generation's readiness is shown
        self.shown_generation = None
        self._option_issues = {}
        self.readiness = {}
        self.buttons = {}
        self.entries = {}
        self.logo_path = None
        self._ready = {}                # check key -> prerequisites met (latest shown generation)
        self.daqd_status = None         # newest DaqdStatus by service revision
        self._daqd_pending = False      # a start/stop request has not been answered yet
        self._init_pending = False
        self._token = None              # this window's requested/running workflow
        self._run_id = None
        self._stop_requested = False
        self._growth_passed = False
        self._shutting_down = False
        self.closed = False
        self.bias_unknown = False
        self.shutdown_timeout_s = SHUTDOWN_TIMEOUT_S
        self.root.title(f"{TITLE} {__version__}")
        self.root.geometry("900x950")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.vars = {name: tk.StringVar(root) for name in (*PROFILE_FIELDS, *DAQ_FIELDS)}
        self.profile_path = tk.StringVar(root)
        self.acq_time = tk.StringVar(root, "10")
        self.hw_trigger = tk.BooleanVar(root, False)
        self.raw_input = tk.StringVar(root)
        self.splits = tk.StringVar(root, "1")
        self.qc_source = tk.StringVar(root, SourceMode.WITH.value)
        self.qc_plots = tk.BooleanVar(root, False)
        self.qc_slabs = tk.BooleanVar(root, False)
        self.safety_vars = {name: tk.StringVar(root) for name in SAFETY_FIELDS}
        self._build()
        for variable in (*self.vars.values(), *self.safety_vars.values(), self.acq_time, self.hw_trigger,
                         self.raw_input, self.splits, self.qc_source, self.qc_plots, self.qc_slabs):
            variable.trace_add("write", self._edited)
        session.log(f"PETsys Manager {__version__}; checkout {session.repo_root}")
        self.profile_state = session.open()
        self._show_profile()
        self._refresh_controls()
        self._poll_id = self.root.after(POLL_MS, self._poll)

    # Layout -------------------------------------------------------------------------------------

    def _build(self):
        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=5)
        self.tabs = []
        for label in TABS:
            self.tabview.add(label)
            body = ctk.CTkScrollableFrame(self.tabview.tab(label), fg_color="transparent")
            body.pack(fill="both", expand=True)
            self.tabs.append(body)
        self._setup_tab(self.tabs[0])
        self._conversion_tab(self.tabs[1])
        self._ldat_tab(self.tabs[2])
        self._lm_tab(self.tabs[3])
        self._qc_tab(self.tabs[4])
        output = ctk.CTkFrame(self.root)
        output.pack(fill="x", padx=10, pady=5)  # the tabs take any extra height
        ctk.CTkLabel(output, text="Output Log:").pack(anchor="w", padx=5, pady=(5, 0))
        self.log_text = ctk.CTkTextbox(output, width=800, height=150, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._add_logo()

    def _frame(self, parent, title):
        frame = ctk.CTkFrame(parent)
        frame.pack(padx=10, pady=5, fill="x")
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(5, 0))
        return frame

    def _row(self, frame, row, label, variable, browse=None, width=None):
        ctk.CTkLabel(frame, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry = ctk.CTkEntry(frame, textvariable=variable, **({"width": width} if width else {}))
        entry.grid(row=row, column=1, padx=5, pady=2, sticky="w" if width else "ew")
        if browse:
            ctk.CTkButton(frame, text="Browse", width=100, command=browse).grid(row=row, column=2, padx=5, pady=2)
        return entry

    def _fields(self, frame, names, first_row=1):
        for row, name in enumerate(names, first_row):
            label, kind = PROFILE_FIELDS[name]
            entry = self._row(frame, row, label, self.vars[name],
                              lambda name=name, kind=kind: self._browse(self.vars[name], kind))
            self.entries.setdefault(name, []).append(entry)

    def _button(self, frame, key, text, **options):
        button = ctk.CTkButton(frame, text=text, state="disabled", **options)
        self.buttons[key] = button
        return button

    def _readiness(self, parent, keys):
        frame = self._frame(parent, "Prerequisites")
        for row, key in enumerate(keys, 1):
            label = ctk.CTkLabel(frame, text=f"{CHECKS[key][1]}: checking prerequisites...",
                                 justify="left", anchor="w", wraplength=780)
            label.grid(row=row, column=0, columnspan=3, sticky="w", padx=10, pady=2)
            self.readiness[key] = label

    def _setup_tab(self, tab):
        # FR-19 warning, shown above everything once a bias-off could not be confirmed.
        self.bias_frame = ctk.CTkFrame(tab, fg_color=("#fde2e1", "#5c1a1a"), border_color=RED, border_width=2)
        self.bias_label = ctk.CTkLabel(self.bias_frame, text="", text_color=STATUS_COLOURS["error"],
                                       font=ctk.CTkFont(size=14, weight="bold"), justify="left", anchor="w",
                                       wraplength=640)
        self.bias_label.pack(side="left", padx=10, pady=8, fill="x", expand=True)
        self.bias_ack = ctk.CTkButton(self.bias_frame, text="I checked the bias", width=140,
                                      command=self.acknowledge_bias)
        self.bias_ack.pack(side="right", padx=10, pady=8)
        profile = self._frame(tab, "Machine Profile")
        self._first_setup_frame = profile
        self.entries["profile"] = [self._row(profile, 1, "Profile File:", self.profile_path, self._browse_profile)]
        buttons = ctk.CTkFrame(profile, fg_color="transparent")
        buttons.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        self.reload_button = ctk.CTkButton(buttons, text="Reload", width=100, command=self.reload_profile)
        self.reload_button.pack(side="left", padx=(0, 10))
        self.save_button = ctk.CTkButton(buttons, text="Save", width=100, command=self.save_profile)
        self.save_button.pack(side="left")
        self.profile_status = ctk.CTkLabel(profile, text="", justify="left", anchor="w", wraplength=780)
        self.profile_status.grid(row=3, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 5))
        settings = self._frame(tab, "Settings")
        self._fields(settings, ("petsys_folder", "ini_file", "data_dir"))
        for row, (name, label) in enumerate(DAQ_FIELDS.items(), 4):
            self.entries[name] = [self._row(settings, row, label, self.vars[name])]
        self._row(settings, 7, "Acq. Time (s):", self.acq_time, width=100)
        safety = self._frame(tab, "Acquisition Safety Limits")
        safety.grid_columnconfigure(3, weight=1)
        for index, (name, (label, _)) in enumerate(SAFETY_FIELDS.items()):
            row, column = 1 + index // 2, 2 * (index % 2)
            ctk.CTkLabel(safety, text=label).grid(row=row, column=column, sticky="e", padx=5, pady=2)
            entry = ctk.CTkEntry(safety, textvariable=self.safety_vars[name], width=100)
            entry.grid(row=row, column=column + 1, sticky="w", padx=5, pady=2)
            self.entries[f"safety.{name}"] = [entry]
        ctk.CTkLabel(safety, text="Acquisition safety policy, not detector QC thresholds; each run records the "
                                  "values it used.", font=ctk.CTkFont(size=11)).grid(
            row=5, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 5))
        processing = self._frame(tab, "Process LDAT files Settings")
        self._fields(processing, ("yaml_file", "processing_root"))
        ctk.CTkLabel(processing, text=f"Relative map paths default to this checkout: {self.session.repo_root}",
                     font=ctk.CTkFont(size=11), justify="left", anchor="w", wraplength=780).grid(
            row=3, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 5))
        columns = ctk.CTkFrame(tab, fg_color="transparent")
        columns.pack(padx=10, pady=5, fill="both", expand=True)
        columns.grid_columnconfigure((0, 1), weight=1)
        left = ctk.CTkFrame(columns, fg_color="transparent")
        left.grid(row=0, column=0, padx=(0, 5), sticky="nsew")
        right = ctk.CTkFrame(columns, fg_color="transparent")
        right.grid(row=0, column=1, padx=(5, 0), sticky="nsew")
        control = self._frame(left, "System Control")
        self.daqd_state = tk.BooleanVar(self.root, False)
        self.buttons["daqd"] = ctk.CTkCheckBox(control, text="DAQD OFF", variable=self.daqd_state, state="disabled",
                                               command=self.toggle_daqd)
        self.buttons["daqd"].grid(row=1, column=0, padx=20, pady=10)
        self._button(control, "initialize", "Initialize System", command=self.initialize).grid(
            row=1, column=1, padx=20, pady=10)
        self.daqd_label = ctk.CTkLabel(control, text="DAQD not started by this manager", justify="left",
                                       anchor="w", wraplength=360, font=ctk.CTkFont(size=11))
        self.daqd_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))
        acquisition = self._frame(left, "Data Acquisition")
        ctk.CTkCheckBox(acquisition, text="Enable Hardware Trigger", variable=self.hw_trigger).grid(
            row=1, column=0, padx=20, pady=10)
        self._button(acquisition, "acquire", "Acquire Data", command=self.acquire).grid(
            row=1, column=1, padx=20, pady=10)
        self.acq_status = ctk.CTkLabel(acquisition, text="", justify="left", anchor="w", wraplength=360,
                                       font=ctk.CTkFont(size=11))
        self.acq_status.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))
        pipeline = self._frame(right, "Complete Automated Pipeline")
        ctk.CTkLabel(pipeline, text="Execute complete workflow:\nAcquire → Convert → Calibrate → Generate LM",
                     font=ctk.CTkFont(size=12)).grid(row=1, column=0, columnspan=2, padx=20, pady=10)
        big = {"font": ctk.CTkFont(size=16, weight="bold"), "height": 50}
        self._button(pipeline, "pipeline", "> RUN COMPLETE PIPELINE", fg_color=GREEN, hover_color=GREEN_HOVER,
                     **big).grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        self._button(pipeline, "stop", "STOP", fg_color=RED, hover_color=RED_HOVER, command=self.stop, **big).grid(
            row=3, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        self.pipeline_status = ctk.CTkLabel(pipeline, text="", font=ctk.CTkFont(size=11), text_color="orange")
        self.pipeline_status.grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        self._readiness(tab, ("daqd", "initialize", "acquire", "pipeline"))

    def _conversion_tab(self, tab):
        selection = self._frame(tab, "Data File Selection")
        self.entries["raw_input"] = [self._row(selection, 1, "RAW Data File:", self.raw_input,
                                               lambda: self._browse(self.raw_input, "file"))]
        settings = self._frame(tab, "Processing Settings")
        self._row(settings, 1, "Number of Split Files:", self.splits, width=100)
        options = self._frame(tab, "Conversion Options")
        self._button(options, "convert_coincidence", "Convert Raw to Coincidence").grid(row=1, column=0, padx=20, pady=10)
        self._button(options, "convert_group", "Convert Raw to Group").grid(row=1, column=1, padx=20, pady=10)
        self._readiness(tab, ("convert_coincidence", "convert_group"))

    def _ldat_tab(self, tab):
        frame = self._frame(tab, "Energy cal file generation")
        self._fields(frame, ("cog_limits_file", "calibration_dir", "report_dir"))
        self._button(frame, "calibrate", "Create Energy cal file").grid(row=4, column=0, columnspan=3, padx=20, pady=10)
        self._readiness(tab, ("calibrate",))

    def _lm_tab(self, tab):
        frame = self._frame(tab, "LM File Generation Settings")
        self._fields(frame, ("calibration_file", "cog_limits_file", "doi_limits_file", "pair_map_file",
                             "region_map_file", "lm_dir"))
        self._button(frame, "listmode", "Generate LM File").grid(row=7, column=0, columnspan=3, padx=20, pady=10)
        self._readiness(tab, ("listmode",))

    def _qc_tab(self, tab):
        settings = self._frame(tab, "Acquisition Settings")
        ctk.CTkLabel(settings, text="Acquisition Mode:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        source = ctk.CTkFrame(settings, fg_color="transparent")
        source.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        for text, mode in (("With Source (1 min)", SourceMode.WITH), ("Without Source (3 min)", SourceMode.WITHOUT)):
            ctk.CTkRadioButton(source, text=text, variable=self.qc_source, value=mode.value).pack(side="left", padx=5)
        self._fields(settings, ("report_dir",), first_row=2)
        validation = self._frame(tab, "Validation Options")
        ctk.CTkCheckBox(validation, text="Generate Plots ( no slab analysis, takes ~5 min )", variable=self.qc_plots,
                        command=self._plots_toggled).grid(row=1, column=0, sticky="w", padx=20, pady=5)
        self.qc_slabs_check = ctk.CTkCheckBox(validation, text="Enable Slab Analysis (requires plots, takes ~30 min)",
                                              variable=self.qc_slabs, state="disabled")
        self.qc_slabs_check.grid(row=2, column=0, sticky="w", padx=40, pady=5)
        control = self._frame(tab, "Quality Control Execution")
        control.grid_columnconfigure(0, weight=1)
        self._button(control, "qc", "Run Quality Control", width=200, height=40, fg_color=GREEN,
                     hover_color=GREEN_HOVER).grid(row=1, column=0, pady=5)
        self._button(control, "qc_stop", "STOP", width=200, height=40, fg_color=RED, hover_color=RED_HOVER).grid(
            row=2, column=0, pady=5)
        status = self._frame(tab, "Status")
        self.qc_status = ctk.CTkLabel(status, text="Quality control is not connected yet",
                                      font=ctk.CTkFont(size=12))
        self.qc_status.grid(row=1, column=0, pady=10)
        self._readiness(tab, ("qc", "qc_analyze"))

    def _add_logo(self):
        try:
            from PIL import Image
            with Image.open(LOGO) as source:
                image = source.convert("RGB")
            width = 150
            self.logo_image = ctk.CTkImage(light_image=image, dark_image=image,
                                           size=(width, max(1, round(image.height * width / image.width))))
            frame = ctk.CTkFrame(self.root, fg_color="transparent")
            frame.pack(side="bottom", fill="x", pady=5)
            ctk.CTkLabel(frame, image=self.logo_image, text="").pack()
            self.logo_path = LOGO
        except Exception as exc:  # Optional decoration never blocks the manager.
            self.session.log(f"Optional logo not shown: {type(exc).__name__}: {exc}")

    # Events (main thread only) --------------------------------------------------------------------

    def _poll(self):
        self._poll_id = None
        lines = []
        for _ in range(MAX_EVENTS_PER_POLL):
            try:
                event = self.session.events.get_nowait()
            except queue.Empty:
                break
            if event.kind == "log":
                lines.append(event.payload)
            elif event.kind == "readiness":
                if event.payload.generation == self._awaiting:
                    self.shown_generation = event.payload.generation
                    self._show_readiness({**event.payload.issues, **self._option_issues})
            elif event.kind == "daqd":
                self._daqd_pending = False
                self._accept_daqd(event.payload, lines)
            elif event.kind == "init_done":
                self._init_pending = False
                outcome = event.payload
                if outcome.status is not None:
                    self._accept_daqd(outcome.status, lines)
                lines.append("System initialized" if outcome.initialized
                             else f"Initialization failed; acquisition stays locked: {outcome.message}")
            elif event.kind == "refused":
                key, message = event.payload
                if key == "daqd":
                    self._daqd_pending = False
                elif key == "initialize":
                    self._init_pending = False
                lines.append(f"{CHECKS[key][1] if key in CHECKS else key} not done: {message}")
            elif event.kind == "workflow":
                self._workflow_event(event.payload)
            elif event.kind == "workflow_done":
                self._workflow_done(event.payload, lines)
            elif event.kind == "shutdown":
                if lines:
                    self.log(*lines)
                    lines = []
                if self._shutdown_done(event.payload):
                    return  # the window is gone
            else:
                lines.append(f"Ignored {event.kind} event")
        if lines:
            self.log(*lines)
        self._refresh_controls()
        if not self._closing:
            self._poll_id = self.root.after(POLL_MS, self._poll)

    def log(self, *messages):
        """Append lines in one widget update, keeping the profile's bounded tail."""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", "".join(f"{message}\n" for message in messages))
        limit = self.session.profile.limits.log_tail_lines
        lines = int(self.log_text.index("end-1c").split(".")[0]) - 1
        if lines > limit:
            self.log_text.delete("1.0", f"{lines - limit + 1}.0")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _show_readiness(self, issues):
        for key, label in self.readiness.items():
            name = CHECKS[key][1]
            found = issues.get(key, ())
            self._ready[key] = not found
            if found:
                label.configure(text=f"{name} unavailable:\n" + "\n".join(f"  • {line}" for line in issue_lines(found)),
                                text_color=STATUS_COLOURS["warn"])
            else:
                suffix = "" if key in CONNECTED else " (control not connected yet)"
                label.configure(text=f"{name}: prerequisites met{suffix}", text_color=STATUS_COLOURS["ok"])
        self._refresh_controls()

    # Backend state (main thread only) -----------------------------------------------------------

    def _accept_daqd(self, status, lines):
        """Keep only the newest service revision: an older status can never restore readiness."""
        current = self.daqd_status
        if current is not None and status.revision <= current.revision:
            return
        if current is None or (status.state, status.message, status.initialized) != (
                current.state, current.message, current.initialized):
            lines.append(f"{DAQD_TEXT[status.state]}: {status.message}" if status.message
                         else DAQD_TEXT[status.state])
        self.daqd_status = status

    def _workflow_event(self, event):
        if self._token is None:
            return  # not ours: no workflow requested by this window
        if event.kind == "workflow_started" and self._run_id is None:
            self._run_id = event.identity.run_id
        if event.identity.run_id != self._run_id:
            return
        kind = event.kind.removeprefix("acquisition_")
        payload = event.payload
        text, tone = None, "info"
        if event.kind == "workflow_started":
            text = f"Run {self._run_id} started"
        elif kind == "attempt_started":
            self._growth_passed = False
            text = f"Attempt {payload['attempt']}/{payload['max_attempts']}: waiting for the RAW file to start writing"
        elif kind == "growth_started":
            text = f"RAW file started writing ({payload['size']:,} bytes)"
        elif kind == "growth_passed":
            self._growth_passed = True
            text, tone = "File growing: growth check passed", "ok"
        elif kind == "rawf_progress":
            text = f"RAW {payload['size'] / 1e6:,.1f} MB, {payload['bytes_per_s'] / 1e6:.2f} MB/s"
            if not payload["growing"]:
                stalled = self._growth_passed
                text += (" - WARNING: RAW file stopped growing" if stalled else " - not growing since the last check")
                tone = "error" if stalled else "warn"
            elif self._growth_passed:
                tone = "ok"
        elif kind in ("aborting", "retry_wait", "attempt_finished", "bias_off"):
            text, tone = event.message, "warn" if kind != "bias_off" else "info"
        elif kind == "bias_unknown":
            self.show_bias_unknown(event.message)
            text, tone = event.message, "error"
        elif event.kind == "workflow_finished":
            text = f"{payload['status']}: {event.message}"
        if text is not None:
            self.acq_status.configure(text=text, text_color=STATUS_COLOURS[tone])

    def _workflow_done(self, result, lines):
        if result.token != self._token:
            lines.append(f"Ignored a stale workflow result (request {result.token})")
            return
        self._token, self._run_id, self._stop_requested = None, None, False
        outcome = result.outcome
        if outcome is None:
            text, tone = result.message, "warn"
        else:
            text = f"Acquisition {outcome.status.value}: {outcome.message}"
            if outcome.run_root is not None:
                text += f"\nRun directory: {outcome.run_root}"
            tone = {ResultStatus.SUCCEEDED: "ok", ResultStatus.CANCELLED: "warn"}.get(outcome.status, "error")
        self.acq_status.configure(text=text, text_color=STATUS_COLOURS[tone])
        lines.append(text)

    def show_bias_unknown(self, message):
        self.bias_unknown = True
        self.bias_label.configure(text=f"SiPM bias state unknown. {message}")
        if not self.bias_frame.winfo_ismapped():
            self.bias_frame.pack(padx=10, pady=5, fill="x", before=self._first_setup_frame)

    def acknowledge_bias(self):
        self.bias_unknown = False
        self.bias_frame.pack_forget()
        self.log("Operator confirmed the SiPM bias state after the unknown-bias warning")
        self._refresh_controls()

    def _refresh_controls(self):
        """Enable only what the newest backend state allows; called after every event batch."""
        if self._closing:
            return
        status = self.daqd_status
        state = status.state if status is not None else DaqdState.OFF
        busy = self._token is not None or self._shutting_down
        initializing = self._init_pending or (status is not None and status.initializing)
        ready = state == DaqdState.READY
        live = state in (DaqdState.STARTING, DaqdState.READY)
        enable = {
            "daqd": not busy and not self._daqd_pending and not initializing and (
                live or (state in (DaqdState.OFF, DaqdState.FAILED) and self._ready.get("daqd", False))),
            "initialize": not busy and ready and not initializing and self._ready.get("initialize", False),
            "acquire": (not busy and ready and status.initialized and not initializing and not self.bias_unknown
                        and self._ready.get("acquire", False)),
            "stop": self._token is not None and not self._stop_requested and not self._shutting_down,
        }
        for key, button in self.buttons.items():
            button.configure(state="normal" if enable.get(key, False) else "disabled")
        self.daqd_state.set(live)
        self.buttons["daqd"].configure(text=DAQD_TEXT[state])
        if status is None:
            text = "DAQD not started by this manager"
        else:
            text = f"{DAQD_TEXT[state]}" + (f" (pid {status.pid})" if status.pid else "")
            if ready:
                text += "; initialized" if status.initialized else "; not initialized"
            if status.message:
                text += f"\n{status.message}"
        self.daqd_label.configure(text=text, text_color=STATUS_COLOURS[
            {DaqdState.READY: "ok", DaqdState.FAILED: "error"}.get(state, "info")])

    # Actions --------------------------------------------------------------------------------------

    def _profile_or_log(self):
        try:
            return self.profile_from_ui()
        except ProfileError as exc:
            self.log(f"Profile invalid: {exc}")
            return None

    def toggle_daqd(self):
        state = self.daqd_status.state if self.daqd_status is not None else DaqdState.OFF
        if state in (DaqdState.STARTING, DaqdState.READY):
            self.session.stop_daqd()
            self._daqd_pending = True
        else:
            profile = self._profile_or_log()
            if profile is not None:
                self.session.start_daqd(profile)
                self._daqd_pending = True
        self._refresh_controls()  # the checkbox shows the backend state, not the click

    def initialize(self):
        profile = self._profile_or_log()
        if profile is not None:
            self.session.initialize(profile)
            self._init_pending = True
        self._refresh_controls()

    def acquire(self):
        profile = self._profile_or_log()
        try:
            options = RunOptions(duration_s=float(self.acq_time.get().strip()), hardware_trigger=self.hw_trigger.get())
        except (ValueError, ProfileError) as exc:
            self.log(f"Acquisition not started: Acq. Time (s): {exc}")
            options = None
        if profile is not None and options is not None:
            self._token = self.session.start_workflow(profile, Action.ACQUIRE, options)
            self._run_id, self._stop_requested = None, False
            self.acq_status.configure(text="Starting acquisition...", text_color=STATUS_COLOURS["info"])
        self._refresh_controls()

    def stop(self):
        if self._token is not None and self.session.stop_workflow():
            self._stop_requested = True
            self.acq_status.configure(text="STOP: terminating the acquisition and switching bias off; "
                                           "no further attempt", text_color=STATUS_COLOURS["warn"])
            self.log("STOP requested")
        self._refresh_controls()

    # Profile and options ------------------------------------------------------------------------

    def _show_profile(self):
        profile = self.session.profile
        self._loading = True
        try:
            self.profile_path.set(str(self.session.profile_path))
            for name in PROFILE_FIELDS:
                self.vars[name].set(getattr(profile, name) or "")
            self.vars["daq_type"].set(profile.daq_type)
            self.vars["cards"].set(", ".join(profile.cards))
            self.vars["socket_path"].set(profile.socket_path)
            for name in SAFETY_FIELDS:
                value = (profile.safety.min_growth_bytes / 1e6 if name == "min_growth_mb"
                         else getattr(profile.safety, name))
                self.safety_vars[name].set(str(value) if type(value) is int else format(value, ".15g"))
        finally:
            self._loading = False
        self._update_profile_status()
        self._run_check()

    def profile_from_ui(self):
        """Edited profile; fields this window does not show keep the loaded values."""
        values = {name: self.vars[name].get().strip() or None for name in PROFILE_FIELDS}
        values["daq_type"] = self.vars["daq_type"].get().strip()
        values["socket_path"] = self.vars["socket_path"].get().strip()
        values["cards"] = tuple(card.strip() for card in self.vars["cards"].get().split(",") if card.strip())
        values["safety"] = self.safety_from_ui()
        return replace(self.session.profile, **values)

    def safety_from_ui(self):
        values = {}
        for name, (label, kind) in SAFETY_FIELDS.items():
            text = self.safety_vars[name].get().strip()
            try:
                values[name] = kind(text)
                if name == "min_growth_mb":
                    values["min_growth_bytes"] = round(values.pop(name) * 1_000_000)
            except (ValueError, OverflowError):
                raise ProfileError(f"{label.rstrip(':')} must be {'an integer' if kind is int else 'a number'}") \
                    from None
        return AcquisitionSafety(**values)  # range checks raise ProfileError

    def requests(self):
        """Prerequisite requests per check plus run-option reasons found while parsing."""
        requests, issues = {}, {}

        def add(key, **options):
            try:
                requests[key] = (CHECKS[key][0], RunOptions(**options))
            except ProfileError as exc:
                issues[key] = (PrerequisiteIssue("options", str(exc)),)

        def number(variable, kind, label, keys):
            try:
                return kind(variable.get().strip())
            except ValueError:
                for key in keys:
                    issues[key] = (PrerequisiteIssue("options", f"{label} must be a positive number"),)

        duration = number(self.acq_time, float, "Acq. Time (s)", ("acquire", "pipeline"))
        splits = number(self.splits, int, "Number of Split Files", ("convert_coincidence", "convert_group"))
        for key in ("daqd", "initialize", "calibrate", "listmode", "qc_analyze"):
            add(key)
        if duration is not None:
            for key in ("acquire", "pipeline"):
                add(key, duration_s=duration, hardware_trigger=self.hw_trigger.get())
        if splits is not None:
            raw = self.raw_input.get().strip() or None
            add("convert_coincidence", splits=splits, raw_input=raw)
            add("convert_group", splits=splits, raw_input=raw, population=Population.GROUP)
        add("qc", source_mode=SourceMode(self.qc_source.get()), plots=self.qc_plots.get(),
            slabs=self.qc_plots.get() and self.qc_slabs.get())
        return requests, issues

    def _edited(self, *_):
        if self._loading or self._closing:
            return
        self._update_profile_status()
        if self._check_id is not None:
            self.root.after_cancel(self._check_id)
        self._check_id = self.root.after(CHECK_DELAY_MS, self._run_check)

    def _run_check(self):
        self._check_id = None
        try:
            profile = self.profile_from_ui()
        except ProfileError as exc:
            self._awaiting = None  # an in-flight answer for an older profile is now stale
            self._show_readiness({key: (PrerequisiteIssue("profile", str(exc)),) for key in CHECKS})
            return
        requests, self._option_issues = self.requests()
        self._awaiting = self.session.check(profile, requests)

    def _update_profile_status(self):
        try:
            edited = self.profile_from_ui() != self.session.profile
            invalid = None
        except ProfileError as exc:
            edited, invalid = True, exc
        state = {"loaded": "loaded", "defaults": "not found; defaults in use",
                 "error": "not loaded (see log); defaults in use"}.get(self.profile_state, self.profile_state)
        text = f"Profile {self.session.profile_path}: {state}"
        if invalid is not None:
            text += f"; edits invalid: {invalid}"
        elif edited or self.profile_path.get().strip() != str(self.session.profile_path):
            text += "; unsaved edits"
        self.profile_status.configure(text=text)

    def reload_profile(self):
        path = self.profile_path.get().strip()
        try:
            self.session.load(path or None)
        except (OSError, ValueError) as exc:
            self.log(f"Profile not reloaded; current values kept: {exc}")
            return False
        self.profile_state = "loaded"
        self.log(f"Loaded profile {self.session.profile_path}")
        self._show_profile()
        return True

    def save_profile(self):
        path = self.profile_path.get().strip()
        try:
            written = self.session.save(self.profile_from_ui(), path or None)
        except (OSError, ValueError) as exc:
            self.log(f"Profile not saved: {exc}")
            return False
        self.profile_state = "loaded"
        self.log(f"Saved profile {written}")
        self._show_profile()
        return True

    def _plots_toggled(self):
        if self.qc_plots.get():
            self.qc_slabs_check.configure(state="normal")
        else:
            self.qc_slabs.set(False)
            self.qc_slabs_check.configure(state="disabled")

    def _browse(self, variable, kind):
        current = variable.get().strip()
        start = str(Path(current).parent if kind == "file" and current else current or Path.home())
        chosen = (filedialog.askopenfilename(initialdir=start) if kind == "file"
                  else filedialog.askdirectory(initialdir=start))
        if chosen:
            variable.set(chosen)

    def _browse_profile(self):
        current = self.profile_path.get().strip()
        chosen = filedialog.askopenfilename(initialdir=str(Path(current).parent) if current else None,
                                            filetypes=[("Manager profile", "*.yaml *.yml"), ("All files", "*")])
        if chosen:
            self.profile_path.set(chosen)
            self.reload_profile()

    def on_close(self):
        """Close at once when idle; otherwise stop the workflow (bias-off included), then DAQD, asynchronously."""
        if self.closed or self._closing:
            return
        if self._shutting_down:
            self.log("Closing is in progress; waiting for the acquisition and DAQD to stop")
            return
        if self.session.idle():
            self._finalize_close()
            return
        self._shutting_down = True
        self.acq_status.configure(text="Closing: stopping the workflow, waiting for bias-off, then stopping DAQD",
                                  text_color=STATUS_COLOURS["warn"])
        self.log("Closing: STOP sent to the workflow; DAQD stops after it has finished")
        self.session.shutdown(self.shutdown_timeout_s)
        self._refresh_controls()

    def _shutdown_done(self, result):
        if result.ok:
            self.log(f"Shutdown complete: {result.message}")
            self._finalize_close()
            return True
        self._shutting_down = False
        self.acq_status.configure(text=f"Close incomplete: {result.message}", text_color=STATUS_COLOURS["error"])
        self.log(f"Close incomplete: {result.message}")
        return False

    def _finalize_close(self):
        self._closing = True
        for job in (self._poll_id, self._check_id):
            if job is not None:
                self.root.after_cancel(job)
        self._poll_id = self._check_id = None
        self.session.close()
        self.root.destroy()
        self.closed = True


def main(argv=None):
    parser = argparse.ArgumentParser(description="PETsys Manager for the Cornell system.")
    parser.add_argument("--profile", help=f"machine profile YAML (default: {default_profile_path()})")
    args = parser.parse_args(argv)
    root = ctk.CTk()
    PETsysManager(root, ManagerSession(args.profile))
    root.mainloop()
