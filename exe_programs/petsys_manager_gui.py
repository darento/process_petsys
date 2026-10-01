"""PETsys Manager window: the five Cornell workflow tabs, command log and machine profile.

Views and main-thread event polling only. Profiles and prerequisite checks live
in the toolkit-free `src.petsys_manager.session`; workers reach this window only
through its queue, drained here with `after` on the Tk thread. DAQD/acquisition,
conversion and processing controls are connected by later spec 003 tasks
(T14-T16); until then their buttons stay disabled beside the prerequisite reasons.
Nothing here launches processes or touches DAQ resources at startup.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

from src.petsys_manager.contracts import Action, Population, SourceMode
from src.petsys_manager.session import ManagerSession
from src.petsys_manager.settings import (PrerequisiteIssue, ProfileError, RunOptions,
                                         default_profile_path)


__version__ = "0.1.0"
TITLE = "PETsys Manager - Cornell"
TABS = ("System Setup & Acquisition", "RAWF to LDAT Conversion", "LDAT Processing",
        "LM File Generation", "System Quality Control")
LOGO = Path(__file__).resolve().parent / "assets" / "onco_logo.jpeg"  # optional, module-relative
POLL_MS = 100
MAX_EVENTS_PER_POLL = 200
CHECK_DELAY_MS = 400
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
ISSUE_LABELS = {name: label.rstrip(":") for name, (label, _) in PROFILE_FIELDS.items()}
ISSUE_LABELS.update({name: label.rstrip(":") for name, label in DAQ_FIELDS.items()})
ISSUE_LABELS.update({
    "platform": "Platform", "raw_input": "RAW Data File", "inputs": "Input LDAT files",
    "map_file": "Map file named by the processing YAML", "output_format": "Conversion format",
    "capabilities.fixed_output_confirmed": "Converter fixed output", "options": "Run options",
    "settings": "Settings", "profile": "Profile",
})
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
        self._build()
        for variable in (*self.vars.values(), self.acq_time, self.hw_trigger, self.raw_input, self.splits,
                         self.qc_source, self.qc_plots, self.qc_slabs):
            variable.trace_add("write", self._edited)
        session.log(f"PETsys Manager {__version__}; checkout {session.repo_root}")
        self.profile_state = session.open()
        self._show_profile()
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
        profile = self._frame(tab, "Machine Profile")
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
        self.buttons["daqd"] = ctk.CTkCheckBox(control, text="DAQD OFF", variable=self.daqd_state, state="disabled")
        self.buttons["daqd"].grid(row=1, column=0, padx=20, pady=10)
        self._button(control, "initialize", "Initialize System").grid(row=1, column=1, padx=20, pady=10)
        acquisition = self._frame(left, "Data Acquisition")
        ctk.CTkCheckBox(acquisition, text="Enable Hardware Trigger", variable=self.hw_trigger).grid(
            row=1, column=0, padx=20, pady=10)
        self._button(acquisition, "acquire", "Acquire Data").grid(row=1, column=1, padx=20, pady=10)
        pipeline = self._frame(right, "Complete Automated Pipeline")
        ctk.CTkLabel(pipeline, text="Execute complete workflow:\nAcquire → Convert → Calibrate → Generate LM",
                     font=ctk.CTkFont(size=12)).grid(row=1, column=0, columnspan=2, padx=20, pady=10)
        big = {"font": ctk.CTkFont(size=16, weight="bold"), "height": 50}
        self._button(pipeline, "pipeline", "> RUN COMPLETE PIPELINE", fg_color=GREEN, hover_color=GREEN_HOVER,
                     **big).grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        self._button(pipeline, "stop", "STOP", fg_color=RED, hover_color=RED_HOVER, **big).grid(
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
            else:
                lines.append(f"Ignored {event.kind} event")
        if lines:
            self.log(*lines)
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
            if found:
                label.configure(text=f"{name} unavailable:\n" + "\n".join(f"  • {line}" for line in issue_lines(found)),
                                text_color=("#a04000", "#f0a050"))
            else:
                label.configure(text=f"{name}: prerequisites met (control not connected yet)",
                                text_color=("#1e7d3a", "#6fcf8a"))

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
        return replace(self.session.profile, **values)

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
        # No workflow can be active before T14 connects them; T14 makes this asynchronous.
        self._closing = True
        for job in (self._poll_id, self._check_id):
            if job is not None:
                self.root.after_cancel(job)
        self._poll_id = self._check_id = None
        self.session.close()
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="PETsys Manager for the Cornell system.")
    parser.add_argument("--profile", help=f"machine profile YAML (default: {default_profile_path()})")
    args = parser.parse_args(argv)
    root = ctk.CTk()
    PETsysManager(root, ManagerSession(args.profile))
    root.mainloop()
