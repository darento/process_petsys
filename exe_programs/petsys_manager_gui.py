"""PETsys Manager window: the five Cornell workflow tabs, command log and machine profile.

Views and main-thread event polling only. Profiles, prerequisite checks, DAQD,
initialization and workflows live in the toolkit-free `src.petsys_manager.session`;
workers reach this window only through its queue, drained here with `after` on
the Tk thread. Controls follow the backend: DAQD readiness/initialization come
from the service's revisioned statuses, a workflow's buttons from its token, and
older statuses or results never re-enable a control. DAQD, Initialize, Acquire
and STOP are connected (T14), as are RAW conversion and the exact ordered LDAT
input lists of the processing tabs (T15) and the calibration, LM, QC and
complete pipeline actions (T16). Each result shows the recorded run directory
and the exact validated outputs; a later stage only ever consumes its
predecessor's recorded outputs, and nothing here edits a field or the profile
on behalf of a run. Conversion duration, splits and hit limit are explicit
settings, never read from a file name. The manager converts and processes
compact coincidence only (FR-10); the operator confirms that listed LDAT files
are compact coincidence, since an extension cannot tell.
Nothing launches at startup. Closing stops the workflow, waits for its bias-off,
then stops the owned DAQD, while the window keeps polling.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
import queue
import re
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

from src.petsys_manager.acquisition import DaqdState
from src.petsys_manager.contracts import (Action, DataFormat, InputDescriptor, Population, ResultStatus,
                                          SourceMode)
from src.petsys_manager.session import ManagerSession
from src.petsys_manager.settings import (AcquisitionSafety, LMMetadata, PrerequisiteIssue, ProfileError,
                                         RunOptions, default_profile_path)
from src.petsys_manager.workflow import format_elapsed, portable_name


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
    "petsys_python": ("PETsys Python (init/acquire/bias):", "file"),   # FR-23; empty: the tools' shebang
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
    "options": "Run options",
    "settings": "Settings", "profile": "Profile",
})
# LM header metadata, saved in the profile (FR-12): name -> (label, type). Empty means unavailable.
LM_FIELDS = {
    "isotope": ("Isotope:", str), "acquisition_time_s": ("Acquisition time (s):", float),
    "measurement_time_s": ("Measurement time (s):", float), "detector_size_x_mm": ("Detector size X (mm):", float),
    "detector_size_y_mm": ("Detector size Y (mm):", float), "module_number": ("Module number:", int),
    "ring_number": ("Ring number:", int), "ring_distance_mm": ("Ring distance (mm):", float),
    "detector_pixels_x": ("Detector pixels X:", int), "detector_pixels_y": ("Detector pixels Y:", int),
    "timestamp_unit": ("Timestamp unit:", str),
}
LIVE_KEYS = ("acquire", "pipeline", "qc")      # need an initialized system from this manager's DAQD
PROCESSING_KEYS = ("calibrate", "listmode", "qc_analyze")
STOP_KEYS = ("stop", "convert_stop", "qc_stop")
ACTION_TEXT = {Action.ACQUIRE: "Acquisition", Action.CONVERT: "Conversion", Action.CALIBRATE: "Energy calibration",
               Action.LISTMODE: "LM generation", Action.QC: "Quality control", Action.QC_ANALYZE: "Offline QC",
               Action.PIPELINE: "Complete pipeline"}
STAGE_TEXT = {"acquisition": "Acquisition", "conversion": "Conversion", "calibration": "Energy calibration",
              "listmode": "LM generation", "qc": "QC analysis"}
ARTIFACT_TEXT = {"encal": "Energy cal file", "calibration_sidecar": "Calibration provenance",
                 "calibration_status": "Fit status per key", "calibration_plot": "Summary plot",
                 "listmode": "LM file", "listmode_provenance": "LM provenance", "listmode_job": "LM job record",
                 "listmode_debug_plot": "LM debug plot", "qc_report": "QC output", "qc_summary": "QC summary"}
HIDDEN_ARTIFACTS = {"processing_request", "processing_result"}
QC_PRESET_S = {SourceMode.WITH: 60.0, SourceMode.WITHOUT: 180.0}   # display of the preflight preset
SPLIT_TIME_OFFSET_S = 0.1  # display of commands.build_conversion's --splitTime rule
# Declared LDAT content: key -> (label, format, population). Bytes/extension cannot prove it.
# The manager's only route is compact coincidence (FR-10, 2026-10-05).
DECLARED = {
    "compact_coincidence": ("Compact coincidence", DataFormat.COMPACT, Population.COINCIDENCE),
}
# Processing input lists: check key -> (frame title, default declaration).
SELECTIONS = {
    "calibrate": ("Input LDAT Files (compact coincidence)", "compact_coincidence"),
    "listmode": ("Input LDAT Files (compact coincidence)", "compact_coincidence"),
    "qc_analyze": ("Existing LDAT Files for Offline QC (compact coincidence)", "compact_coincidence"),
}
# Which processing lists may take a conversion's validated outputs.
OUTPUT_TARGETS = {(DataFormat.COMPACT, Population.COINCIDENCE): ("calibrate", "listmode", "qc_analyze")}
SPLIT_NAME = re.compile(r"(.+)_(\d+)\.ldat\Z")
MAX_FOLDER_ENTRIES = 20000  # bound for the split-sibling folder scan
PROBE_RECORDS = 10000
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
    "calibrate": (Action.CALIBRATE, "Create Energy cal file", 2),
    "listmode": (Action.LISTMODE, "Generate LM File", 3),
    "qc": (Action.QC, "Run Quality Control", 4),
    "qc_analyze": (Action.QC_ANALYZE, "Analyze existing compact LDAT", 4),
}
GREEN, GREEN_HOVER, RED, RED_HOVER = "#2ecc71", "#27ae60", "#e74c3c", "#c0392b"
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


ROUTE_NEEDS = {  # what each processing action consumes (plan route table)
    "calibrate": "Energy calibration takes compact coincidence files",
    "listmode": "LM generation takes compact coincidence files",
    "qc_analyze": "Offline QC takes compact coincidence files",
}
WRONG_ROUTE = re.compile(r"Unsupported format/population for (\w+): ")


def issue_lines(issues):
    """Specific reasons; unselected tools, missing LM metadata and wrong-route inputs collapse to one line each."""
    lines, tools, metadata, routes = [], [], [], {}
    for issue in issues:
        route = WRONG_ROUTE.match(issue.message) if issue.field == "inputs" else None
        if route is not None and route.group(1) in ROUTE_NEEDS:
            routes[route.group(1)] = routes.get(route.group(1), 0) + 1
        elif issue.field.startswith("tool:") and issue.message == "Select the PETsys tools folder":
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
        lines.append(f"LM metadata missing from the profile: {', '.join(metadata)} (LM File Generation tab)")
    for action, count in routes.items():
        lines.append(f"{ISSUE_LABELS['inputs']}: {ROUTE_NEEDS[action]}; {count} listed file(s) are declared "
                     "as another content")
    return lines


def limit_plan_text(plan):
    """Operator text for an event-limit plan (FR-21)."""
    if "error" in plan:
        return f"Event limit unavailable: {plan['error']}"
    files = f"{plan['files']} split(s)" if plan.get("files_from_splits") else f"{plan['files']} file(s)"
    workers = plan.get("workers")
    if isinstance(workers, int):
        files += f", {workers} worker process(es)" if workers > 1 else ", 1 worker (no parallel processing)"
    if plan["limit_mode"] == "reference":
        per_file = plan["limit_per_file"]
        return ("Event limit (reference): " + ("whole files" if per_file is None else
                f"{per_file:,} passing coincidences per file") + f", {files} (cornell_slab_en_cal.py)")
    return (f"Event limit (target): K {plan['mapped_slab_keys']:,} keys x P {plan['positions']} x "
            f"T {plan['target_per_key']:,} = {plan['limit_total']:,} kept sides, {plan['limit_per_file']:,} per "
            f"file over {files}; T is an average, so low-occupancy keys can stay below it")


def limit_used_text(summary):
    """The limit a calibration used and the sides its keys received (FR-21)."""
    plan, coverage = summary.get("limit_plan") or {}, summary.get("coverage") or {}
    text = ""
    if plan:
        inputs = summary.get("inputs") or ()
        events = [item.get("events_passed", 0) for item in inputs]
        sides = [item.get("accepted_sides", 0) for item in inputs]
        text += f"\n  {limit_plan_text(plan)}"
        if events:
            text += (f"; used per file: {min(events):,}-{max(events):,} events, {min(sides):,}-{max(sides):,} "
                     "kept sides")
    decoding = summary.get("decoding") or {}
    if decoding.get("decodings") == 1:
        text += (f"\n  Each file read once: {decoding['pairs_kept_bytes'] / 2 ** 20:,.0f} MB of selected events kept "
                 f"(budget {decoding['memory_budget_bytes'] / 2 ** 20:,.0f} MB)")
    elif decoding.get("decodings") == 2:
        bound, budget = decoding.get("pair_storage_bound_bytes"), decoding.get("memory_budget_bytes")
        why = ("reference mode" if plan.get("limit_mode") == "reference" else
               "no memory budget" if budget is None else
               f"up to {bound / 2 ** 20:,.0f} MB needed, budget {budget / 2 ** 20:,.0f} MB" if bound else "no limit")
        text += f"\n  Each file read twice ({why})"
    if coverage.get("keys_with_sides"):
        below = coverage.get("keys_below_target")
        text += (f"\n  Sides per key ({coverage['keys_with_sides']:,} keys with sides): min {coverage['min_sides']:,}, "
                 f"median {coverage['median_sides']:,.0f}"
                 + ("" if below is None else f"; {below:,} below T {coverage['target_per_key']:,}")
                 + f"; {coverage['keys_below_fit_minimum']:,} below the {coverage['fit_minimum']}-event fit minimum")
    return text


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def split_siblings(path):
    """Other split files of exactly this prefix, ``<prefix>_<n>.ldat`` in the same folder, in split order.

    Returns (siblings, complete); complete is False when the folder exceeds the scan bound.
    Files of another prefix (``<prefix>_extra_2``, ``<prefix>2_1``, an unsplit ``<prefix>.ldat``) never match.
    """
    match = SPLIT_NAME.fullmatch(path.name)
    if match is None:
        return (), True
    pattern = re.compile(re.escape(match.group(1)) + r"_\d+\.ldat\Z")
    found = []
    with os.scandir(path.parent) as entries:
        for count, entry in enumerate(entries):
            if count >= MAX_FOLDER_ENTRIES:
                return tuple(sorted(found, key=natural_key)), False
            if entry.name != path.name and pattern.fullmatch(entry.name) and entry.is_file():
                found.append(Path(entry.path))
    return tuple(sorted(found, key=natural_key)), True


def megabytes(size):
    return f"{size / 1e6:,.1f} MB"


class InputSelection:
    """A processing tab's exact ordered LDAT list, confirmed as compact coincidence (FR-10/FR-11).

    Main thread only. Nothing is matched by wildcard: picking one split file offers
    the other splits of exactly that prefix for confirmation. Structure checks run
    in the session's worker and only the newest result is shown.
    """

    def __init__(self, app, parent, key, title, default):
        self.app, self.key = app, key
        self.paths = []
        self.origin = None          # "conversion run <id>" when filled from validated converter outputs
        self.probe_request = None
        self._setting = False
        root = app.root
        self.declared = tk.StringVar(root, default)
        self.confirmed = tk.BooleanVar(root, False)
        frame = app._frame(parent, title)
        holder = ctk.CTkFrame(frame, fg_color="transparent")
        holder.grid(row=1, column=0, columnspan=4, sticky="ew", padx=10, pady=2)
        holder.grid_columnconfigure(0, weight=1)
        dark = ctk.get_appearance_mode() == "Dark"  # plain Tk widget: match the theme once
        self.listbox = tk.Listbox(holder, height=6, selectmode="browse", exportselection=False, activestyle="none",
                                  bg="#2b2b2b" if dark else "white", fg="#dce4ee" if dark else "black",
                                  selectbackground="#1f6aa5", selectforeground="white", highlightthickness=0)
        self.listbox.grid(row=0, column=0, sticky="ew")
        scroll = ctk.CTkScrollbar(holder, command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scroll.set)
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.grid(row=2, column=0, columnspan=4, sticky="w", padx=10, pady=2)
        self.buttons = {}
        for name, text, command in (("add", "Add files...", self.add), ("remove", "Remove", self.remove),
                                    ("up", "Up", lambda: self.move(-1)), ("down", "Down", lambda: self.move(1)),
                                    ("clear", "Clear", self.clear), ("check", "Check structure", self.check)):
            self.buttons[name] = ctk.CTkButton(row, text=text, width=90, command=command)
            self.buttons[name].pack(side="left", padx=(0, 6))
        self.confirm_check = ctk.CTkCheckBox(
            frame, variable=self.confirmed,
            text="I confirm the listed files are compact coincidence (an .ldat name cannot tell)")
        self.confirm_check.grid(row=4, column=0, columnspan=4, sticky="w", padx=10, pady=2)
        self.summary = ctk.CTkLabel(frame, text="", justify="left", anchor="w", wraplength=780)
        self.summary.grid(row=5, column=0, columnspan=4, sticky="w", padx=10, pady=(2, 0))
        self.feedback = ctk.CTkLabel(frame, text="", justify="left", anchor="w", wraplength=780,
                                     font=ctk.CTkFont(size=11))
        self.feedback.grid(row=6, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 5))
        self.declared.trace_add("write", self._declared_changed)
        self.confirmed.trace_add("write", lambda *_: self._changed(reset_probe=False))
        self.refresh()

    # Declaration -------------------------------------------------------------------------------

    @property
    def declaration(self):
        return DECLARED[self.declared.get()]

    def descriptors(self):
        """The exact ordered list as declared; confirmation is reported separately (confirmation_issue)."""
        _, data_format, population = self.declaration
        return tuple(InputDescriptor(path, data_format, population) for path in self.paths)

    def confirmation_issue(self):
        if self.paths and not self.confirmed.get():
            return "Confirm that the listed files are compact coincidence (an .ldat name cannot tell)"
        return None

    def _declared_changed(self, *_):
        if self._setting:
            return
        self.origin = None  # an operator declaration replaces the converter's
        self._set(confirmed=False)
        self._changed()

    def _set(self, *, declared=None, confirmed=None):
        self._setting = True
        try:
            if declared is not None:
                self.declared.set(declared)
            if confirmed is not None:
                self.confirmed.set(confirmed)
        finally:
            self._setting = False

    # List editing --------------------------------------------------------------------------------

    def add(self, chosen=None):
        """Append operator-chosen files in natural order; offer only same-prefix split siblings."""
        if chosen is None:
            start = str(self.paths[-1].parent) if self.paths else None
            chosen = self.app.ask_files(title="Select LDAT files", initialdir=start,
                                        filetypes=[("LDAT files", "*.ldat"), ("All files", "*")])
        chosen = sorted({Path(item) for item in chosen or ()}, key=natural_key)
        if not chosen:
            return
        additions, offered = list(chosen), {}
        for path in chosen:
            try:
                siblings, complete = split_siblings(path)
            except OSError as exc:
                self.app.log(f"Split files of {path.name} not offered: {exc}")
                continue
            if not complete:
                self.app.log(f"Split files of {path.name} not offered: its folder exceeds {MAX_FOLDER_ENTRIES} entries")
            stem = SPLIT_NAME.fullmatch(path.name).group(1) if siblings else None
            for sibling in siblings:
                if sibling not in additions and sibling not in self.paths:
                    offered.setdefault((path.parent, stem), []).append(sibling)
        for (folder, stem), extra in offered.items():
            extra = sorted(set(extra), key=natural_key)
            names = "\n".join(item.name for item in extra[:12]) + (f"\n... {len(extra) - 12} more" if len(extra) > 12 else "")
            if self.app.ask_yes_no("Add the other split files?",
                                   f"{len(extra)} other split file(s) of exactly {stem}_<n>.ldat in {folder}:\n\n"
                                   f"{names}\n\nAdd them to the list? Files of any other prefix are not offered."):
                additions.extend(extra)
                self.app.log(f"Added {len(extra)} split sibling(s) of {stem}_<n>.ldat after confirmation")
        additions = sorted(set(additions), key=natural_key)
        repeated = [path for path in additions if path in self.paths]
        if repeated:
            self.app.log(f"Already listed, not added again: {', '.join(path.name for path in repeated)}")
        self.paths.extend(path for path in additions if path not in self.paths)
        self.origin = None
        self._set(confirmed=False)  # the confirmation covers the listed files: confirm again
        self._changed()

    def _selected(self):
        chosen = self.listbox.curselection()
        return chosen[0] if chosen else None

    def remove(self):
        index = self._selected()
        if index is not None:
            del self.paths[index]
            self._changed()
            if self.paths:
                self.listbox.selection_set(min(index, len(self.paths) - 1))

    def move(self, delta):
        index = self._selected()
        if index is None or not 0 <= index + delta < len(self.paths):
            return
        self.paths[index], self.paths[index + delta] = self.paths[index + delta], self.paths[index]
        self._changed()
        self.listbox.selection_set(index + delta)

    def clear(self):
        self.paths, self.origin = [], None
        self._set(confirmed=False)
        self._changed()

    def use_outputs(self, descriptors, run_id):
        """Replace the list with a conversion's exact validated outputs and their converter declaration."""
        data_format, population = descriptors[0].format, descriptors[0].population
        key = next(key for key, (_, f, p) in DECLARED.items() if (f, p) == (data_format, population))
        self.paths = [Path(item.path) for item in descriptors]
        self.origin = f"conversion run {run_id}"
        self._set(declared=key, confirmed=True)
        self._changed()

    def _changed(self, reset_probe=True):
        if self._setting:
            return
        if reset_probe:
            self.probe_request = None
            self.feedback.configure(text="", text_color=STATUS_COLOURS["info"])
        self.refresh()
        self.app._edited()

    def refresh(self):
        self.listbox.delete(0, "end")
        total, missing = 0, 0
        for index, path in enumerate(self.paths, 1):
            try:
                size = path.stat().st_size
                total += size
                self.listbox.insert("end", f"{index}. {path}   ({megabytes(size)})")
            except OSError:
                missing += 1
                self.listbox.insert("end", f"{index}. {path}   (MISSING)")
        label = self.declaration[0]
        if not self.paths:
            text = f"No files selected; declared {label.lower()}"
        else:
            text = f"{len(self.paths)} file(s), {megabytes(total)}" + (f", {missing} missing" if missing else "")
            text += f"; declared {label.lower()}"
            text += f" from {self.origin}" if self.origin else ""
            text += "; confirmed" if self.confirmed.get() else "; NOT confirmed"
        self.summary.configure(text=text)
        busy = self.probe_request is not None
        for name in ("remove", "up", "down", "clear"):
            self.buttons[name].configure(state="normal" if self.paths else "disabled")
        self.buttons["check"].configure(state="normal" if self.paths and not busy else "disabled")

    # Structure check -----------------------------------------------------------------------------

    def check(self):
        profile = self.app._profile_or_log()
        if profile is None or not self.paths:
            return
        self.probe_request = self.app.session.probe_inputs(profile, self.key, self.descriptors(),
                                                           max_records=PROBE_RECORDS)
        self.feedback.configure(text=f"Checking the first {PROBE_RECORDS:,} records of each file against the "
                                     "selected map...", text_color=STATUS_COLOURS["info"])
        self.refresh()

    def show_probe(self, result):
        if result.request != self.probe_request:
            return False  # an older check, or the list changed since
        self.probe_request = None
        lines, failed = [], bool(result.message)
        if result.message:
            lines.append(result.message)
        for path, summary, error in result.results:
            name = Path(path).name
            if error is not None:
                failed = True
                lines.append(f"FAILED {name}: {error.removeprefix(f'{path}: ')}")
            elif summary.complete:
                lines.append(f"OK {name}: whole file checked, {summary.records_checked:,} records, "
                             f"{summary.channel_hits_checked:,} hits")
            else:
                total = f" of {summary.records_total:,} (from the file size)" if summary.records_total else ""
                lines.append(f"OK {name}: first {summary.records_checked:,} records{total} pass")
        if not failed:
            lines.append("Partial check only: full validation runs before processing; the bytes cannot prove "
                         "group versus coincidence.")
        self.feedback.configure(text="\n".join(lines), text_color=STATUS_COLOURS["error" if failed else "ok"])
        self.refresh()
        return True


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
        self._action = None             # the action of this window's workflow token
        self._status_label = None       # where that workflow's progress is shown
        self.last_conversion = None     # (run id, exact validated output descriptors) of the last conversion
        self.last_calibration = None    # (run id, recorded .encal) of the last successful calibration stage
        self._last_raw = None           # the RAW input the running conversion reported
        self._stages = ()               # stage ids of this window's running workflow
        self._issues = {}               # check key -> reasons of the newest shown readiness
        self.selections = {}
        self.ask_files = filedialog.askopenfilenames   # dialogs are attributes so checks can answer them
        self.ask_yes_no = messagebox.askyesno
        self.root.title(f"{TITLE} {__version__}")
        self.root.geometry("900x950")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.vars = {name: tk.StringVar(root) for name in (*PROFILE_FIELDS, *DAQ_FIELDS)}
        self.profile_path = tk.StringVar(root)
        self.acq_time = tk.StringVar(root, "10")
        self.acq_name = tk.StringVar(root, "acquisition")    # RAW basename and run-folder name (T29)
        self.hw_trigger = tk.BooleanVar(root, False)
        self.raw_input = tk.StringVar(root)
        self.splits = tk.StringVar(root, "1")
        self.convert_duration = tk.StringVar(root, "10")  # explicit; never parsed from the RAW name
        self.positions = tk.StringVar(root, "5")          # calibration positions per slab (FR-21)
        self.hit_limit = tk.StringVar(root, "16")
        self.qc_source = tk.StringVar(root, SourceMode.WITH.value)
        self.qc_plots = tk.BooleanVar(root, False)
        self.qc_slabs = tk.BooleanVar(root, False)
        self.lm_debug = tk.BooleanVar(root, True)          # the reference LM call always passes -d
        self.limit_mode = tk.StringVar(root, "target")     # FR-21: calibration event limit, per run
        self.target_per_key = tk.StringVar(root)           # FR-21: T, saved in the profile limits
        self.workers = tk.StringVar(root)                  # FR-15: worker processes, saved in the profile limits
        self._limit_plans = {}
        self.safety_vars = {name: tk.StringVar(root) for name in SAFETY_FIELDS}
        self.lm_vars = {name: tk.StringVar(root) for name in LM_FIELDS}
        self._build()
        for variable in (*self.vars.values(), *self.safety_vars.values(), *self.lm_vars.values(), self.acq_time, self.acq_name,
                         self.hw_trigger, self.raw_input, self.splits, self.convert_duration, self.hit_limit,
                         self.positions, self.qc_source, self.limit_mode, self.target_per_key, self.workers,
                         self.qc_plots, self.qc_slabs, self.lm_debug):
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

    @staticmethod
    def _small():
        return {"font": ctk.CTkFont(size=11), "justify": "left", "anchor": "w", "wraplength": 780}

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
        self._fields(settings, ("petsys_folder", "petsys_python", "ini_file", "data_dir"))
        for row, (name, label) in enumerate(DAQ_FIELDS.items(), 5):
            self.entries[name] = [self._row(settings, row, label, self.vars[name])]
        self._row(settings, 8, "Acq. Time (s):", self.acq_time, width=100)
        self._row(settings, 9, "Acquisition Name:", self.acq_name, width=240)
        ctk.CTkLabel(settings, text="Names the RAW file and the run folder of Acquire, the pipeline and live QC "
                                    "(letters, digits, '_' or '-'; up to 48)", **self._small()).grid(
            row=10, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 5))
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
                     command=self.run_pipeline, **big).grid(row=2, column=0, columnspan=2, padx=20, pady=10,
                                                            sticky="ew")
        self._button(pipeline, "stop", "STOP", fg_color=RED, hover_color=RED_HOVER, command=self.stop, **big).grid(
            row=3, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        self.pipeline_plan = ctk.CTkLabel(pipeline, text="", font=ctk.CTkFont(size=11), justify="left", anchor="w",
                                          wraplength=360)
        self.pipeline_plan.grid(row=4, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="w")
        self.pipeline_status = ctk.CTkLabel(pipeline, text="", font=ctk.CTkFont(size=11), justify="left",
                                            anchor="w", wraplength=360)
        self.pipeline_status.grid(row=5, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="w")
        self._readiness(tab, ("daqd", "initialize", "acquire", "pipeline"))

    def _conversion_tab(self, tab):
        small = {"font": ctk.CTkFont(size=11), "justify": "left", "anchor": "w", "wraplength": 780}
        selection = self._frame(tab, "Data File Selection")
        self.entries["raw_input"] = [self._row(selection, 1, "RAW Data File (.rawf):", self.raw_input,
                                               self._browse_raw)]
        self.raw_plan = ctk.CTkLabel(selection, text="", **small)
        self.raw_plan.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 5))
        settings = self._frame(tab, "Processing Settings")
        self._row(settings, 1, "Number of Split Files:", self.splits, width=100)
        self._row(settings, 2, "RAW Acquisition Duration (s):", self.convert_duration, width=100)
        self._row(settings, 3, "Max Hits per Side:", self.hit_limit, width=100)
        self.split_plan = ctk.CTkLabel(settings, text="", **small)
        self.split_plan.grid(row=4, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 5))
        options = self._frame(tab, "Conversion Options")
        ctk.CTkLabel(options, text="Output: compact coincidence (--writeBinaryCompact), for calibration, LM and QC",
                     **small).grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=2)
        self._button(options, "convert_coincidence", "Convert Raw to Coincidence",
                     command=self.convert).grid(row=3, column=0, padx=20, pady=10)
        self._button(options, "convert_stop", "STOP", width=100, fg_color=RED, hover_color=RED_HOVER,
                     command=self.stop).grid(row=3, column=2, padx=20, pady=10)
        status = self._frame(tab, "Conversion Result")
        self.convert_status = ctk.CTkLabel(status, text="No conversion run in this session", **small)
        self.convert_status.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=2)
        self._button(status, "use_outputs", "Use these outputs as processing inputs",
                     command=self.use_conversion_outputs).grid(row=2, column=0, sticky="w", padx=10, pady=(2, 8))
        self._readiness(tab, ("convert_coincidence",))

    def _ldat_tab(self, tab):
        self._selection(tab, "calibrate")
        frame = self._frame(tab, "Energy cal file generation")
        self._fields(frame, ("cog_limits_file", "calibration_dir", "report_dir"))
        self._row(frame, 4, "Positions per Slab:", self.positions, width=100)
        ctk.CTkLabel(frame, text="1 = one factor per slab, ID(t_ch, slab), no COG limits needed; 2 or more = "
                                 "position regions along the slab, ID(time_ch, slab, region), from the COG limits",
                     font=ctk.CTkFont(size=11), justify="left", anchor="w", wraplength=780).grid(
            row=5, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 5))
        modes = ctk.CTkFrame(frame, fg_color="transparent")
        modes.grid(row=6, column=0, columnspan=3, sticky="w", padx=10, pady=2)
        ctk.CTkLabel(modes, text="Event limit:").pack(side="left", padx=(0, 6))
        for value, text in (("target", "Target sides per histogram (default)"),
                            ("reference", "Reference: 10,000,000 per file (cornell_slab_en_cal.py)")):
            ctk.CTkRadioButton(modes, text=text, variable=self.limit_mode, value=value).pack(side="left", padx=4)
        self._row(frame, 7, "Target sides per histogram (T):", self.target_per_key, width=100)
        self._row(frame, 8, "Workers (0 = automatic):", self.workers, width=100)
        self.limit_plan = ctk.CTkLabel(frame, text="", **self._small())
        self.limit_plan.grid(row=9, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 5))
        self._button(frame, "calibrate", "Create Energy cal file", command=self.calibrate).grid(
            row=10, column=0, columnspan=3, padx=20, pady=10)
        status = self._frame(tab, "Energy Calibration Result")
        self.cal_status = ctk.CTkLabel(status, text="No calibration run in this session", **self._small())
        self.cal_status.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=2)
        self._button(status, "use_calibration", "Use this .encal as the LM System Energy cal file",
                     command=self.use_calibration).grid(row=2, column=0, sticky="w", padx=10, pady=(2, 8))
        self._readiness(tab, ("calibrate",))

    def _lm_tab(self, tab):
        self._selection(tab, "listmode")
        frame = self._frame(tab, "LM File Generation Settings")
        self._fields(frame, ("calibration_file", "cog_limits_file", "doi_limits_file", "pair_map_file",
                             "region_map_file", "lm_dir"))
        ctk.CTkLabel(frame, text="Regions come from the calibration file: ID(t_ch, slab) = one factor per slab, a "
                                 "position file = its region count.", **self._small()).grid(
            row=7, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 2))
        ctk.CTkCheckBox(frame, text="Write LM debug plots (reference -d)", variable=self.lm_debug).grid(
            row=8, column=0, columnspan=3, sticky="w", padx=20, pady=2)
        self._button(frame, "listmode", "Generate LM File", command=self.generate_lm).grid(
            row=9, column=0, columnspan=3, padx=20, pady=10)
        metadata = self._frame(tab, "LM Header Metadata (saved in the profile; empty = unavailable)")
        metadata.grid_columnconfigure(3, weight=1)
        for index, (name, (label, _)) in enumerate(LM_FIELDS.items()):
            row, column = 1 + index // 2, 2 * (index % 2)
            ctk.CTkLabel(metadata, text=label).grid(row=row, column=column, sticky="e", padx=5, pady=2)
            entry = ctk.CTkEntry(metadata, textvariable=self.lm_vars[name], width=120)
            entry.grid(row=row, column=column + 1, sticky="w", padx=5, pady=2)
            self.entries[f"lm_metadata.{name}"] = [entry]
        ctk.CTkLabel(metadata, text="Written as given into every LM header; never measured or inferred by the "
                                    "manager. The complete pipeline writes its Acq. Time as both times instead.",
                     **self._small()).grid(
            row=7, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 5))
        status = self._frame(tab, "LM Result")
        self.lm_status = ctk.CTkLabel(status, text="No LM generation run in this session", **self._small())
        self.lm_status.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 8))
        self._readiness(tab, ("listmode",))

    def _selection(self, tab, key):
        title, default = SELECTIONS[key]
        self.selections[key] = InputSelection(self, tab, key, title, default)

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
                     hover_color=GREEN_HOVER, command=self.run_qc).grid(row=1, column=0, pady=5)
        self._button(control, "qc_stop", "STOP", width=200, height=40, fg_color=RED, hover_color=RED_HOVER,
                     command=self.stop).grid(row=2, column=0, pady=5)
        self.qc_plan = ctk.CTkLabel(control, text="", **self._small())
        self.qc_plan.grid(row=3, column=0, sticky="w", padx=10, pady=(0, 5))
        status = self._frame(tab, "Status")
        self.qc_status = ctk.CTkLabel(status, text="No quality control run in this session", **self._small())
        self.qc_status.grid(row=1, column=0, sticky="w", padx=10, pady=(2, 8))
        self._selection(tab, "qc_analyze")
        offline = self._frame(tab, "Offline QC of the listed files")
        self._button(offline, "qc_analyze", "Analyze existing compact LDAT", command=self.analyze_qc).grid(
            row=1, column=0, padx=20, pady=10, sticky="w")
        ctk.CTkLabel(offline, text="Uses the plot/slab options above; results go to a new run folder "
                                   "<data>_qc_<date>_<time> in the Report Destination. Source mode and duration are "
                                   "not recorded for existing files. Files are read in parallel with the Workers "
                                   "setting (LDAT Processing tab).",
                     **self._small()).grid(row=2, column=0, sticky="w", padx=10, pady=(0, 5))
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
                    found = event.payload.issues
                    self._limit_plans = dict(event.payload.limit_plans)
                    self._show_readiness({key: tuple(found.get(key, ())) + tuple(self._option_issues.get(key, ()))
                                          for key in {*found, *self._option_issues}})
                    self._show_limit_plans()
            elif event.kind == "inputs_probed":
                selection = self.selections.get(event.payload.key)
                if selection is not None and selection.show_probe(event.payload):
                    lines.append(f"{CHECKS[event.payload.key][1]} inputs checked: "
                                 + selection.feedback.cget("text").splitlines()[0])
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
        """Append lines in one widget update, each starting with the local time (FR-1); bounded tail."""
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", "".join(f"{stamp} {message}\n" for message in messages))
        limit = self.session.profile.limits.log_tail_lines
        lines = int(self.log_text.index("end-1c").split(".")[0]) - 1
        if lines > limit:
            self.log_text.delete("1.0", f"{lines - limit + 1}.0")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _show_limit_plans(self):
        """Event limit of the next calibration (FR-21), from the readiness worker; shown, never computed here."""
        plan = self._limit_plans.get("calibrate")
        self.limit_plan.configure(text=limit_plan_text(plan) if plan else
                                  "Event limit: shown when the calibration prerequisites are met")
        pipeline = self._limit_plans.get("pipeline")
        text = self.pipeline_plan.cget("text").split("\nCalibration ")[0]
        if pipeline:
            text += "\nCalibration " + limit_plan_text(pipeline)[0].lower() + limit_plan_text(pipeline)[1:]
        self.pipeline_plan.configure(text=text)

    def _show_readiness(self, issues):
        self._issues = {key: tuple(issues.get(key, ())) for key in self.readiness}
        for key in self.readiness:
            self._ready[key] = not self._issues[key]
        self._refresh_controls()

    def _live_hint(self, key):
        """Why a control whose settings are complete is still disabled by the live system state."""
        status = self.daqd_status
        ready = status is not None and status.state == DaqdState.READY
        if key == "initialize" and not ready:
            return " (start DAQD to enable)"
        if key in LIVE_KEYS:
            if self.bias_unknown:
                return " (confirm the SiPM bias state to enable)"
            if not (ready and status.initialized):
                return " (start DAQD and initialize the system to enable)"
        return ""

    def _render_readiness(self):
        for key, label in self.readiness.items():
            name, found = CHECKS[key][1], self._issues.get(key)
            if found is None:
                continue  # not checked yet
            if found:
                text = f"{name} unavailable:\n" + "\n".join(f"  • {line}" for line in issue_lines(found))
                colour = "warn"
            else:
                text, colour = f"{name}: prerequisites met{self._live_hint(key)}", "ok"
            if label.cget("text") != text:
                label.configure(text=text, text_color=STATUS_COLOURS[colour])

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
        stage = event.identity.stage_id
        step = ""
        if stage in self._stages and len(self._stages) > 1:
            step = f"Step {self._stages.index(stage) + 1}/{len(self._stages)} ({STAGE_TEXT.get(stage, stage)}): "
        if event.kind == "workflow_started":
            self._stages = tuple(payload.get("stages", ()))
            text = f"Run {self._run_id} started: {' -> '.join(STAGE_TEXT.get(s, s) for s in self._stages)}" \
                   f"\nRun directory: {payload.get('run_root')}"
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
        elif event.kind == "stage_started" and stage == "conversion":
            self._last_raw = payload.get("raw")
            text = f"{step}{event.message}\nRAW input: {payload.get('raw')}\nWriting to: {payload.get('directory')}"
        elif event.kind == "stage_started":
            text = f"{step}{event.message}\nWriting to: {payload.get('directory')}"
        elif event.kind == "stage_progress":
            index, files, phase = payload.get("file_index"), payload.get("files"), payload.get("phase")
            text = f"{step}{STAGE_TEXT.get(stage, stage)}"
            if phase in ("read", "pass 2"):
                text += f" ({phase})"
            if phase == "fits" and isinstance(payload.get("keys_total"), int):
                text += f": fits {payload.get('keys_done', 0):,}/{payload['keys_total']:,} keys"
            elif isinstance(index, int) and isinstance(files, int):
                text += f": file {index + 1}/{files} {Path(str(payload.get('path'))).name}"
            if isinstance(payload.get("records_read"), int):
                text += f", {payload['records_read']:,} records read"
            if isinstance(payload.get("records_written"), int):
                text += f", {payload['records_written']:,} LM records written"
        elif event.kind == "stage_finished":
            text = f"{step}{event.message}"
        elif event.kind == "workflow_finished":
            text = f"{payload['status']}: {event.message}"
        if text is not None:
            self._status_label.configure(text=text, text_color=STATUS_COLOURS[tone])

    def _workflow_done(self, result, lines):
        if result.token != self._token:
            lines.append(f"Ignored a stale workflow result (request {result.token})")
            return
        action, label, run_id = self._action, self._status_label, self._run_id
        self._token, self._run_id, self._stop_requested, self._stages = None, None, False, ()
        outcome = result.outcome
        if outcome is None:
            text, tone = f"{ACTION_TEXT.get(action, action.value)} {result.message}", "warn"
        else:
            text = f"{ACTION_TEXT.get(action, action.value)} {outcome.status.value}: {outcome.message}"
            if outcome.run_root is not None:
                text += f"\nRun directory: {outcome.run_root}"
            tone = {ResultStatus.SUCCEEDED: "ok", ResultStatus.CANCELLED: "warn"}.get(outcome.status, "error")
            if action == Action.CONVERT:
                text += self._conversion_report(outcome, run_id)
            elif action != Action.ACQUIRE:
                text += self._stage_report(outcome, run_id)
        label.configure(text=text, text_color=STATUS_COLOURS[tone])
        lines.append(text)

    def _stage_report(self, outcome, run_id):
        """Each recorded stage with its status and directory; successful ones list their exact validated outputs."""
        text = ""
        for stage in outcome.stages:
            name = STAGE_TEXT.get(stage.stage_id, stage.stage_id)
            text += f"\n{name}: {stage.status.value}"
            if stage.directory is not None:
                text += f" in {stage.directory}"
            if isinstance(stage.details.get("elapsed_s"), (int, float)):
                text += f"\n  Elapsed: {format_elapsed(stage.details['elapsed_s'])}"
            if stage.status != ResultStatus.SUCCEEDED:
                text += f"\n  {stage.message}"
                if stage.directory is not None:
                    text += "\n  Partial outputs, if any, are kept there and recorded as unvalidated"
                continue
            summary = stage.details.get("summary") or {}
            if stage.stage_id == "acquisition":
                text += f" ({stage.details.get('attempts')} attempt(s))"
            elif stage.stage_id == "conversion":
                text += f" ({len(stage.artifacts)} structure-checked {stage.details.get('format')} "\
                        f"{stage.details.get('population')} file(s))"
            elif stage.stage_id == "calibration" and summary:
                layout = "one factor per slab" if summary.get("layout") == "per_slab" else \
                    f"{summary.get('positions')} position regions per slab"
                counts = ", ".join(f"{key} {value:,}" for key, value in (summary.get("status_counts") or {}).items())
                text += f"\n  {layout}; {summary.get('factors', 0):,} of {summary.get('keys', 0):,} mapped keys " \
                        f"have a factor ({counts})"
                text += limit_used_text(summary)
            elif stage.stage_id == "listmode" and summary:
                text += f"\n  {summary.get('records_written', 0):,} LM records written"
                if summary.get("calibration_non_positive_mu_as_no_factor"):
                    text += f"\n  Warning: {summary['calibration_non_positive_mu_as_no_factor']:,} calibration " \
                            "row(s) with mu <= 0 (failed fits) were read as no factor; their pairs are rejected " \
                            "as missing calibration. The keys are in the LM provenance."
            elif stage.stage_id == "qc":
                findings = stage.details.get("findings") or {}
                summary_path = next((a.path for a in stage.artifacts if a.kind == "qc_summary"), None)
                if summary_path is not None:
                    text += f"\n  Results directory: {Path(summary_path).parent}"
                text += "\n  Processing completed. Findings are observations of the coincidence sample, not a " \
                        "detector verdict:"
                text += "".join(f"\n    {key.replace('_', ' ')}: {value}" for key, value in findings.items())
            widths = {k: v for k, v in (summary.get("zero_width_limit_keys") or {}).items() if v}
            if widths:
                text += "\n  Warning: zero-width limits (left = right) for " + ", ".join(
                    f"{v:,} {k.upper()} key(s)" for k, v in widths.items()) + \
                    "; their sides fall out of range, as in the reference. The keys are in the provenance."
            if stage.stage_id != "conversion":
                for artifact in stage.artifacts:
                    if artifact.kind not in HIDDEN_ARTIFACTS:
                        text += f"\n  {ARTIFACT_TEXT.get(artifact.kind, artifact.kind)}: {artifact.path}"
            else:
                text += "".join(f"\n  {index}. {artifact.path}" for index, artifact in enumerate(stage.artifacts, 1))
            if stage.stage_id == "calibration":
                encal = next((a.path for a in stage.artifacts if a.kind == "encal"), None)
                if encal is not None:
                    self.last_calibration = (run_id, Path(encal))
        return text

    def _conversion_report(self, outcome, run_id):
        """The exact RAW input and the exact recorded outputs, in order; nothing is guessed from names."""
        stage = next((item for item in outcome.stages if item.stage_id == "conversion"), None)
        if stage is None:
            return ""
        text = f"\nRAW input: {self._last_raw}" if self._last_raw else ""
        counts = {item["path"]: item for item in stage.details.get("ldat", ())}
        outputs = outcome.outputs("conversion")
        if outputs:
            descriptor = outputs[0].input_descriptor
            text += (f"\nOutputs ({len(outputs)}, {descriptor.format.value} {descriptor.population.value}, "
                     "structure-checked against the selected map, in this order; each processing stage "
                     "validates the records it reads):")
            for index, artifact in enumerate(outputs, 1):
                item = counts.get(str(artifact.path)) or {}
                records, checked = item.get("records"), item.get("records_checked")
                note = (f"{records:,} records" if records is not None else
                        f"first {checked:,} records checked" if checked is not None else "")
                text += f"\n  {index}. {artifact.path}" + (f"   ({note})" if note else "")
            self.last_conversion = (run_id, tuple(artifact.input_descriptor for artifact in outputs))
        removed = stage.details.get("removed", ())
        if removed:     # T34: after a successful conversion
            splits = [Path(path).name for path in removed if path.endswith(".ldat")]
            text += (f"\nRemoved after conversion (not outputs): "
                     f"{sum(path.endswith('.lidx') for path in removed)} .lidx index file(s)"
                     + (f"; empty split(s): {', '.join(splits)}" if splits else ""))
        empty = [path for path in stage.details.get("empty_ldat", ()) if path not in removed]
        if empty:
            text += "\nEmpty split files (kept, not outputs): " + ", ".join(Path(path).name for path in empty)
        return text

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
        system = ready and status.initialized and not initializing and not self.bias_unknown
        enable = {
            "daqd": not busy and not self._daqd_pending and not initializing and (
                live or (state in (DaqdState.OFF, DaqdState.FAILED) and self._ready.get("daqd", False))),
            "initialize": not busy and ready and not initializing and self._ready.get("initialize", False),
            "use_outputs": not busy and self.last_conversion is not None,
            "use_calibration": not busy and self.last_calibration is not None,
        }
        for key in LIVE_KEYS:
            enable[key] = not busy and system and self._ready.get(key, False)
        for key in ("convert_coincidence", *PROCESSING_KEYS):
            enable[key] = not busy and self._ready.get(key, False)
        for key in STOP_KEYS:
            enable[key] = self._token is not None and not self._stop_requested and not self._shutting_down
        for key, button in self.buttons.items():  # a redraw per poll and button would starve Tk: changes only
            wanted = "normal" if enable.get(key, False) else "disabled"
            if button.cget("state") != wanted:
                button.configure(state=wanted)
        if self.daqd_state.get() != live:
            self.daqd_state.set(live)
        if self.buttons["daqd"].cget("text") != DAQD_TEXT[state]:
            self.buttons["daqd"].configure(text=DAQD_TEXT[state])
        if status is None:
            text = "DAQD not started by this manager"
        else:
            text = f"{DAQD_TEXT[state]}" + (f" (pid {status.pid})" if status.pid else "")
            if ready:
                text += "; initialized" if status.initialized else "; not initialized"
            if status.message:
                text += f"\n{status.message}"
        colour = STATUS_COLOURS[{DaqdState.READY: "ok", DaqdState.FAILED: "error"}.get(state, "info")]
        if (self.daqd_label.cget("text"), self.daqd_label.cget("text_color")) != (text, colour):
            self.daqd_label.configure(text=text, text_color=colour)
        self._render_readiness()

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
            duration = float(self.acq_time.get().strip())
        except ValueError as exc:
            self.log(f"Acquisition not started: Acq. Time (s): {exc}")
            duration = None
        try:
            options = None if duration is None else RunOptions(
                duration_s=duration, hardware_trigger=self.hw_trigger.get(),
                acquisition_name=self.acq_name.get().strip())
        except ProfileError as exc:
            self.log(f"Acquisition not started: {exc}")
            options = None
        if profile is not None and options is not None:
            self._start(profile, Action.ACQUIRE, options, self.acq_status, "Starting acquisition...")
        self._refresh_controls()

    def convert(self):
        """Manual compact coincidence conversion of the selected RAW with the explicit duration/split/hit settings."""
        key = "convert_coincidence"
        profile = self._profile_or_log()
        requests, issues = self.requests()
        if key in issues:
            self.log(f"{CHECKS[key][1]} not started: " + "; ".join(issue.message for issue in issues[key]))
        elif profile is not None:
            self._start(profile, Action.CONVERT, requests[key][1], self.convert_status,
                        "Starting compact coincidence conversion...")
        self._refresh_controls()

    def _start(self, profile, action, options, label, text, inputs=()):
        self._token = self.session.start_workflow(profile, action, options, inputs)
        self._action, self._status_label, self._last_raw = action, label, None
        self._run_id, self._stop_requested, self._stages = None, False, ()
        if action == Action.CONVERT:
            self.last_conversion = None  # "Use these outputs" only ever means the run shown
        if action in (Action.CALIBRATE, Action.PIPELINE):
            self.last_calibration = None
        label.configure(text=text, text_color=STATUS_COLOURS["info"])

    def _run(self, key, label, text):
        """Start the check's exact request (options and ordered inputs) as shown by its prerequisites."""
        profile = self._profile_or_log()
        requests, issues = self.requests()
        if key in issues:
            self.log(f"{CHECKS[key][1]} not started: " + "; ".join(issue.message for issue in issues[key]))
        elif not self._ready.get(key, False):
            self.log(f"{CHECKS[key][1]} not started: prerequisites not met (see its Prerequisites)")
        elif profile is not None and key in requests:
            action, options, inputs = requests[key]
            self._start(profile, action, options, label, text, inputs)
        self._refresh_controls()

    def calibrate(self):
        self._run("calibrate", self.cal_status, "Starting energy calibration of the listed files...")

    def generate_lm(self):
        self._run("listmode", self.lm_status, "Starting LM generation of the listed files...")

    def analyze_qc(self):
        self._run("qc_analyze", self.qc_status, "Starting offline QC of the listed files...")

    def run_qc(self):
        mode = SourceMode(self.qc_source.get())
        self._run("qc", self.qc_status, f"Starting quality control {mode.value} source "
                                        f"({QC_PRESET_S[mode]:g} s acquisition)...")

    def run_pipeline(self):
        self._run("pipeline", self.pipeline_status, "Starting the complete pipeline...")

    def stop(self):
        if self._token is not None and self.session.stop_workflow():
            self._stop_requested = True
            if self._action == Action.ACQUIRE:
                text = "STOP: terminating the acquisition and switching bias off; no further attempt"
            elif self._action in (Action.QC, Action.PIPELINE):
                text = ("STOP: terminating the running stage (an acquisition switches bias off); "
                        "no further attempt or later stage")
            else:
                text = f"STOP: terminating the {ACTION_TEXT[self._action].lower()}; no later stage"
            self._status_label.configure(text=text, text_color=STATUS_COLOURS["warn"])
            self.log("STOP requested")
        self._refresh_controls()

    def use_calibration(self):
        """Operator choice: put the last recorded .encal into the LM field (an unsaved profile edit)."""
        if self.last_calibration is None:
            return
        run_id, path = self.last_calibration
        self.vars["calibration_file"].set(str(path))
        self.log(f"System Energy cal file set to {path} from run {run_id} (unsaved profile edit)")

    def use_conversion_outputs(self):
        """Hand the last conversion's exact validated outputs to the processing lists that accept them."""
        if self.last_conversion is None:
            return
        run_id, descriptors = self.last_conversion
        targets = OUTPUT_TARGETS[(descriptors[0].format, descriptors[0].population)]
        for key in targets:
            replaced = len(self.selections[key].paths)
            self.selections[key].use_outputs(descriptors, run_id)
            self.log(f"{CHECKS[key][1]} inputs: {len(descriptors)} output(s) of conversion run {run_id}"
                     + (f" (replaced {replaced} listed file(s))" if replaced else ""))

    def _update_conversion_plan(self):
        """Exact converter input/output naming and split time for the current fields (display only)."""
        raw = self.raw_input.get().strip()
        if raw:
            path = Path(raw)
            prefix = path.with_suffix("") if path.suffix == ".rawf" else path
            text = f"Converter input (-i): {prefix}  (reads {prefix.name}.rawf and its .idxf index)"
            if path.suffix != ".rawf":
                text += "\nWARNING: select the acquisition's .rawf file"
            text += (f"\nOutputs: a new run folder {portable_name(prefix.name)}_conv_<date>_<time> in the Output "
                     "Data Folder, "
                     f"with {prefix.name}_coincCompact[_<n>].ldat")
        else:
            text = "Select the RAW acquisition (.rawf); any file name is accepted"
        self.raw_plan.configure(text=text)
        try:
            splits, duration = int(self.splits.get().strip()), float(self.convert_duration.get().strip())
            if splits > 1 and duration > 0:
                text = (f"--splitTime {duration / splits + SPLIT_TIME_OFFSET_S:g} s: duration {duration:g} s / "
                        f"{splits} splits + {SPLIT_TIME_OFFSET_S:g} s (duration as entered, not from the file name)")
            elif splits == 1:
                text = "One output file (no --splitTime); split numbering, if any, comes from the converter"
            else:
                text = "Splits and duration must be positive"
        except ValueError:
            text = "Splits must be an integer and duration a number"
        self.split_plan.configure(text=text)
        conversion = (f"{self.splits.get().strip() or '?'} split(s), max {self.hit_limit.get().strip() or '?'} "
                      "hits per side (RAWF to LDAT tab)")
        acq_time = self.acq_time.get().strip() or '?'
        name = self.acq_name.get().strip() or '?'
        self.pipeline_plan.configure(
            text=f"This run: acquire {acq_time} s -> compact coincidence conversion, "
                 f"{conversion} -> calibration, {self.positions.get().strip() or '?'} position(s) per slab (LDAT "
                 f"Processing tab) -> LM with the LM tab files and metadata, header acquisition/measurement time "
                 f"{acq_time} s (Acq. Time). One new run folder {name}_pipeline-P{self.positions.get().strip() or '?'}"
                 "_<date>_<time> in the Output Data Folder, a numbered folder per stage.")
        self._show_limit_plans()
        mode = SourceMode(self.qc_source.get())
        self.qc_plan.configure(
            text=f"This run: acquire {QC_PRESET_S[mode]:g} s {mode.value} source -> compact coincidence conversion, "
                 f"{conversion} -> QC; one new run folder {name}_qc-{mode.value}-source_<date>_<time> in the "
                 "Output Data Folder.")

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
            for name in LM_FIELDS:
                value = getattr(profile.lm_metadata, name)
                self.lm_vars[name].set("" if value is None else format(value, ".15g") if type(value) is float
                                       else str(value))
            self.target_per_key.set(str(profile.limits.calibration_target_per_key))
            self.workers.set(str(profile.limits.workers))
        finally:
            self._loading = False
        self._update_profile_status()
        self._update_conversion_plan()
        self._run_check()

    def profile_from_ui(self):
        """Edited profile; fields this window does not show keep the loaded values."""
        values = {name: self.vars[name].get().strip() or None for name in PROFILE_FIELDS}
        values["daq_type"] = self.vars["daq_type"].get().strip()
        values["socket_path"] = self.vars["socket_path"].get().strip()
        values["cards"] = tuple(card.strip() for card in self.vars["cards"].get().split(",") if card.strip())
        values["safety"] = self.safety_from_ui()
        values["lm_metadata"] = self.lm_metadata_from_ui()
        try:
            target = int(self.target_per_key.get().strip())
        except ValueError:
            raise ProfileError("Target sides per histogram (T) must be a positive integer") from None
        try:
            workers = int(self.workers.get().strip())
        except ValueError:
            raise ProfileError("Workers must be a whole number (0 = automatic)") from None
        values["limits"] = replace(self.session.profile.limits, calibration_target_per_key=target, workers=workers)
        return replace(self.session.profile, **values)

    def lm_metadata_from_ui(self):
        """Typed LM header metadata; an empty field stays unavailable (None), never a default."""
        values = {}
        for name, (label, kind) in LM_FIELDS.items():
            text = self.lm_vars[name].get().strip()
            try:
                values[name] = None if not text else kind(text)
            except (ValueError, OverflowError):
                raise ProfileError(f"LM {label.rstrip(':')} must be {'an integer' if kind is int else 'a number'}") \
                    from None
        return LMMetadata(**values)  # range checks raise ProfileError

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
        """Prerequisite requests per check (action, options, exact inputs) plus reasons found while parsing."""
        requests, issues = {}, {}

        def reason(key, field, message):
            issues[key] = issues.get(key, ()) + (PrerequisiteIssue(field, message),)

        def add(key, inputs=(), **options):
            try:
                requests[key] = (CHECKS[key][0], RunOptions(**options), inputs)
            except ProfileError as exc:
                reason(key, "options", str(exc))

        def number(variable, kind, label, keys):
            try:
                return kind(variable.get().strip())
            except ValueError:
                for key in keys:
                    reason(key, "options", f"{label} must be {'a positive integer' if kind is int else 'a positive number'}")

        duration = number(self.acq_time, float, "Acq. Time (s)", ("acquire", "pipeline"))
        name = self.acq_name.get().strip()     # checked by RunOptions (T29)
        conversions = ("convert_coincidence",)
        splits = number(self.splits, int, "Number of Split Files", (*conversions, "pipeline", "qc"))
        raw_duration = number(self.convert_duration, float, "RAW Acquisition Duration (s)", conversions)
        hits = number(self.hit_limit, int, "Max Hits per Side", (*conversions, "pipeline", "qc"))
        for key in ("daqd", "initialize"):
            add(key)
        positions = number(self.positions, int, "Positions per Slab", ("calibrate", "pipeline"))
        plots = self.qc_plots.get()
        qc = dict(plots=plots, slabs=plots and self.qc_slabs.get())    # slab analysis requires plots
        for key, selection in self.selections.items():
            if key == "calibrate":
                if positions is not None:
                    add(key, selection.descriptors(), regions=positions,
                        calibration_limit_mode=self.limit_mode.get())
            elif key == "listmode":   # compact LM decodes at the conversion hit limit (FR-22)
                if hits is None:
                    reason(key, "options", "Max Hits per Side (RAWF to LDAT tab) must be a positive integer: "
                                           "compact LM decodes at the conversion hit limit")
                else:
                    add(key, selection.descriptors(), debug=self.lm_debug.get(), hit_limit=hits)
            else:
                add(key, selection.descriptors(), **qc)
            problem = selection.confirmation_issue()
            if problem:
                reason(key, "inputs", problem)
        if duration is not None:
            add("acquire", duration_s=duration, hardware_trigger=self.hw_trigger.get(), acquisition_name=name)
            if None not in (splits, hits, positions):   # compact coincidence conversion (FR-10)
                add("pipeline", duration_s=duration, hardware_trigger=self.hw_trigger.get(), splits=splits,
                    acquisition_name=name,
                    hit_limit=hits, regions=positions, debug=self.lm_debug.get(),
                    calibration_limit_mode=self.limit_mode.get())
        if None not in (splits, raw_duration, hits):
            common = dict(splits=splits, duration_s=raw_duration, hit_limit=hits,
                          raw_input=self.raw_input.get().strip() or None)
            add("convert_coincidence", **common)
        if None not in (splits, hits):   # duration and compact format: the preflight's source preset
            add("qc", source_mode=SourceMode(self.qc_source.get()), splits=splits, hit_limit=hits,
                acquisition_name=name, **qc)
        return requests, issues

    def _edited(self, *_):
        if self._loading or self._closing:
            return
        self._update_profile_status()
        self._update_conversion_plan()
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
            self._limit_plans = {}  # never show a plan computed from older values
            self._show_limit_plans()
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

    def _browse_raw(self):
        current = self.raw_input.get().strip() or self.vars["data_dir"].get().strip()
        chosen = filedialog.askopenfilename(initialdir=str(Path(current).parent if current.endswith(".rawf")
                                                           else current or Path.home()),
                                            filetypes=[("PETsys RAW", "*.rawf"), ("All files", "*")])
        if chosen:
            self.raw_input.set(chosen)

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
