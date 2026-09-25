"""RAWInspector-inspired, PETsys-specific offline LDAT workbench."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import multiprocessing
from pathlib import Path
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize, to_rgba
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import numpy as np

from src.ldat_fastread import init_worker
from src.ldat_inspector import (
    FINDING_COLOURS, FINDINGS_POPULATION, OVERVIEW_METRICS, TILE_UNAVAILABLE, TILE_UNPOPULATED, TILE_VALUE,
    FileResult, FindingThresholds, Selection, Settings, apply_calibration, channel_findings, channel_geometry,
    fit_peak, fit_on_display_bins, fit_peak_background, flood_counts, load_setup, merge_results,
    minimodule_layout, minimodule_metrics, overview_grid, pair_dt, pair_mask, pair_matrix, process_file,
    rate_series, pair_offset_series, supermodule_layout, system_channel_findings, uniformity,
)
from src.ldat_memory import estimate_memory


__version__ = "0.1.0"
# Strong colours for plotted channel states; table rows keep the pale RAWInspector colours.
STATE_EDGES = {"OK": "#4c9a5b", "NOT OBSERVED": "#d62728", "HIGH": "#c2185b", "LOW": "#ef8a00",
               "INSUFFICIENT EVENTS": "#8c939a"}
FLOOD_MODE = "Flood maps"
TILE_COLOURS = {TILE_UNPOPULATED: "#d5d8dc", TILE_UNAVAILABLE: "#7d848b"}
SM_TAB = "SuperModule"
COINC_TAB = "Coincidences"
TABS = ("Channel Status", SM_TAB, "System Overview", COINC_TAB, "Timestamps")
ALL_PARTNERS = "All partners"
DT_LABEL = ("Observational: geometry and time of flight contribute to the paired time difference; "
            "it is not a clock offset or a CTR calibration.")
STEP_DEBOUNCE_MS = 150  # wheel and Page Up/Down redraw once the stepping pauses
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def _entry(parent, variable, width=74):
    box = ctk.CTkEntry(parent, width=width, textvariable=variable)
    box.pack(side="left", padx=(3, 9))
    return box


def _label(parent, text):
    ctk.CTkLabel(parent, text=text).pack(side="left", padx=(6, 0))


class _Tooltip:
    """Hover text for a widget (CustomTkinter has no tooltip)."""

    def __init__(self, widget, text):
        self.widget, self.text, self.window = widget, text, None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event=None):
        if self.window is not None:
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.geometry(f"+{self.widget.winfo_rootx() + 8}+"
                             f"{self.widget.winfo_rooty() + self.widget.winfo_height() + 4}")
        tk.Label(self.window, text=self.text, background="#ffffe0", foreground="black", relief="solid",
                 borderwidth=1, font=("Arial", 9), justify="left").pack()

    def _hide(self, _event=None):
        if self.window is not None:
            self.window.destroy()
            self.window = None


class LDATWorkbench(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"LDAT Inspector  v{__version__}")
        self.geometry("1600x960")
        self.minsize(1120, 740)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self.files = []
        self.config_path = ""
        self.calibration_path = ""
        self.calibrated = tk.BooleanVar(value=True)
        self.dataset = None
        self._busy = False
        self._report_busy = False
        self._events = queue.Queue()
        self._abort = threading.Event()
        self._selector = None
        self._colorbar = None
        self._residual_window = None

        # Processing pool, shared cancel flag and per-file progress (spec 002 T4).
        self._pool = None
        self._cancel = None
        self._file_progress = None
        self._workers = 0
        self._reader = None  # checks substitute a slow reader; None uses process_file
        self._estimate_token = 0
        self._estimate_after = None
        self.whole_files = tk.BooleanVar(value=False)

        self.system = tk.StringVar(value="IMAS")
        self.max_pairs = tk.StringVar(value="10000")
        self.min_channels = tk.StringVar(value="1")
        self.min_channel_energy = tk.StringVar(value="0")
        self.module_var = tk.StringVar()
        self.energy_low = tk.StringVar(value="400")
        self.energy_high = tk.StringVar(value="650")
        self.doi_low = tk.StringVar(value="0")
        self.doi_high = tk.StringVar(value="15")
        self.x_low = tk.StringVar(value="0")
        self.x_high = tk.StringVar(value="102")
        self.y_low = tk.StringVar(value="0")
        self.y_high = tk.StringVar(value="102")
        self.bins = tk.StringVar(value="100")
        self.color_min = tk.StringVar(value="0.1")
        self.color_max = tk.StringVar(value="Auto")
        self.experimental = tk.BooleanVar(value=False)
        self._sliders = []
        self.fit_low = tk.StringVar(value="350")
        self.fit_high = tk.StringVar(value="700")
        self.search_low = tk.StringVar(value="425")
        self.search_high = tk.StringVar(value="600")
        self.profile_mode = tk.BooleanVar(value=False)
        self.overview_mode = tk.StringVar(value="Ingest sides")
        # Per-minimodule metrics are computed in a thread and cached per (dataset, selection).
        self._overview_cache = []
        self._overview_token = 0
        self._overview_job = None
        self._overview_grid = None
        self._overview_ax = None
        self._overview_pick = None
        self.time_file = tk.StringVar()
        self.time_view = tk.StringVar(value="Event rate")
        self.time_bins = tk.StringVar(value="60")
        self.target = tk.StringVar(value="511")
        self.tolerance = tk.StringVar(value="10")
        default = FindingThresholds()
        self.thresholds = default
        self.finding_low = tk.StringVar(value=f"{default.low_frac:g}")
        self.finding_high = tk.StringVar(value=f"{default.high_frac:g}")
        self.finding_median = tk.StringVar(value=f"{default.min_median:g}")
        self.finding_events = tk.StringVar(value=str(default.min_events))
        self._findings = {}
        self._channel_sm = None
        # SuperModule stepping and its per-minimodule table (spec 002 T10). One background
        # job at a time; the latest wanted (dataset, selection, SM) runs next.
        self._sm_ids = []
        self._step_after = None
        self._sm_cache = []
        self._sm_job = None
        self._sm_wanted = None
        self._mm_layout = None
        # Tabs whose content is out of date; only the visible tab is redrawn.
        self._stale = set()
        # Coincidences (T11): pair mask + SM x SM matrix from a thread, cached per (dataset, selection).
        self.pair_a = tk.StringVar()
        self.pair_b = tk.StringVar(value=ALL_PARTNERS)
        self.dt_bins = tk.StringVar(value="100")
        self._pair_cache = []
        self._pair_token = 0
        self._pair_job = None
        self._pair_current = None
        self._matrix_ax = self._dt_ax = None

        self._build_top()
        self._build_tabs()
        self._drawers = {SM_TAB: self._refresh_selected, "System Overview": self._draw_overview,
                         COINC_TAB: self._draw_coincidences, "Timestamps": self._draw_timestamps}
        self._log("Select an IMAS or Cornell config and LDAT files; calibration can be applied later.")
        self.bind("<Control-o>", lambda _: self._select_files())
        self.bind("<Control-p>", lambda _: self._start_processing())
        self.bind("<Escape>", lambda _: self._reset_filters())
        self.bind("<Prior>", lambda _: self._key_step(-1))
        self.bind("<Next>", lambda _: self._key_step(1))
        self.after(80, self._poll_events)

    def _card(self, parent, title, *, width=None):
        options = {"width": width} if width is not None else {}
        card = ctk.CTkFrame(parent, corner_radius=7, **options)
        card.pack(side="left", fill="both", padx=5, pady=5)
        ctk.CTkLabel(card, text=title, font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(7, 2))
        return card

    def _build_top(self):
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=8, pady=(5, 0))

        inputs = self._card(top, "Input files and calibration", width=370)
        self.inputs_card = inputs
        self.config_button = ctk.CTkButton(inputs, text="Config...", width=95, command=self._select_config)
        self.config_button.pack(side="top", anchor="w", padx=10, pady=2)
        self.config_text = ctk.CTkLabel(inputs, text="No config", width=340, anchor="w")
        self.config_text.pack(padx=10)
        self.calib_button = ctk.CTkButton(inputs, text="Calibration...", width=115, command=self._select_calibration)
        self.calib_button.pack(anchor="w", padx=10, pady=2)
        self.calib_text = ctk.CTkLabel(inputs, text="No energy calibration", width=340, anchor="w")
        self.calib_text.pack(padx=10)
        self.calib_text.bind("<Button-1>", self._show_calibration_path)
        self.calib_switch = ctk.CTkSwitch(inputs, text="Calib: keV / raw a.u.", variable=self.calibrated,
                                           command=self._toggle_calibration)
        self.calib_switch.pack(anchor="w", padx=10, pady=2)
        file_row = ctk.CTkFrame(inputs, fg_color="transparent")
        file_row.pack(fill="x", padx=8, pady=(5, 0))
        ctk.CTkButton(file_row, text="Add LDAT...", width=106, command=self._select_files).pack(side="left", padx=2)
        ctk.CTkButton(file_row, text="Clear", width=65, command=self._clear_files).pack(side="left", padx=2)
        self.file_count = ctk.CTkLabel(file_row, text="0 files")
        self.file_count.pack(side="left", padx=5)
        self.file_list = ctk.CTkTextbox(inputs, height=51, width=340, font=("Consolas", 10))
        self.file_list.pack(fill="x", padx=10, pady=(3, 6))
        self.file_list.insert("end", "No LDAT files selected")
        inputs.update_idletasks()
        inputs.configure(width=370, height=max(270, inputs.winfo_reqheight()))
        inputs.pack_propagate(False)

        processing = self._card(top, "Processing", width=260)
        row = ctk.CTkFrame(processing, fg_color="transparent")
        row.pack(fill="x", padx=5)
        _label(row, "System")
        ctk.CTkComboBox(row, values=["IMAS", "CORNELL"], variable=self.system,
                        state="readonly", width=120).pack(side="left", padx=8)
        for text, var in (("Coincidence pairs / file", self.max_pairs),
                          ("Min energy channels", self.min_channels),
                          ("Min channel energy (a.u.)", self.min_channel_energy)):
            r = ctk.CTkFrame(processing, fg_color="transparent")
            r.pack(fill="x", padx=5)
            ctk.CTkLabel(r, text=text, width=174, anchor="w").pack(side="left")
            box = _entry(r, var, width=74)
            if var is self.max_pairs:
                self.max_pairs_entry = box
        self.whole_switch = ctk.CTkSwitch(processing, text="Whole files (no pair limit)",
                                          variable=self.whole_files, command=self._whole_files_changed)
        self.whole_switch.pack(anchor="w", padx=10, pady=(2, 0))
        buttons = ctk.CTkFrame(processing, fg_color="transparent")
        buttons.pack(fill="x", padx=10, pady=(5, 2))
        self.process_button = ctk.CTkButton(buttons, text="Process files (Ctrl+P)",
                                            fg_color="#288047", command=self._start_processing)
        self.process_button.pack(side="left", fill="x", expand=True)
        self.cancel_button = ctk.CTkButton(buttons, text="Cancel", width=62, fg_color="#9b3b30",
                                           command=self._cancel_processing, state="disabled")
        self.cancel_button.pack(side="left", padx=(4, 0))
        self.progress = ctk.CTkProgressBar(processing)
        self.progress.pack(fill="x", padx=10, pady=(2, 2))
        self.progress.set(0)
        self.estimate_text = ctk.CTkLabel(processing, text="Estimated memory: add LDAT files",
                                          anchor="w", justify="left", wraplength=240,
                                          font=("Arial", 11))
        self.estimate_text.pack(fill="x", padx=10, pady=(0, 5))
        for var in (self.max_pairs, self.min_channels, self.min_channel_energy):
            var.trace_add("write", lambda *_: self._schedule_estimate())

        actions = self._card(top, "Analysis and reports", width=170)
        self.uniformity_button = ctk.CTkButton(actions, text="Photopeak Uniformity", command=self._uniformity)
        self.uniformity_button.pack(fill="x", padx=8, pady=3)
        self.system_report = ctk.CTkButton(actions, text="Full System Report", command=lambda: self._report(None))
        self.system_report.pack(fill="x", padx=8, pady=3)
        self.module_report = ctk.CTkButton(actions, text="SuperModule Report", command=lambda: self._report(self._selected_sm()))
        self.module_report.pack(fill="x", padx=8, pady=3)
        self.open_report = ctk.CTkButton(actions, text="Open Last Report", command=self._open_report)
        self.open_report.pack(fill="x", padx=8, pady=3)
        self.last_report = None
        for button in (self.uniformity_button, self.system_report, self.module_report, self.open_report):
            button.configure(state="disabled")

        console_card = self._card(top, "Console and status")
        self.console = ctk.CTkTextbox(console_card, height=183, width=490,
                                      fg_color="#101820", text_color="#93e68b",
                                      font=("Consolas", 11))
        self.console.pack(fill="both", expand=True, padx=8, pady=(2, 5))
        self.status = ctk.CTkLabel(console_card, text="Ready", anchor="w")
        self.status.pack(fill="x", padx=8)

    def _build_tabs(self):
        self.tabs = ctk.CTkTabview(self, corner_radius=7, command=self._draw_visible)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=5)
        for name in TABS:
            self.tabs.add(name)
        self._build_channels()
        self._build_explorer()
        self._build_overview()
        self._build_coincidences()
        self._build_timestamps()

    def _build_channels(self):
        tab = self.tabs.tab("Channel Status")
        controls = ctk.CTkFrame(tab)
        controls.pack(fill="x", padx=9, pady=(5, 2))
        for text, variable in (("LOW <", self.finding_low), ("× median   HIGH >", self.finding_high),
                               ("× median   Min median (hits)", self.finding_median),
                               ("Min ingest sides", self.finding_events)):
            _label(controls, text)
            _entry(controls, variable, 58)
        ctk.CTkButton(controls, text="Apply", width=66, command=self._apply_thresholds).pack(side="left", padx=4)
        self.open_sm_button = ctk.CTkButton(controls, text="Open in SuperModule", width=150,
                                            command=self._open_in_supermodule, state="disabled")
        self.open_sm_button.pack(side="right", padx=6)
        self.channel_summary = ctk.CTkLabel(tab, text=f"Channel status • {FINDINGS_POPULATION}; "
                                            "process files to assess channels", anchor="w", justify="left")
        self.channel_summary.pack(fill="x", padx=9)
        columns = ("sm", "events", "mm", "time", "time_flags", "time_median",
                   "energy", "energy_flags", "energy_median", "unexpected", "state")
        # The frame must exist first: a widget packed into a later sibling is stacked below it (hidden).
        frame = ctk.CTkFrame(tab)
        frame.pack(fill="x", padx=9)
        self.channel_tree = ttk.Treeview(frame, columns=columns, show="headings", height=9)
        for column, heading, width in zip(columns,
                ("SM", "Ingest sides", "mM seen/expected", "Time assessed", "Time not obs / low / high",
                 "Time median hits", "Energy assessed", "Energy not obs / low / high", "Energy median hits",
                 "Unexpected hits", "Finding"),
                (60, 100, 115, 100, 160, 115, 110, 170, 125, 110, 150)):
            self.channel_tree.heading(column, text=heading)
            self.channel_tree.column(column, width=width, anchor="center")
        for state, colour in FINDING_COLOURS.items():
            self.channel_tree.tag_configure(state, background=colour, foreground="black")
        self.channel_tree.pack(side="left", fill="x", expand=True)
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.channel_tree.yview)
        bar.pack(side="right", fill="y")
        self.channel_tree.configure(yscrollcommand=bar.set)
        self.channel_tree.bind("<<TreeviewSelect>>", self._select_channel_row)
        detail = ctk.CTkFrame(tab, fg_color="transparent")
        detail.pack(fill="both", expand=True, padx=5, pady=(4, 4))
        self.channel_flags = ctk.CTkTextbox(detail, width=300, font=("Consolas", 10))
        self.channel_flags.pack(side="right", fill="y", padx=(4, 0))
        self.channel_fig = Figure(figsize=(13, 4.2), dpi=100)
        self.channel_canvas = FigureCanvasTkAgg(self.channel_fig, master=detail)
        self.channel_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.channel_axes = {}

    def _build_explorer(self):
        tab = self.tabs.tab(SM_TAB)
        toolbar = ctk.CTkFrame(tab)
        toolbar.pack(fill="x", padx=5, pady=(4, 2))
        self.sm_toolbar = toolbar
        self.prev_button = ctk.CTkButton(toolbar, text="◀ Prev", width=64, state="disabled",
                                         command=lambda: self._step_sm(-1))
        self.prev_button.pack(side="left", padx=(6, 2))
        _Tooltip(self.prev_button, "Previous SuperModule (Page Up, or wheel up over this bar)")
        self.module_combo = ctk.CTkComboBox(toolbar, variable=self.module_var,
                                            values=["—"], width=110, state="readonly",
                                            command=lambda _: self._sm_changed())
        self.module_combo.pack(side="left", padx=2)
        _Tooltip(self.module_combo, "SuperModule shown in this tab; the wheel over this bar steps through them")
        self.next_button = ctk.CTkButton(toolbar, text="Next ▶", width=64, state="disabled",
                                         command=lambda: self._step_sm(1))
        self.next_button.pack(side="left", padx=2)
        _Tooltip(self.next_button, "Next SuperModule (Page Down, or wheel down over this bar)")
        self.sm_position = ctk.CTkLabel(toolbar, text="—", width=62)
        self.sm_position.pack(side="left", padx=(2, 4))
        self.fit_toggle = ctk.CTkCheckBox(toolbar, text="Show background fit", variable=self.experimental,
                                          command=self._refresh_all)
        self.fit_toggle.pack(side="left", padx=8)
        self.fit_settings_button = ctk.CTkButton(toolbar, text="Fit settings...", width=100,
                                                  command=self._fit_settings)
        self.fit_settings_button.pack(side="left", padx=2)
        self.residual_button = ctk.CTkButton(toolbar, text="Residuals", width=76,
                                              command=self._residuals)
        self.residual_button.pack(side="left", padx=2)
        ctk.CTkCheckBox(toolbar, text="Profile region", variable=self.profile_mode,
                        command=self._switch_rectangle).pack(side="left", padx=8)
        ctk.CTkButton(toolbar, text="Reset filters (Esc)", width=140,
                      command=self._reset_filters).pack(side="right", padx=6)

        controls = ctk.CTkFrame(tab, fg_color="transparent")
        controls.pack(fill="x", padx=5, pady=(0, 3))
        groups = []
        for title in ("Energy (keV), both sides", "DOI (light-sharing ratio)", "Spatial ROI (mm) / 2D colour"):
            group = ctk.CTkFrame(controls)
            group.pack(side="left", fill="both", expand=True, padx=3, pady=3)
            heading = ctk.CTkLabel(group, text=title, font=("Arial", 11, "bold"))
            heading.pack(anchor="w", padx=6)
            groups.append((group, heading))
        self.energy_control_group, self.doi_control_group, self.spatial_control_group = (
            group for group, _ in groups)
        self.energy_group_title = groups[0][1]

        def add_slider(group, label, variable, upper):
            row = ctk.CTkFrame(group, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=1)
            ctk.CTkLabel(row, text=label, width=34).pack(side="left")
            slider = ctk.CTkSlider(row, from_=0, to=upper, number_of_steps=300,
                                   command=lambda value: variable.set(f"{value:.2f}"))
            slider.pack(side="left", fill="x", expand=True)
            slider.bind("<ButtonRelease-1>", lambda _: self._refresh_all(), add="+")
            _entry(row, variable, 58)
            self._sliders.append((slider, variable))

        add_slider(groups[0][0], "Lo", self.energy_low, 1500)
        add_slider(groups[0][0], "Hi", self.energy_high, 1500)
        add_slider(groups[1][0], "Lo", self.doi_low, 1)
        add_slider(groups[1][0], "Hi", self.doi_high, 1)
        colour_group = groups[2][0]
        roi_row = ctk.CTkFrame(colour_group, fg_color="transparent")
        roi_row.pack(fill="x", padx=4)
        for text, variable in (("X", self.x_low), ("to", self.x_high),
                               ("Y", self.y_low), ("to", self.y_high)):
            _label(roi_row, text)
            _entry(roi_row, variable, 51)
        add_slider(colour_group, "Min", self.color_min, 10)
        add_slider(colour_group, "Max", self.color_max, 10)
        row = ctk.CTkFrame(colour_group, fg_color="transparent")
        row.pack(fill="x", padx=4)
        _label(row, "2D bins")
        ctk.CTkComboBox(row, variable=self.bins, values=["50", "100", "200", "300"],
                        state="readonly", width=80, command=lambda _: self._refresh_all()).pack(side="left", padx=5)
        ctk.CTkButton(row, text="Apply", command=self._refresh_all, width=66).pack(side="right", padx=7)

        self.explorer_info = ctk.CTkLabel(tab, text="Process files to explore SuperModules", anchor="w")
        self.explorer_info.pack(fill="x", padx=9)
        body = ctk.CTkFrame(tab, fg_color="transparent")
        body.pack(fill="both", expand=True)
        self._build_sm_summary(body)
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.energy_ax, self.doi_ax, self.flood_ax = self.fig.subplots(1, 3)
        self.fig.subplots_adjust(left=0.055, right=0.93, bottom=0.12, top=0.91, wspace=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self._bind_wheel(toolbar)

    def _build_sm_summary(self, parent):
        """Summary panel beside the plots: occupancy, channel findings, per-minimodule table."""
        panel = ctk.CTkFrame(parent, width=430)
        panel.pack(side="right", fill="y", padx=(4, 5), pady=(0, 5))
        panel.pack_propagate(False)
        self.sm_state = ctk.CTkLabel(panel, text="No SuperModule selected", anchor="w", corner_radius=5,
                                     font=("Arial", 12, "bold"))
        self.sm_state.pack(fill="x", padx=6, pady=(6, 3))
        # text above, table below; the sash lets the operator trade one for the other
        split = ttk.PanedWindow(panel, orient="vertical")
        split.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        top = ctk.CTkFrame(split, fg_color="transparent")
        bottom = ctk.CTkFrame(split, fg_color="transparent")
        split.add(top, weight=2)
        split.add(bottom, weight=3)
        self.sm_summary = ctk.CTkTextbox(top, height=60, font=("Consolas", 10), wrap="word")
        self.sm_summary.pack(fill="both", expand=True)
        self.mm_title = ctk.CTkLabel(bottom, text="Per-minimodule counts and photopeak", anchor="w",
                                     justify="left", wraplength=410, font=("Arial", 11, "bold"))
        self.mm_title.pack(fill="x", pady=(4, 1))
        columns = ("mm", "ingest", "selected", "mu", "resolution", "fit")
        frame = ctk.CTkFrame(bottom)  # before the Treeview it holds (see _build_channels)
        frame.pack(fill="both", expand=True)
        self.mm_tree = ttk.Treeview(frame, columns=columns, show="headings", height=4)
        for column, heading, width in zip(columns, ("mM", "Ingest", "Selected", "Centroid keV", "Res. %", "Fit"),
                                          (32, 60, 60, 76, 46, 130)):
            self.mm_tree.heading(column, text=heading)
            self.mm_tree.column(column, width=width, anchor="center", stretch=column == "fit")
        self.mm_tree.tag_configure("unpopulated", background=FINDING_COLOURS["INSUFFICIENT EVENTS"],
                                   foreground="#424952")
        self.mm_tree.tag_configure("unavailable", background="#eceef0", foreground="#424952")
        self.mm_tree.pack(side="left", fill="both", expand=True)
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.mm_tree.yview)
        bar.pack(side="right", fill="y")
        self.mm_tree.configure(yscrollcommand=bar.set)
        self.mm_tree.bind("<<TreeviewSelect>>", self._select_mm_row)

    def _build_overview(self):
        tab = self.tabs.tab("System Overview")
        controls = ctk.CTkFrame(tab)
        controls.pack(fill="x", padx=8, pady=5)
        _label(controls, "Minimodule tiles")
        ctk.CTkComboBox(controls, variable=self.overview_mode, values=[*OVERVIEW_METRICS, FLOOD_MODE],
                        state="readonly", width=210,
                        command=lambda _: self._draw_overview()).pack(side="left", padx=8)
        self.overview_open = ctk.CTkButton(controls, text="Open SM", width=90, state="disabled",
                                           command=self._overview_open_sm)
        self.overview_open.pack(side="right", padx=8)
        self.overview_info = ctk.CTkLabel(controls, text="Click a minimodule tile for its SM, mM and value",
                                          anchor="w")
        self.overview_info.pack(side="left", fill="x", expand=True, padx=8)
        self.overview_fig = Figure(figsize=(15, 5), dpi=100)
        self.overview_canvas = FigureCanvasTkAgg(self.overview_fig, master=tab)
        self.overview_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.overview_canvas.mpl_connect("button_press_event", self._overview_click)

    def _build_coincidences(self):
        tab = self.tabs.tab(COINC_TAB)
        controls = ctk.CTkFrame(tab)
        controls.pack(fill="x", padx=8, pady=5)
        _label(controls, "Δt: SM a")
        self.pair_a_combo = ctk.CTkComboBox(controls, variable=self.pair_a, values=["—"], width=100,
                                            state="readonly", command=lambda _: self._draw_pair_dt())
        self.pair_a_combo.pack(side="left", padx=4)
        _label(controls, "vs SM b")
        self.pair_b_combo = ctk.CTkComboBox(controls, variable=self.pair_b, values=[ALL_PARTNERS], width=130,
                                            state="readonly", command=lambda _: self._draw_pair_dt())
        self.pair_b_combo.pack(side="left", padx=4)
        _label(controls, "Bins")
        ctk.CTkComboBox(controls, variable=self.dt_bins, values=["50", "100", "200", "400"], width=80,
                        state="readonly", command=lambda _: self._draw_pair_dt()).pack(side="left", padx=4)
        self.pair_info = ctk.CTkLabel(controls, text="Click a matrix cell to choose an SM pair", anchor="w")
        self.pair_info.pack(side="left", fill="x", expand=True, padx=8)
        ctk.CTkLabel(tab, text=DT_LABEL, anchor="w").pack(fill="x", padx=10)
        self.coinc_fig = Figure(figsize=(15, 5.5), dpi=100)
        self.coinc_canvas = FigureCanvasTkAgg(self.coinc_fig, master=tab)
        self.coinc_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.coinc_canvas.mpl_connect("button_press_event", self._pair_click)

    def _build_timestamps(self):
        tab = self.tabs.tab("Timestamps")
        controls = ctk.CTkFrame(tab)
        controls.pack(fill="x", padx=8, pady=5)
        _label(controls, "File")
        self.time_combo = ctk.CTkComboBox(controls, variable=self.time_file,
                                          values=["—"], state="readonly", width=260,
                                          command=lambda _: self._draw_timestamps())
        self.time_combo.pack(side="left", padx=8)
        _label(controls, "View")
        ctk.CTkComboBox(controls, variable=self.time_view,
                        values=["Event rate", "Paired delta-t"], state="readonly", width=140,
                        command=lambda _: self._draw_timestamps()).pack(side="left", padx=8)
        _label(controls, "Bins")
        ctk.CTkComboBox(controls, variable=self.time_bins, values=["30", "60", "120", "300"],
                        state="readonly", width=90,
                        command=lambda _: self._draw_timestamps()).pack(side="left")
        ctk.CTkLabel(tab, text="Within-file detector-side activity or paired hit time difference; neither establishes absolute clock drift.",
                     anchor="w").pack(fill="x", padx=10)
        self.time_fig = Figure(figsize=(13, 5), dpi=100)
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, master=tab)
        self.time_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _log(self, text):
        from datetime import datetime
        self.console.insert("end", f"[{datetime.now():%H:%M:%S}] {text}\n")
        self.console.see("end")
        self.status.configure(text=text[:90])

    def _select_config(self):
        if self._busy or self._report_busy:
            return
        path = filedialog.askopenfilename(filetypes=[("YAML configuration", "*.yaml"), ("All files", "*.*")],
                                           initialdir=str(Path(__file__).resolve().parent.parent / "configs"))
        if path:
            self._invalidate_data()
            self.config_path = path
            self.config_text.configure(text=Path(path).name)
            try:
                import yaml
                with open(path, encoding="utf-8") as handle:
                    config = yaml.safe_load(handle)
                if isinstance(config, dict):
                    if self.calibrated.get():
                        low, high = config.get("energy_range") or (400, 650)
                        self.energy_low.set(str(low))
                        self.energy_high.set(str(high))
                    self.min_channels.set(str(config.get("min_ch", 1)))
                    self.min_channel_energy.set(str(config.get("en_min_ch", 0)))
                self.system.set("CORNELL" if "cornell" in Path(path).name.lower() else "IMAS")
            except (OSError, ValueError, TypeError) as exc:
                self._log(f"Config needs review: {exc}")

    def _select_calibration(self):
        if self._busy or self._report_busy:
            return
        path = filedialog.askopenfilename(filetypes=[("Calibration", "*.txt *.encal"), ("All files", "*.*")])
        if path:
            previous = self.calibration_path
            self.calibration_path = path
            self._set_calibration_label()
            if self.dataset and self.calibrated.get():
                self._start_calibration(path, True, previous_path=previous)

    def _set_calibration_label(self):
        if not self.calibration_path:
            self.calib_text.configure(text="No energy calibration")
            return
        name = Path(self.calibration_path).name
        self.calib_text.configure(text=name if len(name) <= 42 else name[:19] + "…" + name[-19:])

    def _show_calibration_path(self, _event=None):
        if self.calibration_path:
            messagebox.showinfo("Selected energy calibration", self.calibration_path, parent=self)

    def _toggle_calibration(self):
        if self._busy or self._report_busy:
            self.calibrated.set(self.dataset.settings.calibrated if self.dataset else True)
            return
        enabled = self.calibrated.get()
        if enabled and not Path(self.calibration_path).is_file():
            self.calibrated.set(False)
            messagebox.showerror("Calibration", "Select an energy calibration file to enable keV", parent=self)
            return
        if self.dataset:
            self._start_calibration(self.calibration_path, enabled)
            return
        if not enabled:
            self.experimental.set(False)
        self._fit_control_state()
        self._reset_energy_range()
        self._sync_sliders()

    def _start_calibration(self, path, enabled, *, previous_path=None):
        dataset = self.dataset
        self._busy = True
        self.calib_switch.configure(state="disabled")
        self.process_button.configure(state="disabled")
        for button in (self.uniformity_button, self.system_report, self.module_report):
            button.configure(state="disabled")
        self._log("Applying energy calibration to retained detector sides..." if enabled
                  else "Switching retained detector sides to raw a.u....")

        def run():
            try:
                result = apply_calibration(dataset, path, enabled)
                self._events.put(("calibrated", result))
            except Exception as exc:
                self._events.put(("calibration_error", str(exc), previous_path))

        threading.Thread(target=run, daemon=True).start()

    def _fit_control_state(self):
        state = "normal" if self.calibrated.get() else "disabled"
        for widget in (self.fit_toggle, self.fit_settings_button, self.residual_button):
            widget.configure(state=state)

    def _reset_energy_range(self):
        if self.calibrated.get():
            low, high = (400, 650)
            if self.dataset:
                low, high = self.dataset.config.get("energy_range") or (400, 650)
            elif self.config_path:
                import yaml
                with open(self.config_path, encoding="utf-8") as handle:
                    low, high = yaml.safe_load(handle).get("energy_range") or (400, 650)
        else:
            low, high = 0, 300
        self.energy_low.set(str(low))
        self.energy_high.set(str(high))
        self.energy_group_title.configure(text="Energy (keV), both detector sides" if self.calibrated.get()
                                           else "Raw energy (a.u.), both sides")

    def _sync_sliders(self):
        if not self._sliders:
            return
        data = self.dataset.modules.get(self._selected_sm()) if self.dataset else None
        energy_max = 1500 if self.calibrated.get() else 300
        colour_max = max(10, float(len(data) / max(int(self.bins.get()) ** 2, 1)) * 10
                         if data is not None else 0)
        doi_max = max(1.0, float(np.max(data.doi)) * 1.1 if data is not None and len(data) else 0)
        for index, (slider, field) in enumerate(self._sliders):
            upper = energy_max if index < 2 else doi_max if index < 4 else colour_max
            slider.configure(to=upper)
            try:
                value = float(field.get())
            except ValueError:  # Auto remains available for the colour maximum entry.
                value = upper
            slider.set(min(max(value, 0), upper))

    def _invalidate_data(self):
        if not self.dataset:
            return
        self.dataset = None
        for button in (self.uniformity_button, self.system_report, self.module_report):
            button.configure(state="disabled")
        self.module_var.set("")
        self.channel_tree.delete(*self.channel_tree.get_children())
        self._findings, self._channel_sm = {}, None
        self.open_sm_button.configure(state="disabled")
        self.channel_flags.delete("1.0", "end")
        self.channel_summary.configure(text=f"Channel status • {FINDINGS_POPULATION}; process files to assess channels")
        self._overview_token += 1  # a running overview job stops at its next SuperModule
        self._overview_cache, self._overview_job, self._overview_grid = [], None, None
        self._overview_ax = self._overview_pick = None
        self.overview_open.configure(state="disabled")
        self.overview_info.configure(text="Click a minimodule tile for its SM, mM and value")
        self._pair_token += 1  # a running pair job's result is ignored
        self._pair_cache, self._pair_job, self._pair_current = [], None, None
        self._matrix_ax = self._dt_ax = None
        self.pair_info.configure(text="Click a matrix cell to choose an SM pair")
        if self._step_after is not None:
            self.after_cancel(self._step_after)
        self._step_after, self._sm_ids, self._stale = None, [], set()
        self._sm_cache, self._sm_wanted, self._mm_layout = [], None, None  # a running SM job is ignored
        self._update_step_controls()
        self._clear_summary("Inputs changed; process the selected files again")
        if self._selector is not None:
            self._selector.set_active(False)
            self._selector.disconnect_events()
            self._selector = None
        for figure, canvas in ((self.fig, self.canvas), (self.channel_fig, self.channel_canvas),
                               (self.overview_fig, self.overview_canvas), (self.coinc_fig, self.coinc_canvas),
                               (self.time_fig, self.time_canvas)):
            figure.clear()
            canvas.draw_idle()
        self._colorbar = None
        self._close_residuals()
        self.energy_ax, self.doi_ax, self.flood_ax = self.fig.subplots(1, 3)
        self.channel_axes = {}
        self.explorer_info.configure(text="Inputs changed; process the selected files again")

    def _select_files(self):
        if self._busy or self._report_busy:
            return
        paths = filedialog.askopenfilenames(filetypes=[("PETsys LDAT", "*.ldat"), ("All files", "*.*")])
        for path in paths:
            if path not in self.files:
                self._invalidate_data()
                self.files.append(path)
        self.file_count.configure(text=f"{len(self.files)} files")
        self._update_file_list()
        if paths:
            self._log(f"Selected {len(self.files)} LDAT files; no file count is tied to CPU cores.")
            self._schedule_estimate()

    def _clear_files(self):
        if self._busy or self._report_busy:
            return
        self._invalidate_data()
        self.files.clear()
        self.file_count.configure(text="0 files")
        self._update_file_list()
        self._log("File selection cleared")
        self._schedule_estimate()

    def _whole_files_changed(self):
        self.max_pairs_entry.configure(state="disabled" if self.whole_files.get() else "normal")
        self._schedule_estimate()

    def _pairs_limit(self):
        """Pairs read per file, or None for whole files."""
        return None if self.whole_files.get() else int(self.max_pairs.get())

    def _current_settings(self):
        settings = Settings(self.config_path, self.calibration_path, self.system.get(),
                            self._pairs_limit(), int(self.min_channels.get()),
                            float(self.min_channel_energy.get()), self.calibrated.get())
        settings.validate()
        return settings

    def _schedule_estimate(self):
        """Refresh the memory estimate shortly after inputs stop changing."""
        if self._estimate_after is not None:
            self.after_cancel(self._estimate_after)
        self._estimate_after = self.after(400, self._start_estimate)

    def _start_estimate(self, *, then_process=None):
        """Estimate in a worker thread; ``then_process`` = (settings, files) to start after."""
        self._estimate_after = None
        files = tuple(self.files)
        if not files:
            self.estimate_text.configure(text="Estimated memory: add LDAT files", text_color=("gray10", "gray90"))
            return
        try:
            settings = self._current_settings()
        except (ValueError, OSError):
            settings = None  # config or cuts incomplete: upper bound without sampling
        try:
            limit = self._pairs_limit()
        except ValueError:
            return
        self._estimate_token += 1
        token = self._estimate_token

        def run():
            try:
                estimate = estimate_memory(files, settings, limit)
                self._events.put(("estimate", token, estimate, then_process))
            except Exception as exc:
                self._events.put(("estimate", token, None, then_process, str(exc)))

        threading.Thread(target=run, daemon=True).start()

    def _show_estimate(self, estimate):
        warn = estimate.exceeds_available
        self.estimate_text.configure(text=estimate.text() + (" — exceeds free RAM" if warn else ""),
                                     text_color="#c0392b" if warn else ("gray10", "gray90"))

    def _update_file_list(self):
        self.file_list.delete("1.0", "end")
        self.file_list.insert("end", "\n".join(Path(p).name for p in self.files)
                              if self.files else "No LDAT files selected")

    def _start_processing(self):
        if self._busy or self._report_busy:
            return
        try:
            if not self.files:
                raise ValueError("Add at least one LDAT file")
            settings = self._current_settings()
            selection = self._selection()
            del selection
        except Exception as exc:
            messagebox.showerror("Input", str(exc), parent=self)
            return
        self._busy = True
        self._abort.clear()
        self.process_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.calib_switch.configure(state="disabled")
        self.progress.set(0)
        self._log("Estimating memory before processing...")
        self._start_estimate(then_process=(settings, tuple(self.files)))

    def _confirm_and_launch(self, estimate, settings, files):
        """After the pre-run estimate: ask when it exceeds free RAM, then start."""
        if self._abort.is_set():
            self._processing_stopped("Processing cancelled before it started")
            return
        if estimate is not None and estimate.exceeds_available and not messagebox.askyesno(
                "Memory", f"{estimate.text()}.\n\nThe estimate exceeds the free physical memory. "
                          "Process anyway?", parent=self):
            self._processing_stopped("Processing not started: estimated memory exceeds free RAM")
            return
        self._launch_pool(settings, files)

    def _launch_pool(self, settings, files):
        context = multiprocessing.get_context("spawn")
        self._cancel = context.Event()
        self._file_progress = context.Array("d", len(files))
        self._workers = max(1, min(len(files), (os.cpu_count() or 1) - 2))
        pool = ProcessPoolExecutor(max_workers=self._workers, mp_context=context,
                                   initializer=init_worker, initargs=(self._cancel, self._file_progress))
        self._pool = pool
        reader = self._reader or process_file
        scope = ("whole files" if settings.max_pairs is None
                 else f"first {settings.max_pairs:,} pairs per file")
        self._log(f"Processing {len(files)} files with {self._workers} worker processes ({scope})")

        def run():
            results = []
            try:
                setup = load_setup(settings)
                futures = {pool.submit(reader, path, settings, i): i for i, path in enumerate(files)}
                for future in as_completed(futures):
                    if self._abort.is_set():
                        break
                    i = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = FileResult(i, files[i], error=str(exc))
                    results.append(result)
                    self._events.put(("file", result, len(results), len(files), settings.max_pairs is None))
                if self._abort.is_set():
                    pool.shutdown(wait=False, cancel_futures=True)
                    self._events.put(("cancelled", len(results), len(files)))
                    return
                pool.shutdown(wait=False)
                results.sort(key=lambda r: r.index)
                self._events.put(("complete", merge_results(settings, results, setup, consume=True)))
            except Exception as exc:
                pool.shutdown(wait=False, cancel_futures=True)
                self._events.put(("error", str(exc)))

        threading.Thread(target=run, daemon=True).start()

    def _cancel_processing(self):
        """Stop workers at their next chunk; the loaded dataset (if any) stays."""
        if not self._busy:
            return
        self._abort.set()
        if self._cancel is not None:
            self._cancel.set()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        self.cancel_button.configure(state="disabled")
        self._log("Cancelling: workers stop after their current chunk...")

    def _processing_stopped(self, message):
        self._busy = False
        self._pool = None
        self.progress.set(0)
        self.process_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.calib_switch.configure(state="normal")
        self._log(message)

    def _poll_events(self):
        try:
            while True:
                event = self._events.get_nowait()
                if event[0] == "estimate":
                    token, estimate, then_process = event[1:4]
                    if estimate is not None and token == self._estimate_token:
                        self._show_estimate(estimate)
                    elif estimate is None:
                        self._log(f"Memory estimate unavailable: {event[4]}")
                    if then_process is not None:
                        self._confirm_and_launch(estimate, *then_process)
                elif event[0] == "file":
                    _, result, done, count, whole = event
                    if self._file_progress is not None and 0 <= result.index < len(self._file_progress):
                        self._file_progress[result.index] = 1.0
                    note = f"{result.pairs_accepted:,}/{result.pairs_read:,} pairs"
                    if result.prefix_limited:
                        note += " (file prefix)"
                    elif whole and result.success:
                        note += " (whole file)"
                    if result.errors:
                        note += "; rejected: " + ", ".join(
                            f"{reason}={number}" for reason, number in result.errors.items())
                    self._log(f"{Path(result.path).name}: {note}" if result.success
                              else f"{Path(result.path).name}: FAILED — {result.error}")
                elif event[0] == "cancelled":
                    kept = "previous dataset kept" if self.dataset else "no dataset loaded"
                    self._processing_stopped(f"Processing cancelled after {event[1]}/{event[2]} files; {kept}")
                elif event[0] == "complete":
                    self.dataset = event[1]
                    self._processing_stopped(f"Processing complete ({self._workers} worker processes)")
                    self.progress.set(1)
                    self._update_after_processing()
                elif event[0] == "calibrated":
                    self.dataset = event[1]
                    self._busy = False
                    self.calibrated.set(self.dataset.settings.calibrated)
                    self.process_button.configure(state="normal")
                    self.calib_switch.configure(state="normal")
                    if not self.calibrated.get():
                        self.experimental.set(False)
                    self._fit_control_state()
                    for button in (self.uniformity_button, self.system_report, self.module_report):
                        button.configure(state="normal" if self.dataset.modules else "disabled")
                    self._reset_energy_range()
                    self._refresh_all()
                    self._log("Energy view updated without rereading LDAT (keV)" if self.calibrated.get()
                              else "Energy view updated without rereading LDAT (a.u.)")
                elif event[0] == "calibration_error":
                    self._busy = False
                    self.process_button.configure(state="normal")
                    self.calib_switch.configure(state="normal")
                    self.calibrated.set(self.dataset.settings.calibrated)
                    if event[2] is not None:
                        self.calibration_path = event[2]
                        self._set_calibration_label()
                    self._log(f"Energy calibration unavailable: {event[1]}")
                    for button in (self.uniformity_button, self.system_report, self.module_report):
                        button.configure(state="normal" if self.dataset.modules else "disabled")
                    messagebox.showerror("Calibration", event[1], parent=self)
                elif event[0] == "overview":
                    self._overview_done(*event[1:])
                elif event[0] == "sm_metrics":
                    self._sm_metrics_done(*event[1:])
                elif event[0] == "pairs":
                    self._pairs_done(*event[1:])
                elif event[0] == "report":
                    self._report_busy = False
                    self.last_report = event[1]
                    self.process_button.configure(state="normal")
                    self.calib_switch.configure(state="normal")
                    self.open_report.configure(state="normal")
                    for button in (self.system_report, self.module_report):
                        button.configure(state="normal")
                    self._log(f"Report saved: {event[1]}")
                elif event[0] == "report_error":
                    self._report_busy = False
                    self.process_button.configure(state="normal")
                    self.calib_switch.configure(state="normal")
                    for button in (self.system_report, self.module_report):
                        button.configure(state="normal")
                    self._log(f"Report failed: {event[1]}")
                    messagebox.showerror("Report", event[1])
                else:
                    self._processing_stopped(f"Processing failed: {event[1]}")
        except queue.Empty:
            pass
        if self._busy and self._file_progress is not None and self._pool is not None:
            fractions = list(self._file_progress)
            self.progress.set(sum(fractions) / max(len(fractions), 1))
        if self.winfo_exists():
            self.after(80, self._poll_events)

    def _update_after_processing(self):
        dataset = self.dataset
        self.calibrated.set(dataset.settings.calibrated)
        self._fit_control_state()
        self._reset_energy_range()
        successful = [f for f in dataset.files if f.success]
        self._log(f"Merged {sum(f.pairs_accepted for f in successful):,} coincidence pairs, "
                  f"{sum(map(len, dataset.modules.values())):,} detector sides, "
                  f"{len(successful)}/{len(dataset.files)} successful files")
        ids = sorted(set(dataset.expected_time) | set(dataset.modules))
        values = [f"SM {sm}" for sm in ids]
        self.module_combo.configure(values=values or ["—"])
        self.module_var.set(values[0] if values else "")
        self._sm_ids, self._sm_cache, self._sm_wanted, self._mm_layout = ids, [], None, None
        self._update_step_controls()
        self.pair_a_combo.configure(values=values or ["—"])
        self.pair_b_combo.configure(values=[ALL_PARTNERS, *values])
        self.pair_a.set(values[0] if values else "")
        self.pair_b.set(ALL_PARTNERS)
        labels = [f"{f.index}: {Path(f.path).name}" for f in successful]
        self.time_combo.configure(values=labels or ["—"])
        self.time_file.set(labels[0] if labels else "")
        self._sync_sliders()
        for button in (self.uniformity_button, self.system_report, self.module_report):
            button.configure(state="normal" if dataset.modules else "disabled")
        # Findings use the ingest population only: independent of display cuts and calibration.
        self._draw_channels()
        self._refresh_all()

    def _selected_sm(self):
        try:
            return int(self.module_var.get().removeprefix("SM "))
        except ValueError:
            return None

    def _selection(self):
        values = [float(var.get()) for var in (self.energy_low, self.energy_high,
                  self.doi_low, self.doi_high, self.x_low, self.x_high,
                  self.y_low, self.y_high)]
        if not np.isfinite(values).all() or any(values[i] >= values[i+1] for i in (0, 2, 4, 6)):
            raise ValueError("Each filter must have a finite low value below its high value")
        return Selection(*values)

    def _refresh_all(self):
        """Validate the cuts, mark every drawn tab stale and redraw the visible one."""
        try:
            self._selection()
            if self.experimental.get():
                self._experimental_settings()
            count = int(self.bins.get())
            if not 10 <= count <= 500:
                raise ValueError("Flood bins must be between 10 and 500")
            self._colour_range()
            self._sync_sliders()
        except (ValueError, OverflowError) as exc:
            messagebox.showerror("Filter", str(exc))
            return
        self._stale.update(self._drawers)
        self._draw_visible()

    def _draw_visible(self):
        """Draw the visible tab if it is stale (also the tab-change callback)."""
        name = self.tabs.get()
        if not self.dataset or name not in self._stale:
            return
        self._stale.discard(name)
        try:
            self._drawers[name]()
        except (ValueError, OverflowError) as exc:
            self._stale.add(name)
            messagebox.showerror("Filter", str(exc))

    def _show_tab(self, name):
        self.tabs.set(name)  # CTkTabview.set does not call the tab-change command
        self._draw_visible()

    def _sm_changed(self):
        """The selected SM changed: only the SuperModule tab depends on it."""
        if self._step_after is not None:
            self.after_cancel(self._step_after)
            self._step_after = None
        self._update_step_controls()
        self._stale.add(SM_TAB)
        self._draw_visible()

    def _update_step_controls(self):
        ids, sm = self._sm_ids, self._selected_sm()
        index = ids.index(sm) if sm in ids else None
        self.sm_position.configure(text="—" if index is None else f"{index + 1} / {len(ids)}")
        self.prev_button.configure(state="normal" if index else "disabled")
        self.next_button.configure(state="normal" if index is not None and index < len(ids) - 1 else "disabled")

    def _step_sm(self, delta, *, debounce=False):
        """Select the previous/next SM, clamped at the ends; True when the selection moved.

        Button clicks redraw at once. The wheel and Page Up/Down redraw
        ``STEP_DEBOUNCE_MS`` after the last step, so a fast scroll draws once.
        """
        ids, sm = self._sm_ids, self._selected_sm()
        if sm not in ids:
            return False
        target = ids[min(max(ids.index(sm) + delta, 0), len(ids) - 1)]
        if target == sm:
            return False
        self.module_var.set(f"SM {target}")
        if not debounce:
            self._sm_changed()
            return True
        self._update_step_controls()
        if self._step_after is not None:
            self.after_cancel(self._step_after)
        self._step_after = self.after(STEP_DEBOUNCE_MS, self._sm_changed)
        return True

    def _key_step(self, delta):
        if self.dataset and self.tabs.get() == SM_TAB:
            self._step_sm(delta, debounce=True)
            return "break"
        return None

    def _wheel_step(self, event):
        up = event.num == 4 or (event.num != 5 and event.delta > 0)
        self._step_sm(-1 if up else 1, debounce=True)
        return "break"

    def _bind_wheel(self, widget):
        """Bind the wheel on a widget and all its inner Tk widgets (CTk widgets are composites)."""
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            tk.Misc.bind(widget, sequence, self._wheel_step, "+")
        for child in tk.Misc.winfo_children(widget):
            self._bind_wheel(child)

    def _refresh_selected(self):
        if not self.dataset:
            return
        selection = self._selection()
        min_count, max_count = self._colour_range()
        if self.experimental.get():
            try:
                interval, search = self._experimental_settings()
            except ValueError as exc:
                self._log(f"Fit settings: {exc}")
                return
        self._close_residuals()
        sm = self._selected_sm()
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        for axis in (self.energy_ax, self.doi_ax, self.flood_ax):
            axis.clear()
        data = self.dataset.modules.get(sm)
        if data is None or len(data) == 0:
            self.explorer_info.configure(text=f"SM {sm}: no accepted coincidence detector sides")
            self.canvas.draw_idle()
            self._draw_summary(sm, selection, 0)
            return
        spatial = selection.mask(data, energy=False)
        chosen = selection.mask(data)
        energies = data.energy[spatial]
        finite_energies = energies[np.isfinite(energies)]
        calibrated = self.dataset.settings.calibrated
        energy_max = 1500 if calibrated else 300
        _, edges, _ = self.energy_ax.hist(finite_energies, bins=160, range=(0, energy_max),
                                            color="#347dc1", alpha=0.75)
        if calibrated:
            self.energy_ax.axvspan(350, 700, color="#e64c46", alpha=0.035, zorder=0)
        self.energy_ax.set(xlabel="Calibrated energy (keV)" if calibrated else "Raw energy (a.u.)",
                           ylabel="Detector sides",
                            title=f"Energy • {len(finite_energies):,}/{len(energies):,} ROI/DOI sides"
                            if calibrated else f"Energy • {len(energies):,} ROI/DOI sides")
        for threshold in (selection.energy_low, selection.energy_high):
            self.energy_ax.axvline(threshold, c="#d35930", lw=1.4, ls="--")
        self.energy_ax.set_xlim(0, energy_max)
        legacy = (fit_peak(finite_energies) if calibrated else
                   {"status": "unavailable: raw a.u. (no keV calibration)"})
        auto_line = None
        fit_readouts = []
        if legacy["status"] == "FIT":
            overlay = fit_on_display_bins(legacy, edges)
            if overlay is not None:
                auto_line, = self.energy_ax.plot(
                    overlay["x"], overlay["total"], c="#e64c46", lw=2,
                    label=f"Photopeak model: {legacy['mu']:.1f} keV, resolution "
                          f"{legacy['resolution']:.1f}% (Gaussian + {legacy['background_model']} background)")
                fit_readouts.append(f"Photopeak: {legacy['mu']:.1f} keV  |  resolution: {legacy['resolution']:.1f}%")
                if self.experimental.get():
                    self.energy_ax.plot(overlay["x"], overlay["gaussian"], c="#ed7792", lw=1.5,
                                        ls="--", label="Gaussian component (above local background)")
                    self.energy_ax.plot(overlay["x"], overlay["background"], c="#ba9bdb", ls=":",
                                        label="Estimated background")
        if self.experimental.get() and calibrated:
            extra = fit_peak_background(finite_energies, interval=interval, search=search)
            if extra["status"] == "FIT":
                overlay = fit_on_display_bins(extra, edges)
                if overlay is not None:
                    same_fit = (auto_line is not None and legacy["background_model"] == "linear"
                                 and tuple(interval) == tuple(legacy["interval"])
                                 and tuple(search) == tuple(legacy["search"]))
                    if same_fit:
                        auto_line.set_label(auto_line.get_label() + "; same background-aware fit")
                        fit_readouts.append(f"Background fit: {extra['mu']:.1f} keV  |  resolution: "
                                            f"{extra['resolution']:.1f}% (same model)")
                    else:
                        self.energy_ax.plot(overlay["x"], overlay["total"], c="#7f3fb3", lw=1.6,
                                            label=f"Background-aware fit: {extra['mu']:.1f} keV, "
                                                  f"resolution {extra['resolution']:.1f}% (linear)")
                        self.energy_ax.plot(overlay["x"], overlay["background"], c="#ba9bdb", ls="--",
                                            label="Adjusted background")
                        fit_readouts.append(f"Background fit: {extra['mu']:.1f} keV  |  resolution: {extra['resolution']:.1f}%")
            else:
                self._log(f"Experimental fit SM {sm}: {extra['status']}")
                fit_readouts.append(f"Background fit: unavailable ({extra['status']})")
        if fit_readouts:
            self.energy_ax.text(0.98, 0.97, "\n".join(fit_readouts), transform=self.energy_ax.transAxes,
                                ha="right", va="top", fontsize=8, color="#9f322c",
                                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#a8a8a8"})
        if self.energy_ax.get_legend_handles_labels()[1]:
            self.energy_ax.legend(loc="lower right", fontsize=8)

        doi_selection = replace(selection, doi_low=-float("inf"), doi_high=float("inf"))
        doi_population = doi_selection.mask(data)
        doi_max = max(1.0, float(np.max(data.doi)) * 1.1)
        self.doi_ax.hist(data.doi[doi_population], bins=55, range=(0, doi_max), color="#a358b0")
        self.doi_ax.set(xlabel="DOI light-sharing ratio (not mm)", ylabel="Detector sides",
                         title=f"DOI • {int(doi_population.sum()):,} energy/ROI sides")
        self.doi_ax.axvline(selection.doi_low, c="#d35930", ls="--")
        self.doi_ax.axvline(selection.doi_high, c="#d35930", ls="--")
        self.doi_ax.set_xlim(0, doi_max)

        if chosen.any():
            counts, xedges, yedges = flood_counts(data.x[chosen], data.y[chosen], int(self.bins.get()))
            cmap = matplotlib.colormaps["plasma"].copy()
            cmap.set_bad("white")
            image = self.flood_ax.pcolormesh(xedges, yedges, counts, cmap=cmap,
                                              vmin=min_count, vmax=max_count)
            self._colorbar = self.fig.colorbar(image, ax=self.flood_ax, pad=0.02, shrink=0.8)
            self._colorbar.set_label("Detector sides / bin")
        self.flood_ax.set(xlabel="Local X (mm)", ylabel="Local Y (mm)",
                          title=f"Flood map • SM {sm}")
        self.flood_ax.set_xlim(0, 102)
        self.flood_ax.set_ylim(0, 102)
        self.flood_ax.add_patch(Rectangle((selection.x_low, selection.y_low),
                                          selection.x_high - selection.x_low,
                                          selection.y_high - selection.y_low,
                                          fill=False, edgecolor="#4ec06c", lw=1.6))
        self.explorer_info.configure(text=f"SM {sm} • {len(data):,} ingested sides • "
                f"{int(chosen.sum()):,} after paired energy + DOI + ROI "
                f"({'keV' if calibrated else 'raw a.u.'}) • fit: {legacy['status']}")
        self._switch_rectangle()
        self.canvas.draw_idle()
        self._draw_summary(sm, selection, int(chosen.sum()))

    def _experimental_settings(self):
        interval = (float(self.fit_low.get()), float(self.fit_high.get()))
        search = (float(self.search_low.get()), float(self.search_high.get()))
        if not (np.isfinite((*interval, *search)).all()
                and interval[0] < search[0] < search[1] < interval[1]):
            raise ValueError("Fit interval must contain the narrower keV peak search interval")
        return interval, search

    def _colour_range(self):
        minimum = float(self.color_min.get())
        maximum = (None if self.color_max.get().strip().lower() == "auto"
                   else float(self.color_max.get()))
        if (not np.isfinite(minimum) or minimum < 0 or
                (maximum is not None and
                 (not np.isfinite(maximum) or maximum <= minimum))):
            raise ValueError("Flood colour min must be nonnegative and below max (or use Auto)")
        return minimum, maximum

    def _fit_settings(self):
        window = ctk.CTkToplevel(self)
        window.title("Experimental calibrated-keV fit")
        window.geometry("450x210")
        ctk.CTkLabel(window, text="Opt-in Gaussian + local continuum display estimate (keV)",
                     wraplength=400).pack(pady=8)
        for label, low, high in (("Fit window", self.fit_low, self.fit_high),
                                 ("Peak search", self.search_low, self.search_high)):
            row = ctk.CTkFrame(window, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=6)
            _label(row, label)
            _entry(row, low, 70)
            _label(row, "to")
            _entry(row, high, 70)

        def apply():
            try:
                self._experimental_settings()
            except ValueError as exc:
                messagebox.showerror("Fit settings", str(exc))
                return
            window.destroy()
            self._refresh_all()

        ctk.CTkButton(window, text="Apply", command=apply).pack(pady=7)
        self._raise_dialog(window)

    def _raise_dialog(self, window):
        """Keep a Toplevel owned by the inspector on Windows after it maps."""
        window.transient(self)
        window.lift(self)

        def focus_after_map():
            try:
                if window.winfo_exists():
                    window.lift(self)
                    window.focus_force()
            except tk.TclError:
                pass  # A dialog closed before the mapping callback.

        window.bind("<Map>", lambda event: window.after_idle(focus_after_map), add="+")
        window.after_idle(focus_after_map)

    def _close_residuals(self):
        if self._residual_window is not None:
            try:
                if self._residual_window.winfo_exists():
                    self._residual_window.destroy()
            except tk.TclError:
                pass
            self._residual_window = None

    def _residuals(self):
        if self.dataset and not self.dataset.settings.calibrated:
            self._log("Residuals unavailable: raw a.u. has no calibrated-keV fit")
            return
        if not self.dataset or not self.experimental.get():
            self._log("Enable the experimental fit to inspect its residuals")
            return
        try:
            interval, search = self._experimental_settings()
        except ValueError as exc:
            messagebox.showerror("Residuals", str(exc))
            return
        data = self.dataset.modules.get(self._selected_sm())
        if data is None:
            return
        energies = data.energy[self._selection().mask(data, energy=False)]
        fit = fit_peak_background(energies, interval=interval, search=search)
        if fit["status"] != "FIT":
            self._log(f"No residuals: {fit['status']}")
            return
        counts, _ = np.histogram(energies, bins=len(fit["x"]), range=interval)
        self._close_residuals()
        self._residual_window = ctk.CTkToplevel(self)
        self._residual_window.title(f"SM {self._selected_sm()} • derived fit residuals")
        self._residual_window.geometry("870x400")
        fig = Figure(figsize=(8.7, 4))
        axis = fig.add_subplot(111)
        axis.plot(fit["x"], counts - fit["total"], lw=1, color="#683eaf")
        axis.axhline(0, color="gray", lw=0.8)
        axis.set(xlabel="Calibrated energy (keV)", ylabel="Measured - model counts",
                 title="Derived residuals (negative values retained)")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self._residual_window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        self._raise_dialog(self._residual_window)

    def _switch_rectangle(self):
        if self._selector is not None:
            self._selector.set_active(False)
            self._selector.disconnect_events()
        if not self.dataset:
            return
        self._selector = RectangleSelector(self.flood_ax, self._on_rectangle,
                                            useblit=False, button=[1], minspanx=1,
                                            minspany=1, spancoords="data", interactive=False)

    def _on_rectangle(self, start, end):
        if start.xdata is None or end.xdata is None or start.ydata is None or end.ydata is None:
            return
        xlo, xhi = sorted((start.xdata, end.xdata))
        ylo, yhi = sorted((start.ydata, end.ydata))
        if self.profile_mode.get():
            self._show_profile(xlo, xhi, ylo, yhi)
            return
        for var, value in ((self.x_low, xlo), (self.x_high, xhi),
                           (self.y_low, ylo), (self.y_high, yhi)):
            var.set(f"{value:.2f}")
        self._refresh_all()

    def _show_profile(self, xlo, xhi, ylo, yhi):
        data = self.dataset.modules.get(self._selected_sm())
        if data is None:
            return
        mask = self._selection().mask(data)
        mask &= ((data.x >= xlo) & (data.x <= xhi)
                 & (data.y >= ylo) & (data.y <= yhi))
        if not mask.any():
            self._log("Profile region has no selected detector sides")
            return
        window = ctk.CTkToplevel(self)
        window.title(f"SM {self._selected_sm()} • ROI profile • {int(mask.sum()):,} sides")
        window.geometry("900x480")
        fig = Figure(figsize=(9, 4.5))
        left, right = fig.subplots(1, 2)
        left.hist(data.x[mask], bins=60, color="#3f8dc1")
        left.set(xlabel="X (mm)", ylabel="Detector sides", title="X profile")
        right.hist(data.y[mask], bins=60, color="#9656ad")
        right.set(xlabel="Y (mm)", ylabel="Detector sides", title="Y profile")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        self._raise_dialog(window)

    def _reset_filters(self):
        self._reset_energy_range()
        for var, text in ((self.doi_low, "0"), (self.doi_high, "15"),
                          (self.x_low, "0"), (self.x_high, "102"),
                          (self.y_low, "0"), (self.y_high, "102"),
                          (self.color_min, "0.1"),
                          (self.color_max, "Auto")):
            var.set(text)
        self.experimental.set(False)
        for var, text in ((self.fit_low, "350"), (self.fit_high, "700"),
                          (self.search_low, "425"), (self.search_high, "600")):
            var.set(text)
        self.profile_mode.set(False)
        self._refresh_all()

    def _apply_thresholds(self):
        try:
            events = float(self.finding_events.get())
            thresholds = FindingThresholds(float(self.finding_low.get()), float(self.finding_high.get()),
                                           float(self.finding_median.get()),
                                           int(events) if events.is_integer() else events).validate()
        except ValueError as exc:
            messagebox.showerror("Channel thresholds", str(exc))
            return
        self.thresholds = thresholds
        self._log(f"Channel thresholds: {thresholds.text()}")
        self._draw_channels()
        self._stale.add(SM_TAB)  # its summary shows the findings
        self._draw_visible()

    def _draw_channels(self):
        if not self.dataset:
            return
        tree = self.channel_tree
        keep = self._channel_sm
        tree.delete(*tree.get_children())
        rows = system_channel_findings(self.dataset, self.thresholds)
        self._findings = {row["sm"]: row for row in rows}
        states, channel_states = Counter(), Counter()

        def assessed(kind):
            return "insufficient" if kind["insufficient"] else f"{len(kind['channels'])}"

        def flags(kind):
            return "—" if kind["insufficient"] else " / ".join(
                str(kind["counts"].get(state, 0)) for state in ("NOT OBSERVED", "LOW", "HIGH"))

        def median(kind):
            return "—" if kind["median"] is None else f"{kind['median']:,.0f}"

        for row in rows:
            sm, t, e = row["sm"], row["time"], row["energy"]
            data = self.dataset.modules.get(sm)
            seen_mm = len(np.unique(data.mm)) if data is not None else 0
            unexpected = len(t["unexpected"]) + len(e["unexpected"])
            tree.insert("", "end", iid=str(sm), tags=(row["state"],), values=(
                f"SM {sm}", f"{row['events']:,}", f"{seen_mm}/{len(self.dataset.expected_mm.get(sm, ()))}",
                assessed(t), flags(t), median(t), assessed(e), flags(e), median(e),
                f"{unexpected} ch" if unexpected else "—", row["state"]))
            states[row["state"]] += 1
            channel_states.update(t["states"])
            channel_states.update(e["states"])
        order = ("NOT OBSERVED", "HIGH", "LOW", "INSUFFICIENT EVENTS", "NO DATA", "OK")
        self.channel_summary.configure(text=(
            f"{len(rows)} SuperModules: " + ", ".join(f"{states[s]} {s}" for s in order if states[s])
            + "  •  channels: " + ", ".join(f"{channel_states[s]:,} {s}" for s in order if channel_states[s])
            + f"\n{FINDINGS_POPULATION} (not a dead/hot hardware verdict) • {self.thresholds.text()}"))
        if keep in self._findings:
            tree.selection_set(str(keep))
            self._draw_channel_detail(keep)

    def _select_channel_row(self, _event):
        selection = self.channel_tree.selection()
        if selection and (int(selection[0]) != self._channel_sm or not self.channel_axes):
            self._draw_channel_detail(int(selection[0]))

    def _open_in_supermodule(self):
        if self._channel_sm is None:
            return
        self._open_sm(self._channel_sm)

    def _open_sm(self, sm):
        """Show one SM in the SuperModule tab (from Channel Status or System Overview)."""
        self.module_var.set(f"SM {sm}")
        self._update_step_controls()
        self._stale.add(SM_TAB)
        self._show_tab(SM_TAB)

    def _draw_channel_detail(self, sm):
        """Channel map (time: vertical at fine X; energy: horizontal at fine Y) and bars."""
        self._channel_sm = sm
        self.open_sm_button.configure(state="normal")
        findings = self._findings.get(sm) or channel_findings(self.dataset, sm, self.thresholds)
        geometry = channel_geometry(self.dataset, sm)
        figure = self.channel_fig
        figure.clear()
        grid = figure.add_gridspec(2, 2, width_ratios=(1, 2.9), hspace=0.42, wspace=0.16,
                                   left=0.005, right=0.99, top=0.9, bottom=0.04)
        self.channel_axes = {}
        norm = Normalize(0, 2)
        cmap = matplotlib.colormaps["viridis"]
        flag_lines = []
        for row, kind in enumerate(("time", "energy")):
            found = findings[kind]
            state_of = dict(zip(found["channels"], found["states"]))
            hits_of = dict(zip(found["channels"], found["hits"].tolist()))
            scale = found["median"] if found["median"] else max(max(hits_of.values(), default=0), 1)
            placed = geometry[kind]
            ids = [ch for ch, *_ in placed]
            map_ax = figure.add_subplot(grid[row, 0])
            bar_ax = figure.add_subplot(grid[row, 1])
            self.channel_axes[kind] = (map_ax, bar_ax)

            # map: minimodule boxes (hatched when unpopulated) and one segment per channel
            for mm, info in geometry["minimodules"].items():
                x0, x1, y0, y1 = info["box"]
                map_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=not info["populated"],
                                           facecolor="#e4e6e9", hatch=None if info["populated"] else "//",
                                           edgecolor="#7c8288", lw=0.8))
            if kind == "time":
                segments = [[(pos, lo), (pos, hi)] for _, pos, lo, hi, _ in placed]
            else:
                segments = [[(lo, pos), (hi, pos)] for _, pos, lo, hi, _ in placed]
            flagged = [i for i, ch in enumerate(ids) if state_of.get(ch, "OK") != "OK"]
            if flagged:
                map_ax.add_collection(LineCollection([segments[i] for i in flagged], linewidths=5,
                                                     colors=[STATE_EDGES[state_of[ids[i]]] for i in flagged]))
            lines = LineCollection(segments, linewidths=1.8, cmap=cmap, norm=norm)
            lines.set_array(np.array([hits_of.get(ch, 0) / scale for ch in ids], dtype=float))
            map_ax.add_collection(lines)
            map_ax.set_xlim(0, 103.6)
            map_ax.set_ylim(0, 103.6)
            map_ax.set_aspect("equal")
            map_ax.set_xticks([])
            map_ax.set_yticks([])
            map_ax.set_title(f"{kind.capitalize()} channels at their {'fine X' if kind == 'time' else 'fine Y'}",
                             fontsize=9)
            colorbar = figure.colorbar(lines, cax=map_ax.inset_axes((1.04, 0.0, 0.05, 1.0)), ticks=(0, 1, 2))
            colorbar.ax.set_yticklabels(("0", "median", "≥ 2×"), fontsize=6)
            colorbar.set_label("hits / type median", fontsize=7)

            # bars grouped by minimodule, in map order
            hits = [hits_of.get(ch, 0) for ch in ids]
            bar_ax.bar(np.arange(len(ids)), hits, width=0.85,
                       color=[STATE_EDGES[state_of.get(ch, "OK")] for ch in ids])
            zero = [i for i, n in enumerate(hits) if n == 0]
            if zero:
                bar_ax.plot(zero, [0] * len(zero), "x", color=STATE_EDGES["NOT OBSERVED"], ms=6, mew=1.6)
            boundaries = [i for i in range(1, len(placed)) if placed[i][4] != placed[i - 1][4]]
            for b in boundaries:
                bar_ax.axvline(b - 0.5, color="#b8bcc1", lw=0.7)
            for a, b in zip([0] + boundaries, boundaries + [len(placed)]):
                bar_ax.text((a + b - 1) / 2, 1.0, f"mM{placed[a][4]}", transform=bar_ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=6, color="#555b61")
            if found["median"] is not None:
                bar_ax.axhline(found["median"], color="#303438", lw=1, label=f"median {found['median']:,.0f}")
                bar_ax.axhline(self.thresholds.low_frac * found["median"], color=STATE_EDGES["LOW"], ls="--",
                               lw=1, label=f"low {self.thresholds.low_frac:g} × median")
                bar_ax.axhline(self.thresholds.high_frac * found["median"], color=STATE_EDGES["HIGH"], ls="--",
                               lw=1, label=f"high {self.thresholds.high_frac:g} × median")
                bar_ax.legend(fontsize=7, loc="upper right", ncol=3)
            bar_ax.set_xlim(-0.6, len(ids) - 0.4)
            bar_ax.set_xticks([])
            status = found["insufficient"] or ", ".join(
                f"{found['counts'][s]} {s}" for s in ("NOT OBSERVED", "HIGH", "LOW") if found["counts"][s]) \
                or "all OK"
            bar_ax.set_ylabel("Hits", fontsize=8)
            bar_ax.set_title(f"SM {sm} • {kind} channel hits • {status}", fontsize=9, pad=12)
            for ch in ids:
                state = state_of.get(ch, "OK")
                if state not in ("OK", "INSUFFICIENT EVENTS"):
                    flag_lines.append(f"{kind:<6} {ch:>7} {hits_of.get(ch, 0):>9,}  {state}")
            for ch, n in found["unexpected"].items():
                flag_lines.append(f"{kind:<6} {ch:>7} {n:>9,}  unexpected (unpopulated mM)")
        self.channel_canvas.draw_idle()
        self.channel_flags.delete("1.0", "end")
        header = [f"SM {sm} • {findings['state']}", f"{findings['events']:,} ingest sides",
                  "", f"{'type':<6} {'channel':>7} {'hits':>9}  state"]
        self.channel_flags.insert("end", "\n".join(header + (flag_lines or ["no flagged channels"])))

    def _overview_metrics(self, selection, fits):
        """Cached per-minimodule metrics, or None after starting a background job for them."""
        for entry in self._overview_cache:
            if entry["dataset"] is self.dataset and entry["selection"] == selection and (entry["fits"] or not fits):
                return entry["metrics"]
        job = (id(self.dataset), selection, fits)
        if self._overview_job == job:
            return None
        self._overview_token += 1
        token, dataset = self._overview_token, self.dataset
        self._overview_job = job

        def run():
            import time
            start = time.perf_counter()
            try:
                metrics = minimodule_metrics(
                    dataset, selection, fits=fits,
                    cancelled=lambda: token != self._overview_token or self._abort.is_set())
            except Exception as exc:  # reported on the Tk thread
                self._events.put(("overview", token, dataset, selection, fits, None, str(exc)))
                return
            if metrics is not None:
                self._events.put(("overview", token, dataset, selection, fits, metrics,
                                  time.perf_counter() - start))

        threading.Thread(target=run, daemon=True).start()
        return None

    def _overview_done(self, token, dataset, selection, fits, metrics, detail):
        if token != self._overview_token:
            return  # superseded; the newer job reports instead
        self._overview_job = None
        if metrics is None:
            self._log(f"System Overview metrics unavailable: {detail}")
            return
        self._overview_cache = [{"dataset": dataset, "selection": selection, "fits": fits,
                                 "metrics": metrics}] + self._overview_cache[:3]
        what = "counts and photopeak fits" if fits and dataset.settings.calibrated else "counts"
        self._log(f"System Overview: {what} for {len(metrics):,} minimodules in {detail:.1f} s")
        if dataset is self.dataset:
            self._stale.add("System Overview")  # drawn now if visible, else when shown
            self._draw_visible()

    def _draw_overview(self):
        if not self.dataset:
            return
        mode = self.overview_mode.get()
        if mode == FLOOD_MODE:
            self._draw_overview_floods()
            return
        selection = self._selection()
        field_name, label = OVERVIEW_METRICS[mode]
        fits = field_name in ("mu", "resolution")
        metrics = self._overview_metrics(selection, fits)
        figure = self.overview_fig
        figure.clear()
        axis = figure.add_subplot(111)
        self._overview_ax = axis
        if metrics is None:
            self._overview_grid = None
            axis.text(0.5, 0.5, f"Computing per-minimodule {'photopeak fits' if fits else 'counts'}…",
                      transform=axis.transAxes, ha="center", va="center", fontsize=12, color="#555b61")
            axis.set_axis_off()
            self.overview_canvas.draw_idle()
            return
        grid = overview_grid(self.dataset, metrics, mode)
        self._overview_grid = {**grid, "metrics": metrics, "mode": mode}
        kind, values = grid["kind"], grid["values"]
        background = np.zeros((*kind.shape, 4))
        for code, colour in TILE_COLOURS.items():
            background[kind == code] = to_rgba(colour)
        axis.imshow(background, interpolation="nearest")
        finite = values[np.isfinite(values)]
        if fits and finite.size:
            vmin, vmax = np.percentile(finite, [1, 99])
        else:  # counts: zero is the colormap minimum
            vmin, vmax = 0.0, float(finite.max()) if finite.size else 1.0
        cmap = matplotlib.colormaps["viridis"].copy()
        cmap.set_bad((0, 0, 0, 0))
        image = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-9),
                            interpolation="nearest")
        for (row, col), (sm, mm) in grid["cells"].items():
            if kind[row, col] == TILE_UNPOPULATED:
                axis.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, hatch="////",
                                         edgecolor="#9aa0a6", lw=0))
        for sm, (top, left) in grid["origins"].items():
            axis.text(left - 0.42, top - 0.42, str(sm), fontsize=6, va="top", ha="left", color="white",
                      bbox={"facecolor": "#303438", "alpha": 0.55, "edgecolor": "none", "pad": 0.6})
        rows, cols = grid["sm_shape"]
        mm_rows, mm_cols = grid["mm_shape"]
        cornell = self.dataset.settings.system == "CORNELL"
        axis.set_xticks([c * (mm_cols + 1) + (mm_cols - 1) / 2 for c in range(cols)], [str(c) for c in range(cols)],
                        fontsize=7)
        axis.set_yticks([r * (mm_rows + 1) + (mm_rows - 1) / 2 for r in range(rows)], [str(r) for r in range(rows)],
                        fontsize=7)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set(xlabel="Cassette" if cornell else "Azimuthal SuperModule", ylabel="Ring")
        units = "keV" if self.dataset.settings.calibrated else "raw a.u."
        population = {"ingest": "all accepted sides, before display cuts",
                      "selected": f"paired energy + DOI + ROI cuts ({units})"}.get(
            field_name, "ROI/DOI sides, energy window off; spec 001 fit guards")
        unavailable = int((kind == TILE_UNAVAILABLE).sum())
        note = ""
        if fits and not self.dataset.settings.calibrated:
            note = " • raw a.u.: no keV calibration, fits unavailable"
        elif fits:
            note = f" • {unavailable} unavailable fits"
        axis.set_title(f"{label} per minimodule • {population}{note}", fontsize=10)
        figure.colorbar(image, ax=axis, shrink=0.85, pad=0.01, label=label)
        handles = [Patch(facecolor=TILE_COLOURS[TILE_UNPOPULATED], hatch="////", edgecolor="#9aa0a6",
                         label="unpopulated (config)")]
        if fits:
            handles.append(Patch(facecolor=TILE_COLOURS[TILE_UNAVAILABLE], label="fit unavailable"))
        else:
            handles.append(Patch(facecolor=cmap(0.0), label="0 sides (colormap minimum)"))
        axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.06), ncol=2, fontsize=8,
                    frameon=False)
        if self._overview_pick in grid["cells"].values():
            self._highlight_tile(*self._overview_pick)
        figure.subplots_adjust(left=0.03, right=1.0, top=0.93, bottom=0.12)
        self.overview_canvas.draw_idle()

    def _highlight_tile(self, sm, mm):
        grid = self._overview_grid
        pixel = next(p for p, cell in grid["cells"].items() if cell == (sm, mm))
        for patch in [p for p in self._overview_ax.patches if p.get_gid() == "pick"]:
            patch.remove()
        self._overview_ax.add_patch(Rectangle((pixel[1] - 0.5, pixel[0] - 0.5), 1, 1, fill=False,
                                              edgecolor="#e64c46", lw=2, gid="pick"))

    def _overview_tile_text(self, sm, mm):
        grid = self._overview_grid
        pixel = next(p for p, cell in grid["cells"].items() if cell == (sm, mm))
        kind = grid["kind"][pixel]
        if kind == TILE_UNPOPULATED:
            return f"SM {sm} · mM {mm} · unpopulated in the config (no sensors)"
        row = grid["metrics"].get((sm, mm)) or {}
        parts = [f"SM {sm} · mM {mm}", f"ingest {row.get('ingest', 0):,} sides",
                 f"selected {row.get('selected', 0):,}"]
        fit = row.get("fit")
        if fit is not None:
            parts.append(f"photopeak {fit['mu']:.1f} keV, resolution {fit['resolution']:.1f} %"
                         if fit.get("status") == "FIT" else f"fit: {fit['status']}")
        elif OVERVIEW_METRICS[grid["mode"]][0] not in ("mu", "resolution"):
            parts.append("fits: choose a centroid or resolution view")
        return " · ".join(parts)

    def _overview_click(self, event):
        grid = self._overview_grid
        if grid is None or event.inaxes is not self._overview_ax or event.xdata is None:
            return
        cell = grid["cells"].get((int(round(event.ydata)), int(round(event.xdata))))
        if cell is None:
            return
        self._select_overview_tile(*cell)

    def _select_overview_tile(self, sm, mm):
        self._overview_pick = (sm, mm)
        self.overview_info.configure(text=self._overview_tile_text(sm, mm))
        self.overview_open.configure(state="normal")
        self._highlight_tile(sm, mm)
        self.overview_canvas.draw_idle()

    def _overview_open_sm(self):
        if self._overview_pick is None:
            return
        self._open_sm(self._overview_pick[0])

    def _pair_entry(self, selection):
        """Cached pair mask and SM x SM matrix, or None after starting a background job."""
        for entry in self._pair_cache:
            if entry["dataset"] is self.dataset and entry["selection"] == selection:
                return entry
        job = (id(self.dataset), selection)
        if self._pair_job == job:
            return None
        self._pair_token += 1
        token, dataset = self._pair_token, self.dataset
        self._pair_job = job

        def run():
            import time
            start = time.perf_counter()
            try:
                mask = pair_mask(dataset, selection)
                matrix = pair_matrix(dataset, selection, mask=mask)
            except Exception as exc:  # reported on the Tk thread
                self._events.put(("pairs", token, dataset, selection, None, None, str(exc)))
                return
            self._events.put(("pairs", token, dataset, selection, mask, matrix, time.perf_counter() - start))

        threading.Thread(target=run, daemon=True).start()
        return None

    def _pairs_done(self, token, dataset, selection, mask, matrix, detail):
        if token != self._pair_token:
            return  # superseded or inputs changed
        self._pair_job = None
        entry = {"dataset": dataset, "selection": selection, "mask": mask, "matrix": matrix,
                 "error": None if matrix is not None else detail}
        self._pair_cache = [entry] + self._pair_cache[:1]  # masks are one bool per side: keep two
        if matrix is None:
            self._log(f"Coincidence matrix unavailable: {detail}")
        else:
            self._log(f"Coincidences: {matrix['pairs']:,} of {matrix['ingest_pairs']:,} pairs pass the cuts "
                      f"({detail:.1f} s)")
        self._stale.add(COINC_TAB)
        self._draw_visible()

    def _draw_coincidences(self):
        if not self.dataset:
            return
        entry = self._pair_entry(self._selection())
        figure = self.coinc_fig
        figure.clear()
        self._pair_current = self._matrix_ax = self._dt_ax = None
        if entry is None or entry["error"]:
            axis = figure.add_subplot(111)
            axis.text(0.5, 0.5, "Computing the SM × SM pair matrix…" if entry is None
                      else f"Coincidence matrix unavailable: {entry['error']}", transform=axis.transAxes,
                      ha="center", va="center", fontsize=12, color="#555b61")
            axis.set_axis_off()
            self.coinc_canvas.draw_idle()
            return
        self._pair_current = entry
        matrix = entry["matrix"]
        sms, counts = matrix["sms"], matrix["counts"]
        grid = figure.add_gridspec(1, 2, width_ratios=(1, 1.3), wspace=0.28, left=0.05, right=0.98,
                                   top=0.9, bottom=0.12)
        axis = self._matrix_ax = figure.add_subplot(grid[0, 0])
        self._dt_ax = figure.add_subplot(grid[0, 1])
        cmap = matplotlib.colormaps["viridis"].copy()
        cmap.set_bad("#eceef0")
        top = int(counts.max()) if counts.size else 0
        image = axis.imshow(np.ma.masked_equal(counts, 0), cmap=cmap, norm=LogNorm(1, max(top, 2)),
                            interpolation="nearest")
        step = max(1, -(-len(sms) // 30))
        ticks = list(range(0, len(sms), step))
        axis.set_xticks(ticks, [str(sms[i]) for i in ticks], fontsize=7, rotation=90)
        axis.set_yticks(ticks, [str(sms[i]) for i in ticks], fontsize=7)
        axis.set(xlabel="SM b", ylabel="SM a")
        units = "keV" if self.dataset.settings.calibrated else "raw a.u."
        axis.set_title(f"Accepted pairs, both sides pass the display cuts ({units})\n"
                       f"{matrix['pairs']:,} of {matrix['ingest_pairs']:,} pairs, each counted once • "
                       "grey: 0 pairs", fontsize=9)
        figure.colorbar(image, ax=axis, shrink=0.85, pad=0.02, label="pairs (log scale)")
        self._draw_pair_dt()

    def _pair_sms(self):
        try:
            sm_a = int(self.pair_a.get().removeprefix("SM "))
        except ValueError:
            return None, None
        b = self.pair_b.get()
        return sm_a, None if b == ALL_PARTNERS else int(b.removeprefix("SM "))

    def _draw_pair_dt(self):
        """Δt histogram of the chosen pair (or SM a against all partners), from the cached mask."""
        entry = self._pair_current
        if entry is None or entry["dataset"] is not self.dataset or self._dt_ax is None:
            return
        sm_a, sm_b = self._pair_sms()
        axis = self._dt_ax
        axis.clear()
        for patch in [p for p in self._matrix_ax.patches if p.get_gid() == "pick"]:
            patch.remove()
        if sm_a is None:
            self.coinc_canvas.draw_idle()
            return
        result = pair_dt(self.dataset, sm_a, sm_b, mask=entry["mask"])
        partner = "all partners" if sm_b is None else f"SM {sm_b}"
        if result["count"]:
            dt = result["dt_ns"]
            # core ± 4 central-68 % widths, within the data; pairs outside are counted, not hidden
            width = result["width68"] or 1.0
            low = max(float(dt.min()), result["p16"] - 4 * width)
            high = min(float(dt.max()), result["p84"] + 4 * width)
            if high <= low:
                low, high = low - 1.0, high + 1.0
            outside = int(((dt < low) | (dt > high)).sum())
            axis.hist(dt, bins=int(self.dt_bins.get()), range=(low, high), color="#347dc1", alpha=0.8)
            if outside:
                axis.text(0.02, 0.97, f"{outside:,} pairs outside the plotted range\n"
                          f"(all {dt.min():+.1f} to {dt.max():+.1f} ns; statistics use every pair)",
                          transform=axis.transAxes, ha="left", va="top", fontsize=8, color="#555b61")
            axis.axvline(result["median"], color="#e64c46", lw=1.6, label=f"median {result['median']:+.2f} ns")
            axis.axvspan(result["p16"], result["p84"], color="#e64c46", alpha=0.08,
                         label=f"central 68 %: {result['width68']:.2f} ns wide")
            axis.legend(fontsize=8, loc="upper right")
            text = (f"SM {sm_a} ↔ {partner}: {result['count']:,} pairs • median {result['median']:+.2f} ns • "
                    f"central 68 % width {result['width68']:.2f} ns")
        else:
            axis.text(0.5, 0.5, "No pairs pass the cuts", transform=axis.transAxes, ha="center", color="#555b61")
            text = f"SM {sm_a} ↔ {partner}: no pairs pass the cuts"
        same = " (both sides in one SM: sign arbitrary)" if sm_b == sm_a else ""
        axis.set(xlabel="t(SM a) − t(partner) (ns; PETsys ps timestamps)", ylabel="Pairs")
        axis.set_title(f"Paired hit time difference • SM {sm_a} − {partner}{same}\n"
                       "geometry and time of flight contribute; not a clock or CTR calibration", fontsize=9)
        self.pair_info.configure(text=text)
        sms = entry["matrix"]["sms"]
        if sm_a in sms:
            row = sms.index(sm_a)
            cells = [(row, sms.index(sm_b)), (sms.index(sm_b), row)] if sm_b in sms else []
            boxes = [((col - 0.5, r - 0.5), 1, 1) for r, col in cells] or [((-0.5, row - 0.5), len(sms), 1)]
            for origin, width, height in boxes:
                self._matrix_ax.add_patch(Rectangle(origin, width, height, fill=False, edgecolor="#e64c46",
                                                    lw=1.6, gid="pick"))
        self.coinc_canvas.draw_idle()

    def _pair_click(self, event):
        entry = self._pair_current
        if entry is None or event.inaxes is not self._matrix_ax or event.xdata is None:
            return
        sms = entry["matrix"]["sms"]
        row, col = int(round(event.ydata)), int(round(event.xdata))
        if 0 <= row < len(sms) and 0 <= col < len(sms):
            self.pair_a.set(f"SM {sms[row]}")
            self.pair_b.set(f"SM {sms[col]}")
            self._draw_pair_dt()

    def _draw_overview_floods(self):
        """Spec 001 per-SM flood maps, placed by ``supermodule_layout``."""
        rows, cols, placement = supermodule_layout(self.dataset)
        self._overview_grid = self._overview_ax = None
        self.overview_fig.clear()
        axes = self.overview_fig.subplots(rows, cols, squeeze=False)
        selection = self._selection()
        for sm, (row, col) in placement.items():
            axis = axes[row, col]
            data = self.dataset.modules.get(sm)
            if data is not None:
                mask = selection.mask(data)
                if mask.any():
                    counts, xedges, yedges = flood_counts(data.x[mask], data.y[mask], 28)
                    cmap = matplotlib.colormaps["plasma"].copy()
                    cmap.set_bad("white")
                    axis.pcolormesh(xedges, yedges, counts, cmap=cmap, vmin=0.1)
            axis.text(0.03, 0.97, str(sm), transform=axis.transAxes, va="top", fontsize=7,
                      bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
        self.overview_fig.subplots_adjust(left=0.01, right=0.99, top=0.98,
                                           bottom=0.02, wspace=0.08, hspace=0.07)
        self.overview_canvas.draw_idle()

    def _draw_timestamps(self):
        if not self.dataset:
            return
        self.time_fig.clear()
        axis = self.time_fig.add_subplot(111)
        try:
            index = int(self.time_file.get().split(":", 1)[0])
        except ValueError:
            return
        view = self.time_view.get()
        plotted = 0
        if view == "Event rate":
            for sm in sorted(self.dataset.modules):
                rate = rate_series(self.dataset, sm, index, bins=int(self.time_bins.get()))
                if rate is not None:
                    edges, values = rate
                    axis.plot(edges[:-1], values, lw=0.8, alpha=0.65,
                              label=f"SM {sm}" if len(self.dataset.modules) < 9 else None)
                    plotted += 1
            if plotted and len(self.dataset.modules) < 9:
                axis.legend(fontsize=8)
            axis.set(ylabel="Observed detector sides / s",
                     title=f"Within-file coincidence activity • {self.time_file.get()}")
        else:
            rows = []
            ids = []
            edges = None
            for sm in sorted(self.dataset.modules):
                series = pair_offset_series(self.dataset, sm, index, bins=int(self.time_bins.get()))
                if series is not None:
                    edges, medians = series
                    rows.append(medians)
                    ids.append(sm)
            if rows:
                matrix = np.asarray(rows)
                valid = np.abs(matrix[np.isfinite(matrix)])
                limit = max(float(np.percentile(valid, 98)), 1) if valid.size else 1
                image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="coolwarm",
                                    vmin=-limit, vmax=limit, origin="lower",
                                    extent=[edges[0], edges[-1], -0.5, len(ids) - 0.5])
                axis.set_yticks(range(len(ids)))
                axis.set_yticklabels([str(sm) if len(ids) < 35 or i % 4 == 0 else ""
                                      for i, sm in enumerate(ids)], fontsize=7)
                self.time_fig.colorbar(image, ax=axis, label="Median local - paired hit time (ns)")
                plotted = len(ids)
            axis.set(ylabel="SuperModule", title="Paired coincidence hit time difference (not clock drift)")
        if plotted == 0:
            axis.text(0.5, 0.5, "Insufficient valid timestamps in this file",
                      transform=axis.transAxes, ha="center")
        axis.set_xlabel("Time from first observed file event (s; PETsys ps stamps)")
        axis.grid(alpha=0.2)
        self.time_fig.tight_layout()
        self.time_canvas.draw_idle()

    def _clear_summary(self, text):
        self.sm_state.configure(text=text, fg_color="transparent", text_color=("gray10", "gray90"))
        self.sm_summary.delete("1.0", "end")
        self.mm_tree.delete(*self.mm_tree.get_children())
        self.mm_title.configure(text="Per-minimodule counts and photopeak")

    def _draw_summary(self, sm, selection, selected):
        """Occupancy and channel findings of one SM (spec 001's Status tab content), then its mM table."""
        dataset = self.dataset
        data = dataset.modules.get(sm)
        ingest = len(data) if data is not None else 0
        total = sum(len(m) for m in dataset.modules.values())
        findings = self._findings.get(sm) or channel_findings(dataset, sm, self.thresholds)
        state = findings["state"]
        self.sm_state.configure(text=f"SM {sm} • {dataset.settings.system} • {state}",
                                fg_color=FINDING_COLOURS.get(state, "transparent"), text_color="black")
        units = "keV" if dataset.settings.calibrated else "raw a.u."
        expected = dataset.expected_mm.get(sm, set())
        seen = set(np.unique(data.mm).tolist()) if ingest else set()
        share = f"  ({100 * ingest / total:.1f} % of system)" if total else ""
        lines = ["OCCUPANCY",
                 f"  Ingest sides    {ingest:>11,}{share}",
                 f"  Selected sides  {selected:>11,}  (energy + DOI + ROI, {units})",
                 f"  Minimodules seen {len(seen & expected)}/{len(expected)} expected"
                 + (f"; {len(seen - expected)} unpopulated with sides" if seen - expected else ""),
                 "", "CHANNEL FINDINGS",
                 f"  {FINDINGS_POPULATION}, before display cuts"]
        for kind in ("time", "energy"):
            found = findings[kind]
            observed = int((found["hits"] > 0).sum())
            lines.append(f"  {kind.upper():<6} {observed}/{len(found['channels'])} observed"
                         + ("" if found["median"] is None else f", median {found['median']:,.0f} hits"))
            if found["insufficient"]:
                lines.append(f"         insufficient: {found['insufficient']}")
                groups = (("0 hits", [ch for ch, n in zip(found["channels"], found["hits"]) if n == 0]),)
            else:
                counts = found["counts"]
                lines.append(f"         {counts['NOT OBSERVED']} not observed / {counts['LOW']} low / "
                             f"{counts['HIGH']} high")
                groups = [(label.lower(), [ch for ch, s in zip(found["channels"], found["states"]) if s == label])
                          for label in ("NOT OBSERVED", "LOW", "HIGH")]
            for label, ids in groups:
                if ids:
                    lines.append(f"         {label}: {', '.join(map(str, ids))}")
            if found["unexpected"]:
                lines.append("         unexpected hits (unpopulated mM): " + ", ".join(
                    f"{ch} ({n:,})" for ch, n in found["unexpected"].items()))
        lines += [f"  {self.thresholds.text()}",
                  "  Observational: no channel is declared dead or hot from a coincidence sample."]
        self.sm_summary.delete("1.0", "end")
        self.sm_summary.insert("end", "\n".join(lines))
        self._draw_mm_table(sm, selection)

    def _layout_populated(self, sm):
        if self._mm_layout is None:
            self._mm_layout = minimodule_layout(self.dataset)
        return self._mm_layout.get(sm, {}).get("populated", {})

    def _draw_mm_table(self, sm, selection):
        tree = self.mm_tree
        tree.delete(*tree.get_children())
        entry = self._sm_metrics(sm, selection)
        calibrated = self.dataset.settings.calibrated
        scope = ("fits on ROI/DOI sides, energy window off" if calibrated
                 else "fits unavailable: raw a.u. (no keV calibration)")
        if entry is None:
            self.mm_title.configure(text=f"Per-minimodule counts and photopeak • computing… • {scope}")
            return
        if entry.get("error"):
            self.mm_title.configure(text=f"Per-minimodule table unavailable: {entry['error']}")
            return
        metrics = entry["metrics"]
        populated = self._layout_populated(sm)
        mms = sorted(set(populated) | {mm for s, mm in metrics if s == sm})
        fitted = 0
        for mm in mms:
            row = metrics.get((sm, mm))
            ingest = f"{row['ingest']:,}" if row else "0"
            selected = f"{row['selected']:,}" if row else "0"
            fit = row["fit"] if row else None
            if not populated.get(mm, True):
                values, tag = (mm, ingest, selected, "—", "—", "unpopulated (config)"), "unpopulated"
            elif fit is not None and fit["status"] == "FIT":
                values, tag = (mm, ingest, selected, f"{fit['mu']:.1f}", f"{fit['resolution']:.1f}", "FIT"), "fit"
                fitted += 1
            else:
                status = fit["status"] if fit is not None else "not computed"
                values, tag = (mm, ingest, selected, "—", "—", status), "unavailable"
            tree.insert("", "end", iid=str(mm), values=values, tags=(tag,))
        assessed = sum(1 for mm in mms if populated.get(mm, True))
        self.mm_title.configure(text=f"Per-minimodule counts and photopeak • {fitted}/{assessed} fitted • {scope}"
                                if calibrated else f"Per-minimodule counts • {scope}")

    def _select_mm_row(self, _event):
        """Show the full fit status of the chosen minimodule (the column is narrow)."""
        chosen = self.mm_tree.selection()
        if chosen:
            values = self.mm_tree.item(chosen[0], "values")
            self.mm_title.configure(text=f"SM {self._selected_sm()} · mM {values[0]} · ingest {values[1]} · "
                                         f"selected {values[2]} · fit: {values[5]}")

    def _sm_metrics(self, sm, selection):
        """One SM's minimodule metrics (counts + fits): a cache entry, or None while a job runs.

        A System Overview result with fits for the same selection is reused.
        """
        for entry in self._overview_cache:
            if entry["dataset"] is self.dataset and entry["selection"] == selection and entry["fits"]:
                return entry
        for entry in self._sm_cache:
            if entry["dataset"] is self.dataset and entry["selection"] == selection and entry["sm"] == sm:
                return entry
        self._sm_wanted = (self.dataset, selection, sm)
        if self._sm_job is None:
            self._start_sm_job(*self._sm_wanted)
        return None

    def _start_sm_job(self, dataset, selection, sm):
        self._sm_job = (dataset, selection, sm)

        def run():
            try:
                metrics = minimodule_metrics(dataset, selection, sms=[sm])
            except Exception as exc:  # reported on the Tk thread
                self._events.put(("sm_metrics", dataset, selection, sm, None, str(exc)))
                return
            self._events.put(("sm_metrics", dataset, selection, sm, metrics, None))

        threading.Thread(target=run, daemon=True).start()

    def _sm_metrics_done(self, dataset, selection, sm, metrics, error):
        self._sm_job = None
        if dataset is not self.dataset:
            return  # inputs changed while it ran
        self._sm_cache = [{"dataset": dataset, "selection": selection, "sm": sm, "metrics": metrics,
                           "error": error}] + self._sm_cache[:31]
        if error:
            self._log(f"SM {sm} minimodule metrics unavailable: {error}")
        wanted = self._sm_wanted
        if wanted is None or wanted[0] is not self.dataset:
            return
        if (wanted[1], wanted[2]) != (selection, sm):
            self._start_sm_job(*wanted)  # the operator moved on while it ran
            return
        self._sm_wanted = None
        if self._selected_sm() == sm and SM_TAB not in self._stale:
            self._draw_mm_table(sm, selection)

    def _uniformity(self):
        if not self.dataset:
            return
        window = ctk.CTkToplevel(self)
        window.title("Photopeak Uniformity • calibrated keV" if self.dataset.settings.calibrated
                     else "Photopeak Uniformity • unavailable in raw a.u. mode")
        window.geometry("900x600")
        controls = ctk.CTkFrame(window)
        controls.pack(fill="x", padx=8, pady=7)
        _label(controls, "Target keV")
        _entry(controls, self.target, 85)
        _label(controls, "Tolerance %")
        _entry(controls, self.tolerance, 85)
        tree = ttk.Treeview(window, columns=("sm", "events", "peak", "res", "dev", "status"), show="headings")
        for col, heading in (("sm", "SM"), ("events", "Sides"), ("peak", "Centroid (keV)"),
                             ("res", "Resolution %"), ("dev", "Deviation %"), ("status", "Result")):
            tree.heading(col, text=heading)
            tree.column(col, width=120, anchor="center")
        tree.tag_configure("in_tolerance", background="#d3f0d8", foreground="#14552a")
        tree.tag_configure("out_of_tolerance", background="#fbd2c3", foreground="#752615")
        tree.tag_configure("unavailable", background="#e3e5e8", foreground="#424952")
        tree.pack(fill="both", expand=True, padx=10, pady=8)
        legend = ctk.CTkFrame(window, fg_color="transparent")
        legend.pack(fill="x", padx=10, pady=(0, 7))
        for text, colour in (("In tolerance", "#369b54"),
                             ("Out of tolerance", "#d76540"),
                             ("Unavailable fit", "#7e858d")):
            ctk.CTkLabel(legend, text=f"● {text}", text_color=colour).pack(side="left", padx=(4, 16))

        def fill():
            try:
                rows = uniformity(self.dataset, self._selection(), float(self.target.get()),
                                  float(self.tolerance.get()))
            except ValueError as exc:
                messagebox.showerror("Uniformity", str(exc))
                return
            tree.delete(*tree.get_children())
            for row in rows:
                fit = row["fit"]
                tag = {"IN TOLERANCE": "in_tolerance",
                       "OUT OF TOLERANCE": "out_of_tolerance",
                       "UNAVAILABLE": "unavailable"}[row["result"]]
                tree.insert("", "end", values=(row["sm"], row["events"],
                    f"{fit['mu']:.1f}" if fit["mu"] is not None else "—",
                    f"{fit['resolution']:.1f}" if fit["resolution"] is not None else "—",
                    f"{row['deviation_pct']:+.1f}" if row["deviation_pct"] is not None else "—",
                    row["result"] if fit["status"] == "FIT" else fit["status"]), tags=(tag,))
        ctk.CTkButton(controls, text="Measure", command=fill, width=95).pack(side="right", padx=10)
        fill()
        self._raise_dialog(window)

    def _report(self, sm):
        if not self.dataset or self._report_busy or self._busy:
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF report", "*.pdf")],
                                            initialfile=f"LDAT_{self.dataset.settings.system}_"
                                            f"{'system' if sm is None else 'SM'+str(sm)}.pdf")
        if not path:
            return
        try:
            selection = self._selection()
            target, tolerance = float(self.target.get()), float(self.tolerance.get())
        except ValueError as exc:
            messagebox.showerror("Report", str(exc))
            return
        self._report_busy = True
        self.process_button.configure(state="disabled")
        self.calib_switch.configure(state="disabled")
        for button in (self.system_report, self.module_report):
            button.configure(state="disabled")
        dataset = self.dataset
        self._log(f"Writing {'system' if sm is None else 'SM '+str(sm)} PDF in background...")

        def run():
            try:
                from src.ldat_report import write_report
                write_report(path, dataset, selection, sm=sm,
                             target=target, tolerance_pct=tolerance)
                self._events.put(("report", path))
            except Exception as exc:
                self._events.put(("report_error", str(exc)))

        threading.Thread(target=run, daemon=True).start()

    def _open_report(self):
        if self.last_report and Path(self.last_report).exists():
            os.startfile(self.last_report)

    def _close(self):
        """Close without waiting for whole-file workers: they stop at their next chunk."""
        self._abort.set()
        if self._cancel is not None:
            self._cancel.set()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        self.destroy()


def main():
    app = LDATWorkbench()
    app.mainloop()


if __name__ == "__main__":
    main()
