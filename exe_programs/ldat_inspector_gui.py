"""RAWInspector-inspired, PETsys-specific offline LDAT workbench."""

from __future__ import annotations

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
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import numpy as np

from src.ldat_fastread import init_worker
from src.ldat_inspector import (
    FileResult, Selection, Settings, apply_calibration, channel_status, fit_peak,
    fit_on_display_bins, fit_peak_background, flood_counts, load_setup, merge_results, process_file, rate_series,
    pair_offset_series, uniformity,
)
from src.ldat_memory import estimate_memory


__version__ = "0.1.0"
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def _entry(parent, variable, width=74):
    box = ctk.CTkEntry(parent, width=width, textvariable=variable)
    box.pack(side="left", padx=(3, 9))
    return box


def _label(parent, text):
    ctk.CTkLabel(parent, text=text).pack(side="left", padx=(6, 0))


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
        self.overview_mode = tk.StringVar(value="Counts")
        self.time_file = tk.StringVar()
        self.time_view = tk.StringVar(value="Event rate")
        self.time_bins = tk.StringVar(value="60")
        self.target = tk.StringVar(value="511")
        self.tolerance = tk.StringVar(value="10")

        self._build_top()
        self._build_tabs()
        self._log("Select an IMAS or Cornell config and LDAT files; calibration can be applied later.")
        self.bind("<Control-o>", lambda _: self._select_files())
        self.bind("<Control-p>", lambda _: self._start_processing())
        self.bind("<Escape>", lambda _: self._reset_filters())
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
        self.tabs = ctk.CTkTabview(self, corner_radius=7)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=5)
        for name in ("Channel Status", "SuperModule Explorer", "System Overview", "Timestamps", "SuperModule Status"):
            self.tabs.add(name)
        self._build_channels()
        self._build_explorer()
        self._build_overview()
        self._build_timestamps()
        self._build_status()

    def _build_channels(self):
        tab = self.tabs.tab("Channel Status")
        ctk.CTkLabel(tab, text="PETsys SuperModule summary • coincidence detector sides; mapped channel/minimodule coverage uses the ingest population",
                      anchor="w").pack(fill="x", padx=9, pady=5)
        columns = ("sm", "events", "selected", "mm", "time", "energy", "median", "doi", "state")
        self.channel_tree = ttk.Treeview(tab, columns=columns, show="headings", height=12)
        for column, heading, width in zip(columns,
                ("SM", "Ingest sides", "Selected sides", "mM seen/mapped", "Time seen/mapped",
                 "Energy seen/mapped", "Median energy", "Median DOI ratio", "Finding"),
                (60, 115, 115, 115, 120, 130, 130, 125, 150)):
            self.channel_tree.heading(column, text=heading)
            self.channel_tree.column(column, width=width, anchor="center")
        self.channel_tree.tag_configure("warn", background="#ffe7ba")
        self.channel_tree.tag_configure("ok", background="#d8eedb")
        self.channel_tree.tag_configure("none", background="#dce0e4")
        frame = ctk.CTkFrame(tab)
        frame.pack(fill="x", padx=9)
        self.channel_tree.pack(in_=frame, side="left", fill="x", expand=True)
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.channel_tree.yview)
        bar.pack(side="right", fill="y")
        self.channel_tree.configure(yscrollcommand=bar.set)
        self.channel_tree.bind("<<TreeviewSelect>>", self._select_channel_row)
        self.channel_fig = Figure(figsize=(13, 3.5), dpi=100)
        self.channel_ax = self.channel_fig.add_subplot(111)
        self.channel_canvas = FigureCanvasTkAgg(self.channel_fig, master=tab)
        self.channel_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=6)

    def _build_explorer(self):
        tab = self.tabs.tab("SuperModule Explorer")
        toolbar = ctk.CTkFrame(tab)
        toolbar.pack(fill="x", padx=5, pady=(4, 2))
        _label(toolbar, "SM")
        self.module_combo = ctk.CTkComboBox(toolbar, variable=self.module_var,
                                            values=["—"], width=110, state="readonly",
                                            command=lambda _: self._refresh_all())
        self.module_combo.pack(side="left", padx=4)
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
        self.fig = Figure(figsize=(16, 6), dpi=100)
        self.energy_ax, self.doi_ax, self.flood_ax = self.fig.subplots(1, 3)
        self.fig.subplots_adjust(left=0.045, right=0.93, bottom=0.12, top=0.91, wspace=0.28)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_overview(self):
        tab = self.tabs.tab("System Overview")
        controls = ctk.CTkFrame(tab)
        controls.pack(fill="x", padx=8, pady=5)
        ctk.CTkLabel(controls, text="Geometry-aware IMAS / Cornell overview").pack(side="left", padx=8)
        ctk.CTkComboBox(controls, variable=self.overview_mode,
                        values=["Counts", "Flood maps"], state="readonly", width=115,
                        command=lambda _: self._draw_overview()).pack(side="right", padx=8)
        self.overview_fig = Figure(figsize=(15, 5), dpi=100)
        self.overview_canvas = FigureCanvasTkAgg(self.overview_fig, master=tab)
        self.overview_canvas.get_tk_widget().pack(fill="both", expand=True)

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

    def _build_status(self):
        tab = self.tabs.tab("SuperModule Status")
        self.detail = ctk.CTkTextbox(tab, font=("Consolas", 12))
        self.detail.pack(fill="both", expand=True, padx=10, pady=10)

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
        if self._selector is not None:
            self._selector.set_active(False)
            self._selector.disconnect_events()
            self._selector = None
        for figure, canvas in ((self.fig, self.canvas), (self.channel_fig, self.channel_canvas),
                               (self.overview_fig, self.overview_canvas), (self.time_fig, self.time_canvas)):
            figure.clear()
            canvas.draw_idle()
        self._colorbar = None
        self._close_residuals()
        self.energy_ax, self.doi_ax, self.flood_ax = self.fig.subplots(1, 3)
        self.channel_ax = self.channel_fig.add_subplot(111)
        self.detail.delete("1.0", "end")
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
        labels = [f"{f.index}: {Path(f.path).name}" for f in successful]
        self.time_combo.configure(values=labels or ["—"])
        self.time_file.set(labels[0] if labels else "")
        self._sync_sliders()
        for button in (self.uniformity_button, self.system_report, self.module_report):
            button.configure(state="normal" if dataset.modules else "disabled")
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
        try:
            self._selection()
            if self.experimental.get():
                self._experimental_settings()
            count = int(self.bins.get())
            if not 10 <= count <= 500:
                raise ValueError("Flood bins must be between 10 and 500")
            self._colour_range()
            self._sync_sliders()
            self._refresh_selected()
            self._draw_channels()
            self._draw_overview()
            self._draw_timestamps()
        except (ValueError, OverflowError) as exc:
            messagebox.showerror("Filter", str(exc))

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
            self._draw_detail()
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
        self._draw_detail()

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

    def _draw_channels(self):
        if not self.dataset:
            return
        tree = self.channel_tree
        tree.delete(*tree.get_children())
        rows, selected_rows = [], []
        selection = self._selection()
        unit = "keV" if self.dataset.settings.calibrated else "a.u."
        tree.heading("median", text=f"Median energy ({unit})")
        for sm in sorted(set(self.dataset.expected_time) | set(self.dataset.modules)):
            status = channel_status(self.dataset, sm)
            rows.append((sm, status["events"]))
            data = self.dataset.modules.get(sm)
            selected = selection.mask(data) if data is not None else np.zeros(0, dtype=bool)
            count = int(selected.sum())
            selected_rows.append(count)
            mm_count = len(np.unique(data.mm)) if data is not None else 0
            valid_energy = data.energy[selected & np.isfinite(data.energy)] if data is not None else []
            median_energy = f"{np.median(valid_energy):.1f}" if len(valid_energy) else "—"
            median_doi = f"{np.median(data.doi[selected]):.3f}" if count else "—"
            tag = "none" if status["events"] == 0 else "warn" if status["state"] != "OBSERVED" else "ok"
            tree.insert("", "end", iid=str(sm), values=(f"SM {sm}", f"{status['events']:,}", f"{count:,}",
                        f"{mm_count}/{len(self.dataset.expected_mm.get(sm, set()))}",
                        f"{len(status['active_time'])}/{len(status['expected_time'])}",
                        f"{len(status['active_energy'])}/{len(status['expected_energy'])}",
                        median_energy, median_doi, status["state"]), tags=(tag,))
        self.channel_ax.clear()
        if rows:
            ids, totals = zip(*rows)
            self.channel_ax.bar(ids, totals, color="#5188c4", label="Ingested")
            self.channel_ax.bar(ids, selected_rows, color="#48aa76", label="Selected")
            self.channel_ax.legend(fontsize=8)
        self.channel_ax.set(xlabel="SuperModule", ylabel="Coincidence detector sides",
                            title="PETsys SuperModule participation")
        self.channel_ax.grid(axis="y", alpha=0.25)
        self.channel_fig.tight_layout()
        self.channel_canvas.draw_idle()

    def _select_channel_row(self, _event):
        selection = self.channel_tree.selection()
        if selection:
            self.module_var.set(f"SM {selection[0]}")
            self.tabs.set("SuperModule Explorer")
            self._refresh_all()

    def _layout(self):
        config = self.dataset.config
        if self.dataset.settings.system == "CORNELL":
            rings = len(config.get("ring_z") or [0, 1, 2])
            ncols = max(len(config.get("ring_yx") or {}), 1)
            return rings, ncols, lambda sm: (sm % rings, sm // rings)
        ncols = max(len(config.get("ring_yx") or {}), 1)
        rings = max(len(config.get("ring_z") or []), 1)
        return rings, ncols, lambda sm: (sm // ncols, sm % ncols)

    def _draw_overview(self):
        if not self.dataset:
            return
        rows, cols, locate = self._layout()
        self.overview_fig.clear()
        selected = set(self.dataset.expected_time) | set(self.dataset.modules)
        if self.overview_mode.get() == "Counts":
            matrix = np.full((rows, cols), np.nan)
            for sm in selected:
                row, col = locate(sm)
                if row < rows and col < cols:
                    matrix[row, col] = len(self.dataset.modules[sm]) if sm in self.dataset.modules else 0
            axis = self.overview_fig.add_subplot(111)
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#b8bcc1")
            image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap)
            axis.set(xlabel="Cassette" if self.dataset.settings.system == "CORNELL" else "Azimuthal SuperModule",
                     ylabel="Ring", title="Observed detector sides per mapped SuperModule")
            self.overview_fig.colorbar(image, ax=axis, shrink=0.85, label="Detector sides")
        else:
            axes = self.overview_fig.subplots(rows, cols, squeeze=False)
            selection = self._selection()
            for sm in selected:
                row, col = locate(sm)
                if row >= rows or col >= cols:
                    continue
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

    def _draw_detail(self):
        if not self.dataset:
            return
        sm = self._selected_sm()
        if sm is None:
            return
        status = channel_status(self.dataset, sm)
        lines = [f"SUPERMODULE {sm} • {self.dataset.settings.system}", "",
                 f"Ingested coincidence detector sides: {status['events']:,}",
                 "Channel occupancy uses the ingest population (before display energy/DOI/ROI cuts).",
                 f"Finding: {status['state']} (minimum 100 sides for an observational finding)", "",
                 f"TIME CHANNELS: {len(status['active_time'])}/{len(status['expected_time'])} observed",
                 f"Not observed: {sorted(status['unobserved_time'])}", "",
                 f"ENERGY CHANNELS: {len(status['active_energy'])}/{len(status['expected_energy'])} observed",
                 f"Not observed: {sorted(status['unobserved_energy'])}", "",
                 "No channel is declared dead from an unobserved coincidence sample."]
        data = self.dataset.modules.get(sm)
        if data is not None:
            from collections import Counter
            counts = Counter(data.mm.tolist())
            lines.extend(("", "MINIMODULE OCCUPANCY:",
                          ", ".join(f"mM{mm}: {n:,}" for mm, n in sorted(counts.items()))))
        self.detail.delete("1.0", "end")
        self.detail.insert("end", "\n".join(lines))

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
