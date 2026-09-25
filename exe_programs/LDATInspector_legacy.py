#!/usr/bin/env python3
"""
Simple LDAT Inspector GUI for PETsys data processing
This GUI provides a simplified interface for processing .ldat files without acquisition or calibration components.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import yaml
from collections import defaultdict
import sys
from datetime import datetime

# Import PETsys processing modules
from src.read_compact import read_binary_file
from src.mapping_generator import map_factory, ChannelType
from src.detector_features import (
    calculate_centroid,
    calculate_DOI,
)
from src.filters import filter_min_ch, filter_total_energy
from src.utils import (
    get_maxEnergy_sm_mM,
    get_max_en_channel,
    KevConverter,
    get_slab_cornell,
)

CORNELL_SM_MM_INACTIVE = {sm: (2, 3, 6, 7, 10, 11, 14, 15) for sm in range(20, 30)}


def process_single_file(args):
    """
    Process a single LDAT file - this function will run in separate processes
    """
    (
        file_path,
        config_path,
        slab_energy_path,
        max_events,
        min_ch,
        en_min_ch,
        en_min,
        en_max,
        system_type,
    ) = args

    try:
        # Load configuration for this process
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Load mapping for this process
        map_file = config["map_file"]
        local_coord_dict, sm_mM_map, chtype_map, FEM_instance = map_factory(map_file)

        # Load energy calibration for this process
        if system_type == "IMAS":
            kev_converter = KevConverter(slab_energy_path, file_type="mu")
        elif system_type == "CORNELL":
            kev_converter = KevConverter(slab_energy_path, file_type="cornell")

        # Process this file
        sm_data = defaultdict(list)
        reader = read_binary_file(file_path, en_min_ch)

        event_count = 0
        filtered_count = 0

        for event in reader:
            if event_count >= max_events:
                break

            det1, det2 = event

            # Apply minimum channel filters first
            min_ch_filter1 = filter_min_ch(
                det1, min_ch, chtype_map, FEM_instance.sum_rows_cols
            )
            min_ch_filter2 = filter_min_ch(
                det2, min_ch, chtype_map, FEM_instance.sum_rows_cols
            )

            if not (min_ch_filter1 and min_ch_filter2):
                event_count += 1
                continue

            # Get minimodule with maximum energy for each detector
            max_det1, energy_det1 = get_maxEnergy_sm_mM(det1, sm_mM_map, chtype_map)
            max_det2, energy_det2 = get_maxEnergy_sm_mM(det2, sm_mM_map, chtype_map)

            # Apply minimum channel filter to max energy minimodules
            min_ch_maxdet1 = filter_min_ch(
                max_det1, min_ch, chtype_map, FEM_instance.sum_rows_cols
            )
            min_ch_maxdet2 = filter_min_ch(
                max_det2, min_ch, chtype_map, FEM_instance.sum_rows_cols
            )

            if not (min_ch_maxdet1 and min_ch_maxdet2):
                event_count += 1
                continue

            # Get slab/time channel IDs
            time_info_det1 = get_max_en_channel(max_det1, chtype_map, ChannelType.TIME)
            time_info_det2 = get_max_en_channel(max_det2, chtype_map, ChannelType.TIME)
            tch_det1 = time_info_det1[2]
            tch_det2 = time_info_det2[2]

            # Get SM IDs from the slab mapping
            sm_det1 = sm_mM_map[tch_det1][0]
            sm_det2 = sm_mM_map[tch_det2][0]
            mm_det1 = sm_mM_map[tch_det1][1]
            mm_det2 = sm_mM_map[tch_det2][1]

            if system_type == "CORNELL":
                slab_id_det1, int_flag1, x_det1 = get_slab_cornell(
                    max_det1, chtype_map, local_coord_dict
                )
                slab_id_det2, int_flag2, x_det2 = get_slab_cornell(
                    max_det2, chtype_map, local_coord_dict
                )

            # Convert energy to keV using calibration
            try:
                if system_type == "IMAS":
                    energy_det1_kev = kev_converter.convert(tch_det1, energy_det1)
                    energy_det2_kev = kev_converter.convert(tch_det2, energy_det2)
                elif system_type == "CORNELL":
                    energy_det1_kev = kev_converter.convert(
                        (tch_det1, slab_id_det1), energy_det1
                    )
                    energy_det2_kev = kev_converter.convert(
                        (tch_det2, slab_id_det2), energy_det2
                    )
            except (KeyError, TypeError):
                event_count += 1
                continue

            # Apply energy filters in keV
            en_filter1 = filter_total_energy(energy_det1_kev, en_min, en_max)
            en_filter2 = filter_total_energy(energy_det2_kev, en_min, en_max)

            if not (en_filter1 and en_filter2):
                event_count += 1
                continue

            # Calculate positions and DOI
            x_det1, y_det1 = calculate_centroid(
                max_det1, local_coord_dict, 1, 2, chtype_map
            )
            x_det2, y_det2 = calculate_centroid(
                max_det2, local_coord_dict, 1, 2, chtype_map
            )

            doi_det1 = calculate_DOI(
                max_det1,
                local_coord_dict,
                FEM_instance.sum_rows_cols,
                chtype_map,
            )
            doi_det2 = calculate_DOI(
                max_det2,
                local_coord_dict,
                FEM_instance.sum_rows_cols,
                chtype_map,
            )

            tch_list_det1 = []
            tch_list_det2 = []
            ech_list_det1 = []
            ech_list_det2 = []
            for ch in max_det1:
                chan_id = ch[2]
                if ChannelType.TIME in chtype_map[chan_id]:
                    tch_list_det1.append(chan_id)
                if ChannelType.ENERGY in chtype_map[chan_id]:
                    ech_list_det1.append(chan_id)
            for ch in max_det2:
                chan_id = ch[2]
                if ChannelType.TIME in chtype_map[chan_id]:
                    tch_list_det2.append(chan_id)
                if ChannelType.ENERGY in chtype_map[chan_id]:
                    ech_list_det2.append(chan_id)

            # Store data by SM
            sm_data[sm_det1].append(
                {
                    "energy": energy_det1_kev,
                    "x": x_det1,
                    "y": y_det1,
                    "doi": doi_det1,
                    "tchs": tch_list_det1,
                    "echs": ech_list_det1,
                    "mm": mm_det1,
                    "file": os.path.basename(file_path),  # Track source file
                }
            )

            sm_data[sm_det2].append(
                {
                    "energy": energy_det2_kev,
                    "x": x_det2,
                    "y": y_det2,
                    "doi": doi_det2,
                    "tchs": tch_list_det2,
                    "echs": ech_list_det2,
                    "mm": mm_det2,
                    "file": os.path.basename(file_path),  # Track source file
                }
            )

            filtered_count += 1
            event_count += 1

        return {
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "sm_data": dict(sm_data),
            "total_events": event_count,
            "filtered_events": filtered_count,
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "sm_data": {},
            "total_events": 0,
            "filtered_events": 0,
            "success": False,
            "error": str(e),
        }


class ConsoleRedirect:
    """Redirect stdout to both console and GUI text widget"""

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.stdout = sys.stdout

    def write(self, string):
        # Write to original stdout (terminal)
        self.stdout.write(string)
        self.stdout.flush()

        # Write to GUI console with timestamp
        if string.strip():  # Only add timestamp for non-empty strings
            timestamp = datetime.now().strftime("[%H:%M:%S]")
            self.text_widget.after(0, self._write_to_gui, f"{timestamp} {string}")
        else:
            self.text_widget.after(0, self._write_to_gui, string)

    def _write_to_gui(self, string):
        """Thread-safe way to write to GUI"""
        try:
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.insert(tk.END, string)
            self.text_widget.see(tk.END)  # Auto-scroll to bottom
            self.text_widget.config(state=tk.DISABLED)
        except:
            pass  # Ignore errors if widget is destroyed

    def flush(self):
        self.stdout.flush()


class LDATInspector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LDAT Inspector (Multi-File)")
        self.geometry("1800x900")  # Made wider for the new tab

        # Data variables
        self.file_paths = []
        self.config_path = None
        self.slab_energy_path = None
        self.config = None
        self.local_coord_dict = None
        self.sm_mM_map = None
        self.chtype_map = None
        self.FEM_instance = None
        self.kev_converter = None

        # Processing variables
        self.num_events = tk.StringVar(value="10000")
        self.min_ch_filter = tk.IntVar(value=1)
        self.en_min_ch = tk.DoubleVar(value=0.0)
        self.energy_min = tk.DoubleVar(value=400.0)
        self.energy_max = tk.DoubleVar(value=650.0)

        self.system_type = tk.StringVar(value="IMAS")

        # Multiprocessing variables
        cpu_count = multiprocessing.cpu_count()
        self.max_cores = min(
            max(1, cpu_count - 2),
            61,
            cpu_count - 4,
        )
        self.processing_results = []

        # Results storage
        self.processed_data = {}
        self.sm_data = defaultdict(list)
        self.sm_status = {}  # New: Store SM status information
        self.processing_thread = None
        self.processing_active = False

        # Console redirect
        self.console_redirect = None

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface with tabbed layout"""
        # Set up close protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for controls - SET FIXED WIDTH
        control_frame = ttk.Frame(main_frame, width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Right panel for tabbed interface
        tab_frame = ttk.Frame(main_frame)
        tab_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_control_panel(control_frame)
        self.setup_tabbed_interface(tab_frame)

    def setup_control_panel(self, parent):
        """Setup the control panel with file selection and parameters"""
        # Main control container
        control_container = ttk.Frame(parent)
        control_container.pack(fill=tk.BOTH, expand=True)

        # Top part for existing controls
        controls_frame = ttk.Frame(control_container)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # File selection section
        file_frame = ttk.LabelFrame(controls_frame, text="File Selection")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        # Config file selection
        ttk.Label(file_frame, text="Config File:").pack(anchor=tk.W, padx=5, pady=2)
        config_frame = ttk.Frame(file_frame)
        config_frame.pack(fill=tk.X, padx=5, pady=2)

        self.config_label = ttk.Label(
            config_frame,
            text="No config selected",
            background="white",
            relief="sunken",
            width=35,
        )
        self.config_label.pack(side=tk.LEFT, expand=False, padx=(0, 5))

        ttk.Button(config_frame, text="Browse", command=self.select_config_file).pack(
            side=tk.RIGHT
        )

        # Slab energy calibration file selection
        ttk.Label(file_frame, text="Slab Energy File:").pack(
            anchor=tk.W, padx=5, pady=2
        )
        slab_frame = ttk.Frame(file_frame)
        slab_frame.pack(fill=tk.X, padx=5, pady=2)

        self.slab_energy_label = ttk.Label(
            slab_frame,
            text="No calibration file selected",
            background="white",
            relief="sunken",
            width=35,
        )
        self.slab_energy_label.pack(side=tk.LEFT, expand=False, padx=(0, 5))

        ttk.Button(
            slab_frame, text="Browse", command=self.select_slab_energy_file
        ).pack(side=tk.RIGHT)

        # LDAT file selection
        ttk.Label(file_frame, text="LDAT Files:").pack(anchor=tk.W, padx=5, pady=2)
        ldat_frame = ttk.Frame(file_frame)
        ldat_frame.pack(fill=tk.X, padx=5, pady=2)

        # File list display
        self.file_listbox = tk.Listbox(ldat_frame, height=4)
        self.file_listbox.pack(fill=tk.X, padx=5, pady=2)

        # File selection buttons
        file_button_frame = ttk.Frame(ldat_frame)
        file_button_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(
            file_button_frame, text="Add Files", command=self.select_ldat_files
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_button_frame, text="Clear All", command=self.clear_files).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        # File counter label
        self.file_counter_label = ttk.Label(
            file_button_frame, text="Files: 0", foreground="blue"
        )
        self.file_counter_label.pack(side=tk.RIGHT)

        # Core info
        core_info_label = ttk.Label(
            file_frame,
            text=f"Max parallel files: {self.max_cores} (CPU cores: {multiprocessing.cpu_count()})",
        )
        core_info_label.pack(anchor=tk.W, padx=5, pady=2)

        # Processing parameters section
        params_frame = ttk.LabelFrame(controls_frame, text="Processing Parameters")
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Number of events (PER FILE) - UPDATE THE LABEL
        ttk.Label(params_frame, text="Coincidence Events per File:").pack(
            anchor=tk.W, padx=5, pady=2
        )
        ttk.Entry(params_frame, textvariable=self.num_events).pack(
            fill=tk.X, padx=5, pady=2
        )

        # Add explanation label
        ttk.Label(
            params_frame,
            text="(Each coincidence creates 2 detector events)",
            font=("Arial", 8),
            foreground="gray",
        ).pack(anchor=tk.W, padx=5, pady=(0, 5))

        # Minimum channels filter
        ttk.Label(params_frame, text="Min Channels per Event:").pack(
            anchor=tk.W, padx=5, pady=2
        )
        ttk.Entry(params_frame, textvariable=self.min_ch_filter).pack(
            fill=tk.X, padx=5, pady=2
        )

        # Energy threshold per channel
        ttk.Label(params_frame, text="Min Energy per Channel:").pack(
            anchor=tk.W, padx=5, pady=2
        )
        ttk.Entry(params_frame, textvariable=self.en_min_ch).pack(
            fill=tk.X, padx=5, pady=2
        )

        # Energy range
        ttk.Label(params_frame, text="Energy Range (Min):").pack(
            anchor=tk.W, padx=5, pady=2
        )
        ttk.Entry(params_frame, textvariable=self.energy_min).pack(
            fill=tk.X, padx=5, pady=2
        )

        ttk.Label(params_frame, text="Energy Range (Max):").pack(
            anchor=tk.W, padx=5, pady=2
        )
        ttk.Entry(params_frame, textvariable=self.energy_max).pack(
            fill=tk.X, padx=5, pady=2
        )

        ttk.Label(params_frame, text="System Type:").pack(
            anchor=tk.W, padx=5, pady=(6, 2)
        )

        system_combo = ttk.Combobox(
            params_frame,
            textvariable=self.system_type,
            state="readonly",
            values=["IMAS", "CORNELL"],
            width=20,
        )
        system_combo.pack(fill=tk.X, padx=5, pady=(0, 6))

        # Process button
        self.process_button = ttk.Button(
            params_frame, text="Process All Files", command=self.start_processing
        )
        self.process_button.pack(fill=tk.X, padx=5, pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            params_frame, variable=self.progress_var, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)

        # Status label
        self.status_label = ttk.Label(params_frame, text="Ready")
        self.status_label.pack(padx=5, pady=2)

        # Console section at the bottom
        console_frame = ttk.LabelFrame(control_container, text="Console Output")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Console text widget with scrollbar
        console_content_frame = ttk.Frame(console_frame)
        console_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Console text widget
        self.console_text = tk.Text(
            console_content_frame,
            height=8,  # Fixed height for console
            wrap=tk.WORD,
            font=("Consolas", 8),  # Monospace font for console
            bg="#1e1e1e",  # Dark background
            fg="#ffffff",  # White text
            insertbackground="#ffffff",  # White cursor
        )

        # Console scrollbar
        console_scrollbar = ttk.Scrollbar(
            console_content_frame, orient=tk.VERTICAL, command=self.console_text.yview
        )
        self.console_text.configure(yscrollcommand=console_scrollbar.set)

        # Pack console components
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Console control buttons
        console_button_frame = ttk.Frame(console_frame)
        console_button_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Button(
            console_button_frame, text="Clear Console", command=self.clear_console
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            console_button_frame, text="Save Log", command=self.save_console_log
        ).pack(side=tk.LEFT, padx=(0, 5))

        # Console info label
        self.console_info_label = ttk.Label(
            console_button_frame,
            text="Console ready",
            font=("Arial", 8),
            foreground="gray",
        )
        self.console_info_label.pack(side=tk.RIGHT)

        # Initial console message
        self.console_text.config(state=tk.NORMAL)
        welcome_msg = f"=== LDAT Inspector Console ===\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nReady for processing...\n\n"
        self.console_text.insert(tk.END, welcome_msg)
        self.console_text.config(state=tk.DISABLED)

        # Setup console redirection
        self.setup_console_redirect()

    def setup_console_redirect(self):
        """Setup console output redirection"""
        self.console_redirect = ConsoleRedirect(self.console_text)
        sys.stdout = self.console_redirect

        # Add welcome message
        print("Console output redirection active")
        print(
            f"Max workers: {self.max_cores}, CPU cores: {multiprocessing.cpu_count()}"
        )

    def clear_console(self):
        """Clear the console output"""
        self.console_text.config(state=tk.NORMAL)
        self.console_text.delete(1.0, tk.END)
        self.console_text.config(state=tk.DISABLED)

        # Add cleared message
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, f"{timestamp} Console cleared\n")
        self.console_text.config(state=tk.DISABLED)

    def save_console_log(self):
        """Save console log to file"""
        try:
            from tkinter import filedialog

            filename = filedialog.asksaveasfilename(
                title="Save Console Log",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"ldat_inspector_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            )

            if filename:
                with open(filename, "w", encoding="utf-8") as f:
                    # Get all text from console
                    content = self.console_text.get(1.0, tk.END)
                    f.write(content)

                self.console_info_label.config(
                    text=f"Log saved: {os.path.basename(filename)}"
                )
                print(f"Console log saved to: {filename}")

                # Reset info label after a few seconds
                self.after(
                    3000, lambda: self.console_info_label.config(text="Console ready")
                )

        except Exception as e:
            print(f"Error saving log: {e}")
            # Use try-except for messagebox too in case of issues
            try:
                messagebox.showerror("Error", f"Failed to save log: {e}")
            except:
                print("Failed to show error dialog")

    def processing_error(self, error_msg):
        """Called when processing encounters an error"""
        print(f"PROCESSING ERROR: {error_msg}")

        try:
            self.processing_active = False
            self.process_button.config(state="normal")
            self.update_status(f"Error: {error_msg}")
            messagebox.showerror("Error", f"Processing failed: {error_msg}")
        except Exception as e:
            print(f"Error in processing_error method: {e}")
            # Fallback - at least reset the processing state
            try:
                self.processing_active = False
                self.process_button.config(state="normal")
            except:
                pass

    def clear_files(self):
        """Clear all selected files"""
        self.file_paths = []
        self.update_file_listbox()

    def update_status(self, message):
        """Update the status label with a message"""
        try:
            self.status_label.config(text=message)
            print(f"Status: {message}")
        except Exception as e:
            print(f"Failed to update status: {e}")

    def process_files_parallel(self):
        """Process multiple files in parallel using multiprocessing"""
        try:
            self.update_status(
                f"Starting parallel processing of {len(self.file_paths)} files using {self.max_cores} workers..."
            )

            # Clear previous results
            self.processing_results = []
            self.sm_data.clear()

            # Prepare arguments for each file
            max_events = int(self.num_events.get())
            min_ch = self.min_ch_filter.get()
            en_min_ch = self.en_min_ch.get()
            en_min = self.energy_min.get()
            en_max = self.energy_max.get()
            system_type = self.system_type.get()

            # Create argument tuples for each file
            process_args = []
            for file_path in self.file_paths:
                args = (
                    file_path,
                    self.config_path,
                    self.slab_energy_path,
                    max_events,
                    min_ch,
                    en_min_ch,
                    en_min,
                    en_max,
                    system_type,
                )
                process_args.append(args)

            # Process files in parallel with safe worker count
            total_files = len(process_args)
            completed_files = 0

            # Ensure we don't exceed the number of files or system limits
            actual_workers = min(self.max_cores, total_files)

            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                # Submit all jobs
                future_to_file = {
                    executor.submit(process_single_file, args): args[0]
                    for args in process_args
                }

                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        self.processing_results.append(result)

                        completed_files += 1
                        progress = (completed_files / total_files) * 100
                        self.progress_var.set(progress)

                        if result["success"]:
                            self.update_status(
                                f"Completed {completed_files}/{total_files}: {result['filename']} "
                                f"({result['filtered_events']} events)"
                            )
                        else:
                            self.update_status(
                                f"Failed {completed_files}/{total_files}: {result['filename']} - {result['error']}"
                            )

                    except Exception as e:
                        self.update_status(
                            f"Error processing {os.path.basename(file_path)}: {str(e)}"
                        )

            # Combine results and update UI
            self.after(0, self.processing_complete_parallel)

        except Exception as e:
            self.after(0, self.processing_error, str(e))

    def on_closing(self):
        """Handle application closing"""
        print("Application closing...")

        # Restore stdout
        if self.console_redirect:
            sys.stdout = self.console_redirect.stdout

        # Stop any running processing
        if self.processing_active:
            print("Stopping active processing...")
            self.processing_active = False

        self.destroy()

    def setup_tabbed_interface(self, parent):
        """Setup the tabbed interface for plots and SM status"""
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Full System Status
        self.system_status_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.system_status_tab, text="Full System Status")
        self.setup_system_status_tab(self.system_status_tab)

        # Tab 2: SuperModule Status
        self.status_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.status_tab, text="SuperModule Status")
        self.setup_status_tab(self.status_tab)

        # Tab 3: SuperModule Plots (existing functionality)
        self.plots_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plots_tab, text="SuperModule Plots")
        self.setup_plots_tab(self.plots_tab)

    def setup_system_status_tab(self, parent):
        """Setup the Full System Status tab with plots always visible and properly sized text below"""
        # Main container - no scrollbar here
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top frame for title and plots - always visible, no scrolling
        top_section = ttk.Frame(main_container)
        top_section.pack(fill=tk.X, padx=10, pady=10)

        # Title and refresh button
        title_frame = ttk.Frame(top_section)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            title_frame, text="Full System Status Overview", font=("Arial", 14, "bold")
        ).pack(side=tk.LEFT)

        ttk.Button(
            title_frame,
            text="Refresh System Status",
            command=self.update_system_status_display,
        ).pack(side=tk.RIGHT)

        # Plots frame - always visible, no scrolling
        plots_frame = ttk.LabelFrame(top_section, text="System Overview Plots")
        plots_frame.pack(fill=tk.X, pady=(0, 10))

        # Create matplotlib figure for the plots with custom sizing
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.gridspec as gridspec

        # Create figure with proper aspect ratios
        self.system_fig = Figure(
            figsize=(14.5, 4)
        )  # Slightly taller for better visibility

        # Use GridSpec for precise control over subplot sizes
        gs = gridspec.GridSpec(
            1,
            3,
            figure=self.system_fig,
            width_ratios=[1, 1.2, 1.2],  # Energy plot smaller, hits plot wider
            hspace=0.2,
            wspace=0.3,
        )

        # Energy spectrum: takes left column (more square/tall aspect)
        self.system_energy_ax = self.system_fig.add_subplot(gs[0, 0])  # First column

        # SM hits: takes right 2 columns (rectangular/wide aspect)
        self.system_hits_ax = self.system_fig.add_subplot(gs[0, 1:])  # Columns 1 and 2

        # Create canvas for the plots
        self.system_canvas = FigureCanvasTkAgg(self.system_fig, plots_frame)
        self.system_canvas.get_tk_widget().pack(
            fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        # Initialize empty plots
        self.system_energy_ax.text(
            0.5,
            0.5,
            "Process data to see energy spectrum",
            ha="center",
            va="center",
            transform=self.system_energy_ax.transAxes,
        )
        self.system_energy_ax.set_title("Total Energy Spectrum")
        self.system_energy_ax.set_xlabel("Energy (keV)")
        self.system_energy_ax.set_ylabel("Counts")

        self.system_hits_ax.text(
            0.5,
            0.5,
            "Process data to see SM hits",
            ha="center",
            va="center",
            transform=self.system_hits_ax.transAxes,
        )
        self.system_hits_ax.set_title("Hits per SuperModule")
        self.system_hits_ax.set_xlabel("SuperModule")
        self.system_hits_ax.set_ylabel("Row")

        # Apply custom layout adjustments
        self.system_fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15)
        self.system_canvas.draw()

        # Bottom section for detailed status - THIS gets all remaining space
        bottom_section = ttk.Frame(main_container)
        bottom_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Detailed status frame with its own scrollbar - USE ALL AVAILABLE SPACE
        status_frame = ttk.LabelFrame(bottom_section, text="Detailed System Status")
        status_frame.pack(fill=tk.BOTH, expand=True)

        # Create the text widget directly with scrollbar - NO extra containers
        text_container = ttk.Frame(status_frame)
        text_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create the actual text widget with scrollbar
        self.system_status_text = tk.Text(
            text_container,
            wrap=tk.WORD,
            font=("Courier", 9),
            bg="#f8f9fa",
            fg="#212529",
            # REMOVE fixed height and width - let it expand to fill available space
        )

        # Scrollbar for the text widget
        status_scrollbar = ttk.Scrollbar(
            text_container, orient="vertical", command=self.system_status_text.yview
        )
        self.system_status_text.configure(yscrollcommand=status_scrollbar.set)

        # Pack text widget and scrollbar
        self.system_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure color tags for system health status
        self.system_status_text.tag_configure(
            "excellent",
            background="#d4edda",
            foreground="#155724",
            font=("Courier", 9, "bold"),
        )
        self.system_status_text.tag_configure(
            "good",
            background="#fff3cd",
            foreground="#856404",
            font=("Courier", 9, "bold"),
        )
        self.system_status_text.tag_configure(
            "warning",
            background="#f8d7da",
            foreground="#721c24",
            font=("Courier", 9, "bold"),
        )
        self.system_status_text.tag_configure(
            "critical",
            background="#d1ecf1",
            foreground="#0c5460",
            font=("Courier", 9, "bold"),
        )

        # Configure individual SM status colors
        self.system_status_text.tag_configure(
            "sm_excellent", background="#d4edda", foreground="#155724"
        )
        self.system_status_text.tag_configure(
            "sm_good", background="#fff3cd", foreground="#856404"
        )
        self.system_status_text.tag_configure(
            "sm_warning", background="#f8d7da", foreground="#721c24"
        )
        self.system_status_text.tag_configure(
            "sm_critical", background="#f5c6cb", foreground="#721c24"
        )
        self.system_status_text.tag_configure(
            "sm_offline", background="#e2e3e5", foreground="#383d41"
        )

        # Initial message
        self.system_status_text.insert(
            tk.END, "Process data to see Full System status information."
        )
        self.system_status_text.config(state=tk.DISABLED)

        # Bind mousewheel scrolling to the text widget
        def _on_mousewheel(event):
            self.system_status_text.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind mousewheel when mouse enters the text area
        def _bind_mousewheel(event):
            self.system_status_text.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            self.system_status_text.unbind_all("<MouseWheel>")

        self.system_status_text.bind("<Enter>", _bind_mousewheel)
        self.system_status_text.bind("<Leave>", _unbind_mousewheel)

    def setup_status_tab(self, parent):
        """Setup the SuperModule status tab"""
        # Top frame for SM selection and controls
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        # SM selection for status
        ttk.Label(
            top_frame, text="Select SuperModule:", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))

        self.status_sm_var = tk.StringVar()
        self.status_sm_combo = ttk.Combobox(
            top_frame, textvariable=self.status_sm_var, state="readonly", width=30
        )
        self.status_sm_combo.pack(anchor=tk.W, pady=(0, 10))
        self.status_sm_combo.bind("<<ComboboxSelected>>", self.update_sm_status_display)

        # Refresh button
        ttk.Button(
            top_frame, text="Refresh Status", command=self.update_sm_status_display
        ).pack(anchor=tk.W, pady=(0, 10))

        # Main content frame with scrollbar
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Create scrollable text widget for status display
        self.status_text = tk.Text(
            content_frame,
            wrap=tk.WORD,
            font=("Courier", 10),
            bg="#f8f9fa",
            fg="#212529",
        )

        # Configure color tags for individual SM status
        self.status_text.tag_configure(
            "excellent",
            background="#d4edda",
            foreground="#155724",
            font=("Courier", 10, "bold"),
        )
        self.status_text.tag_configure(
            "good",
            background="#fff3cd",
            foreground="#856404",
            font=("Courier", 10, "bold"),
        )
        self.status_text.tag_configure(
            "warning",
            background="#f8d7da",
            foreground="#721c24",
            font=("Courier", 10, "bold"),
        )
        self.status_text.tag_configure(
            "critical",
            background="#f5c6cb",
            foreground="#721c24",
            font=("Courier", 10, "bold"),
        )
        self.status_text.tag_configure(
            "active",
            background="#d4edda",
            foreground="#155724",
            font=("Courier", 10, "bold"),
        )
        self.status_text.tag_configure(
            "inactive",
            background="#e2e3e5",
            foreground="#383d41",
            font=("Courier", 10, "bold"),
        )

        # Scrollbar for text widget
        status_scrollbar = ttk.Scrollbar(
            content_frame, orient=tk.VERTICAL, command=self.status_text.yview
        )
        self.status_text.configure(yscrollcommand=status_scrollbar.set)

        # Pack text widget and scrollbar
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Initial message
        self.status_text.insert(
            tk.END, "Process data to see SuperModule status information."
        )
        self.status_text.config(state=tk.DISABLED)

    def setup_plots_tab(self, parent):
        """Setup the plots tab (existing functionality moved here)"""
        # Top frame for plot controls
        plot_control_frame = ttk.Frame(parent)
        plot_control_frame.pack(fill=tk.X, padx=10, pady=10)

        # SM selection for plots
        ttk.Label(
            plot_control_frame, text="Select SuperModule:", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))

        self.sm_var = tk.StringVar()
        self.sm_combo = ttk.Combobox(
            plot_control_frame, textvariable=self.sm_var, state="readonly", width=30
        )
        self.sm_combo.pack(anchor=tk.W, pady=(0, 10))
        self.sm_combo.bind("<<ComboboxSelected>>", self.on_sm_selected)

        # Plot type selection
        ttk.Label(
            plot_control_frame, text="Plot Type:", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=(5, 5))

        self.plot_type = tk.StringVar(value="Energy Spectrum")
        plot_combo = ttk.Combobox(
            plot_control_frame,
            textvariable=self.plot_type,
            values=["Energy Spectrum", "Flood Map", "DOI Distribution"],
            state="readonly",
            width=30,
        )
        plot_combo.pack(anchor=tk.W, pady=(0, 10))
        plot_combo.bind("<<ComboboxSelected>>", self.update_plot)

        # Update plot button
        ttk.Button(
            plot_control_frame, text="Update Plot", command=self.update_plot
        ).pack(anchor=tk.W, pady=(0, 10))

        # Plot panel
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.setup_plot_panel(plot_frame)

    def setup_plot_panel(self, parent):
        """Setup the plotting panel"""
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial empty plot
        ax = self.fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Load and process data to see plots",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=16,
        )
        self.canvas.draw()

    def truncate_filename(self, filepath, max_length=30):
        """Truncate filename if too long, keeping the extension visible"""
        if not filepath:
            return filepath

        filename = os.path.basename(filepath)
        if len(filename) <= max_length:
            return filename

        # Split name and extension
        name, ext = os.path.splitext(filename)

        # Calculate how much space we have for the name part
        available_space = max_length - len(ext) - 3  # 3 for "..."

        if available_space > 0:
            return f"{name[:available_space]}...{ext}"
        else:
            return f"...{ext}"

    def select_config_file(self):
        """Select configuration file"""
        file_path = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            initialdir="configs/",
        )
        if file_path:
            self.config_path = file_path
            # Truncate filename for display
            display_name = self.truncate_filename(file_path)
            self.config_label.config(text=display_name)
            self.load_config()

    def select_slab_energy_file(self):
        """Select slab energy calibration file"""
        file_path = filedialog.askopenfilename(
            title="Select Slab Energy Calibration File",
            filetypes=[
                ("Energy calibration (*.txt, *.encal)", ("*.txt", "*.encal")),
                ("Text files", "*.txt"),
                ("ENCal files", "*.encal"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        self.slab_energy_path = file_path
        # Truncate filename for display
        display_name = self.truncate_filename(file_path)
        self.slab_energy_label.config(text=display_name)

        # Try to ensure system_type matches the slab file.
        # Auto-detect from filename (best-effort).
        guessed = None
        # 1) Prefer guess from config filename
        if self.config_path:
            cfg_name = os.path.basename(self.config_path).lower()
            if "cornell" in cfg_name or "corn" in cfg_name:
                guessed = "CORNELL"
            elif "imas" in cfg_name or "mu" in cfg_name:
                guessed = "IMAS"

        # 2) Fall back to slab filename if config didn't help
        if not guessed:
            fname = os.path.basename(file_path).lower()
            if "cornell" in fname or "corn" in fname:
                guessed = "CORNELL"
            elif "mu" in fname or "imas" in fname:
                guessed = "IMAS"

        if guessed:
            if guessed != (self.system_type.get() or "IMAS"):
                # auto-switch and inform user
                self.system_type.set(guessed)
                messagebox.showinfo(
                    "System Type Auto-selected",
                    f"Auto-selected system type '{guessed}' based on the config filename.",
                )
        else:
            # Use a modal dialog with two explicit buttons instead of Yes/No
            dlg = tk.Toplevel(self)
            dlg.title("Select System Type")
            dlg.transient(self)
            dlg.grab_set()
            ttk.Label(
                dlg,
                text="Could not detect system type from the filename.\n\nClick a button to select system type:",
                wraplength=360,
                justify=tk.LEFT,
                padding=10,
            ).pack(fill=tk.BOTH, expand=True)

            btn_frame = ttk.Frame(dlg)
            btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

            def _choose_system(s):
                self.system_type.set(s)
                dlg.destroy()

            imas_btn = ttk.Button(
                btn_frame, text="IMAS", command=lambda: _choose_system("IMAS")
            )
            cornell_btn = ttk.Button(
                btn_frame, text="CORNELL", command=lambda: _choose_system("CORNELL")
            )
            imas_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
            cornell_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

            # Center the dialog over the main window
            self.update_idletasks()
            x = (
                self.winfo_rootx()
                + (self.winfo_width() // 2)
                - (dlg.winfo_reqwidth() // 2)
            )
            y = (
                self.winfo_rooty()
                + (self.winfo_height() // 2)
                - (dlg.winfo_reqheight() // 2)
            )
            try:
                dlg.geometry(f"+{x}+{y}")
            except Exception:
                pass

            # Wait for user to choose; if dialog closed without choice, cancel load
            self.wait_window(dlg)
            chosen = self.system_type.get()
            if chosen not in ("IMAS", "CORNELL"):
                # user closed dialog without selecting -> do not attempt load
                return

        # Attempt to load the calibration (will use current self.system_type)
        try:
            self.load_energy_calibration()
        except Exception as e:
            # keep the path so user can retry, but show the error
            messagebox.showerror(
                "Calibration Load Error",
                f"Failed to load energy calibration: {e}\n\n"
                "Please check the selected file and the System Type selection.",
            )

    def select_ldat_files(self):
        """Select multiple LDAT files"""
        file_paths = filedialog.askopenfilenames(
            title="Select LDAT Files",
            filetypes=[("LDAT files", "*.ldat"), ("All files", "*.*")],
        )

        if file_paths:
            # Limit to max_cores files OR practical limit for GUI responsiveness
            max_files = min(
                self.max_cores, 61
            )  # Limit to 61 files max for GUI responsiveness

            if len(file_paths) > max_files:
                messagebox.showwarning(
                    "Too Many Files",
                    f"Selected {len(file_paths)} files, but only {max_files} can be processed in parallel.\n"
                    f"Using first {max_files} files.\n\n"
                    f"System limits: CPU cores={multiprocessing.cpu_count()}, Max workers={self.max_cores}",
                )
                file_paths = file_paths[:max_files]

            self.file_paths = list(file_paths)
            self.update_file_listbox()

    def update_file_listbox(self):
        """Update the file listbox display and counter"""
        self.file_listbox.delete(0, tk.END)
        for file_path in self.file_paths:
            filename = self.truncate_filename(file_path, 50)
            self.file_listbox.insert(tk.END, filename)

        # Update file counter
        file_count = len(self.file_paths)
        counter_text = f"Files: {file_count}"
        if file_count > 0:
            counter_text += f"/{self.max_cores}"
        self.file_counter_label.config(text=counter_text)

        # Change color based on file count
        if file_count == 0:
            self.file_counter_label.config(foreground="gray")
        elif file_count <= self.max_cores:
            self.file_counter_label.config(foreground="blue")
        else:
            self.file_counter_label.config(
                foreground="red"
            )  # Should not happen due to limit

    def load_config(self):
        """Load configuration file and setup mapping"""
        try:
            print(f"Loading configuration from: {self.config_path}")
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)

            # Load mapping
            map_file = self.config["map_file"]
            print(f"Loading mapping from: {map_file}")
            (
                self.local_coord_dict,
                self.sm_mM_map,
                self.chtype_map,
                self.FEM_instance,
            ) = map_factory(map_file)

            print(
                f"Loaded mapping: {len(self.local_coord_dict)} coordinates, {len(self.sm_mM_map)} SM mappings, {len(self.chtype_map)} channel types"
            )

            # Update parameters from config
            if "energy_range" in self.config:
                self.energy_min.set(float(self.config["energy_range"][0]))
                self.energy_max.set(float(self.config["energy_range"][1]))
                print(f"Set energy range: {self.config['energy_range']}")
            if "min_ch" in self.config:
                self.min_ch_filter.set(int(self.config["min_ch"]))
                print(f"Set min channels: {self.config['min_ch']}")
            if "en_min_ch" in self.config:
                self.en_min_ch.set(float(self.config["en_min_ch"]))
                print(f"Set min energy per channel: {self.config['en_min_ch']}")

            self.status_label.config(text="Config loaded successfully")
            print("Configuration loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load config: {str(e)}"
            print(f"ERROR: {error_msg}")
            messagebox.showerror("Error", error_msg)

    def load_energy_calibration(self):
        """Load energy calibration file"""
        try:
            print(f"Loading energy calibration from: {self.slab_energy_path}")
            system = self.system_type.get()
            if system not in ["IMAS", "CORNELL"]:
                raise ValueError("System type must be IMAS or CORNELL")
            if system == "IMAS":
                file_type = "mu"
            elif system == "CORNELL":
                file_type = "cornell"
            self.kev_converter = KevConverter(
                self.slab_energy_path, file_type=file_type
            )
            self.status_label.config(text="Energy calibration loaded successfully")
            print("Energy calibration loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load energy calibration: {str(e)}"
            print(f"ERROR: {error_msg}")
            messagebox.showerror("Error", error_msg)
            self.kev_converter = None

    def start_processing(self):
        """Start parallel data processing"""
        if not self.file_paths or not self.config_path:
            warning_msg = "Please select config file and at least one LDAT file"
            print(f"WARNING: {warning_msg}")
            messagebox.showwarning("Warning", warning_msg)
            return

        if not self.kev_converter:
            warning_msg = "Please select energy calibration file"
            print(f"WARNING: {warning_msg}")
            messagebox.showwarning("Warning", warning_msg)
            return

        if self.processing_active:
            warning_msg = "Processing is already running"
            print(f"WARNING: {warning_msg}")
            messagebox.showwarning("Warning", warning_msg)
            return

        print("=== STARTING DATA PROCESSING ===")
        print(f"Files to process: {len(self.file_paths)}")
        print(f"Events per file: {self.num_events.get()}")
        print(f"Worker processes: {self.max_cores}")

        self.processing_active = True
        self.process_button.config(state="disabled")
        self.progress_var.set(0)

        # Start processing in separate thread
        self.processing_thread = threading.Thread(target=self.process_files_parallel)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def processing_complete_parallel(self):
        """Called when parallel processing is complete - MERGE ALL DATA"""
        print("=== PROCESSING COMPLETE ===")

        self.processing_active = False
        self.process_button.config(state="normal")
        self.progress_var.set(100)

        # Analyze results
        successful_files = [r for r in self.processing_results if r["success"]]
        failed_files = [r for r in self.processing_results if not r["success"]]

        total_filtered_events = sum(r["filtered_events"] for r in successful_files)
        print(
            f"Successful files: {len(successful_files)}/{len(self.processing_results)}"
        )
        print(f"Total filtered events: {total_filtered_events}")

        if failed_files:
            print(f"Failed files: {[r['filename'] for r in failed_files]}")

        # MERGE ALL DATA FROM ALL FILES
        self.sm_data.clear()

        # Combine SM data from all successful files
        for result in successful_files:
            for sm_id, sm_events in result["sm_data"].items():
                self.sm_data[sm_id].extend(sm_events)

        print(f"Merged data for {len(self.sm_data)} SuperModules")

        # Analyze SuperModule status
        print("Starting SuperModule status analysis...")
        self.analyze_sm_status()

        # Update both SM combo boxes
        if self.sm_data:
            sm_list = sorted(self.sm_data.keys())
            sm_options = [f"SM {sm}" for sm in sm_list]

            # Update plots tab SM selection
            self.sm_combo["values"] = sm_options
            if sm_list:
                self.sm_combo.current(0)

            # Update status tab SM selection
            self.status_sm_combo["values"] = sm_options
            if sm_list:
                self.status_sm_combo.current(0)
                self.update_sm_status_display()

            # Update system status display
            self.update_system_status_display()
            print("GUI updated with processed data")

        # Show completion message
        if failed_files:
            failed_names = [r["filename"] for r in failed_files]
            messagebox.showwarning(
                "Processing Complete with Errors",
                f"Successfully processed: {len(successful_files)}/{len(self.processing_results)} files\n"
                f"Total filtered events: {total_filtered_events}\n\n"
                f"Failed files: {', '.join(failed_names)}\n\n"
                f"Data from all successful files has been merged.\n"
                f"Check the SuperModule Status tab for detailed channel analysis.",
            )
        else:
            messagebox.showinfo(
                "Processing Complete",
                f"Successfully processed all {len(successful_files)} files!\n"
                f"Total filtered events: {total_filtered_events}\n\n"
                f"Data from all files has been merged by Super Module.\n"
                f"Check the SuperModule Status tab for detailed channel analysis.",
            )

        self.update_status(
            f"Processing complete: {len(successful_files)} files, {total_filtered_events} total events merged"
        )
        print("=== ALL PROCESSING COMPLETE ===")

    def on_sm_selected(self, event=None):
        """Called when a SM is selected"""
        self.update_plot()

    def update_plot(self, event=None):
        """Update the plot based on current selection - SIMPLIFIED"""
        if not self.sm_data:
            return

        # Get selected SM
        selected_sm_idx = self.sm_combo.current()
        if selected_sm_idx < 0:
            return

        sm_list = sorted(self.sm_data.keys())
        if selected_sm_idx >= len(sm_list):
            return

        sm_id = sm_list[selected_sm_idx]
        data = self.sm_data[sm_id]

        if not data:
            return

        # Clear previous plot
        self.fig.clear()

        plot_type = self.plot_type.get()

        if plot_type == "Energy Spectrum":
            self.plot_energy_spectrum_merged(data, sm_id)
        elif plot_type == "Flood Map":
            self.plot_flood_map_merged(data, sm_id)
        elif plot_type == "DOI Distribution":
            self.plot_doi_distribution_merged(data, sm_id)

        self.canvas.draw()

    def plot_energy_spectrum_merged(self, data, sm_id):
        """Plot fancy energy spectrum for the selected SM with merged data using Gaussian fitting"""
        ax = self.fig.add_subplot(111)

        energies = [d["energy"] for d in data]

        # Count events per minimodule and per file
        mm_counts = defaultdict(int)
        file_counts = defaultdict(int)
        for d in data:
            mm_counts[d["mm"]] += 1
            file_counts[d["file"]] += 1

        # Calculate number of bins based on energy range with better resolution
        energy_range = self.energy_max.get() - self.energy_min.get()
        num_bins = max(50, min(200, int(energy_range * 2)))  # 0.5 keV per bin

        # Create main histogram with gradient fill
        n, bins, patches = ax.hist(
            energies,
            bins=num_bins,
            alpha=0.8,
            edgecolor="navy",
            linewidth=0.8,
            range=(self.energy_min.get(), self.energy_max.get()),
        )

        # Apply gradient colors to bars based on height
        cm = plt.colormaps.get_cmap("viridis")
        bin_max = n.max()
        for i, p in enumerate(patches):
            color = cm(n[i] / bin_max)
            p.set_facecolor(color)

        # Use fits.py for Gaussian fitting (similar to imas_listmode.py debug_plots)
        fit_results = None
        try:
            from src.fits import fit_gaussian

            # Perform Gaussian fit like in imas_listmode.py
            bin_centers, gaussian_curve, pars, pcov, chi_ndf = fit_gaussian(
                n, bins, cb=8, min_peak=50, pk_finder="max", gaussian_str="gaussian"
            )

            # Extract fit parameters
            amplitude, mu, sigma = pars[0], pars[1], pars[2]

            # Calculate energy resolution (FWHM/peak * 100%)
            fwhm = 2.355 * sigma  # FWHM = 2.355 * sigma for Gaussian
            energy_resolution = (fwhm / mu) * 100

            # Plot the Gaussian fit
            ax.plot(
                bin_centers,
                gaussian_curve,
                "red",
                linewidth=2.5,
                alpha=0.9,
                label=f"Gaussian Fit\nμ = {mu:.1f} keV\nσ = {sigma:.1f} keV\nFWHM = {fwhm:.1f} keV\nRes = {energy_resolution:.1f}%",
            )

            # Store fit results for statistics box
            fit_results = {
                "mu": mu,
                "sigma": sigma,
                "fwhm": fwhm,
                "resolution": energy_resolution,
                "chi_ndf": chi_ndf,
                "amplitude": amplitude,
            }

        except Exception as e:
            print(f"Gaussian fitting failed: {e}")
            # Fallback to simple peak finding
            hist_counts, hist_bins = np.histogram(energies, bins=num_bins)
            peak_idx = np.argmax(hist_counts)
            peak_energy = (hist_bins[peak_idx] + hist_bins[peak_idx + 1]) / 2
            fit_results = {
                "mu": peak_energy,
                "sigma": 0,
                "fwhm": 0,
                "resolution": 0,
                "chi_ndf": 0,
            }

        # Styling
        ax.set_xlabel("Energy (keV)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Counts", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Energy Spectrum - SM {sm_id}\n{len(energies)} events from {len(file_counts)} files",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Enhanced grid
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

        # Set background color
        ax.set_facecolor("#f8f9fa")

        # Create enhanced statistics box with fit results
        if energies and fit_results:
            mean_energy = np.mean(energies)
            std_energy = np.std(energies)

            stats_text = (
                f"Fit Results:\n"
                f"Peak (μ): {fit_results['mu']:.1f} keV\n"
                f"Resolution: {fit_results['resolution']:.1f}%\n"
                f"─────────────────\n"
                f"Data Stats:\n"
                f"Total Events: {len(energies):,}\n"
                f"Files: {len(file_counts)}\n"
            )

            # Create fancy text box
            props = dict(
                boxstyle="round,pad=0.5",
                facecolor="lightblue",
                alpha=0.9,
                edgecolor="navy",
                linewidth=1.5,
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=props,
                family="monospace",
            )

            # Add legend for the fit
            ax.legend(loc="upper right", framealpha=0.9, fancybox=True, shadow=True)

        # Set axis limits with some padding
        if energies:
            x_min, x_max = min(energies), max(energies)
            x_range = x_max - x_min
            ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)

        # Add colorbar for the gradient
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=bin_max))
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label("Counts per bin", rotation=270, labelpad=15, fontweight="bold")

        # Tight layout
        self.fig.tight_layout()

    def plot_flood_map_merged(self, data, sm_id):
        """Plot flood map for the selected SM with merged data"""
        ax = self.fig.add_subplot(111)

        x_coords = [d["x"] for d in data]
        y_coords = [d["y"] for d in data]

        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(
            x_coords, y_coords, bins=200, range=[[0, 102], [0, 102]]
        )
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        im = ax.imshow(
            h.T, extent=extent, origin="lower", aspect="auto", cmap="viridis"
        )
        ax.set_xlabel("X Position (mm)")
        ax.set_ylabel("Y Position (mm)")

        # Count files and minimodules
        file_counts = defaultdict(int)
        mm_counts = defaultdict(int)
        for d in data:
            file_counts[d["file"]] += 1
            mm_counts[d["mm"]] += 1

        ax.set_title(
            f"Flood Map - SM {sm_id} ({len(x_coords)} events from {len(file_counts)} files)"
        )
        self.fig.colorbar(im, ax=ax, label="Counts")

        # Add minimodule and file information
        mm_info = ", ".join(
            [f"mM{mm}: {count}" for mm, count in sorted(mm_counts.items())]
        )
        file_info = f"Files: {len(file_counts)}"

        ax.text(
            0.02,
            0.02,
            f"{mm_info}\n{file_info}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    def plot_doi_distribution_merged(self, data, sm_id):
        """Plot DOI distribution for the selected SM with merged data"""
        ax = self.fig.add_subplot(111)

        dois = [d["doi"] for d in data]

        # Count events per minimodule for color coding
        mm_data = defaultdict(list)
        file_counts = defaultdict(int)
        for d in data:
            mm_data[d["mm"]].append(d["doi"])
            file_counts[d["file"]] += 1

        # Plot histogram for each minimodule with different colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(mm_data)))

        for i, (mm, mm_dois) in enumerate(sorted(mm_data.items())):
            ax.hist(
                mm_dois,
                bins=30,
                alpha=0.7,
                edgecolor="black",
                label=f"mM {mm} ({len(mm_dois)} events)",
                color=colors[i],
            )

        ax.set_xlabel("DOI")
        ax.set_ylabel("Counts")
        ax.set_title(f"DOI Distribution - SM {sm_id} (from {len(file_counts)} files)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics
        if dois:
            mean_doi = np.mean(dois)
            std_doi = np.std(dois)
            ax.text(
                0.02,
                0.98,
                f"Overall Mean: {mean_doi:.3f}\n"
                f"Overall Std: {std_doi:.3f}\n"
                f"Files: {len(file_counts)}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

    def analyze_sm_status(self):
        """Analyze SuperModule status including missing channels using proper mapping - OPTIMIZED"""
        if not self.chtype_map or not self.sm_mM_map or not self.local_coord_dict:
            print("Warning: Missing mapping data for status analysis")
            return

        print("Starting SM status analysis...")
        self.sm_status = {}

        # OPTIMIZED: Build expected channels mapping more efficiently
        # Instead of iterating through ALL channels in local_coord_dict,
        # use the sm_mM_map which is already filtered
        expected_channels_by_sm = defaultdict(lambda: {"time": set(), "energy": set()})

        print(f"Processing {len(self.sm_mM_map)} mapped channels...")

        system = self.system_type.get()
        # Use sm_mM_map directly - it already contains the SM assignments
        for channel_id, (sm_id, mm_id) in self.sm_mM_map.items():
            # Only process channels that have coordinate info AND type info
            if system == "CORNELL" and sm_id in CORNELL_SM_MM_INACTIVE:
                if mm_id in CORNELL_SM_MM_INACTIVE[sm_id]:
                    continue
            if channel_id in self.local_coord_dict and channel_id in self.chtype_map:
                channel_type = self.chtype_map[channel_id]

                if ChannelType.TIME in channel_type:
                    expected_channels_by_sm[sm_id]["time"].add(channel_id)
                elif ChannelType.ENERGY in channel_type:
                    expected_channels_by_sm[sm_id]["energy"].add(channel_id)

        # OPTIMIZED: Build active channels mapping more efficiently
        active_channels_by_sm = defaultdict(lambda: {"time": set(), "energy": set()})

        print("Analyzing active channels from event data...")

        # Process each SM's events to find active channels
        for sm_id, sm_events in self.sm_data.items():
            # Track unique slab channels (time channels) from events
            active_tchs = set()
            active_echs = set()
            active_minimodules = set()

            for event in sm_events:
                for ch in event.get("tchs", []):
                    active_tchs.add(ch)
                for ch in event.get("echs", []):
                    active_echs.add(ch)
                active_minimodules.add(event["mm"])

            # Add time channels (slabs) to active list
            for time_ch in active_tchs:
                active_channels_by_sm[sm_id]["time"].add(time_ch)

            # Add energy channels (slabs) to active list
            for energy_ch in active_echs:
                active_channels_by_sm[sm_id]["energy"].add(energy_ch)

        # Analyze each SM (combine expected SMs and SMs with data)
        all_sm_ids = set(expected_channels_by_sm.keys()) | set(self.sm_data.keys())

        print(f"Analyzing {len(all_sm_ids)} SuperModules...")

        for sm_id in sorted(all_sm_ids):
            sm_events = self.sm_data.get(sm_id, [])

            # Get expected and active channels
            expected_time = expected_channels_by_sm[sm_id]["time"]
            expected_energy = expected_channels_by_sm[sm_id]["energy"]
            active_time = active_channels_by_sm[sm_id]["time"]
            active_energy = active_channels_by_sm[sm_id]["energy"]

            # Calculate missing channels
            missing_time = expected_time - active_time
            missing_energy = expected_energy - active_energy

            # Get minimodule information
            active_mms = set()
            for event in sm_events:
                active_mms.add(event["mm"])

            # Calculate statistics
            total_events = len(sm_events)
            files_contributing = (
                len(set(event["file"] for event in sm_events)) if sm_events else 0
            )

            # Energy statistics
            energies = [event["energy"] for event in sm_events]
            mean_energy = np.mean(energies) if energies else 0
            std_energy = np.std(energies) if energies else 0

            # Calculate channel efficiencies
            time_efficiency = (
                (len(active_time) / len(expected_time) * 100) if expected_time else 0
            )
            energy_efficiency = (
                (len(active_energy) / len(expected_energy) * 100)
                if expected_energy
                else 0
            )

            self.sm_status[sm_id] = {
                "total_events": total_events,
                "files_contributing": files_contributing,
                "active_minimodules": sorted(active_mms),
                "active_time_channels": sorted(active_time),
                "active_energy_channels": sorted(active_energy),
                "missing_time_channels": sorted(missing_time),
                "missing_energy_channels": sorted(missing_energy),
                "expected_time_channels": sorted(expected_time),
                "expected_energy_channels": sorted(expected_energy),
                "mean_energy": mean_energy,
                "std_energy": std_energy,
                "channel_efficiency": {
                    "time": time_efficiency,
                    "energy": energy_efficiency,
                },
                # Additional debugging info
                "expected_total_channels": len(expected_time) + len(expected_energy),
                "active_total_channels": len(active_time) + len(active_energy),
                "missing_total_channels": len(missing_time) + len(missing_energy),
            }

        print(f"Analysis complete. Found status for {len(self.sm_status)} SMs")

        # Print summary for debugging
        total_expected = sum(
            status["expected_total_channels"] for status in self.sm_status.values()
        )
        total_active = sum(
            status["active_total_channels"] for status in self.sm_status.values()
        )

        print(
            f"System totals: {total_expected} expected channels, {total_active} active channels"
        )

    def update_system_status_display(self, event=None):
        """Update the Full System Status display with colored status indicators"""
        # First update the plots
        self.update_system_plots()

        if not self.sm_status:
            print("No SM status data available")
            return

        # Clear and update system status display
        self.system_status_text.config(state=tk.NORMAL)
        self.system_status_text.delete(1.0, tk.END)

        # Calculate system-wide statistics
        total_sms = len(self.sm_status)
        active_sms = sum(
            1 for status in self.sm_status.values() if status["total_events"] > 0
        )
        total_events = sum(status["total_events"] for status in self.sm_status.values())
        total_files = max(
            (status["files_contributing"] for status in self.sm_status.values()),
            default=0,
        )

        # Channel statistics
        total_expected_time = sum(
            len(status["expected_time_channels"]) for status in self.sm_status.values()
        )
        total_active_time = sum(
            len(status["active_time_channels"]) for status in self.sm_status.values()
        )
        total_missing_time = sum(
            len(status["missing_time_channels"]) for status in self.sm_status.values()
        )

        total_expected_energy = sum(
            len(status["expected_energy_channels"])
            for status in self.sm_status.values()
        )
        total_active_energy = sum(
            len(status["active_energy_channels"]) for status in self.sm_status.values()
        )
        total_missing_energy = sum(
            len(status["missing_energy_channels"]) for status in self.sm_status.values()
        )

        # Overall efficiency
        overall_time_efficiency = (
            (total_active_time / total_expected_time * 100)
            if total_expected_time
            else 0
        )
        overall_energy_efficiency = (
            (total_active_energy / total_expected_energy * 100)
            if total_expected_energy
            else 0
        )

        # System health assessment with color tags
        if overall_time_efficiency > 90 and overall_energy_efficiency > 90:
            system_health_text = "EXCELLENT"
            system_health_tag = "excellent"
        elif overall_time_efficiency > 80 and overall_energy_efficiency > 80:
            system_health_text = "GOOD"
            system_health_tag = "good"
        elif overall_time_efficiency > 60 and overall_energy_efficiency > 60:
            system_health_text = "WARNING"
            system_health_tag = "warning"
        else:
            system_health_text = "CRITICAL"
            system_health_tag = "critical"

        # Format system status information
        status_info_header = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════
    ║                           FULL SYSTEM STATUS OVERVIEW
    ╚══════════════════════════════════════════════════════════════════════════════════

    🏥 SYSTEM HEALTH: """

        self.system_status_text.insert(tk.END, status_info_header)

        # Insert colored system health status
        self.system_status_text.insert(tk.END, system_health_text, system_health_tag)

        status_info_body = f"""
    ┌─────────────────────────────────────────────────────────────────────────────────
    │ Overall Time Channel Efficiency: {overall_time_efficiency:.1f}%
    │ Overall Energy Channel Efficiency: {overall_energy_efficiency:.1f}%
    │ Active SuperModules: {active_sms}/{total_sms}
    │ Total Events Processed: {total_events:,}
    │ Data Files Analyzed: {total_files}
    └─────────────────────────────────────────────────────────────────────────────────

    📊 CHANNEL SUMMARY
    ┌─────────────────────────────────────────────────────────────────────────────────
    │ ⏱️  TIME CHANNELS:
    │    ├─ Expected: {total_expected_time:,}
    │    ├─ Active: {total_active_time:,} ({overall_time_efficiency:.1f}%)
    │    └─ Missing: {total_missing_time:,}
    │
    │ ⚡ ENERGY CHANNELS:
    │    ├─ Expected: {total_expected_energy:,}
    │    ├─ Active: {total_active_energy:,} ({overall_energy_efficiency:.1f}%)
    │    └─ Missing: {total_missing_energy:,}
    └─────────────────────────────────────────────────────────────────────────────────

    🔍 SUPERMODULE BREAKDOWN
    """

        self.system_status_text.insert(tk.END, status_info_body)

        # Add individual SM status with colored indicators
        for sm_id, status in sorted(self.sm_status.items()):
            time_eff = status["channel_efficiency"]["time"]
            energy_eff = status["channel_efficiency"]["energy"]

            # Determine SM health indicator and color tag
            if status["total_events"] == 0:
                sm_health_text = "OFFLINE"
                sm_health_tag = "sm_offline"
            elif time_eff > 90 and energy_eff > 90:
                sm_health_text = "EXCELLENT"
                sm_health_tag = "sm_excellent"
            elif time_eff > 80 and energy_eff > 80:
                sm_health_text = "GOOD"
                sm_health_tag = "sm_good"
            elif time_eff > 60 and energy_eff > 60:
                sm_health_text = "WARNING"
                sm_health_tag = "sm_warning"
            else:
                sm_health_text = "CRITICAL"
                sm_health_tag = "sm_critical"

            sm_line_prefix = f"\n├─ SM {sm_id:2d} "
            sm_line_suffix = f" │ Events: {status['total_events']:6,} │ Time: {time_eff:5.1f}% │ Energy: {energy_eff:5.1f}% │ mMs: {len(status['active_minimodules'])}"

            self.system_status_text.insert(tk.END, sm_line_prefix)
            self.system_status_text.insert(tk.END, sm_health_text, sm_health_tag)
            self.system_status_text.insert(tk.END, sm_line_suffix)

        # Continue with rest of the status display...
        critical_issues_header = f"""

    🚨 CRITICAL ISSUES
    ┌─────────────────────────────────────────────────────────────────────────────────"""

        self.system_status_text.insert(tk.END, critical_issues_header)

        # Find SMs with issues
        critical_sms = []
        warning_sms = []
        offline_sms = []

        for sm_id, status in self.sm_status.items():
            time_eff = status["channel_efficiency"]["time"]
            energy_eff = status["channel_efficiency"]["energy"]

            if status["total_events"] == 0:
                offline_sms.append(sm_id)
            elif time_eff < 60 or energy_eff < 60:
                critical_sms.append(sm_id)
            elif time_eff < 80 or energy_eff < 80:
                warning_sms.append(sm_id)

        issues_content = ""
        if offline_sms:
            issues_content += f"\n│ "
            self.system_status_text.insert(tk.END, issues_content)
            self.system_status_text.insert(tk.END, "OFFLINE", "sm_offline")
            self.system_status_text.insert(tk.END, f" SuperModules: {offline_sms}")
            issues_content = ""

        if critical_sms:
            issues_content += f"\n│ "
            self.system_status_text.insert(tk.END, issues_content)
            self.system_status_text.insert(tk.END, "CRITICAL", "sm_critical")
            self.system_status_text.insert(tk.END, f" SuperModules: {critical_sms}")
            issues_content = ""

        if warning_sms:
            issues_content += f"\n│ "
            self.system_status_text.insert(tk.END, issues_content)
            self.system_status_text.insert(tk.END, "WARNING", "sm_warning")
            self.system_status_text.insert(tk.END, f" SuperModules: {warning_sms}")
            issues_content = ""

        if not offline_sms and not critical_sms and not warning_sms:
            issues_content += f"\n│ ✅ No critical issues detected"
            self.system_status_text.insert(tk.END, issues_content)

        # Continue with missing channels section...
        missing_channels_section = f"""
    └─────────────────────────────────────────────────────────────────────────────────

    📋 DETAILED MISSING CHANNELS
    """

        self.system_status_text.insert(tk.END, missing_channels_section)

        # List missing channels by SM
        for sm_id, status in sorted(self.sm_status.items()):
            if status["missing_time_channels"] or status["missing_energy_channels"]:
                missing_section = f"""
    ┌─ SM {sm_id} Missing Channels:"""
                self.system_status_text.insert(tk.END, missing_section)

                if status["missing_time_channels"]:
                    missing_time_chunks = [
                        status["missing_time_channels"][i : i + 15]
                        for i in range(0, len(status["missing_time_channels"]), 15)
                    ]
                    time_header = (
                        f"\n│  ⏱️  Time ({len(status['missing_time_channels'])}): "
                    )
                    self.system_status_text.insert(tk.END, time_header)
                    for chunk in missing_time_chunks:
                        chunk_text = f"\n│      {', '.join(map(str, chunk))}"
                        self.system_status_text.insert(tk.END, chunk_text)

                if status["missing_energy_channels"]:
                    missing_energy_chunks = [
                        status["missing_energy_channels"][i : i + 15]
                        for i in range(0, len(status["missing_energy_channels"]), 15)
                    ]
                    energy_header = (
                        f"\n│  ⚡ Energy ({len(status['missing_energy_channels'])}): "
                    )
                    self.system_status_text.insert(tk.END, energy_header)
                    for chunk in missing_energy_chunks:
                        chunk_text = f"\n│      {', '.join(map(str, chunk))}"
                        self.system_status_text.insert(tk.END, chunk_text)

        footer = """

    ═══════════════════════════════════════════════════════════════════════════════════
    """
        self.system_status_text.insert(tk.END, footer)
        self.system_status_text.config(state=tk.DISABLED)

        print("System status display updated with plots and detailed information")

    def update_sm_status_display(self, event=None):
        """Update the SuperModule status display with enhanced information and colors"""
        if not self.sm_status:
            # Clear the text and show message
            self.status_text.config(state=tk.NORMAL)
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(
                tk.END, "No SuperModule status data available. Process data first."
            )
            self.status_text.config(state=tk.DISABLED)
            return

        # Get selected SM
        selected_sm_idx = self.status_sm_combo.current()
        if selected_sm_idx < 0:
            self.status_text.config(state=tk.NORMAL)
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(
                tk.END, "Please select a SuperModule from the dropdown."
            )
            self.status_text.config(state=tk.DISABLED)
            return

        sm_list = sorted(self.sm_status.keys())
        if selected_sm_idx >= len(sm_list):
            return

        sm_id = sm_list[selected_sm_idx]
        status = self.sm_status[sm_id]

        # Clear and update status display
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)

        # Enhanced status information with validation
        expected_total = status["expected_total_channels"]
        active_total = status["active_total_channels"]
        missing_total = status["missing_total_channels"]

        # Validate expected channel counts
        expected_time_count = len(status["expected_time_channels"])
        expected_energy_count = len(status["expected_energy_channels"])

        # Determine overall health color
        min_efficiency = min(
            status["channel_efficiency"]["time"], status["channel_efficiency"]["energy"]
        )
        if status["total_events"] == 0:
            health_text = "No Data"
            health_tag = "inactive"
        elif min_efficiency > 90:
            health_text = "Excellent"
            health_tag = "excellent"
        elif min_efficiency > 70:
            health_text = "Good"
            health_tag = "good"
        elif min_efficiency > 50:
            health_text = "Warning"
            health_tag = "warning"
        else:
            health_text = "Critical"
            health_tag = "critical"

        # Processing status
        processing_text = "Active" if status["total_events"] > 0 else "No Data"
        processing_tag = "active" if status["total_events"] > 0 else "inactive"

        status_info_header = f"""
    ╔══════════════════════════════════════════════════════════════════════
    ║                    SUPERMODULE {sm_id} STATUS REPORT
    ╚══════════════════════════════════════════════════════════════════════

    📊 GENERAL STATISTICS
    ├─ Total Events: {status['total_events']:,}
    ├─ Contributing Files: {status['files_contributing']}
    ├─ Active Minimodules: {status['active_minimodules']}
    ├─ Mean Energy: {status['mean_energy']:.1f} ± {status['std_energy']:.1f} keV
    └─ Processing Status: """

        self.status_text.insert(tk.END, status_info_header)
        self.status_text.insert(tk.END, processing_text, processing_tag)

        status_info_middle = f"""

    🔌 CHANNEL STATUS OVERVIEW
    ├─ Expected Total Channels: {expected_total} (Time: {expected_time_count}, Energy: {expected_energy_count})
    ├─ Active Total Channels: {active_total} ({(active_total/expected_total*100) if expected_total > 0 else 0:.1f}%)
    ├─ Missing Total Channels: {missing_total}
    ├─ Time Channel Efficiency: {status['channel_efficiency']['time']:.1f}%
    │  ({len(status['active_time_channels'])}/{len(status['expected_time_channels'])} active)
    ├─ Energy Channel Efficiency: {status['channel_efficiency']['energy']:.1f}%
    │  ({len(status['active_energy_channels'])}/{len(status['expected_energy_channels'])} active)
    └─ Overall Health: """

        self.status_text.insert(tk.END, status_info_middle)
        self.status_text.insert(tk.END, health_text, health_tag)

        # Continue with the rest of the display...
        channels_section = f"""

    ⏱️  TIME CHANNELS (Expected: {expected_time_count})
    ├─ Active Channels: {len(status['active_time_channels'])}
    ├─ Missing Channels: {len(status['missing_time_channels'])}
    │
    ├─ Active Channel IDs (showing first 50):
    """

        self.status_text.insert(tk.END, channels_section)

        # Add active time channels (limit display for readability)
        if status["active_time_channels"]:
            display_channels = status["active_time_channels"][:50]  # Limit display
            chunks = [
                display_channels[i : i + 10]
                for i in range(0, len(display_channels), 10)
            ]
            for chunk in chunks:
                self.status_text.insert(tk.END, f"│  {', '.join(map(str, chunk))}\n")
            if len(status["active_time_channels"]) > 50:
                self.status_text.insert(
                    tk.END,
                    f"│  ... and {len(status['active_time_channels']) - 50} more\n",
                )
        else:
            self.status_text.insert(tk.END, "│  None\n")

        self.status_text.insert(
            tk.END, "│\n├─ Missing Channel IDs (showing first 50):\n"
        )

        # Add missing time channels (limit display)
        if status["missing_time_channels"]:
            display_channels = status["missing_time_channels"][:50]
            chunks = [
                display_channels[i : i + 10]
                for i in range(0, len(display_channels), 10)
            ]
            for i, chunk in enumerate(chunks):
                prefix = "│  " if i < len(chunks) - 1 else "└─ "
                self.status_text.insert(
                    tk.END, f"{prefix}{', '.join(map(str, chunk))}\n"
                )
            if len(status["missing_time_channels"]) > 50:
                self.status_text.insert(
                    tk.END,
                    f"└─ ... and {len(status['missing_time_channels']) - 50} more missing\n",
                )
        else:
            self.status_text.insert(tk.END, "└─ None\n")

        energy_section = f"""
    ⚡ ENERGY CHANNELS (Expected: {expected_energy_count})
    ├─ Active Channels: {len(status['active_energy_channels'])}
    ├─ Missing Channels: {len(status['missing_energy_channels'])}
    │
    ├─ Active Channel IDs (showing first 50):
    """

        self.status_text.insert(tk.END, energy_section)

        # Add active energy channels (limit display)
        if status["active_energy_channels"]:
            display_channels = status["active_energy_channels"][:50]
            chunks = [
                display_channels[i : i + 10]
                for i in range(0, len(display_channels), 10)
            ]
            for chunk in chunks:
                self.status_text.insert(tk.END, f"│  {', '.join(map(str, chunk))}\n")
            if len(status["active_energy_channels"]) > 50:
                self.status_text.insert(
                    tk.END,
                    f"│  ... and {len(status['active_energy_channels']) - 50} more\n",
                )
        else:
            self.status_text.insert(tk.END, "│  None\n")

        self.status_text.insert(
            tk.END, "│\n├─ Missing Channel IDs (showing first 50):\n"
        )

        # Add missing energy channels (limit display)
        if status["missing_energy_channels"]:
            display_channels = status["missing_energy_channels"][:50]
            chunks = [
                display_channels[i : i + 10]
                for i in range(0, len(display_channels), 10)
            ]
            for i, chunk in enumerate(chunks):
                prefix = "│  " if i < len(chunks) - 1 else "└─ "
                self.status_text.insert(
                    tk.END, f"{prefix}{', '.join(map(str, chunk))}\n"
                )
            if len(status["missing_energy_channels"]) > 50:
                self.status_text.insert(
                    tk.END,
                    f"└─ ... and {len(status['missing_energy_channels']) - 50} more missing\n",
                )
        else:
            self.status_text.insert(tk.END, "└─ None\n")

        # Add validation warnings
        if expected_total != 256:
            warning_section = f"""
    ⚠️  CONFIGURATION WARNING
    └─ Expected {expected_total} channels, but standard SM should have 256 channels
    This may indicate a mapping configuration issue.
    """
            self.status_text.insert(tk.END, warning_section)

        footer = """
    ══════════════════════════════════════════════════════════════════════
    """
        self.status_text.insert(tk.END, footer)
        self.status_text.config(state=tk.DISABLED)

    def update_system_plots(self):
        """Update the system overview plots with GridSpec layout - CLEAN VERSION"""
        if not self.sm_data:
            return

        # SIMPLE AND CLEAN: Just clear the entire figure and recreate everything
        # This avoids ALL colorbar management issues
        self.system_fig.clear()

        # Recreate the GridSpec layout fresh each time
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(
            1,
            3,
            figure=self.system_fig,
            width_ratios=[1, 1.2, 1.2],
            hspace=0.2,
            wspace=0.3,
        )

        # Recreate the subplots fresh
        self.system_energy_ax = self.system_fig.add_subplot(gs[0, 0])
        self.system_hits_ax = self.system_fig.add_subplot(gs[0, 1:])

        # Reset colorbar reference - no need to remove anything
        self._system_colorbar = None

        try:
            # 1. Total Energy Spectrum Plot
            all_energies = []
            sm_hits = defaultdict(lambda: defaultdict(int))

            # QUADRANT_MAP from imas_listmode.py
            QUADRANT_MAP = {
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 0,
                5: 0,
                6: 1,
                7: 1,
                8: 2,
                9: 2,
                10: 3,
                11: 3,
                12: 2,
                13: 2,
                14: 3,
                15: 3,
            }

            for sm_id, sm_events in self.sm_data.items():
                for event in sm_events:
                    all_energies.append(event["energy"])
                    mm = event["mm"]
                    if mm in QUADRANT_MAP:
                        quadrant = QUADRANT_MAP[mm]
                        sm_hits[sm_id][quadrant] += 1

            if all_energies:
                # Create energy histogram with better binning
                energy_range = self.energy_max.get() - self.energy_min.get()
                num_bins = max(50, min(200, int(energy_range)))  # Adaptive binning
                n, bins = np.histogram(
                    all_energies,
                    bins=num_bins,
                    range=(self.energy_min.get(), self.energy_max.get()),
                )
                bin_centers = (bins[:-1] + bins[1:]) / 2

                # Plot histogram
                self.system_energy_ax.bar(
                    bin_centers,
                    n,
                    width=bins[1] - bins[0],
                    align="center",
                    color="blue",
                    alpha=0.7,
                )

                # Try to fit Gaussian
                try:
                    from src.fits import fit_gaussian

                    x, y, pars, _, _ = fit_gaussian(
                        n, bins, cb=8, pk_finder="max", gaussian_str="gaussian"
                    )
                    mu, sigma = pars[1], pars[2]
                    self.system_energy_ax.plot(x, y, "-r", linewidth=2)
                    resolution = round(2.35 * sigma / mu * 100, 2)
                    centroid = round(mu, 2)

                    # Add fit information box
                    fit_text = f"Energy res: {resolution}%\nCentroid {centroid} keV"
                    self.system_energy_ax.text(
                        0.98,
                        0.98,
                        fit_text,
                        transform=self.system_energy_ax.transAxes,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8
                        ),
                        fontsize=9,
                        fontweight="bold",
                    )
                except Exception as e:
                    print(f"Could not fit Gaussian: {e}")
                    # Fallback to basic statistics
                    mean_energy, std_energy = np.mean(all_energies), np.std(
                        all_energies
                    )
                    stats_text = (
                        f"Mean: {mean_energy:.1f} keV\nStd: {std_energy:.1f} keV"
                    )
                    self.system_energy_ax.text(
                        0.98,
                        0.98,
                        stats_text,
                        transform=self.system_energy_ax.transAxes,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8
                        ),
                        fontsize=9,
                        fontweight="bold",
                    )

                # Style the energy plot
                self.system_energy_ax.set_xlabel(
                    "Energy (keV)", fontsize=11, fontweight="bold"
                )
                self.system_energy_ax.set_ylabel(
                    "Counts", fontsize=11, fontweight="bold"
                )
                self.system_energy_ax.set_title(
                    f"Total Energy Spectrum ({len(all_energies):,} events)",
                    fontsize=12,
                    fontweight="bold",
                )
                self.system_energy_ax.grid(True, alpha=0.3)

            # 2. SuperModule Hits Plot
            if sm_hits:

                system = (self.system_type.get() or "IMAS").upper()
                if system == "IMAS":
                    # IMAS layout: 120 SMs * 4 quadrants = 480 cells -> reshape (10,48)
                    hits_list = [0] * 480

                    for sm in sorted(sm_hits.keys()):
                        for q in sorted(sm_hits[sm].keys()):
                            row = sm // 24
                            u_d = q // 2
                            l_r = q % 2
                            index = sm * 2 + l_r + u_d * 48 + row * 48
                            if index < len(hits_list):
                                hits_list[index] = sm_hits[sm][q]

                    hits_array = np.array(hits_list).reshape((10, 48))

                    # Create the heatmap with proper aspect
                    im = self.system_hits_ax.imshow(
                        hits_array,
                        cmap="viridis",
                        aspect="auto",
                        interpolation="nearest",
                    )

                    # Set up ticks and labels for better readability
                    x_ticks = range(0, 48, 6)  # Every 6th position
                    x_labels = [f"{i//2}" for i in x_ticks]
                    self.system_hits_ax.set_xticks(x_ticks)
                    self.system_hits_ax.set_xticklabels(x_labels, fontsize=10)

                    y_ticks = range(0, 10, 2)  # Every 2nd row
                    y_labels = [f"SM {i//2*24}" for i in y_ticks]
                    self.system_hits_ax.set_yticks(y_ticks)
                    self.system_hits_ax.set_yticklabels(y_labels, fontsize=10)

                elif system == "CORNELL":
                    # CORNELL SM numbering is cassette-first (see mod_feb_map in
                    # maps/cornell_map_*.yaml and ring_r/ring_z/ring_yx in
                    # configs/cornell_full_system.yaml): sm = cassette_idx * N_RINGS
                    # + ring_idx, where ring_idx in {0,1,2} is the axial position
                    # within a cassette (3 SMs/cassette, one per FEB/D port 1/3/5)
                    # and cassette_idx is the azimuthal cassette position. Rows of
                    # this plot are rings (axial), columns are cassettes (azimuth).
                    N_RINGS = 3
                    n_cassettes = (
                        max(sm // N_RINGS for sm in sm_hits) + 1 if sm_hits else 1
                    )

                    hits_array = np.zeros((N_RINGS * 2, n_cassettes * 2))

                    for sm in sorted(sm_hits.keys()):
                        cassette_idx = sm // N_RINGS
                        ring_idx = sm % N_RINGS
                        for q in sorted(sm_hits[sm].keys()):
                            u_d = q % 2
                            l_r = q // 2
                            row = ring_idx * 2 + u_d
                            col = cassette_idx * 2 + l_r
                            hits_array[row, col] = sm_hits[sm][q]

                    # Create the heatmap with proper aspect
                    im = self.system_hits_ax.imshow(
                        hits_array,
                        cmap="viridis",
                        aspect="auto",
                        interpolation="nearest",
                    )

                    # Set up ticks and labels for better readability
                    x_ticks = range(0, n_cassettes * 2, 2)
                    x_labels = [f"{i//2}" for i in x_ticks]
                    self.system_hits_ax.set_xticks(x_ticks)
                    self.system_hits_ax.set_xticklabels(x_labels, fontsize=10)

                    y_ticks = range(0, N_RINGS * 2, 2)
                    y_labels = [f"Ring {i//2}" for i in y_ticks]
                    self.system_hits_ax.set_yticks(y_ticks)
                    self.system_hits_ax.set_yticklabels(y_labels, fontsize=10)
                # Style the hits plot
                if system == "CORNELL":
                    self.system_hits_ax.set_xlabel(
                        "Cassette", fontsize=11, fontweight="bold"
                    )
                    self.system_hits_ax.set_ylabel(
                        "Ring", fontsize=11, fontweight="bold"
                    )
                else:
                    self.system_hits_ax.set_xlabel(
                        "SuperModule", fontsize=11, fontweight="bold"
                    )
                    self.system_hits_ax.set_ylabel(
                        "Row", fontsize=11, fontweight="bold"
                    )
                self.system_hits_ax.set_title(
                    "Hits per SuperModule", fontsize=12, fontweight="bold"
                )

                # Add colorbar - fresh figure, no conflicts possible
                self._system_colorbar = self.system_fig.colorbar(
                    im, ax=self.system_hits_ax, shrink=0.8, aspect=20, pad=0.02
                )
                self._system_colorbar.set_label(
                    "Hits per SM quadrant", rotation=270, labelpad=15, fontweight="bold"
                )

            # Apply consistent layout to the fresh figure
            self.system_fig.subplots_adjust(
                left=0.08, right=0.92, top=0.90, bottom=0.15, wspace=0.3
            )

            # Redraw the canvas
            self.system_canvas.draw()

        except Exception as e:
            print(f"Error updating system plots: {e}")
            # Show error on both axes
            self.system_energy_ax.text(
                0.5,
                0.5,
                f"Error creating plots:\n{str(e)}",
                ha="center",
                va="center",
                transform=self.system_energy_ax.transAxes,
            )
            self.system_hits_ax.text(
                0.5,
                0.5,
                f"Error creating plots:\n{str(e)}",
                ha="center",
                va="center",
                transform=self.system_hits_ax.transAxes,
            )
            self.system_canvas.draw()


def main():
    """Run the original Tk inspector for historical reference."""
    app = LDATInspector()
    app.mainloop()


if __name__ == "__main__":
    main()
