# PETsys Manager: Cornell deployment

PETsys Manager runs the Cornell acquisition, RAW conversion, energy calibration, list-mode (LM) and quality-control (QC) workflows from this repository. It is a separate application from LDAT Inspector and replaces the sibling `gui_cornell` GUI only after operator acceptance ([spec003](../specs/003-petsys-manager-migration/spec.md), FR-17/FR-18). Until then the sibling repository, its scripts and any desktop shortcuts stay as they are. Retiring them is a separate owner decision.

## Status

| Area | State |
|---|---|
| Offline processing (calibration, LM, offline QC) | Checked on synthetic fixtures, and on six real January 2026 Cornell splits against the reference scripts (Windows, 2026-10-02) |
| Hardware workflows (DAQD, initialize, acquire, live QC, pipeline) | Accepted on the Cornell machine on 2026-10-06 (T19): initialization, acquisition, STOP and bias-off, retries, live QC, complete pipeline and close ([`acceptance.md`](../specs/003-petsys-manager-migration/acceptance.md)) |
| Linux runtime environment | Clean-checkout audit (5/5, production process backend) and dummy process checks (16/16) pass on the Cornell machine (2026-10-02) |
| Comparison with the reference scripts on real data | Pass on the January 2026 Cornell data: calibration, LM and QC identical to the reference; P = 1 calibration and QC also identical on Cornell F18 compact data (2026-10-05) ([`acceptance.md`](../specs/003-petsys-manager-migration/acceptance.md)) |

## Prerequisites (Cornell Linux machine)

- **The `process_petsys` conda environment** from `process_petsys.yml`. It covers numpy, scipy, matplotlib, pandas, pyyaml, numba, openpyxl, reportlab, tqdm and customtkinter; Pillow comes in through matplotlib/reportlab. After pulling, update an existing environment with `conda env update --name process_petsys --file process_petsys.yml`.
  - Packages in a user's `~/.local` site-packages can hide gaps in the environment: on the Cornell machine `customtkinter` was found there and not in the environment.
  - Check with `conda run -n process_petsys python -s -c "import numpy, yaml, scipy, matplotlib, pandas, numba, openpyxl, reportlab, tqdm, customtkinter, PIL, tkinter"`; `-s` ignores `~/.local`.
  - Install anything missing with `conda run -n process_petsys python -s -m pip install <package>==<version from process_petsys.yml>`. Plain `pip install` reports a `~/.local` copy as already installed.
- **Smooth fonts on Linux:** conda-forge's default `tk` is built without Xft, so Tk shows only bitmap fonts (pixelated text; on the Cornell machine 47 font families against 336 for the system Python). Install the Xft build: `conda install -n process_petsys -c conda-forge "tk=8.6.13=xft*"`; check with `conda run -n process_petsys python -c "import tkinter; r=tkinter.Tk(); print(len(r.tk.call('font','families')))"`. Linux only, so it is not in the shared `process_petsys.yml`.
- **Legacy editable install:** an old `event-petsys` editable install (from the package's former name) makes `conda list`/`conda install` fail with `Expected exactly one egg-info directory`. Replace it: `pip uninstall -y event-petsys`, move `event_petsys.egg-info` out of the checkout, then `python -s -m pip install --no-deps -e .`.
- **A desktop session** for the Tk window. Processing runs in background child processes and does not need the display.
- **The PETsys `sw_daq_tofpet2` tools folder**, containing `daqd`, `init_system`, `acquire_sipm_data`, `set_bias` and `convert_raw_to_coincidence`. The manager converts to compact coincidence only (`--writeBinaryCompact`, stock PETsys); calibration, LM, QC and the pipeline all use it.
- **The Python PETsys was installed for.** `init_system`, `acquire_sipm_data` and `set_bias` are Python scripts (`#!/usr/bin/env python3`) that need that interpreter's packages (`bitarray`, `pandas`). On the Cornell machine this is the system `/usr/bin/python3`; the `process_petsys` env lacks `bitarray`. Set `petsys_python` in the profile so the manager runs them with it (FR-23); check with `/usr/bin/python3 -c "import bitarray, pandas"`.
- **DAQ character devices** for `PFP_KX7` (one or two; the default is `/dev/psdaq1`, `/dev/psdaq0`).
- **The default DAQD socket `/tmp/d.sock`.** A custom socket stays refused unless `capabilities.custom_socket_confirmed` is set after checking that every tool honours it.

Hardware actions are refused on any platform other than Linux.

## Launch

```bash
conda activate process_petsys
python exe_programs/PETsysManager.py                     # default profile
python exe_programs/PETsysManager.py --profile /path/to/petsys_manager.yaml
```

It can be started from any working directory. Processing stages run as `python -u -m src.cornell.cli <action>` children of the same interpreter, from the checkout root. Nothing is launched at startup.

## Machine profile

All per-machine settings live in one YAML profile:

- **Default location:** `$XDG_CONFIG_HOME/process_petsys/petsys_manager.yaml`, or `~/.config/process_petsys/petsys_manager.yaml`.
- **Missing default file:** the manager starts with an empty profile.
- **Saving:** **Save** in the GUI writes the profile in use. A new file is created exclusively, and an existing file is replaced only if it is already a valid manager profile.
- **What it never writes:** the processing YAML, maps or calibration files.

Relative paths resolve against `processing_root`, which resolves against the checkout. Empty values (`null`) are unavailable: they block the actions that need them and are never replaced by defaults.

```yaml
schema_version: 1
petsys_folder: /opt/sw_daq_tofpet2          # tools folder (example path)
petsys_python: /usr/bin/python3             # runs init_system/acquire_sipm_data/set_bias; null: their shebang
processing_root: null                       # base for relative paths below (default: the checkout)
ini_file: /path/to/config.ini               # PETsys INI used by init/acquire/convert
yaml_file: configs/<cornell config>.yaml    # processing config; its map_file selects the map
data_dir: /data/acquisitions                # RAW, conversion and pipeline/QC run folders
calibration_dir: /data/encal                # manual calibrations
report_dir: /data/reports                   # QC and LM debug outputs
lm_dir: /data/listmode                      # manual LM jobs
cog_limits_file: /path/to/cog_limits.txt    # position calibration (P >= 2) and LM
doi_limits_file: /path/to/doi_limits.txt    # LM
calibration_file: /path/to/system.encal     # manual LM (per-slab or position .encal)
pair_map_file: /path/to/pairs.txt           # LM module-pair IDs
region_map_file: /path/to/regions.tsv       # LM regions
daq_type: PFP_KX7
cards: [/dev/psdaq1, /dev/psdaq0]
socket_path: /tmp/d.sock
shared_memory_path: /dev/shm/daqd_shm
safety:                                      # acquisition monitoring (editable in the GUI)
  startup_timeout_s: 45.0
  growth_window_s: 20.0
  poll_interval_s: 5.0
  min_growth_bytes: 20000000
  max_loss_percent: 5.0
  max_attempts: 3
  retry_delay_s: 2.0
  terminate_grace_s: 3.0
limits:
  workers: 0                                 # worker processes for calibration, LM and QC; 0 = CPU count - 2
  batch_records: 5000
  calibration_event_limit: 10000000          # reference mode: passing coincidences per file
  calibration_target_per_key: 3000           # target mode: T kept sides per histogram
  calibration_memory_mb: 8192                # target mode: read each file once if its selected events fit
  lm_seed: 0                                 # LM: seed of the per-file random streams (ambiguous slabs)
  qc_seed: 0                                 # QC: seed of the per-file random streams (single-time-channel slabs)
  qc_pair_limit: 1000001                     # accepted pairs per file (reference)
  log_tail_lines: 1000
capabilities:
  custom_socket_confirmed: false
  installed_version: null
lm_metadata:                                 # written into every LM header; all required
  isotope: null                              # e.g. Na22 (16 bytes max)
  acquisition_time_s: null                   # manual LM only; the pipeline writes its Acq. Time
  measurement_time_s: null                   # manual LM only; the pipeline writes its Acq. Time
  detector_size_x_mm: null
  detector_size_y_mm: null
  module_number: null
  ring_number: null
  ring_distance_mm: null
  detector_pixels_x: null                    # 1..127
  detector_pixels_y: null
  timestamp_unit: null
```

## Private input checklist

These files are machine- and system-specific, are not in the repository, and are selected in the profile:

- [ ] PETsys INI for the installed system.
- [ ] Processing YAML for the Cornell system and its map (`map_file`). The Cornell and IMAS layouts differ; use the map of the system being processed.
- [ ] COG limits and DOI limits for that map.
- [ ] LM pair map and region map.
- [ ] A system energy calibration (`.encal`) for manual LM, either per slab (`ID(t_ch, slab)`) or by position (`ID(time_ch, slab, region)`). The pipeline makes its own.
- [ ] LM header metadata confirmed for the reconstruction software (geometry, pixels, isotope, timestamp unit).

## Workflows and format routes

| Action | Input | Output (location) |
|---|---|---|
| Start DAQD / Initialize | profile tools, cards, INI | owned `daqd`; initialization valid for this daemon, INI and cards |
| Acquire | Acq. Time, Acquisition Name, optional hardware trigger | `<name>.rawf`/`.idxf` (`data_dir`) |
| Convert, coincidence | `.rawf` | compact coincidence `.ldat` (`data_dir`) |
| Create energy cal file | compact coincidence `.ldat`; positions per slab P; event limit (target T per histogram, or the reference 10 M per file) | P = 1: per-slab `.encal`; P ≥ 2: position `.encal` with COG regions; plus status, sidecar and plot (`calibration_dir/<run>`) |
| Generate LM file | compact coincidence `.ldat` (read at the Max Hits per Side of its conversion), `.encal`, limits, pair/region maps, LM metadata | `.lm` with provenance and optional debug plots (`lm_dir/<run>`) |
| Run quality control (live) | 60 s with source or 180 s without (fixed presets), Acquisition Name | compact coincidence conversion, then QC report (run folder in `data_dir`) |
| Analyze existing compact LDAT | **compact coincidence** `.ldat` | QC report (`report_dir/<run>`) |
| Run complete pipeline | Acq. Time, Acquisition Name, splits, max hits, positions per slab | acquire → compact coincidence conversion → calibration → LM, all in one run folder in `data_dir` |

`.ldat` files carry no reliable format marker, so every input list must be confirmed as compact coincidence before use. Each stage checks the records it reads (hit counts, truncation, channels of the selected map, finite energies) in the same pass, before publishing anything from them, and never reads a file only to check it (FR-24): conversion checks the first 10,000 records of each output and records them as structure-checked; LM reads and checks whole files once; calibration stops at its passing-event limit; QC checks its sample. Records a stage does not read are not checked, and its provenance says so (QC: records in the file are known only when the whole file was read). A defect after the first 10,000 records fails the first stage that reads it.

Not available: singles counts (LDAT coincidence records contain two detectors, not a singles population), keV energies in QC (QC reports raw a.u.), calibrated DOI depth (the DOI value is a light-sharing ratio), and hardware actions off the Cornell Linux machine.

## Safety semantics

- **DAQD ownership:** the manager starts and owns one `daqd`. An existing `/tmp/d.sock` or `/dev/shm/daqd_shm` blocks start and is never deleted: the installed `daqd` removes another daemon's shared memory when it fails its own exclusive create.
  - READY requires the owned process to answer the DAQD shared-memory query, not just a socket file.
  - Daemon death, or a change of INI or cards, invalidates initialization.
  - Stop and close signal only the owned process.
  - The daemon's own messages appear in the **DAQD Output** box, not in the Output Log. Its per-second `CNT` counter lines (one per card, once acquisition is on) only update the latest counters under the DAQD status in System Control. **Save Log** writes the Output Log, DAQD Output and counters to a new `petsys_manager_log_<date>_<time>.txt` in the Report Destination.
- **Acquisition:**
  - A monitored attempt needs the RAW file to start within `startup_timeout_s` and grow by `min_growth_bytes` within `growth_window_s`, with frame loss at most `max_loss_percent`.
  - Only startup timeout, no growth, no data and frame loss are retried, up to `max_attempts`, each to a new path. A nonzero exit is a failure and is not retried.
  - The GUI shows RAW started, growth passed, live size and rate, and a stall warning.
- **SiPM bias:** after an attempt that did not end normally (STOP, abort, nonzero exit), the manager runs `set_bias --power off` before anything else. If that fails, a persistent "SiPM bias state unknown" warning appears, no retry starts, and Acquire stays locked until the operator confirms the bias state. On close, a running bias-off finishes before DAQD stops.
- **PETsys Python tools:** with `petsys_python` set, `init_system`, `acquire_sipm_data` and `set_bias` run as `<petsys_python> <tool>` without the manager's conda/Python activation (`CONDA_*`, `PYTHONHOME`, `PYTHONPATH`, `VIRTUAL_ENV` removed; the interpreter's folder first on `PATH`), as when the env was deactivated by hand. `daqd`, the converters and processing are unaffected. Changing it invalidates initialization.
- **STOP:** stops the running stage, and no later stage of that run starts.
- **Outputs:**
  - Every run gets one new folder in its destination, created exclusively, named `<data>_<action>[-<options>]_<YYYY-MM-DD>_<HHMM>`; a second run with the same name in the same minute gets `_2`, `_3` …. Existing files and folders are never overwritten or reused.
    - Data: the RAW name (conversion), the Acquisition Name (Acquire, pipeline, live QC; default `acquisition`), otherwise the common start of the input file names without the `_coincCompact` and split suffix.
    - Action and options: `acq`, `conv`, `cal-P<n>-<target|reference>`, `lm-P<n>`, `qc` (offline), `qc-<with|without>-source` (live), `pipeline-P<n>`.
    - Examples: `run_0024_lm-P5_2026-10-04_0936`, `20260930_F18_950uCi_Run1_60s_cal-P5-target_2026-10-05_0533`, `F18_Run1_pipeline-P5_2026-10-05_0533`.
  - Single-stage runs write their outputs directly in the run folder. The pipeline and live QC use one numbered folder per stage: `1_acquisition/`, `2_conversion/`, `3_calibration/`, `4_listmode/` (QC: `3_qc/`). Only acquisition has `attempt-N/` folders, one per attempt.
  - `run.json` in each run folder is the run's record: the settings snapshot, the exact inputs, every attempt and every output with its hash; it is replaced by each update. `.history/` keeps every earlier version. Later stages use only the outputs recorded for this run, never similarly named files.
  - `runs.tsv` in each destination gets one line per finished run (also failed and stopped ones): finish time, run folder, action, inputs, status, main output. It is only appended to.
  - Failed or partial outputs are kept and marked unvalidated.
  - After a successful conversion, the converter's `.lidx` index files and empty split files (e.g. `<name>_coincCompact_00000000.ldat`) of that conversion are removed; `run.json` lists each as `removed`. A failed or stopped conversion keeps them.
  - Run folders from before 2026-10-05 (`<action>-<stamp>-<id>/<stage>/attempt-1/`, `manifest-NNNNNN.json`) stay as they are and remain readable.
- **Calibration:** a new `.encal` is never applied automatically. The GUI offers it for LM as an unsaved profile edit.
  - **Event limit:** *target* (default) keeps S = K × P × T sides in total, ⌈S / n⌉ from each of the n files; a file stops once its kept sides (sides that enter a key's histogram) exceed its share (K = time-channel × slab keys of the selected map, T = target sides per histogram, saved in the profile). The LDAT Processing tab shows S and the share before the run. *Reference* stops each file after 10,000,000, as `cornell_slab_en_cal.py`; use it to compare with the reference script. The result and sidecar record the limit used and the sides each key received (minimum, median, keys below T and below the 200-event fit minimum); T is an average, so low-occupancy keys can stay below it.
  - **Reading once:** in target mode each file is read once when the selected events (at most (share + 2) sides per file × 16 bytes) fit `calibration_memory_mb`; otherwise, and in reference mode, each file is read twice, as the reference does. Both give identical files; the result says which was used.
  - **Workers:** calibration reads files and fits keys in `workers` processes (0 = automatic, CPU count − 2). The files do not depend on the worker count; the sidecar records it. The status line shows the phase (read, pass 2, fits k/K keys).
  - An existing `.encal` row with μ ≤ 0 (a failed legacy fit) is read as "no factor", as the reference LM does: its pairs are rejected as missing calibration, and the keys are listed in the LM provenance and as a warning in the log. Nothing is estimated in their place; to fill them from neighbours, make a new calibration, whose status file labels borrowed and estimated keys.
  - A COG or DOI limits row with left = right is used unchanged, as in the reference, so the sides of that slab fall out of range and are counted; the keys are listed in the provenance and as a warning. A row with right < left still refuses the file.
- **QC:** findings are observations of the coincidence sample (occupancy, fits in a.u.), not a detector PASS/FAIL verdict.
- **Parallel LM and seed:** LM processes its files in `workers` processes and merges them in the same order. The random slab choice for ambiguous sides uses one stream per file, derived from the profile's `lm_seed` and the file's position, so the same inputs and seed give the same `.lm` on every run and for any worker count; the provenance records the seeds. (The reference used one unseeded stream, so its reruns differed slightly.)
- **Parallel QC and seed:** QC samples its files in `workers` processes (the same profile setting) and merges them in the input order. The random slab choice for sides with one time channel uses one stream per file, derived from the profile's `qc_seed` (default 0) and the file's position, so the same inputs and seed give the same QC results on every run and for any worker count; `qc_summary.json` records the seeds and workers. (The reference used one unseeded stream continuing across files.) Live QC has one file per split, so it gains only with several splits.
- **Compact LM:** compact input is decoded at the Max Hits per Side of its conversion (a side with more hits refuses the file) and gives the same `.lm` file as the reference's fixed input of the same events (checked in the library).
- **LM header:** the 11 metadata fields come from the profile. Empty fields block LM; there are no hardcoded values. The pipeline writes its own Acq. Time as the acquisition and measurement time.

## Runtime module map

| Path | Role |
|---|---|
| `exe_programs/PETsysManager.py` | Launcher (`--profile`) |
| `exe_programs/petsys_manager_gui.py` | CustomTkinter window: tabs, controls, progress, results; updates widgets on the Tk thread only |
| `exe_programs/assets/onco_logo.jpeg` | Optional logo; a load failure never blocks startup |
| `src/petsys_manager/contracts.py` | Actions, formats/populations, routes, typed results |
| `src/petsys_manager/settings.py` | Profile schema, load/save, prerequisites (preflight), run snapshots |
| `src/petsys_manager/commands.py` | Literal argv/cwd for PETsys tools and the internal CLI (no shell) |
| `src/petsys_manager/runner.py` | Owned child processes: bounded logs, cancellation, process-group TERM/KILL (Linux) |
| `src/petsys_manager/acquisition.py` | DAQD ownership/readiness, initialization, monitored acquisition attempts, bias-off |
| `src/petsys_manager/artifacts.py` | Exclusive run/stage/attempt folders, `run.json` and `.history/` records, retained outputs |
| `src/petsys_manager/workflow.py` | Fail-closed stage graphs (manual actions, QC, pipeline), run-folder names, `runs.tsv` |
| `src/petsys_manager/session.py` | GUI-facing session: event queue, background preflight, shutdown |
| `src/cornell/cli.py` | Headless processing CLI: request/result JSON, progress events, exit codes |
| `src/cornell/inputs.py` | Processing config/map/limits/calibration loading, bounded LDAT validation |
| `src/cornell/calibration.py` | Per-slab/position energy calibration (two bounded passes) |
| `src/cornell/listmode.py` | Streamed LM writer with the reference header/record layout |
| `src/cornell/qc.py`, `src/cornell/qc_report.py` | QC analysis and its PDF/Excel/plot reports |
| shared `src/` modules | Readers, mapping (`mapping_generator.py`), fits and helpers reused from the package |

`scripts/`, `scripts_cornell/` and `scripts_imas/` are local and untracked; the runtime never imports them. Development checks for the manager live there (`scripts/petsys_manager_*check.py`). For example, `python scripts/petsys_manager_checkout_check.py --tracked-runtime` audits that a copy of only the tracked runtime files imports and processes fixtures without private settings.
