# PETsys Manager: Cornell deployment

PETsys Manager runs the Cornell acquisition, RAW conversion, energy calibration, list-mode (LM) and quality-control (QC) workflows from this repository. It is a separate application from LDAT Inspector and replaces the sibling `gui_cornell` GUI only after operator acceptance ([spec003](../specs/003-petsys-manager-migration/spec.md), FR-17/FR-18). Until then the sibling repository, its scripts and any desktop shortcuts stay as they are. Retiring them is a separate owner decision.

## Status

| Area | State |
|---|---|
| Offline processing (calibration, LM, offline QC) | Checked on synthetic fixtures and read-only Cornell structure probes (Windows) |
| Hardware workflows (DAQD, initialize, acquire, live QC, pipeline) | Checked with fake and dummy processes only; **not yet run on the Cornell machine** (T19) |
| Linux runtime environment | Clean-checkout audit (5/5, production process backend) and dummy process checks (16/16) pass on the Cornell machine (2026-10-02) |
| Comparison with the installed reference scripts on real data | **Pending** (T19) |

## Prerequisites (Cornell Linux machine)

- **The `process_petsys` conda environment** from `process_petsys.yml`. It covers numpy, scipy, matplotlib, pandas, pyyaml, numba, openpyxl, reportlab, tqdm and customtkinter; Pillow comes in through matplotlib/reportlab. After pulling, update an existing environment with `conda env update --name process_petsys --file process_petsys.yml`.
  - Packages in a user's `~/.local` site-packages can hide gaps in the environment: on the Cornell machine `customtkinter` was found there and not in the environment.
  - Check with `conda run -n process_petsys python -s -c "import numpy, yaml, scipy, matplotlib, pandas, numba, openpyxl, reportlab, tqdm, customtkinter, PIL, tkinter"`; `-s` ignores `~/.local`.
  - Install anything missing with `conda run -n process_petsys python -s -m pip install <package>==<version from process_petsys.yml>`. Plain `pip install` reports a `~/.local` copy as already installed.
- **A desktop session** for the Tk window. Processing runs in background child processes and does not need the display.
- **The PETsys `sw_daq_tofpet2` tools folder**, containing `daqd`, `init_system`, `acquire_sipm_data`, `set_bias`, `convert_raw_to_coincidence` and `convert_raw_to_group`. Fixed coincidence output needs the converter build with `--writeBinaryFixed`; confirm it in the profile (`capabilities.fixed_output_confirmed`) after checking the installed converter.
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
  workers: 1
  batch_records: 5000
  calibration_event_limit: 10000000          # passing coincidences per file (reference)
  qc_pair_limit: 1000001                     # accepted pairs per file (reference)
  log_tail_lines: 1000
capabilities:
  fixed_output_confirmed: false              # set true only after checking the converter
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
| Acquire | Acq. Time, optional hardware trigger | `.rawf`/`.idxf` (`data_dir`) |
| Convert, coincidence | `.rawf` | fixed **or** compact coincidence `.ldat`, chosen explicitly (`data_dir`) |
| Convert, group | `.rawf` | fixed group `.ldat` (`data_dir`) |
| Create energy cal file | fixed coincidence, fixed group or compact coincidence `.ldat`; positions per slab P | P = 1: per-slab `.encal`; P ≥ 2: position `.encal` with COG regions; plus status, sidecar and plot (`calibration_dir/<run>`) |
| Generate LM file | **fixed coincidence** `.ldat`, `.encal`, limits, pair/region maps, LM metadata | `.lm` with provenance and optional debug plots (`lm_dir/<run>`) |
| Run quality control (live) | 60 s with source or 180 s without (fixed presets) | compact coincidence conversion, then QC report (run folder in `data_dir`) |
| Analyze existing compact LDAT | **compact coincidence** `.ldat` | QC report (`report_dir/<run>`) |
| Run complete pipeline | Acq. Time, splits, max hits, positions per slab | acquire → **fixed** coincidence conversion → calibration → LM, all in one run folder in `data_dir` |

`.ldat` files carry no reliable format marker, so every input list states its format and population and must be confirmed before use. Inputs are validated structurally, within bounds, against the selected map before any result is published.

Not available: singles counts (LDAT coincidence records contain two detectors, not a singles population), keV energies in QC (QC reports raw a.u.), calibrated DOI depth (the DOI value is a light-sharing ratio), and hardware actions off the Cornell Linux machine.

## Safety semantics

- **DAQD ownership:** the manager starts and owns one `daqd`. An existing `/tmp/d.sock` or `/dev/shm/daqd_shm` blocks start and is never deleted: the installed `daqd` removes another daemon's shared memory when it fails its own exclusive create.
  - READY requires the owned process to answer the DAQD shared-memory query, not just a socket file.
  - Daemon death, or a change of INI or cards, invalidates initialization.
  - Stop and close signal only the owned process.
- **Acquisition:**
  - A monitored attempt needs the RAW file to start within `startup_timeout_s` and grow by `min_growth_bytes` within `growth_window_s`, with frame loss at most `max_loss_percent`.
  - Only startup timeout, no growth, no data and frame loss are retried, up to `max_attempts`, each to a new path. A nonzero exit is a failure and is not retried.
  - The GUI shows RAW started, growth passed, live size and rate, and a stall warning.
- **SiPM bias:** after an attempt that did not end normally (STOP, abort, nonzero exit), the manager runs `set_bias --power off` before anything else. If that fails, a persistent "SiPM bias state unknown" warning appears, no retry starts, and Acquire stays locked until the operator confirms the bias state. On close, a running bias-off finishes before DAQD stops.
- **STOP:** stops the running stage, and no later stage of that run starts.
- **Outputs:**
  - Every run, stage and attempt gets a new folder, created exclusively; existing files are never overwritten.
  - Failed or partial outputs are kept and marked unvalidated.
  - A run manifest records the settings snapshot, the exact inputs and every output with its hash. Later stages use only the outputs recorded for this run, never similarly named files.
- **Calibration:** a new `.encal` is never applied automatically. The GUI offers it for LM as an unsaved profile edit.
- **QC:** findings are observations of the coincidence sample (occupancy, fits in a.u.), not a detector PASS/FAIL verdict.
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
| `src/petsys_manager/artifacts.py` | Exclusive run/stage/attempt folders, manifests, retained outputs |
| `src/petsys_manager/workflow.py` | Fail-closed stage graphs (manual actions, QC, pipeline) |
| `src/petsys_manager/session.py` | GUI-facing session: event queue, background preflight, shutdown |
| `src/cornell/cli.py` | Headless processing CLI: request/result JSON, progress events, exit codes |
| `src/cornell/inputs.py` | Processing config/map/limits/calibration loading, bounded LDAT validation |
| `src/cornell/calibration.py` | Per-slab/position energy calibration (two bounded passes) |
| `src/cornell/listmode.py` | Streamed LM writer with the reference header/record layout |
| `src/cornell/qc.py`, `src/cornell/qc_report.py` | QC analysis and its PDF/Excel/plot reports |
| shared `src/` modules | Readers, mapping (`mapping_generator.py`), fits and helpers reused from the package |

`scripts/`, `scripts_cornell/` and `scripts_imas/` are local and untracked; the runtime never imports them. Development checks for the manager live there (`scripts/petsys_manager_*check.py`). For example, `python scripts/petsys_manager_checkout_check.py --tracked-runtime` audits that a copy of only the tracked runtime files imports and processes fixtures without private settings.
