# PROCESS_PETSYS

This project provides a Python module to read and process compact data from detectors. It includes filtering and mapping capabilities, and can handle both grouped and ungrouped events.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed the latest version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

## Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/darento/process_petsys.git
```

Then, navigate to the project directory and install the required dependencies:

```bash
cd process_petsys/
conda env create -f process_petsys.yml
```

or, if updating the existing environment:

```bash
conda env update --name process_petsys --file process_petsys.yml
```

After creating or updating the environment, you can activate it using:

```bash
conda activate process_petsys
```

## Configuration

You can configure the behavior of the script by modifying the YAML files in the `configs/` directory. The `maps/` directory contains mapping files that can be used to map the data to different formats.

## Usage

The idea behind it is for everyone to write their own analysis script, starting from `scripts/template.py`, with the functions they need from `src` and a `config.yaml` file along with the
matching `map.yaml`.

### PETsys Manager (Cornell Linux)

Acquisition, RAW conversion, energy calibration, list-mode and QC workflows for the Cornell system, as a separate application from LDAT Inspector:

```bash
python exe_programs/PETsysManager.py [--profile /path/to/petsys_manager.yaml]
```

- **Machine settings:** one YAML profile (default `~/.config/process_petsys/petsys_manager.yaml`).
- **Format routes:**
  - Calibration takes fixed coincidence, fixed group or compact coincidence files.
  - LM takes fixed coincidence files.
  - QC takes compact coincidence files.
  - The complete pipeline always converts to fixed coincidence.
- **Hardware actions:** they need the PETsys tools and DAQ cards and run only on the Cornell Linux machine. They were accepted there on 2026-10-06; PETsys Manager has replaced the sibling `gui_cornell`, which is retired.

Prerequisites, the profile fields, the private input checklist, safety behaviour and the module map are in [`docs/petsys_manager.md`](docs/petsys_manager.md).

### LDAT Inspector (offline)

Run the multi-file PETsys inspector with the `process_petsys` environment:

```powershell
conda run -n process_petsys --no-capture-output python exe_programs/LDATInspector.py
```

Choose an IMAS or Cornell config, its matching energy calibration (or raw a.u.), and one or more `.ldat` coincidence files. The accepted detector sides stay in memory, so the energy, DOI and local X/Y cuts can change without rereading the files.

- **Reading:** a numba reader uses the available cores. Read a prefix per file (the default; 10k to 30M pairs) or tick **Whole files**. A worker cannot return a file result over 4 GiB (about 60 M Cornell pairs), so **Whole files** is refused for a file estimated above 30M pairs: split larger acquisitions into several files; the six-file Cornell set (~15 M pairs) takes about 12 s. An estimated memory use is shown before processing, with a warning above the free RAM, and **Cancel** keeps the previous dataset. Existing environments need `conda env update` for `numba`.
- **Channel Status:** per-SM channels not observed, low or high against that SM's median hit count, with editable thresholds. This is occupancy in the coincidence sample, not a dead/hot hardware verdict.
- **SuperModule:** step through SMs (Prev/Next, mouse wheel, PgUp/PgDn). The tab shows energy, DOI and flood plots, a summary panel, and a per-minimodule table of counts and photopeak fits.
- **System Overview:** every minimodule as a tile coloured by counts, photopeak centroid or resolution, or per-SM flood maps, all on one colour scale for the whole system. Cornell is drawn as the unrolled cylinder of the config geometry: rows are Z, columns are the cassette angle.
- **Coincidences:** SM × SM matrix of accepted pairs and the paired Δt for a chosen SM or SM pair. This is observational (geometry and time of flight contribute), not a CTR calibration.
- **Cornell options:**
  - optional COG/DOI limits files add a slab-assigned flood map and a DOI in mm (a linear light-sharing mapping, not a validated depth);
  - **Slab rule** can recover non-adjacent time-channel pairs;
  - a `_status.txt` next to the `.encal` shows which keV factors were fitted, borrowed or estimated, with a **Fitted keV factors only** cut.
- **Reports:** the system and SuperModule PDFs record provenance (files, scope, calibration, cuts, limits, slab rule), the channel findings and the per-minimodule tables.

The photopeak fit uses the shaded 350–700 keV window, and curves show counts per **displayed** histogram bin. Cornell sides with one time channel keep the legacy random neighbouring-slab assignment, so repeated runs can differ slightly. A prefix does not describe full-run rates, and the input has no singles. The complete behaviour and its checks are in [`specs/001-ldatinspector-parity`](specs/001-ldatinspector-parity/spec.md) and [`specs/002-ldatinspector-scale-views`](specs/002-ldatinspector-scale-views/spec.md).

`scripts/` is not tracked, apart from `scripts/template.py`: copy it to a new name there to write your own analysis on top of the `src` modules (reading, merging, channel findings, photopeak fits).

Inspector implementation lives in `src/ldat_inspector/`: `engine.py` (analysis), `fastread.py` (parallel reader), `memory.py` (estimates) and `report.py` (PDFs). Import from these modules directly, e.g. `from src.ldat_inspector.engine import Settings`; the package itself exports nothing, and the old `src.ldat_fastread`, `src.ldat_memory` and `src.ldat_report` paths no longer exist. Shared readers, mapping and calibration helpers stay in `src/`; the GUI and launcher stay in `exe_programs/`. The move and validation are recorded in [spec004](specs/004-ldat-inspector-package/spec.md).

Feature work throughout this repo follows [spec-driven development](docs/prompts.md); project constraints are in [AGENTS.md](AGENTS.md).

## Documentation

Anyone can access the documentation of the code by simply compiling the docstrings from `docs/` directory such:

```bash
cd docs/
make html
```

Go to `docs/_builds/`, and open the index.html in a browser. You are good to go :).
