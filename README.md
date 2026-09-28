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

The idea behind it is for everyone to create their own `main.py` script with the desired functionalities taking the necessary functions from the module and defining the `config.yaml` file along with the
matching `map.yaml`.

### LDAT Inspector (offline)

Run the multi-file PETsys inspector with the `process_petsys` environment:

```powershell
conda run -n process_petsys --no-capture-output python exe_programs/LDATInspector.py
```

Choose an IMAS or Cornell config, its matching energy calibration (or raw a.u.), and one or more `.ldat` coincidence files. The accepted detector sides stay in memory, so the energy, DOI and local X/Y cuts can change without rereading the files.

- **Reading:** a numba reader uses the available cores. Read a prefix per file (the default) or tick **Whole files**; the six-file Cornell set (~15 M pairs) takes about 12 s. An estimated memory use is shown before processing, with a warning above the free RAM, and **Cancel** keeps the previous dataset. Existing environments need `conda env update` for `numba`.
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

Feature work throughout this repo follows [spec-driven development](docs/prompts.md); project constraints are in [AGENTS.md](AGENTS.md).

You can run the main script with the following command:

```bash
python main.py configs\<your_config.yml>
```

## Documentation

Anyone can access the documentation of the code by simply compiling the docstrings from `docs/` directory such:

```bash
cd docs/
make html
```

Go to `docs/_builds/`, and open the index.html in a browser. You are good to go :).
