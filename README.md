# PROCESS_PETSYS

This project provides a Python module to read and process compact data from detectors. It includes filtering and mapping capabilities, and can handle both grouped and ungrouped events.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed the latest version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

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
conda env update --name myenv --file process_petsys.yml
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

Choose an IMAS or Cornell config, its matching energy calibration, and one or more `.ldat` files. The interface retains the mapped coincidence detector sides so you can adjust energy, DOI-ratio, and local X/Y filters after processing, inspect channel occupancy, explore SuperModules, compare photopeaks, and generate PDFs. The photopeak fit applies to the shaded 350–700 keV window; curves show counts per **displayed** histogram bin, with a fitted local continuum. Cornell sides with only one time channel retain the existing random neighbouring-slab assignment before keV calibration, so repeated runs can differ slightly. File event limits are **per file**; timestamp plots of a limited prefix do not describe full-run rate. The input has no singles bucket.

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

