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

