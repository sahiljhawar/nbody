# Direct N-body Simulation

This project implements a direct N-body simulation using C++ or JAX as computing backend and Python. This project is developed as my submission for the role of Software Engineer at Leiden University.


## Installation
```bash
git clone git@github.com:sahiljhawar/nbody.git
cd nbody
```

On mac
```bash
CONDA_SUBDIR=osx-arm64 conda create -n nbody python=3.11 -y
conda activate nbody
conda config --env --set subdir osx-arm64
```
On linux
```bash
conda create -n nbody python=3.11 -y
conda activate nbody
```

Install the package
```bash
pip install -r requirements.txt
pip install .
```
## Usage
To run the simulation, use the following command:
```bash
export BACKEND=cpp # or jax
python nbody.py
```

## GitHub Actions
This project uses GitHub Actions for continuous integration. The workflow is defined in `.github/workflows/deploy.yml`. The action runs the `nbody.py` script and produces plots for the simulation results. The plots are then uploaded as artifacts for each run here: https://sahiljhawar.in/nbody/