# Direct N-body Simulation

This project implements a direct N-body simulation using C++ and Python. This project is developed as my submission for the role of Software Engineer at Leiden University.


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


write github action to run the script which creates the movie for multiple configurations and post the video on my website as sahiljhawar.in/nbody/