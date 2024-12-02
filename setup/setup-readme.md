# Environment Setup

## Prerequisites
- Conda or Miniconda installed
- Ubuntu 20.04 or later recommended

## Conda Environment Setup

1. Create a new conda environment:
```bash
conda create -n go2_control python=3.8
conda activate go2_control
```

2. Install required packages:
```bash
# Core dependencies
conda install numpy scipy matplotlib

# Robotics packages
conda install -c conda-forge pinocchio
conda install -c conda-forge eigenpy

# Optional visualization
conda install -c conda-forge meshcat-python
```

## Python Package Dependencies
All Python package dependencies are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```

## Required System Libraries
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    python3-dev \
    python3-numpy
```

## Troubleshooting
[Common issues and solutions here]