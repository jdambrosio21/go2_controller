# Go2 MPC Controller

A Model Predictive Controller (MPC) implementation for the Unitree Go2 quadruped robot.

## Overview
This repository contains a Python implementation of a Convex Model Predictive Controller for quadruped locomotion, based on the MIT Cheetah 3 paper.

## Prerequisites
First install the Unitree Mujoco simulator from:
https://github.com/unitreerobotics/unitree_mujoco

## Quick Start
1. Clone this repository
```bash
git clone https://github.com/yourusername/go2_controller.git
cd go2_controller
```

2. Set up the environment:
- See [setup/README.md](setup/README.md) for detailed environment setup instructions

3. Install dependencies:
```bash
conda install --yes --file requirements.txt
```

## Project Structure
- `controllers/`: MPC and force mapping implementations
- `planners/`: Gait scheduling and trajectory planning
- `state_estimation/`: Robot state estimation
- `utils/`: Helper functions and utilities
- `setup/`: Environment setup instructions and scripts

## Documentation
- Environment Setup: [setup/README.md](setup/README.md)
- Submodule Documentation: [docs/SUBMODULES.md](docs/SUBMODULES.md)

## Usage
[Basic usage instructions here]

## References
- Di Carlo et al. "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"
