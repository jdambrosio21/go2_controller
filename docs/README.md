# Submodules Documentation

## Controllers

### MPC Controller
- Location: `controllers/convex_mpc.py`
- Purpose: Implements Model Predictive Control for quadruped locomotion
- Key Classes:
  - `ConvexMPC`: Main MPC implementation
  - `MPCParams`: Configuration parameters for MPC

### Force Mapper
- Location: `controllers/force_mapper.py`
- Purpose: Maps desired forces to joint torques
- Key Classes:
  - `ForceMapper`: Handles swing and stance force mapping

## Planners

### Gait Scheduler
- Location: `planners/gait_scheduler.py`
- Purpose: Manages gait timing and phase
- Key Classes:
  - `GaitScheduler`: Handles contact scheduling

### Footstep Planner
- Location: `planners/footstep_planner.py`
- Purpose: Plans future footstep locations
- Key Classes:
  - `FootstepPlanner`: Generates footstep trajectories

## State Estimation

### Go2 State Estimator
- Location: `state_estimation/go2_state_estimator.py`
- Purpose: Estimates robot state from sensor data
- Key Classes:
  - `Go2StateEstimator`: Main state estimation implementation

## Utilities

### Quadruped Utils
- Location: `utils/quadruped.py`
- Purpose: Common utility functions for quadruped robots
- Key Functions:
  - Robot kinematics
  - Frame transformations
  - Math utilities