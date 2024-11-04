import numpy as np
import casadi as ca
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MPCParams:
    """Parameters for MPC"""
    dt: float = 0.02            # Timestep
    horizon_steps: int = 10     # Length of horizon

    # Robot params
    mass: float = 12.0          # Mass (should get from pinocchio instead)
    gravity: float = 9.81
    mu: float = 0.04            # Friction Coeff.

    # Force limits
    f_min: float = 0.0          # Min vertical force
    f_max: int = 400.0          # Max vertical force

    # Weights for QP
    w_position = 100.0       # Position tracking weight
    w_orientation = 100.0    # Orientation tracking weight
    w_velocity = 10.0        # Velocity tracking weight 
    w_angular_vel = 10.0     # Angular velocity tracking weight
    w_force = 0.001         # Minimize forces weight


class ConvexMPC():
    def __init__(self, params: MPCParams):
        self.params = params

        # Setup optimization
        self.setup_QP()

    def setup_QP(self):
        """Setup the QP using CasADi"""

        # Create solver instance
        self.opti = ca.Opti('Conic') # Convex QP

        # Decision variables
        self.n_states = 13 # [pos(3), ori(3), lin_vel(3), ang_vel(3), gravity(1)]
        self.n_inputs = 12 # 4 Forces for each foot (xyz)

        # Setup optimization vairables
        self.X = self.opti.variable(self.n_states, self.params.horizon_steps + 1)
        self.U = self.opti.variable(self.n_inputs, self.params.horizon_steps)

        # Parameters that will be set at each solving instance
        self.x0 = self.opti.parameter(self.n_states)
        self.x_ref = self.opti.parameter(self.n_states, self.params.horizon_steps + 1)
        self.contact_sched = self.opti.parameter(4, self.params.horizon_steps) # 4 feet
        self.foot_positions = []    # To hold foot position params

        for k in range(self.params.horizon_steps):
            foot_pos_k = []
            for foot in range(4): # 4 Feet
                pos = self.opti.parameter(3) # xyz
                foot_pos_k.append(pos)
            self.foot_positions.append(foot_pos_k)
        
        self.setup_cost()

        self.add_constraints()
    
    def setup_cost(self):
        """Method to setup QP cost"""

        # Initialize Cost function
        cost = 0

        Q = ca.diag([
            self.params.w_position, self.params.w_position, self.params.w_position,         # Position
            self.params.w_orientation, self.params.w_orientation, self.params.w_orientation, # Orientation
            self.params.w_velocity, self.params.w_velocity, self.params.w_velocity,         # Linear velocity
            self.params.w_angular_vel, self.params.w_angular_vel, self.params.w_angular_vel, # Angular velocity
            0  # gravity state
        ])
        
        R = self.params.w_force * ca.diag(self.n_inputs)
        
        # Add costs for the entire horizon
        for k in range(self.params.horizon_steps):
            # State error cost
            cost += (self.X[:, k] - self.x_ref[:, k]).T @ Q @ (self.X[:, k] - self.x_ref[:, k])

            # Control cost
            cost += self.U[:, k].T @ R @ self.U[:, k]
        
 
    def add_constraints(self):
        """Method to add constraints to the MPC QP"""
        self.add_inital_state_constraint()

        self.add_friction_cone_constraints()

        self.add_dynamics_constraints