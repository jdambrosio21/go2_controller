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
        