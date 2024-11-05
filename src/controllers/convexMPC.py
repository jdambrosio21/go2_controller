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

        self.add_dynamics_constraints()

        self.add_contact_constraints()

    def add_inital_state_constraint(self):
        """Adds the constraint that the first state decision variable equals the robot's starting state"""
        self.opti.subject_to(self.X[:, 0] == self.x0)
    
    def add_dynamics_constraints(self):
        """Adds dynamics constraints to ensure the next state decision variable matches the simplified dynamics"""

        # over the entire horizon
        for k in range(self.params.horizon_steps):
            # Get current state elements
            pos = self.X[0:3, k]
            ori = self.X[3:6, k]
            lin_vel = self.X[6:9, k]
            ang_vel = self.X[9:12, k]
            g = self.X[12, k]

            # Dynamics constraints (equation 16 from paper)
            # Only use yaw rotation (small angle approximation for roll/pitch)
            yaw = ori[2]
            R_yaw = ca.vertcat(
                ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
                ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0),
                ca.horzcat(0, 0, 1)
            )

            # Position dynamics
            pos_next = pos + self.params.dt * lin_vel

            # Simplified orientation dynamics
            ori_next = ori + self.params.dt + ang_vel

            # Velocity dynamics

            # Extract forces for each foot
            forces = []
            for i in range(4):
                f_i = self.U[i * 3:(i + 1) * 3, k]
                forces.append(f_i)

            total_force = sum(forces)
            lin_vel_next = lin_vel + self.params.dt * (
                total_force/self.params.mass - 
                ca.vertcat(0, 0, self.params.gravity)
            )

            # Simplified angular velocity dynamics
            # Calc torques from forces and foot positions
            total_torque = ca.DM.zeros(3)
            for i, (f, p) in enumerate(zip(forces, self.foot_positions[k])):
                    if self.contact_sched[i, k]:
                        total_torque += ca.cross(p, f)
            
            # Get inertia matrix from pinocchio (implement)
            I_body = ca.diag([0.3, 0.3, 0.3])
            ang_vel_next = ang_vel + self.params.dt * ca.solve(I_body, total_torque)

            # Combine next state
            X_next = ca.vertcat(pos_next, ori_next, lin_vel_next, ang_vel_next, g)
            self.opti.subject_to(self.X[:, k + 1] == X_next)
    
    def add_contact_constraints(self):
        """Adds constraints on forces for contact feet and friction cone"""
        
        for k in range(self.params.horizon_steps):
            # Extract forces 
            forces = []
            for i in range(4): # For each foot
                f_i = self.U[i * 3:(i + 1) * 3, k]
                forces.append(f_i)

                # Add force constraints for each foot in contact
                self.opti.bounded(self.params.f_min, self.contact_sched[i, k] * f_i[2], self.params.f_max)

                # Friction cone constraints
                self.opti.bounded(-self.params.mu * f_i[2], self.contact_sched[i, k] * f_i[0], self.params.mu * f_i[2])
                self.opti.bounded(-self.params.mu * f_i[2], self.contact_sched[i, k] * f_i[1], self.params.mu * f_i[2])