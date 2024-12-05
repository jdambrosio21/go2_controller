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
    mass: float = 15.0          # Mass (should get from pinocchio instead)
    gravity: float = 9.81
    mu: float = 1.0           # Friction Coeff.
    I_body: np.ndarray = None

    # Force limits
    f_min: float = 10.0          # Min vertical force
    f_max: float = 232.0          # Max vertical force

    # Weights for QP
    w_position = 100.0       # Position tracking weight
    w_orientation = 100.0    # Orientation tracking weight
    w_velocity = 1.0        # Velocity tracking weight 
    w_angular_vel = 1.0     # Angular velocity tracking weight
    w_force = 1e-6          # Minimize forces weight


class ConvexMPC():
    def __init__(self, params: MPCParams):
        self.params = params
        self.I_body_ca = ca.DM(self.params.I_body)

        # Setup optimization
        self.setup_QP()

    def setup_QP(self):
        """Setup the QP using CasADi"""

        # Create solver instance
        self.opti = ca.Opti('conic') # Convex QP

        # Decision variables
        self.n_states = 13 # [ori(3), pos(3), ang_vel(3), lin_vel(3), gravity(1)]
        self.n_inputs = 12 # 4 Forces for each foot (xyz)

        # Setup optimization vairables
        self.X = self.opti.variable(self.n_states, self.params.horizon_steps + 1)
        self.U = self.opti.variable(self.n_inputs, self.params.horizon_steps) # [FL(3), FR(3), RL (3), RR (3)]

        # Parameters that will be set at each solving instance
        self.x0 = self.opti.parameter(self.n_states)
        self.x_ref = self.opti.parameter(self.n_states, self.params.horizon_steps + 1)
        self.contact_sched = self.opti.parameter(4, self.params.horizon_steps) # 4 feet

        # Initialize foot positions as a symbolic MX matrix
        self.foot_positions = self.opti.parameter(12, self.params.horizon_steps)
        
        self.add_constraints()
        
        self.setup_cost()

    
    def setup_cost(self):
        """Method to setup QP cost"""

        # Initialize Cost function
        cost = 0

        # self.Q = ca.diag([
        #     self.params.w_position, self.params.w_position, self.params.w_position,         # Position
        #     self.params.w_orientation, self.params.w_orientation, self.params.w_orientation, # Orientation
        #     self.params.w_velocity, self.params.w_velocity, self.params.w_velocity,         # Linear velocity
        #     self.params.w_angular_vel, self.params.w_angular_vel, self.params.w_angular_vel, # Angular velocity
        #     0  # gravity state
        # ])
        self.Q = ca.diag([100,100,50,1,1,1,0,0,1,1,1,1,0]) 
        #self.Q = ca.diag([0.25, 0.25, 10, 2, 2, 50, 0, 0, 0.3, 0.2, 0.2, 0.1, 0.0])
        
        self.R = self.params.w_force * ca.DM.eye(self.n_inputs)
        
        # Add costs for the entire horizon
        for k in range(self.params.horizon_steps): #k-1
            # State error cost
            cost += (self.X[:, k+1] - self.x_ref[:, k+1]).T @ self.Q @ (self.X[:, k+1] - self.x_ref[:, k+1])

            # Control cost
            cost += self.U[:, k].T @ self.R @ self.U[:, k]
        
        self.opti.minimize(cost)
        
 
    def add_constraints(self):
        """Method to add constraints to the MPC QP"""
        self.add_inital_state_constraint()

        self.add_dynamics_constraints()

        self.add_contact_constraints()

    def add_inital_state_constraint(self):
        """Adds the constraint that the first state decision variable equals the robot's starting state"""
        self.opti.subject_to(self.X[:, 0] == self.x0)
    
    # TODO: Get discretized dynamics from reference trajectory and footstep planner 
    def get_discretized_dynamics(self, yaw: float, r: np.ndarray) -> tuple[ca.DM, ca.DM]:
        """
            Calculates linearized discrete time dynamics 

            Args:
                yaw: The desired yaw value at the n-th point in the reference trajectory
                r: a [4 x 3] matrix, where each row is a xyz vector from the CoM to the ith contact point
            
            Returns:
                Ad: A [13 x 13] Matrix representing the discrete time dynamics of the state at the nth point
                Bd: A [13 x 12] Matrix representing the discrete time dynamics of the control input at the nth point
        """
        # Create A matrix for state augmented with gravity
        A = ca.MX.zeros((13, 13))
        # n_k = self.x_ref.shape[1]  # Number of columns (k-values)
        # avg_yaw = ca.sum1(self.x_ref[5, :]) / n_k
        R_z_yaw = ca.vertcat(
            ca.horzcat(ca.cos(yaw), ca.sin(yaw), 0),
            ca.horzcat(-ca.sin(yaw), ca.cos(yaw), 0),
            ca.horzcat(0, 0, 1)
        )

        A[0:3, 6:9] = R_z_yaw.T
        A[3:6, 9:12] = ca.DM_eye(3)
        A[-1, -1] = 1

        # Create B matrix (13 x 3n) n = 4 contact points for 4 legs
        B = ca.MX.zeros((13, 12))
        # Use r directly without nested unpacking
        I_r_x = [ca.mtimes(ca.inv(ca.diag(self.I_body_ca)), ca.skew(r[3 * i : 3 * (i + 1)])) for i in range(4)]
        B[6:9, :] = ca.horzcat(*I_r_x)
        B[9:12, :] = ca.repmat(ca.MX.eye(3) * (1 / self.params.mass), 1, 4)


        # Discretize matrix using matrix exponential
        M = ca.vertcat(
            ca.horzcat(A, B),
            ca.horzcat(ca.DM.zeros((12, 13)), ca.DM_eye(12))
        )
        #  M = ca.vertcat(
        #     ca.horzcat(A, B),
        #     ca.horzcat(ca.DM.zeros((12, 13)), ca.DM.zeros((12,12))))

        M_d = matrix_exp(M * self.params.dt)

        # Extract discrete-time Ad and Bd
        Ad = M_d[:13, :13]
        Bd = M_d[:13, 13:]

        return Ad, Bd

    def add_dynamics_constraints(self):
        """Adds dynamics constraints to ensure the next state decision variable matches the simplified dynamics"""

        # over the entire horizon
        for k in range(self.params.horizon_steps): #k-1
            # Get current state elements
            yaw = self.X[2, k]

            # Get  CoM position from optimization variable, not reference
            com_pos = self.X[3:6, k]  # Current state in optimization

            # Get foot positions for this timestep (12,)
            foot_pos_k = self.foot_positions[:, k]
        
            # Reshape foot positions to 4x3 
            foot_pos_reshaped = foot_pos_k.reshape((4, 3))
        
            # Create r vectors (4x3) from CoM to feet
            r_vectors = foot_pos_reshaped - ca.repmat(com_pos.reshape((1, 3)), 4)

            # Flatten r_vectors to 12, to match input expected by get_discretized_dynamics
            r = ca.vec(r_vectors)
    
            Ad, Bd = self.get_discretized_dynamics(yaw, r)

            X_next = Ad @ self.X[:, k] + Bd @ self.U[:, k]

            self.opti.subject_to(self.X[:, k+1] == X_next)
    
    def add_contact_constraints(self):
        """Adds constraints on forces for contact feet and friction cone"""
        
        for k in range(self.params.horizon_steps): #k-1
            for i in range(4): # For each foot
                f_i = self.U[i * 3:(i + 1) * 3, k]
                contact = self.contact_sched[i, k]

                # Apply vertical force constraint conditionally
                self.opti.subject_to(
                    f_i[2] >= ca.if_else(contact == 1, self.params.f_min, 0)
                )
                self.opti.subject_to(
                    f_i[2] <= ca.if_else(contact == 1, self.params.f_max, 0)
                )

                # Friction cone constraint
                self.opti.subject_to(
                    f_i[0] >= ca.if_else(contact == 1, -self.params.mu * f_i[2], 0)
                )
                self.opti.subject_to(
                    f_i[0] <= ca.if_else(contact == 1, self.params.mu * f_i[2], 0)
                )
                self.opti.subject_to(
                    f_i[1] >= ca.if_else(contact == 1, -self.params.mu * f_i[2], 0)
                )
                self.opti.subject_to(
                    f_i[1] <= ca.if_else(contact == 1, self.params.mu * f_i[2], 0)
                )

                # Forces must be zero during swing phase
                self.opti.subject_to(f_i == ca.if_else(contact == 1, f_i, [0, 0, 0]))


    def solve(self,
              x0: NDArray,
              x_ref: NDArray,
              contact_schedule: NDArray,
              foot_positions: List[List[NDArray]]) -> Optional[NDArray]:
        """Solve the MPC Problem

        Args:
            x0: Initial state [13]
            x_ref: Reference trajectory [13 x horizon_steps + 1]
            contact_schedule: Contact flags [4 x horizon steps]
            foot_positions: Foot positions for horizon [12 x horizon steps]
        
        Returns:
            Optimal forces for first time step [12] or None if failed
        """

        opts = {
            'verbose': False,
            # 'warm_start': True,
            # 'polish': True,
            # 'adaptive_rho': True
        }
        
        self.opti.solver('osqp', opts)
        #self.opti.solver('qpoases', opts)

        try:
            # Set values
            x0 = ca.DM(x0.tolist())
            x_ref = ca.DM(x_ref.tolist())
            contact_schedule = ca.DM(contact_schedule.tolist())
            
            self.opti.set_value(self.x0, x0)
            self.opti.set_value(self.x_ref, x_ref)
            self.opti.set_value(self.contact_sched, contact_schedule)
            self.opti.set_value(self.foot_positions, foot_positions)

            # Get previous solution if available
            if hasattr(self, 'prev_sol'):
                self.opti.set_initial(self.prev_sol.value_variables())
            
            print("\nAttempting to solve...")
            sol = self.opti.solve()
            self.prev_sol = sol # Warm Start
            
            forces = sol.value(self.U)[:, 0]
            for i in range(4):
                print(f"Leg {i} forces: {forces[i*3:(i+1)*3]}")
            
            return forces

        except Exception as e:
            print(f"\nMPC solve failed with error: {e}")
            import traceback
            traceback.print_exc()

def matrix_exp(A, order=0):
    """
    Symbolic matrix exponential using Padé approximation
    A: Input matrix
    order: Order of Padé approximation (higher = more accurate)
    """
    n = A.shape[0]
    I = ca.DM.eye(n)
    A_powers = [I]
    A_current = I
    factorial = 1
    
    # Compute Taylor series terms
    for i in range(1, order + 1):
        factorial *= i
        A_current = A_current @ A
        A_powers.append(A_current / factorial)
    
    # Padé approximation
    N = sum(A_powers[i] for i in range(0, order + 1, 2))
    D = sum(A_powers[i] for i in range(1, order + 1, 2))
    
    return (I + D) @ ca.inv(I - D)