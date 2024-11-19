import math
import numpy as np
from planners.gait_scheduler import GaitScheduler
from quadruped import Quadruped

class FootstepPlanner:
    def __init__(self, urdf_path: str, k_raibert = 0.03):
      self.n_legs = 4
      self.k_raibert = k_raibert
      self.next_footholds = np.zeros((4, 3))
      self.quadruped = Quadruped(urdf_path)

    def plan_footsteps(self, 
                       com_state: tuple[np.ndarray, np.ndarray, np.ndarray], 
                       desired_vel: np.ndarray, 
                       q_current: np.ndarray, 
                       gait_scheduler: GaitScheduler) -> np.ndarray:
        """
        Plan footsteps using Raibert Heuristic

        Args:
            com_state: Current CoM Pos, Vel, and Yaw
            desired_vel: Desired CoM Velocity from the reference trajectory
            q_current: current robot configuration
            gait_scheduler: Gait Scheduler to define gait info
        
        Returns: 
            next_footholds: Next foot placements based on Raibert Heuristic in world frame
        """
        # Get stance time for Planning
        stance_duration = gait_scheduler.get_stance_duration()

        # Get which feer are in stance or swing
        contact_state = gait_scheduler.get_current_contact_state()

        # Plan landing positions for swing feet using Raibert heuristic
        for leg in range(self.n_legs):
            if contact_state[leg] == 0: # In swing
                # Get hip position in world frame from pinocchio
                p_hip = self.quadruped.get_hip_position(q_current, leg)

                # Raibert Heuristic
                p_des = (p_hip + com_state[1] * (stance_duration / 2) + self.k_raibert * (com_state[1] - desired_vel))

                self.next_footholds[leg] = p_des
            
        return self.next_footholds
    
    def get_foot_positions_for_mpc(self,
                                   current_positions: np.ndarray,
                                   gait_scheduler: GaitScheduler, 
                                   horizon_length: int,
                                   dt: float) -> np.ndarray:
        """
        Plans footsteps for the entire MPC horizon

        Params:
            current_positions: current robot state
            gait_scheduler: Gait Scheduler to define gait info
            horizon_length: length of the MPC horizon
            dt: MPC update rate

        Returns:
            foot_positions: Planned foot positions for length of the entire horizon
        """
        # Get stance states across horizon
        horizon_states = gait_scheduler.predict_horizon_contact_state(dt, horizon_length)

        # For each timestep in horizon
        foot_positions = np.zeros((horizon_length, self.n_legs * 3))
        for k in range(horizon_length):
            for leg in range(self.n_legs):
                if horizon_states[k, leg] == 1: # If in stance
                    # Use current robot position (from state estimate or ground truth in sim?)
                    foot_positions[k, leg*3:(leg+1)*3] = current_positions[leg]
                else: # If in swing
                    # Use planned landing position
                    foot_positions[k, leg*3:(leg+1)*3] = self.next_footholds[leg]

        return foot_positions

    

