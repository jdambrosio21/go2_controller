import math
import numpy as np
from planners.gait_scheduler import GaitScheduler
from utils.quadruped import Quadruped

class FootstepPlanner:
    def __init__(self, urdf_path: str, k_raibert = 0.1):
      self.n_legs = 4
      self.k_raibert = k_raibert
      self.next_footholds = np.zeros((4, 3))
      self.quadruped = Quadruped(urdf_path)

    def raibert_heuristic(self, leg: int, stance_dur: float, com_vel: np.ndarray, desired_vel: np.ndarray, q: np.ndarray, for_mpc: bool = False):
        """
        Plans footsteps using Raibert Heuristic

        Args: 
            leg: the leg we are planning for
            stance_dur: How long the foot swill spend on the ground
            com_vel: [3 x 1] vector of the Robot's CoM velocity
            desired_vel: [3 x 1] vector of the desired CoM velocity from the refernce traj (optional)
            q: [19 x 1] vector of the robots current state

        Returns:
            p_des: [3 x 1] desired position of robots foot
        """

        # Get hip position
        # Get hip position in appropriate frame
        if for_mpc:
            p_ref = self.quadruped.get_hip_position_world(q, leg)
        else:
            p_ref = self.quadruped.get_hip_position(q, leg)
        com_vel[-1] = 0

        p_vel = com_vel * (stance_dur / 2)
        # will be zero for planned steps, and non zero for instantaneous footstep planning
        p_correction = (com_vel - desired_vel) * self.k_raibert 

        p_des = p_ref + p_vel + p_correction

        return p_des
    
    def plan_current_footsteps(self, com_vel: np.ndarray, desired_vel: np.ndarray, q_curr: np.ndarray, gait_scheduler: GaitScheduler):
        """
            Plans current footsteps using state estimator info

            Args:
                com_vel: [3 x 1] vector of the robots current CoM Vel from State Estimate
                desired_vel: [3 x 1] vector of the robots desired vel from the Traj
                q_curr: [19 x 1] current robot configuration from state estimate
                gait_scheduler: GaitScheduler Object to get Contact State and Stance Duration
                contact_stance: [4 x 1] of which feet are currently in contact (FL, FR, RL, RR)

            Returns:
                footsteps: [4 x 3] vector of the foosteps for each leg based on the current state
        """
        stance_duration = gait_scheduler.get_stance_duration()
        contact_state = gait_scheduler.get_current_contact_state()

        footsteps = np.zeros((4, 3))
        # For each leg
        for i in range(self.n_legs):
            # If in stance
            if contact_state[i] == 1:
                # Get current foot position
                footsteps[i, :] = self.quadruped.get_foot_positions(q_curr)[i, :]
            
            # If in swing:
            else:
                # Plan next footstep with raibert heuristic
                footsteps[i, :] = self.raibert_heuristic(i, stance_duration, com_vel, desired_vel, q_curr, for_mpc=False)
        
        return footsteps
    
    def plan_horizon_footsteps(self, dt: float, horizon: int, ref_traj: np.ndarray, q_nom: np.ndarray, q_curr: np.ndarray, gait_scheduler: GaitScheduler):
        """
            Plans footsteps for entire MPC horizon to be used in calculations

            Args:
                dt: MPC update rate
                horizon: length of MPC Horizon
                ref_traj: [13 x k] containing CoM Pos, Ori, Ang Vel, Vel for each knot point along the trajectory
                q_nom: [12 x 1] vector containing the joint angles for the Go2's Nominal Posture
                gait_scheduler: Gait Scheduler object used to get stance duration and contact state for horizon

            Returns:
                horizon_footsteps: [12 x k] vector containing the footsteps for each foot along the entire MPC Horizon
        """
        horizon_footsteps = np.zeros((12, horizon))
        q_plan = np.zeros(19)
        q_plan[3:7] = np.array([0, 0, 0, 1]) # Use identity quaternion for now can adapt once i test more arbitrary trajectories
        q_plan[7:] = q_nom
        stance_dur = gait_scheduler.get_stance_duration()
        horizon_contact_state = gait_scheduler.predict_horizon_contact_state(dt, horizon)

        # For each knot point, get configuration to get hip positions
        for k in range(horizon):
            q_plan[0:3] = ref_traj[3:6, k] # CoM Pos
            com_vel = ref_traj[9:12, k]

            # Check each leg at current configuration
            for i in range(4):
                if horizon_contact_state[i, k] == 1: # In stance
                    horizon_footsteps[i*3:(i+1)*3, k] = self.quadruped.get_foot_positions_world(q_plan)[i]
                
                else: # In swing
                    p_des = self.raibert_heuristic(i, stance_dur, com_vel, com_vel, q_plan, for_mpc=True)
                    horizon_footsteps[i*3:(i+1)*3, k] = p_des

        return horizon_footsteps 
                    
            


    