import numpy as np
from quadruped import Quadruped
from planners.gait_scheduler import GaitScheduler
from controllers.convex_mpc import ConvexMPC, MPCParams
from planners.footstep_planner import FootstepPlanner

#TODO: Need to get state info for floating base and joints!
#TODO: Get foot and hip position from state estimator vs pinocchio?

class Go2Controller:
    def __init__(self, urdf_path: str):
        # Init componenets
        self.quadruped = Quadruped(urdf_path)
        self.gait = GaitScheduler(total_period=0.5, gait_type="trot")
        self.footstep_planner = FootstepPlanner()

        # Init MPC
        mpc_params = MPCParams(
            dt=0.2,
            horizon_steps=10,
            mass=self.quadruped.mass,
            I_body=self.quadruped.inertia
        )

        self.mpc = ConvexMPC(mpc_params)

    def run(self,
            q: np.ndarray,
            dq: np.ndarray,
            desired_vel: np.ndarray,
            dt: float):
        
        # 1. Update gait
        self.gait.update(dt)    

        # 2. Get current state
        com_pos = q[0:3]
        com_ori = q[3:6] # Convert to quaternion
        com_vel = dq[0:3]
        com_ang_vel = dq[3:6]

        # 3. Plan footsteps
        next_footholds = self.footstep_planner.plan_footsteps(
            com_state=(com_pos, com_vel, com_ori),
            desired_vel=desired_vel,
            robot_model=self.quadruped,
            q_current=q,
            gait_scheduler=self.gait
        )
        current_foot_pos = np.zeros((4, 3))
        for leg in range(4):
            current_foot_pos[leg, :] = self.quadruped.get_foot_positions(q, leg)

        # 4. Get foot positions for MPC horizon
        foot_positions = self.footstep_planner.get_foot_positions_for_mpc(
            current_positions=current_foot_pos, 
            gait_scheduler=self.gait,
            horizon_length=self.mpc.params.horizon_steps,
            dt=dt
            )
        
        # 5. Get contact schedule for horizon
        contact_schedule = self.gait.predict_horizon_contact_state(dt=dt, horizon_length=self.mpc.params.horizon_steps)

        # 6. Create reference trajectory
        x_ref = self._create_reference_trajectory(current_state=q, desired_vel=desired_vel)

        # 7. Get current state for MPC
        x0 = np.concatenate([
            q[0:3],
            q[3:6],
            dq[0:3],
            dq[3:6],
            [9.81]
        ])

        # 8. solve MPC
        forces = self.mpc.solve(
            x0=x0,
            x_ref=x_ref,
            contact_schedule=contact_schedule,
            foot_positions=foot_positions
        )

        return forces#[0]
    
    def _create_reference_trajectory(self, current_state, desired_vel):
        """Create simple reference trajectory"""
        x_ref = np.zeros((13, self.mpc.params.horizon_steps + 1))

        for k in range(self.mpc.params.horizon_steps + 1):
            # Simple ref: constant desired velocity
            x_ref[:, k] = np.concatenate([
                current_state[0:3] + desired_vel * k * self.mpc.params.dt,
                np.zeros(3),
                desired_vel,
                np.zeros(3),
                [9.81]
            ])