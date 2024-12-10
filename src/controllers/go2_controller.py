import numpy as np
import time
import sys
from enum import Enum
from typing import List, Optional

# Unitree SDK
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# Implemented classes
from controllers.convex_mpc import ConvexMPC, MPCParams
from controllers.force_mapper import ForceMapper
from planners.gait_scheduler import GaitScheduler
from planners.footstep_planner import FootstepPlanner
from planners.foot_swing_trajectory import FootSwingTrajectory
from state_estimation.go2_state_estimator import Go2StateEstimator
from state_estimation import unitree_legged_const as go2
from utils.quadruped import Quadruped


class ControlState(Enum):
    STAND_UP = 1
    NORMAL_OPERATION = 2


class Go2Controller:
    def __init__(self, urdf_path: str):
        # Initialize all control and planning components
        self.state_estimator = Go2StateEstimator()
        self.gait_scheduler = GaitScheduler(total_period=0.3, gait_type="trot")
        self.footstep_planner = FootstepPlanner(urdf_path)
        self.force_mapper = ForceMapper(urdf_path)
        self.robot = Quadruped(urdf_path)

        # Control timing (match MIT Cheetah)
        self.dt = 0.001  # 1kHz base control rate
        self.iterationsBetweenMPC = 30  # Run MPC every 40 control iterations
        self.mpc_dt = self.dt * self.iterationsBetweenMPC  # MPC timestep
        self.iteration_counter = 0

        # MPC Parameters
        self.alpha = 4e-5  # Regularization
        self.mpc_horizon_steps = 16
        self.mpc_params = MPCParams(
            mass=self.robot.mass,
            I_body=self.robot.inertia,
            dt=self.mpc_dt,  # Use MPC timestep
            horizon_steps=self.mpc_horizon_steps
        )

        self.mpc = ConvexMPC(self.mpc_params)

        # Initialize state variables
        self.current_mpc_forces = None
        self.swing_trajectories = {leg: None for leg in ["FL", "FR", "RL", "RR"]}

        # Standing poses
        self.stand_up_joint_pos = np.array(
            [
                0.00571868,
                0.608813,
                -1.21763,  # FR
                -0.00571868,
                0.608813,
                -1.21763,  # FL
                0.00571868,
                0.608813,
                -1.21763,  # RR
                -0.00571868,
                0.608813,
                -1.21763,  # RL
            ]
        )

        self.stand_down_joint_pos = np.array(
            [
                0.0473455,
                1.22187,
                -2.44375,
                -0.0473455,
                1.22187,
                -2.44375,
                0.0473455,
                1.22187,
                -2.44375,
                -0.0473455,
                1.22187,
                -2.44375,
            ]
        )

        # Define in order for pinocchio
        self.q_nom = np.zeros(12)
        self.q_nom[0:3] = self.stand_up_joint_pos[3:6]  # FL
        self.q_nom[3:6] = self.stand_up_joint_pos[0:3]  # FR
        self.q_nom[6:9] = self.stand_up_joint_pos[9:]  # RL
        self.q_nom[9:] = self.stand_up_joint_pos[6:9]  # RR

        # Initialize torque storage
        self.current_torques = {
            'FL': np.zeros(3),
            'FR': np.zeros(3),
            'RL': np.zeros(3),
            'RR': np.zeros(3)
        }

        self.leg_torques1 = {
                    'FL': np.array([5.62967769, -13.88385192, -8.77028072]),
                    'FR': np.array([-6.06997512, 23.7, 9.82761581]),
                    'RL': np.array([0.34408537, 0.41120293, -0.70623759]),
                    'RR': np.array([-23.7, 11.66, 6.4715]),
                }
        self.leg_torques2 = {
                    'FL': np.array([0.62503146, 0.45349752, -0.80300164]),
                    'FR': np.array([20.68708318, -23.7, -45.4]),
                    'RL': np.array([23.7, 23.7, 36.76795121]),
                    'RR': np.array([2.00157831, 0.40239766, -0.71061104]),
                }


        # Create leg order mappings
        self.mpc_to_go2_mapping = {
            "FL": (3, 6),  # Maps to go2_torques[0:3]
            "FR": (0, 3),  # Maps to go2_torques[3:6]
            "RL": (9, 12),  # Maps to go2_torques[6:9]
            "RR": (6, 9),  # Maps to go2_torques[9:]
        }

        # Initialize communication components
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.initialize_command()
        self.crc = CRC()

        self.control_state = ControlState.STAND_UP
        self.running_time = 0
        self.last_mpc_time = 0
        self.toggle = True  # Initial toggle state
        self.phase = 0

    def initialize_command(self):
        """Initialize command message"""
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0

        # Initialize all motors
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

    def run_control_loop(self):
        """Main 1kHz low-level Control Loop"""
        while True:
            step_start = time.perf_counter()

            try:
                # Update running time
                self.running_time += self.dt
                self.iteration_counter += 1

                # Stand up for first 3 seconds
                if self.running_time < 3.0:
                    self.execute_stand_up()
                    
                    
                # Main control loop
                else:
                    # Get state at 1 kHz
                    q, dq = self.state_estimator.get_state()

                    # Update MPC and get new forces
                    x_ref = self._create_reference_trajectory(q, np.array([1.0, 0, 0]))

                    self.run_mpc_update(q, dq, x_ref)
                    #time.sleep(0.001)  # Optional: give solver and system a short break
                        

                    # # Control layer - 1 kHz operations
                    if self.current_mpc_forces is not None:
                        # Get contact state
                        contact_state = self.gait_scheduler.get_current_contact_state()

                        # Update swing trajectories
                        self.update_swing_trajectories(q, dq, x_ref, contact_state)

                        # Compute new torques
                        self.compute_leg_torques(q, dq, contact_state)

                        # Apply current torques
                        self.send_torques()

                # Send command with CRC
                self.cmd.crc = self.crc.Crc(self.cmd)
                self.pub.Write(self.cmd)

                # Enforce loop timing with explicit sleep
                elapsed = time.perf_counter() - step_start
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"Control loop error: {e}")
                import traceback

                traceback.print_exc(e)
                break

    def run_mpc_update(self, q: np.ndarray, dq: np.ndarray, x_ref: np.ndarray):
        """Run MPC Update at desired Hz (50)"""

        if (self.iteration_counter % self.iterationsBetweenMPC) == 0:
            print(f"\nMPC Update at iter {self.iteration_counter}")

            self.gait_scheduler.update(self.mpc_dt)

            # Get contact schedule for horizon
            contact_schedule = self.gait_scheduler.predict_horizon_contact_state(
                self.mpc_dt, self.mpc_horizon_steps
            )

            # Plan footsteps for horizon
            foot_pos_horizon = self.footstep_planner.plan_horizon_footsteps(
                self.mpc_dt, self.mpc_horizon_steps, x_ref, self.q_nom, self.gait_scheduler
            )

            # Run MPC
            rpy = self.robot.quat_to_rpy(q[3:7])
            x0 = np.concatenate(
                [
                    rpy,  # orientation
                    q[0:3],  # position
                    dq[3:6],  # angular velocity
                    dq[0:3],  # linear velocity
                    [9.81],  # gravity
                ]
            )

            # Only update forces if solve succeeds
            try:
                new_forces = self.mpc.solve(x0, x_ref, contact_schedule, foot_pos_horizon)
                if new_forces is not None:
                    self.current_mpc_forces = new_forces
                    print("New MPC solution computed")
            except Exception as e:
                print(f"MPC solve failed: {e}")
                # Keep using previous forces

    def update_swing_trajectories(
        self, q: np.ndarray, dq: np.ndarray, x_ref: np.ndarray, contact_state: List[int]
    ):
        """Update swing trajectories for each leg at 1 kHz"""
        # Get current foot positions using Pinocchio FK
        foot_position = self.robot.get_foot_positions(q)

        # Plan next footsteps for any legs currently in swing
        next_footholds = self.footstep_planner.plan_current_footsteps(
            dq[0:3], x_ref[6:9, 0], q, self.gait_scheduler
        )

        # Update each leg's swing trajectory
        for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
            if contact_state[i] == 0:  # Swing
                # Get current swing phase and duration
                phase = self.gait_scheduler.get_swing_phase(i)
                swing_duration = self.gait_scheduler.get_swing_duration()

                # Create new traj at the start of the swing phase (phase = 0)
                # Ensuring we get new traj each time the leg starts swinging
                if phase < 0.01 or self.swing_trajectories[leg] is None:
                    self.swing_trajectories[leg] = FootSwingTrajectory(
                        foot_position[i], next_footholds[i], 0.5
                    )

                # Update the trajectory
                self.swing_trajectories[leg].compute_swing_trajectory_bezier(
                    phase, swing_duration
                )

            else:
                # Stance, clear the trajectory
                self.swing_trajectories[leg] = None


    def compute_leg_torques(self, q, dq, contact_state):
        """Compute torques from MPC forces and store them"""
        for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
            idx = slice(i*3, (i+1)*3)
            if contact_state[i] == 1:  # Stance
                force = self.current_mpc_forces[idx]
                print(f"\n{leg} Stance Force: {force}")
                self.current_torques[leg] = self.force_mapper.compute_stance_torques(leg, q, dq, force)
                print(f"{leg} Stance Torques: {self.current_torques[leg]}")
            else:  # Swing
                traj = self.swing_trajectories[leg]
                if traj is not None:
                    self.current_torques[leg] = self.force_mapper.compute_swing_torques(
                        leg, q, dq, traj.p, traj.v, traj.a
                    )
                    print(f"{leg} Swing Torques: {self.current_torques[leg]}")

    def send_torques(self):  # torques: np.ndarray):
        """Apply computed torques to the robot"""
        # First, configure all motors for torque control
        for i in range(12):
            self.cmd.motor_cmd[i].q = self.stand_up_joint_pos[i]
            self.cmd.motor_cmd[i].dq = 0.0

        # Then apply the computed torques
        for i in range(3):
            self.cmd.motor_cmd[i].tau = float(self.current_torques['FR'][i])
            self.cmd.motor_cmd[i + 3].tau = float(self.current_torques['FL'][i])
            self.cmd.motor_cmd[i + 6].tau = float(self.current_torques['RR'][i])
            self.cmd.motor_cmd[i + 9].tau = float(self.current_torques['RL'][i])
        


    def execute_stand_up(self):
        """Execute standing sequence"""
        self.phase = np.tanh(self.running_time / 1.2)
        for i in range(12):
            self.cmd.motor_cmd[i].q = (
                self.phase * self.stand_up_joint_pos[i]
                + (1 - self.phase) * self.stand_down_joint_pos[i]
            )
            self.cmd.motor_cmd[i].kp = self.phase * 50.0 + (1 - self.phase) * 20.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 3.5
            self.cmd.motor_cmd[i].tau = 0.0

    def _create_reference_trajectory(self, current_state, desired_vel):
        """Create linear reference trajectory for MPC horizon"""
        x_ref = np.zeros((13, self.mpc.params.horizon_steps + 1))

        # Set initial position to current position
        x_ref[0:3, 0] = current_state[0:3]  # Starting COM position

        # Set desired height
        desired_height = 0.25
        x_ref[2, :] = desired_height  # Maintain constant height

        # Use a fixed dt for trajectory generation (e.g. 0.02s)
        traj_dt = 0.02  
        
        # Propagate position linearly based on desired velocity
        for k in range(1, self.mpc.params.horizon_steps + 1):
            dt = k * traj_dt  # Use fixed dt instead of mpc_dt
            x_ref[0:3, k] = current_state[0:3] + desired_vel * dt

        # Set desired velocities (constant)
        x_ref[6:9, :] = np.tile(desired_vel.reshape(3, 1), 
                            (1, self.mpc.params.horizon_steps + 1))

        # Set gravity state
        x_ref[12, :] = 9.81

        return x_ref