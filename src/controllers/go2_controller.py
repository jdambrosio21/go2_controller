import numpy as np
import time
import sys
from typing import List, Optional

# Unitree SDK
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
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

class Go2Controller:
    def __init__(self, urdf_path: str):
        # Initialize all control and planning components
        self.state_estimator = Go2StateEstimator()
        self.gait_scheduler = GaitScheduler(total_period=0.4, gait_type="trot")
        self.footstep_planner = FootstepPlanner(urdf_path)
        self.force_mapper = ForceMapper(urdf_path)
        self.robot = Quadruped(urdf_path)

        # Control Frequencies
        self.mpc_dt = 0.02      # 50 Hz for MPC 
        self.control_dt = 0.001 # 1 kHz for leg control and planning

        # MPC Parameters
        self.mpc_horizon_steps = 10
        mpc_params = MPCParams(
            mass=self.robot.mass,
            I_body=self.robot.inertia,
            dt=self.mpc_dt,
            horizon_steps=self.mpc_horizon_steps
        )

        self.mpc = ConvexMPC(mpc_params)

        # Initialize state variables
        self.last_mpc_time = 0
        self.current_mpc_forces = None
        self.swing_trajectories = {leg: None for leg in ["FL", "FR", "RL", "RR"]}

        # Standing poses
        self.stand_up_joint_pos = np.array([
            0.00571868, 0.608813, -1.21763,     # FR
            -0.00571868, 0.608813, -1.21763,    # FL
            0.00571868, 0.608813, -1.21763,     # RR
            -0.00571868, 0.608813, -1.21763     # RL
        ])

        self.stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, 
            -0.0473455, 1.22187, -2.44375, 
            0.0473455, 1.22187, -2.44375, 
            -0.0473455, 1.22187, -2.44375
        ])

        
        # Define in order for pinocchio
        self.q_nom = np.zeros(12)
        self.q_nom[0:3] = self.stand_up_joint_pos[3:6]  # FL
        self.q_nom[3:6] = self.stand_up_joint_pos[0:3]  # FR
        self.q_nom[6:9] = self.stand_up_joint_pos[9:]   # RL
        self.q_nom[9:]  = self.stand_up_joint_pos[6:9]  # RR

        # Initialize torque arrays with explicit shapes and names
        self.torques = np.zeros(12)  # [FL, FR, RL, RR] order
        self.go2_torques = np.zeros(12)  # Go2 hardware order
    
        # Create leg order mappings
        self.mpc_to_go2_mapping = {
            'FL': (3, 6),   # Maps to go2_torques[0:3]
            'FR': (0, 3),   # Maps to go2_torques[3:6]
            'RL': (9, 12),  # Maps to go2_torques[6:9]
            'RR': (6, 9)    # Maps to go2_torques[9:]
        }

        # Initialize communication components
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.initialize_command()
        self.crc = CRC()

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
        last_mpc_update = time.perf_counter()
        last_state = None  # Cache state computation results
        last_state_time = 0
    
        while True:
            loop_start = time.perf_counter()
            try:
                # State update at lower rate (100Hz)
                current_time = time.perf_counter()
                if last_state is None or current_time - last_state_time >= 0.001:
                    q, dq = self.state_estimator.get_state()
                    if q is None or dq is None:
                        continue
                    last_state = (q, dq)
                    last_state_time = current_time
                else:
                    q, dq = last_state

                # Update gait at 1kHz
                self.gait_scheduler.update(self.control_dt)  # Using control_dt instead of mpc_dt

                 ##**Compute x_ref before the if block**
                x_ref = self._create_reference_trajectory(q, np.array([0.1, 0, 0]))

            
                
                # 2. Run MPC if enough time has passed (50 Hz)
                if current_time - last_mpc_update >= self.mpc_dt:
                    #x_ref = self._create_reference_trajectory(q, np.array([0.1, 0, 0]))

                    self.run_mpc_update(q, dq, x_ref)
                    #self.last_mpc_time = current_time
                    last_mpc_update = current_time

                    # Update gait timing if needed
                    #self.gait_scheduler.update(self.mpc_dt)
                
                # 3. Update gait and foot trajectories (1 kHz)
                # 3. Fast Loop (1 kHz) - Use stored MPC forces
                if self.current_mpc_forces is not None:
                # Get current contact state from gait scheduler
                    contact_state = self.gait_scheduler.get_current_contact_state()
                    
                    # Update swing trajectories
                    self.update_swing_trajectories(q, dq, x_ref, contact_state) #???

                    # 4. Compute and apply torques using current MPC forces (1 kHz)
                    #self.torques = 
                    self.compute_leg_torques(q, dq, contact_state)
            
                    # Apply torques
                    #self.apply_torques(self.torques)
                    self.apply_torques()

                # Strict timing enforcement
                elapsed = time.perf_counter() - loop_start
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
                else:
                    print(f"Control loop overtime: {elapsed*1000:.2f} ms")

            except Exception as e:
                print(f"Control loop error: {e}")
                import traceback
                traceback.print_exc(e)
                break

    def run_mpc_update(self, q: np.ndarray, dq: np.ndarray, x_ref: np.ndarray):
        """Run MPC Update at desired Hz (50)"""

        # Update gait scheduler
        #self.gait_scheduler.update(self.mpc_dt)

        # Get contact schedule for horizon
        contact_schedule = self.gait_scheduler.predict_horizon_contact_state(self.mpc_dt, self.mpc_horizon_steps)

        # Plan footsteps for horizon
        foot_pos_horizon = self.footstep_planner.plan_horizon_footsteps(
            self.mpc_dt, 
            self.mpc_horizon_steps, 
            x_ref, 
            self.q_nom, 
            self.gait_scheduler
            )
        
        # Run MPC
        rpy = self.robot.quat_to_rpy(q[3:7])
        x0 = np.concatenate([
            q[0:3],     # position
            rpy,        # orientation
            dq[0:3],    # linear velocity
            dq[3:6],    # angular velocity
            [9.81],     # gravity
        ])

        self.current_mpc_forces = self.mpc.solve(x0, x_ref, contact_schedule, foot_pos_horizon)

    def update_swing_trajectories(self, q: np.ndarray, dq: np.ndarray, x_ref: np.ndarray, contact_state: List[int]):
        """Update swing trajectories for each leg at 1 kHz"""
        # Get current foot positions using Pinocchio FK
        foot_position = self.robot.get_foot_positions(q)

        # Plan next footsteps for any legs currently in swing
        next_footholds = self.footstep_planner.plan_current_footsteps(dq[0:3], x_ref[6:9, 0], q, self.gait_scheduler)

        # Update each leg's swing trajectory
        for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
            if contact_state[i] == 0: # Swing
                # Get current swing phase and duration
                phase = self.gait_scheduler.get_swing_phase(i)
                swing_duration = self.gait_scheduler.get_swing_duration()

                # Create new traj at the start of the swing phase (phase = 0)
                # Ensuring we get new traj each time the leg starts swinging
                if phase < 0.01 or self.swing_trajectories[leg] is None:
                    self.swing_trajectories[leg] = FootSwingTrajectory(
                        foot_position[i],
                        next_footholds[i],
                        0.05
                    )

                # Update the trajectory
                self.swing_trajectories[leg].compute_swing_trajectory_bezier(phase, swing_duration)
            
            else:
                # Stance, clear the trajectory
                self.swing_trajectories[leg] = None

    def compute_leg_torques(self, q: np.ndarray, dq: np.ndarray, contact_state: List[int]) -> np.ndarray:
        """Compute torques for all legs at 1 kHz"""

        for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
            idx = slice(i*3, (i+1)*3)  # Create slice object for cleaner indexing
            if contact_state[i] == 1:  # Stance
                # Get forces for this leg from MPC horizon
                force = self.current_mpc_forces[idx]
                self.torques[idx] = self.force_mapper.compute_stance_torques(leg, q, dq, force)
            else:  # Swing
                traj = self.swing_trajectories[leg]
                if traj is not None:
                    self.torques[idx] = self.force_mapper.compute_swing_torques(leg, q, dq, traj.p, traj.v, traj.a)

    
    def apply_torques(self): #torques: np.ndarray):
        """Apply computed torques to the robot"""

        # Re order torques
        # Reorder using slice mapping
        for leg, (start, end) in self.mpc_to_go2_mapping.items():
            self.go2_torques[start:end] = self.torques[start:end]
        
        # Apply to motors
        for i in range(12):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = float(self.go2_torques[i])

        # Get torques in MPC order [FL, FR, RL, RR]
        # Map to Go2's motor indices using LegID
        # FL torques
        # self.cmd.motor_cmd[go2.LegID["FL_0"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["FL_1"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["FL_2"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["FL_0"]].tau = float(torques[0])
        # self.cmd.motor_cmd[go2.LegID["FL_1"]].tau = float(torques[1])
        # self.cmd.motor_cmd[go2.LegID["FL_2"]].tau = float(torques[2])

        # # FR torques
        # self.cmd.motor_cmd[go2.LegID["FR_0"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["FR_1"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["FR_2"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["FR_0"]].tau = float(torques[3])
        # self.cmd.motor_cmd[go2.LegID["FR_1"]].tau = float(torques[4])
        # self.cmd.motor_cmd[go2.LegID["FR_2"]].tau = float(torques[5])

        # # RL torques
        # self.cmd.motor_cmd[go2.LegID["RL_0"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["RL_1"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["RL_2"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["RL_0"]].tau = float(torques[6])
        # self.cmd.motor_cmd[go2.LegID["RL_1"]].tau = float(torques[7])
        # self.cmd.motor_cmd[go2.LegID["RL_2"]].tau = float(torques[8])

        # # RR torques
        # self.cmd.motor_cmd[go2.LegID["RR_0"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["RR_1"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["RR_2"]].mode = 0x01
        # self.cmd.motor_cmd[go2.LegID["RR_0"]].tau = float(torques[9])
        # self.cmd.motor_cmd[go2.LegID["RR_1"]].tau = float(torques[10])
        # self.cmd.motor_cmd[go2.LegID["RR_2"]].tau = float(torques[11])

        # # Set other parameters to zero for pure torque control
        # for i in range(20):
        #     self.cmd.motor_cmd[i].q = 0.0
        #     self.cmd.motor_cmd[i].kp = 0.0
        #     self.cmd.motor_cmd[i].dq = 0.0
        #     self.cmd.motor_cmd[i].kd = 0.0
            
        # self.cmd.crc = self.crc.Crc(self.cmd)
        # self.pub.Write(self.cmd)

        # Set all motor modes to torque control
        for i in range(12):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = float(self.go2_torques[i])
        
        # Send single command with all torques
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def execute_stand_up(self, phase):
        """Execute standing sequence"""
        for i in range(12):
            self.cmd.motor_cmd[i].q = phase * self.stand_up_joint_pos[i] + \
                                     (1 - phase) * self.stand_down_joint_pos[i]
            self.cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 3.5
            
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

        # Add small delay at end of standup to ensure stability
        if phase > 0.99:
            time.sleep(100.0)  # Let robot settle

    def _create_stance_ref(self):
        """Create standing reference for MPC"""
        x_ref = np.zeros((13, self.mpc.params.horizon_steps + 1))
        x_ref[2, :] = 0.3  # Desired COM height
        return x_ref
    
    def _create_reference_trajectory(self, current_state, desired_vel):
        """Create linear reference trajectory for MPC horizon
        
        Args:
            current_state: Current robot state [13]
            desired_vel: Desired velocity [3] - e.g. [0.5, 0, 0] for 0.5 m/s forward
        """
        x_ref = np.zeros((13, self.mpc.params.horizon_steps + 1))

        # Set initial position to current position
        x_ref[0:3, 0] = current_state[0:3]  # Starting COM position
        
        # Set desired height
        desired_height = 0.3  # Or whatever your nominal height is
        x_ref[2, :] = desired_height  # Maintain constant height
        
        # Propagate position linearly based on desired velocity
        for k in range(1, self.mpc.params.horizon_steps + 1):
            dt = k * self.mpc.params.dt
            x_ref[0:3, k] = current_state[0:3] + desired_vel * dt
        
        # Set desired velocities (constant)
        x_ref[6:9, :] = np.tile(desired_vel.reshape(3,1), (1, self.mpc.params.horizon_steps + 1))
        
        # Set gravity state
        x_ref[12, :] = 9.81

        # Debug print
        # print("\nReference Trajectory Debug:")
        # print(f"Initial position: {x_ref[0:3, 0]}")
        # print(f"Final position: {x_ref[0:3, -1]}")
        # print(f"Desired velocity: {desired_vel}")

        return x_ref


    def _get_leg_indices(self, leg: str) -> List[int]:
        """Get motor indices for given leg"""
        indices = {
            "FL": [0, 1, 2],
            "FR": [3, 4, 5],
            "RL": [6, 7, 8],
            "RR": [9, 10, 11]
        }
        return indices[leg]
    
    def test_stance_torques(self):
        """Test applying just constant stance torques"""
        # Initialize command
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0

        # Apply simple upward torque to all legs
        for leg in ["FR", "FL", "RR", "RL"]:
            # For each motor in leg
            hip_idx = go2.LegID[f"{leg}_0"]
            thigh_idx = go2.LegID[f"{leg}_1"]
            calf_idx = go2.LegID[f"{leg}_2"]

            # Zero position control
            self.cmd.motor_cmd[hip_idx].mode = 0x01
            self.cmd.motor_cmd[hip_idx].q = 0
            self.cmd.motor_cmd[hip_idx].kp = 0
            self.cmd.motor_cmd[hip_idx].dq = 0
            self.cmd.motor_cmd[hip_idx].kd = 0
            
            self.cmd.motor_cmd[thigh_idx].mode = 0x01
            self.cmd.motor_cmd[thigh_idx].q = 0
            self.cmd.motor_cmd[thigh_idx].kp = 0
            self.cmd.motor_cmd[thigh_idx].dq = 0
            self.cmd.motor_cmd[thigh_idx].kd = 0
            
            self.cmd.motor_cmd[calf_idx].mode = 0x01
            self.cmd.motor_cmd[calf_idx].q = 0
            self.cmd.motor_cmd[calf_idx].kp = 0
            self.cmd.motor_cmd[calf_idx].dq = 0
            self.cmd.motor_cmd[calf_idx].kd = 0

            # Apply constant torques
            self.cmd.motor_cmd[hip_idx].tau = 0.0  # No hip torque
            self.cmd.motor_cmd[thigh_idx].tau = 15.0  # Push up with thigh
            self.cmd.motor_cmd[calf_idx].tau = 10.0  # Push up with calf
        
        time.sleep(1.0)
            
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

