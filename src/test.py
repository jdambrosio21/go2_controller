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
from quadruped import Quadruped

class Go2Controller:
    def __init__(self, urdf_path: str):
        # Initialize robot model and estimator
        self.robot = Quadruped(urdf_path)
            
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        
        # Initialize command
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.initialize_command()
        self.crc = CRC()

        self.state_estimator = Go2StateEstimator()

        # Initialize gait scheduler
        self.gait_scheduler = GaitScheduler(total_period=0.5, gait_type="trot")  # Or whatever gait

        # Initialize Footstep Planner
        self.footstep_planner = FootstepPlanner(urdf_path)

        # Initialize MPC
        mpc_params = MPCParams(
            mass=self.robot.mass,
            I_body=self.robot.inertia,
            dt=0.0333  # 30hz
        )
        self.mpc = ConvexMPC(mpc_params)

        # Initialize Foot swing Trajectory Generator
        #fself.footswing_traj = FootSwingTrajectory()
        self.swing_trajectories = {leg: None for leg in ["FL", "FR", "RL", "RR"]}
        
        # Initialize force mapper
        self.force_mapper = ForceMapper(urdf_path)
        
        # Control parameters
        self.dt = 0.002
        self.state = "STANDUP"
        
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

    def run(self):
        """Main control loop"""
        running_time = 0.0
        print("Starting control loop...")
        
        while True:
            step_start = time.perf_counter()
            
            try:
                if self.state == "STANDUP":
                    phase = np.tanh(running_time / 1.2)
                    if running_time < 1.5:
                        self.execute_stand_up(phase)
                    else:
                        print("Switching to MPC control...")
                        self.state = "CONTROL"
                        
                elif self.state == "CONTROL":
                    self.run_mpc_control()
                    
                running_time += self.dt
                
                # Maintain control frequency
                time_until_next = self.dt - (time.perf_counter() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
                    
            except Exception as e:
                print(f"Control loop failed: {e}")
                self.state = "STANDUP"

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

    def run_mpc_control(self):
        """Run single iteration of MPC control"""
        try:
            # Get current state
            q, dq = self.state_estimator.get_state()
            
            if q is None or dq is None:
                print("Waiting for valid state...")
                time.sleep(0.01)  # Sleep for 10ms and retry
                return  # Just return and try again next cycle
            
            # Init reference traj
            x_ref = self._create_reference_trajectory(q, np.array([0.5, 0, 0]))
            #x_ref = self._create_stance_ref()

            
            # Update gait
            self.gait_scheduler.update(self.dt)

            # Get contact schedules and states
            contact_state = self.gait_scheduler.get_current_contact_state()
            contact_schedule = self.gait_scheduler.predict_horizon_contact_state(
                dt=self.mpc.params.dt,
                horizon_length=self.mpc.params.horizon_steps
            )
            
            # Get foot positions using Pinocchio method
            # NOTE: May be more stable and robust to disturbances if we get the
            # foot position directly from Go2?
            foot_position = self.robot.get_foot_positions(q)

            rpy = self.robot.quat_to_rpy(q[3:7])

            # Plan current next footsteps for swing legs
            next_footholds = self.footstep_planner.plan_current_footsteps(dq[0:3], x_ref[6:9, 0], q, self.gait_scheduler)


            # Update swing trajectories for current swing legs
            for leg_idx, leg in enumerate(["FL", "FR", "RL", "RR"]):
                if contact_state[leg_idx] == 0: # In swing
                    if self.swing_trajectories[leg] is None:
                        self.swing_trajectories[leg] = FootSwingTrajectory(
                            p0=foot_position[leg_idx, :],
                            pf=next_footholds[leg_idx, :],
                            height=0.1
                        )

                    phase = self.gait_scheduler.get_swing_phase(leg_idx)
                    swing_duration = self.gait_scheduler.get_swing_duration()
                    self.swing_trajectories[leg].compute_swing_trajectory_bezier(phase, swing_duration)
                else:
                    self.swing_trajectories[leg] = None # In stance

            # Plan footholds for MPC Horizon
            foot_pos_horizon = self.footstep_planner.plan_horizon_footsteps(self.mpc.params.dt, self.mpc.params.horizon_steps, x_ref, self.q_nom, self.gait_scheduler)
            
            print(foot_pos_horizon)
            
            # 2. Run MPC
            x0 = np.concatenate([
                q[0:3],     # position
                rpy,  # orientation
                dq[0:3],    # linear velocity
                dq[3:6],    # angular velocity
                [9.81]      # gravity
            ])         
            
            mpc_forces = self.mpc.solve(x0, x_ref, contact_schedule, foot_pos_horizon)
            q_joints = q[7:]
            dq_joints = dq[6:]
            torques = np.zeros(12)

            # For each leg, calculate the torques to be applied from the MPC output
            for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
                # Get the force for leg i
                force = mpc_forces[i*3:(i+1)*3] # xyz for each leg

                # Check if leg is in stance or swing
                if contact_state[i] == 1: # In stance
                    # Use stance torque control
                    torque = self.force_mapper.compute_leg_torques(leg, q_joints, dq_joints, 'stance', force=force)
                    # Do i apply the torque now? or all at once
                    torques[i*3:(i+1)*3] = torque
                
                else: # In swing
                    self.swing_trajectories[leg] = FootSwingTrajectory(foot_position[i], next_footholds[i], 0.1)
                    traj = self.swing_trajectories[leg]

                    # Calc torque to track swing traj
                    torque = self.force_mapper.compute_leg_torques(leg, q_joints, dq_joints, 'swing', 
                                             p_ref=traj.p, 
                                             v_ref=traj.v, 
                                             a_ref=traj.a)
                    torques[i*3:(i+1)*3] = torque

            #Re order torques for unitree order
            go2_torques = np.zeros(12)
            go2_torques[0:3] = torques[3:6]
            go2_torques[3:6] = torques[0:3]
            go2_torques[6:9] = torques[9:]
            go2_torques[9:] = torques[6:9]
            # Send a single command with all torques
            # print("\nSending unified command for all motors...")
            for i in range(12):
                self.cmd.motor_cmd[i].mode = 0x01
                self.cmd.motor_cmd[i].tau = go2_torques[i]
                self.cmd.motor_cmd[i].kp = 1.0
                self.cmd.motor_cmd[i].kd = 1.0
                
            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)
                    
        except Exception as e:
            print(f"Control loop failed: {e}")
            import traceback
            traceback.print_exc()
            self.state = "STANDUP"

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
        print("\nReference Trajectory Debug:")
        print(f"Initial position: {x_ref[0:3, 0]}")
        print(f"Final position: {x_ref[0:3, -1]}")
        print(f"Desired velocity: {desired_vel}")

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

if __name__ == "__main__":
    print("Starting controller...")
    
    # Initialize DDS first
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])
    
    # Then create controller
    input("Press enter to start")  # Like in working example
    
    controller = Go2Controller("/home/parallels/go2_controller/robots/go2_description/xacro/go2_generated.urdf")

    
    controller.run()