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
        self.gait_scheduler = GaitScheduler(total_period=0.2, gait_type="trot")  # Or whatever gait

        # Initialize Footstep Planner
        self.footstep_planner = FootstepPlanner(urdf_path)

        # Initialize MPC
        mpc_params = MPCParams(
            mass=self.robot.mass,
            I_body=self.robot.inertia,
            dt=0.002  # 500Hz
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
            0.00571868, 0.608813, -1.21763,
            -0.00571868, 0.608813, -1.21763,
            0.00571868, 0.608813, -1.21763,
            -0.00571868, 0.608813, -1.21763
        ])
        
        self.stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375,
            -0.0473455, 1.22187, -2.44375,
            0.0473455, 1.22187, -2.44375,
            -0.0473455, 1.22187, -2.44375
        ])

        
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
            x_ref = self._create_reference_trajectory(q, np.array([2, 0, 0]))
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

            # Plan footsteps for next swing legs
            next_footholds = self.footstep_planner.plan_footsteps(
                com_state=(q[0:3], dq[0:3]),
                desired_vel=x_ref[6:9, 0],
                q_current=q,
                gait_scheduler=self.gait_scheduler
            )

            # Update swing trajectories for current swing legs
            for leg_idx, leg in enumerate(["FL", "FR", "RL", "RR"]):
                if contact_state[leg_idx] == 0: # In swing
                    if self.swing_trajectories[leg] is None:
                        self.swing_trajectories[leg] = FootSwingTrajectory(
                            p0=foot_position[leg_idx],
                            pf=next_footholds[leg_idx],
                            height=0.05
                        )

                    phase = self.gait_scheduler.get_swing_phase(leg_idx)
                    swing_duration = self.gait_scheduler.get_swing_duration()
                    self.swing_trajectories[leg].compute_swing_trajectory_bezier(phase, swing_duration)
                else:
                    self.swing_trajectories[leg] = None # In stance

            # Create foot positions for MPC horizon
            foot_pos_horizon = np.zeros((12, self.mpc.params.horizon_steps))
            for k in range(self.mpc.params.horizon_steps):
                for leg in range(4):
                    if contact_schedule[leg, k] == 1:
                        # Foot in stance
                        foot_pos_horizon[(leg*3):(leg+1)*3, k] = foot_position[leg]
                    else:
                        # Foot in swing
                        foot_pos_horizon[(leg*3):(leg+1)*3, k] = foot_position[leg]
            
            # 2. Run MPC
            x0 = np.concatenate([
                q[0:3],     # position
                self.robot.quat_to_rpy(q[3:7]),  # orientation
                dq[0:3],    # linear velocity
                dq[3:6],    # angular velocity
                [9.81]      # gravity
            ])
            
            
            mpc_forces = self.mpc.solve(
                x0=x0,
                x_ref=x_ref,
                contact_schedule=contact_schedule,
                foot_positions=foot_pos_horizon
            )

            # Define motor ID mapping for each leg
            leg_motor_map = {
                "FL": [0, 1, 2],
                "FR": [3, 4, 5],
                "RL": [6, 7, 8],
                "RR": [9, 10, 11]
            }
        
            
            # print("\nDebug MPC -> Motor Mapping")
            # Calculate all torques first
            q_joints = q[7:]
            dq_joints = dq[6:]
            
            # Initialize torque array for all 12 motors
            all_torques = np.zeros(12)
            
            # Calculate torques for all legs at once
            for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
                # print(f"\n{leg}:")
                motor_ids = leg_motor_map[leg]
                # print(f"Motor IDs: {motor_ids}")
                
                if contact_state[i] == 1:  # Leg in stance
                    force = mpc_forces[i*3:(i+1)*3]
                    # print(f"Using force: {force}")
                    leg_torques = self.force_mapper.compute_leg_torques(
                        leg_id=leg, q=q_joints, v=dq_joints,
                        mode='stance', force=force, R=np.eye(3)
                    )
                else:  # Leg in swing
                    if self.swing_trajectories[leg] is None:
                        current_pos = foot_position[i]
                        target_pos = next_footholds[i]
                        
                        self.swing_trajectories[leg] = FootSwingTrajectory(
                            p0=current_pos,
                            pf=target_pos,
                            height=0.1
                        )
                    
                    traj = self.swing_trajectories[leg]
                    if traj is None:
                        print(f"Warning: No trajectory for {leg} in swing")
                        continue
                        
                    leg_torques = self.force_mapper.compute_leg_torques(
                        leg_id=leg, q=q_joints, v=dq_joints,
                        mode='swing',
                        p_ref=traj.p, v_ref=traj.v, a_ref=traj.a
                    )
                
                # print(f"Computed torques: {leg_torques}")
                # Store torques in the complete array using correct motor IDs
                for j in range(3):
                    all_torques[motor_ids[j]] = leg_torques[j]
            
            # Send a single command with all torques
            # print("\nSending unified command for all motors...")
            for i in range(12):
                self.cmd.motor_cmd[i].mode = 0x01
                self.cmd.motor_cmd[i].tau = all_torques[i]
                self.cmd.motor_cmd[i].kp = 1.0
                self.cmd.motor_cmd[i].kd = 0.0
                
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