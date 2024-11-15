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
from planners.gait_scheduler import GaitScheduler
from controllers.force_mapper import ForceMapper
from state_estimation.go2_state_estimator import Go2StateEstimator
from quadruped import Quadruped

class Go2Controller:
    def __init__(self, urdf_path: str):
        # Initialize robot model and estimator
        self.robot = Quadruped(urdf_path)

        # # Initialize communication
        # if len(sys.argv) < 2:
        #     ChannelFactoryInitialize(1, "lo")
        # else:
        #     ChannelFactoryInitialize(0, sys.argv[1])
            
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        
        # Initialize command
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.initialize_command()
        self.crc = CRC()

        self.state_estimator = Go2StateEstimator()

        # Initialize MPC
        mpc_params = MPCParams(
            mass=self.robot.mass,
            I_body=self.robot.inertia,
            dt=0.002  # 500Hz
        )
        self.mpc = ConvexMPC(mpc_params)
        
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
                return  # Just return and try again next cycle

            # Get foot positions using Pinocchio method
            foot_position = self.robot.get_foot_positions(q)
            
            # Create contact schedule for full horizon
            horizon_steps = self.mpc.params.horizon_steps
            contact_schedule = np.ones((4, horizon_steps))
            
            # Create foot positions for horizon
            foot_pos_horizon = []
            for _ in range(horizon_steps):
                foot_pos_horizon.append(foot_position) # assume same foot position across horizon to start
            
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
                x_ref=self._create_stance_ref(),
                contact_schedule=contact_schedule,
                foot_positions=foot_pos_horizon
            )
            
            # 3. Map forces to torques
            for i, leg in enumerate(["FR", "FL", "RR", "RL"]):
                force = mpc_forces[i*3:(i+1)*3]
                tau = self.force_mapper.compute_leg_torques(
                    leg_id=leg,
                    q=q,
                    mode='stance',
                    force=force,
                    R=np.eye(3)
                )
                
                # Send to motors
                indices = self._get_leg_indices(leg)
                for j, idx in enumerate(indices):
                    self.cmd.motor_cmd[idx].mode = 0x01  # Torque control
                    self.cmd.motor_cmd[idx].tau = tau[j]
                    self.cmd.motor_cmd[idx].kp = 0
                    self.cmd.motor_cmd[idx].kd = 0
            
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

    def _get_leg_indices(self, leg: str) -> List[int]:
        """Get motor indices for given leg"""
        indices = {
            "FR": [0, 1, 2],
            "FL": [3, 4, 5],
            "RR": [6, 7, 8],
            "RL": [9, 10, 11]
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