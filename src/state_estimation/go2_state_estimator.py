import time
import sys
import numpy as np
from . import unitree_legged_const as go2
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_, unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

class Go2StateEstimator:
    def __init__(self, network_interface: str = None):
        # # Init DDS
        # if network_interface:
        #     ChannelFactoryInitialize(0, network_interface)
        # else:
        #     # Use simulator domain ID and interface
        #     ChannelFactoryInitialize(1, "lo")
        # Setup subscribers
        self.low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.high_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)

        # Init states
        self.low_state = None
        self.high_state = None

        # Init subscribers/ handlers()
        self.low_sub.Init(self.low_state_callback, 10)
        self.high_sub.Init(self.high_state_callback, 10)
        # Allow time for first messages to arrive
        time.sleep(0.5)  # Wait 500ms to ensure initial state messages are received


        # Define leg indices 
        self.leg_indices = {
            "FL": [go2.LegID["FL_0"], [go2.LegID["FL_1"]], [go2.LegID["FL_2"]]],
            "FR": [go2.LegID["FR_0"], [go2.LegID["FR_1"]], [go2.LegID["FR_2"]]],
            "BL": [go2.LegID["RL_0"], [go2.LegID["RL_1"]], [go2.LegID["RL_2"]]],
            "BR": [go2.LegID["RR_0"], [go2.LegID["RR_1"]], [go2.LegID["RR_2"]]]
        }
        
    def low_state_callback(self, msg: LowState_):
        self.low_state = msg
        #print("Got low state")  # Debug print
        #print("IMU data:", msg.imu_state)  # Debug the IMU data structure

    def high_state_callback(self, msg: SportModeState_):
        self.high_state = msg
        #print("Got high state")


    def get_state(self):
        """
            Get current robot state

            Returns:
                Returns:
                q: [base_pos(3), quat(4), joint_angles(12)]
                dq: [base_vel(3), base_angvel(3), joint_vels(12)]
        """
        if self.low_state is None or self.high_state is None:
            print("No state received yet")
            return None, None
        
        try:
            print("Received state:")
        
            # Base state from sport mode
            q = np.zeros(19) # 7 base + 12 joint
            q[0:3] = self.high_state.position

            #NOTE: May just use RPY, but pinocchio uses quaternion (x, y, z, w)
            quat = self.low_state.imu_state.quaternion
            #print("Quaternion:", quat)  # Debug print
            q[3] = float(quat[1])  # x
            q[4] = float(quat[2])  # y
            q[5] = float(quat[3])  # z
            q[6] = float(quat[0])  # w

            # FR
            q[7] = self.low_state.motor_state[go2.LegID["FL_0"]].q
            q[8] = self.low_state.motor_state[go2.LegID["FL_1"]].q
            q[9] = self.low_state.motor_state[go2.LegID["FL_2"]].q
            # FL
            q[10] = self.low_state.motor_state[go2.LegID["FR_0"]].q
            q[11] = self.low_state.motor_state[go2.LegID["FR_1"]].q
            q[12] = self.low_state.motor_state[go2.LegID["FR_2"]].q
            # RR
            q[13] = self.low_state.motor_state[go2.LegID["RL_0"]].q
            q[14] = self.low_state.motor_state[go2.LegID["RL_1"]].q
            q[15] = self.low_state.motor_state[go2.LegID["RL_2"]].q
            # RL
            q[16] = self.low_state.motor_state[go2.LegID["RR_0"]].q
            q[17] = self.low_state.motor_state[go2.LegID["RR_1"]].q
            q[18] = self.low_state.motor_state[go2.LegID["RR_2"]].q

            dq = np.zeros(18)
            dq[0:3] = self.high_state.velocity
            dq[3:6] = self.high_state.yaw_speed

            # Joint velocities
            # FR
            dq[6] = self.low_state.motor_state[go2.LegID["FL_0"]].dq
            dq[7] = self.low_state.motor_state[go2.LegID["FL_1"]].dq
            dq[8] = self.low_state.motor_state[go2.LegID["FL_2"]].dq
            # FL
            dq[9] = self.low_state.motor_state[go2.LegID["FR_0"]].dq
            dq[10] = self.low_state.motor_state[go2.LegID["FR_1"]].dq
            dq[11] = self.low_state.motor_state[go2.LegID["FR_2"]].dq
            # RR
            dq[12] = self.low_state.motor_state[go2.LegID["RL_0"]].dq
            dq[13] = self.low_state.motor_state[go2.LegID["RL_1"]].dq
            dq[14] = self.low_state.motor_state[go2.LegID["RL_2"]].dq
            # RL
            dq[15] = self.low_state.motor_state[go2.LegID["RR_0"]].dq
            dq[16] = self.low_state.motor_state[go2.LegID["RR_1"]].dq
            dq[17] = self.low_state.motor_state[go2.LegID["RR_2"]].dq

            # foot pos and velocity
            p_foot = np.zeros(12)
            v_foot = np.zeros(12)

            p_foot_temp = self.high_state.foot_position_body
            v_foot_temp = self.high_state.foot_speed_body

            

            #re-order to match our indices
            p_foot[0:3] = p_foot_temp[3:6]
            p_foot[3:6] = p_foot_temp[0:3]
            p_foot[6:9] = p_foot_temp[9:]
            p_foot[9:]  = p_foot_temp[6:9]

            v_foot[0:3] = v_foot_temp[3:6]
            v_foot[3:6] = v_foot_temp[0:3]
            v_foot[6:9] = v_foot_temp[9:]
            v_foot[9:]  = v_foot_temp[6:9]

            return q, dq, p_foot, v_foot
        
        except Exception as e:
            print(f"Error getting state {e}")
            import traceback
            traceback.print_exc()
            return None, None
