import time
import sys
import numpy as np
from . import unitree_legged_const as go2
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_, unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

class Go2StateEstimator:
    def __init__(self, network_interface: str = None):
        # Init DDS
        if network_interface:
            ChannelFactoryInitialize(0, network_interface)
        else:
            ChannelFactoryInitialize(0)

        # Setup subscribers
        self.low_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.high_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)

        # Init states
        self.low_state = None
        self.high_state = None

        # Init subscribers/ handlers()
        self.low_sub.Init(self.low_state_callback, 10)
        self.high_sub.Init(self.high_state_callback, 10)

        # Define leg indices 
        self.leg_indices = {
            "FL": [go2.LegID["FL_0"], [go2.LegID["FL_1"]], [go2.LegID["FL_2"]]],
            "FR": [go2.LegID["FR_0"], [go2.LegID["FR_1"]], [go2.LegID["FR_2"]]],
            "BL": [go2.LegID["RL_0"], [go2.LegID["RL_1"]], [go2.LegID["RL_2"]]],
            "BR": [go2.LegID["RR_0"], [go2.LegID["RR_1"]], [go2.LegID["RR_2"]]]
        }
        
    def low_state_callback(self, msg: LowState_):
        self.low_state = msg

    def high_state_callback(self, msg: SportModeState_):
        self.high_state = msg

    def get_state(self):
        """
            Get current robot state

            Returns:
                Returns:
                q: [base_pos(3), quat(4), joint_angles(12)]
                dq: [base_vel(3), base_angvel(3), joint_vels(12)]
        """
        if self.low_state is None or self.high_state is None:
            return None, None, None
        
        # Base state from sport mode
        q = np.zeros(19) # 7 base + 12 joint
        q[0:3] = self.high_state.position

        #NOTE: May just use RPY, but pinocchio uses quaternion (x, y, z, w)
        q[3:7] = [
            self.high_state.quaternion[1], # x
            self.high_state.quaternion[2], # y
            self.high_state.quaternion[3], # z
            self.high_state.quaternion[0]  # w 
        ]

        joint_angles = []
        joint_velocities = []
        for leg in ["FL", "FR", "BL", "BR"]:
            for i in self.leg_indices:
                joint_angles.append(self.low_state.motor_state[i].q)
                joint_velocities.append(self.low_state.motor_state[i].dq)

        q[7:] = joint_angles

        dq = np.zeros(18)
        dq[0:3] = self.high_state.velocity
        dq[3:6] = self.high_state.yaw_speed
        dq[6:] = joint_velocities