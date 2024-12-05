import numpy as np
import pinocchio as pin

class ForceMapper:
    def __init__(self, urdf_path: str):
        # Load robot model
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        # Set up foot frame IDs
        self.foot_frame_ids = {
            'FR': self.model.getFrameId('FR_foot'),
            'FL': self.model.getFrameId('FL_foot'),
            'RR': self.model.getFrameId('RR_foot'),
            'RL': self.model.getFrameId('RL_foot')
        }

        # Define joint indices for each leg
        self.leg_joint_indices = {
            'FL': [6, 7, 8],    # FL_hip/thigh/calf
            'FR': [9, 10, 11],  # FR_hip/thigh/calf
            'RL': [12, 13, 14], # RL_hip/thigh/calf
            'RR': [15, 16, 17]  # RR_hip/thigh/calf
        }

        # Swing control gains - from paper
        self.Kp = np.diag([700, 700, 150]) /10  # Position gains
        self.Kd = np.diag([7, 7, 7])      # Velocity gains

        # Torque limits from Go2 specs (Nm)
        self.tau_max = np.array([23.7, 23.7, 45.4])  # [hip, thigh, calf]

    def compute_swing_torques(self, leg_id: str, q: np.ndarray, v: np.ndarray, 
                            p_des: np.ndarray, v_des: np.ndarray, a_des: np.ndarray) -> np.ndarray:
        """
        Compute swing leg torques using operational space control
        """
        # Update kinematics
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, v)
        pin.crba(self.model, self.data, q)  # Mass matrix
        pin.computeCoriolisMatrix(self.model, self.data, q, v)
        pin.computeGeneralizedGravity(self.model, self.data, q)

        # Get current foot state
        frame_id = self.foot_frame_ids[leg_id]
        joint_ids = self.leg_joint_indices[leg_id]
        p_current = self.data.oMf[frame_id].translation
        v_current = pin.getFrameVelocity(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL).linear

        # Get full Jacobian and explicitly select columns for this leg
        J_full = pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL)
        J = J_full[:3, joint_ids]  # This should give us a 3x3 matrix

        dJ_full = pin.getFrameJacobianTimeVariation(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL)
        dJ = dJ_full[:3, joint_ids]
        dJ_dq = dJ @ v[joint_ids]  # Corrected computation

        tau_pd = J.T @ (self.Kp @ (p_des - p_current) + self.Kd @ (v_des - v_current))

        # Compute feed forward torque
        M_leg = self.data.M[np.ix_(joint_ids, joint_ids)]
        C_leg = self.data.C[np.ix_(joint_ids, joint_ids)]  # Extracted Coriolis matrix for the leg
        G_leg = self.data.g[joint_ids]  # Extracted gravity vector for the leg
        Lambda = np.linalg.inv(J @ np.linalg.inv(M_leg) @ J.T)

        tau_ff = J.T @ Lambda @ (a_des - dJ_dq) + C_leg @ v[joint_ids] + G_leg

        tau = tau_pd + tau_ff

        return np.clip(tau, -self.tau_max, self.tau_max)

    
    def compute_stance_torques(self, leg_id: str, q: np.ndarray, v: np.ndarray, 
                             force_des: np.ndarray) -> np.ndarray:
        """
        Map desired force to joint torques for stance leg
        
        Args:
            leg_id: Leg identifier (FL, FR, RL, RR)
            q: Full robot joint positions
            v: Full robot joint velocities
            force_des: Desired force in world frame [3,]
            
        Returns:
            tau: Joint torques for the leg [3,]
        """
        # Update kinematics
        pin.forwardKinematics(self.model,self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, q)
        

        frame_id = self.foot_frame_ids[leg_id]
        joint_ids = self.leg_joint_indices[leg_id]
        
        # Get full Jacobian and explicitly select columns for this leg
        J_full = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.ReferenceFrame.LOCAL)
        J = J_full[:3, joint_ids]  # This should give us a 3x3 matrix
        
        base_frame_id = self.model.getFrameId('floating_base')
        R_world_base = self.data.oMf[base_frame_id].rotation
        
        # Make sure force_des is the right shape
        force_des = np.array(force_des).reshape(3, 1)
        
        # Compute torques
        tau = J.T @ R_world_base.T @ force_des
        tau = tau.flatten()  # Ensure output is 1D array
        print(tau)
        
        return np.clip(tau, -self.tau_max, self.tau_max)
    
if __name__ == "__main__":

    force_mapper = ForceMapper("/home/parallels/go2_controller/robots/go2_description/xacro/go2_generated.urdf")

    q_stand = np.array([
            0.00571868, 0.608813, -1.21763,     # FR
            -0.00571868, 0.608813, -1.21763,    # FL
            0.00571868, 0.608813, -1.21763,     # RR
            -0.00571868, 0.608813, -1.21763     # RL
        ])
    force_mapper.compute_stance_torques("FR", q_stand, np.zeros(12), np.array([-6, 6, 10]))
    

    