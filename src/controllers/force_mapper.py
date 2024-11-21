import numpy as np
import pinocchio as pin

class ForceMapper:
    def __init__(self, urdf_path: str):
        # Load robot model
        self.model = pin.buildModelFromUrdf(urdf_path)
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
            'FL': [0, 1, 2],    # FL_hip/thigh/calf
            'FR': [3, 4, 5],    # FR_hip/thigh/calf
            'RL': [6, 7, 8],    # RL_hip/thigh/calf
            'RR': [9, 10, 11]   # RR_hip/thigh/calf
        }

        # Swing control gains - from paper
        self.Kp_swing = np.diag([700, 700, 150])  # Position gains
        self.Kd_swing = np.diag([70, 70, 70])     # Velocity gains

        # Torque limits from Go2 specs (Nm)
        self.tau_max = np.array([23.7, 23.7, 45.4])  # [hip, thigh, calf]

    def compute_swing_torques(self, leg_id: str, q: np.ndarray, v: np.ndarray, 
                            p_des: np.ndarray, v_des: np.ndarray, a_des: np.ndarray) -> np.ndarray:
        """
        Compute swing leg torques using operational space control
        
        Args:
            leg_id: Leg identifier (FL, FR, RL, RR)
            q: Full robot joint positions
            v: Full robot joint velocities
            p_des: Desired foot position in world frame
            v_des: Desired foot velocity in world frame
            a_des: Desired foot acceleration in world frame
            
        Returns:
            tau: Joint torques for the leg [3,]
        """
        # Update pinocchio model
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.crba(self.model, self.data, q)  # Mass matrix
        pin.computeCoriolisMatrix(self.model, self.data, q, v)
        pin.computeGeneralizedGravity(self.model, self.data, q)

        # Get current foot state
        frame_id = self.foot_frame_ids[leg_id]
        p_current = self.data.oMf[frame_id].translation
        
        # Get Jacobian for this leg
        J = pin.getFrameJacobian(self.model, self.data, frame_id, 
                                pin.ReferenceFrame.WORLD)[:3, self.leg_joint_indices[leg_id]]
        
        # Current foot velocity
        v_current = J @ v[self.leg_joint_indices[leg_id]]
        
        # Get mass matrix for just this leg
        joint_ids = self.leg_joint_indices[leg_id]
        M_leg = self.data.M[np.ix_(joint_ids, joint_ids)]
        
        # Compute operational space inertia matrix
        Lambda = np.linalg.inv(J @ np.linalg.inv(M_leg) @ J.T)
        
        # Get Coriolis and gravity terms for this leg
        C_leg = self.data.C[np.ix_(joint_ids, joint_ids)]
        g_leg = self.data.g[joint_ids]
        
        # Compute feedforward dynamics
        J_dot_v = pin.getFrameClassicalAcceleration(
            self.model, self.data, frame_id).linear
        
        # Compute control law:
        # 1. Feedback terms
        F_fb = Lambda @ (
            self.Kp_swing @ (p_des - p_current) + 
            self.Kd_swing @ (v_des - v_current)
        )
        
        # 2. Feedforward terms
        F_ff = Lambda @ (a_des - J_dot_v)
        
        # 3. Compensation terms
        tau_comp = C_leg @ v[joint_ids] + g_leg
        
        # Combine everything
        tau = J.T @ (F_fb + F_ff) + tau_comp
        
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
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get base rotation (world_R_body)
        R = self.data.oMi[1].rotation  # Get rotation of floating base
        
        # Get Jacobian for this leg
        frame_id = self.foot_frame_ids[leg_id]
        J = pin.getFrameJacobian(self.model, self.data, frame_id, 
                                pin.ReferenceFrame.WORLD)[:3, self.leg_joint_indices[leg_id]]

        # Get dynamics terms
        joint_ids = self.leg_joint_indices[leg_id]
        C_leg = self.data.C[np.ix_(joint_ids, joint_ids)]
        g_leg = self.data.g[joint_ids]
        
        # Add rotation transformation like in paper
        tau = J.T @ R.T @ force_des + C_leg @ v[joint_ids] + g_leg
        
        return np.clip(tau, -self.tau_max, self.tau_max)
    
    def get_foot_position(self, leg_id: str, q: np.ndarray) -> np.ndarray:
        """Get current foot position in world frame"""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.foot_frame_ids[leg_id]].translation

    def get_foot_velocity(self, leg_id: str, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Get current foot velocity in world frame"""
        pin.computeJointJacobians(self.model, self.data, q)
        J = pin.getFrameJacobian(self.model, self.data, self.foot_frame_ids[leg_id], 
                                pin.ReferenceFrame.WORLD)[:3, self.leg_joint_indices[leg_id]]
        return J @ v[self.leg_joint_indices[leg_id]]