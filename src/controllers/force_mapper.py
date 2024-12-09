import numpy as np
import pinocchio as pin

class ForceMapper:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        self.foot_frame_ids = {
            'FR': self.model.getFrameId('FR_foot'),
            'FL': self.model.getFrameId('FL_foot'),
            'RR': self.model.getFrameId('RR_foot'),
            'RL': self.model.getFrameId('RL_foot')
        }

        self.leg_joint_indices = {
            'FL': [6, 7, 8],    
            'FR': [9, 10, 11],  
            'RL': [12, 13, 14], 
            'RR': [15, 16, 17]  
        }

        # Cache for stance control
        self.cached_J = {leg: None for leg in self.foot_frame_ids.keys()}
        self.cached_R = {leg: None for leg in self.foot_frame_ids.keys()}
        self.last_stance_q = {leg: None for leg in self.foot_frame_ids.keys()}

        # Control gains
        self.Kp = np.diag([700, 700, 150])  
        self.Kd = np.diag([7, 7, 7])      
        #self.omega = 2.0  # Natural frequency for gain adaptation

        # Torque limits
        self.tau_max = np.array([23.7, 23.7, 45.4])

    def update_stance_cache(self, leg_id: str, q: np.ndarray):
        """Update cached matrices for stance control if configuration changed"""
        if (self.last_stance_q[leg_id] is None or 
            not np.allclose(q, self.last_stance_q[leg_id])):
            
            pin.computeJointJacobians(self.model, self.data, q)
            
            frame_id = self.foot_frame_ids[leg_id]
            joint_ids = self.leg_joint_indices[leg_id]
            
            J_full = pin.getFrameJacobian(self.model, self.data, frame_id, 
                                         pin.ReferenceFrame.LOCAL)
            self.cached_J[leg_id] = J_full[:3, joint_ids]
            
            base_frame_id = self.model.getFrameId('floating_base')
            self.cached_R[leg_id] = self.data.oMf[base_frame_id].rotation
            self.last_stance_q[leg_id] = q.copy()

    def compute_operational_space_inertia(self, leg_id: str, q: np.ndarray):
        """Compute operational space inertia matrix (Lambda)"""
        frame_id = self.foot_frame_ids[leg_id]
        joint_ids = self.leg_joint_indices[leg_id]
        
        pin.computeJointJacobians(self.model, self.data, q)
        J_full = pin.getFrameJacobian(self.model, self.data, frame_id, 
                                     pin.ReferenceFrame.LOCAL)
        J = J_full[:3, joint_ids]
        
        pin.crba(self.model, self.data, q)
        M_leg = self.data.M[np.ix_(joint_ids, joint_ids)]
        Lambda = np.linalg.inv(J @ np.linalg.inv(M_leg) @ J.T)
        return Lambda, J

    def compute_swing_torques(self, leg_id: str, q: np.ndarray, v: np.ndarray, 
                            p_des: np.ndarray, v_des: np.ndarray, a_des: np.ndarray):
        """Compute swing leg torques using operational space control"""
        frame_id = self.foot_frame_ids[leg_id]
        joint_ids = self.leg_joint_indices[leg_id]
        
        # Update dynamics
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get operational space inertia and update gains
        Lambda, J = self.compute_operational_space_inertia(leg_id, q)
        # for i in range(3):
        #     self.Kp[i,i] = self.omega**2 * Lambda[i,i]
        
        # Current state
        p_current = self.data.oMf[frame_id].translation
        v_current = pin.getFrameVelocity(self.model, self.data, frame_id, 
                                       pin.ReferenceFrame.LOCAL).linear
        
        # Feedback term
        feedback = self.Kp @ (p_des - p_current) + self.Kd @ (v_des - v_current)
        
        # Feedforward compensation
        pin.computeCoriolisMatrix(self.model, self.data, q, v)
        pin.computeGeneralizedGravity(self.model, self.data, q)
        
        C_leg = self.data.C[np.ix_(joint_ids, joint_ids)]
        G_leg = self.data.g[joint_ids]
        
        J_dot = pin.getFrameJacobianTimeVariation(self.model, self.data, frame_id,
                                                 pin.ReferenceFrame.LOCAL)[:3, joint_ids]
        feedforward = Lambda @ (a_des - J_dot @ v[joint_ids]) + C_leg @ v[joint_ids] + G_leg
        
        tau = J.T @ (feedback + feedforward)
        return np.clip(tau, -self.tau_max, self.tau_max)

    def compute_stance_torques(self, leg_id: str, q: np.ndarray, dq: np.ndarray, 
                             force_des: np.ndarray):
        """Compute stance leg torques using cached Jacobian"""
        self.update_stance_cache(leg_id, q)
        force_des = np.array(force_des).reshape(3, 1)

        # Current state
        joint_ids = self.leg_joint_indices[leg_id]
        J = self.cached_J[leg_id]
        R = self.cached_R[leg_id]

        # Get current foot velocity from Jacobian 
        v_foot = J @ dq[joint_ids]  # Current foot velocity in leg frame
        
        # Compute desired foot velocity - you can pass this in if you want
        # or compute from reference trajectory
        # For now let's use zero desired velocity in stance
        v_des = np.zeros(3)
        
        # Compute force with feedback
        force = force_des + self.Kd @ (v_des - v_foot).reshape(3,1)
        
        # Map to torques through Jacobian transpose
        tau = J.T @ R.T @ force

        # Add joint damping (from MIT impl)
        # kd_joint = 0.2  # Joint damping gain
        # tau_damping = -kd_joint * dq[joint_ids].reshape(3,1)
        # tau = tau + tau_damping

        return np.clip(tau.flatten(), -self.tau_max, self.tau_max)

if __name__ == "__main__":

    force_mapper = ForceMapper("/home/parallels/go2_controller/robots/go2_description/xacro/go2_generated.urdf")

    q_stand = np.array([0, 0, 0.33, 0, 0, 0, 1,
            0.00571868, 0.608813, -1.21763,     # FR
            -0.00571868, 0.608813, -1.21763,    # FL
            0.00571868, 0.608813, -1.21763,     # RR
            -0.00571868, 0.608813, -1.21763     # RL
        ])
    torques = force_mapper.compute_stance_torques("RR", q_stand, np.zeros(12), np.array([-40, 40, 100]))
    print(torques)

    