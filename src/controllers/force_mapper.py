import numpy as np
import pinocchio as pin

class ForceMapper:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Setup foot frame IDs
        self.foot_frame_ids = {
            'FR': self.model.getFrameId('FR_foot'),
            'FL': self.model.getFrameId('FL_foot'),
            'RR': self.model.getFrameId('RR_foot'),
            'RL': self.model.getFrameId('RL_foot')
        }

        # Match URDF joint ordering
        self.leg_joint_indices = {
            'FL': [0, 1, 2],    # FL_hip/thigh/calf are first
            'FR': [3, 4, 5],    # FR_hip/thigh/calf second
            'RL': [6, 7, 8],    # RL_hip/thigh/calf third
            'RR': [9, 10, 11]   # RR_hip/thigh/calf last
        }

        # Control gains
        # Stance gains should be zero (MPC handles this)
        self.Kp_stance = np.zeros((3,3))
        self.Kd_stance = np.zeros((3,3))

        # Swing gains should be high for tracking
        self.Kp_swing = np.diag([700, 700, 150])  # Position gains
        self.Kd_swing = np.diag([70, 70, 70])     # Velocity gains

        # Maximum torque limit (from Go2 Specifications)
        self.tau_max = np.array([23.7, 23.7, 45.4])  # [hip, thigh, calf]

    def get_rotation_matrix(self, q):
        """Get rotation matrix from base frame to world frame"""
        # Update pinocchio model
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get base rotation matrix (world_R_body)
        R = self.data.oMi[0].rotation  # Using frame 0 the base/floating joint
        return R

    def compute_swing_torques(self, leg_id, q, v, p_ref, v_ref, a_ref):
        """
        Compute torques for swing leg control

        Returns:
            torque: [3 x 1] vector of torques for each joint of the swing leg
        """

        # Update model
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.crba(self.model, self.data, q)  # Compute mass matrix
        pin.computeCoriolisMatrix(self.model, self.data, q, v)
        pin.computeGeneralizedGravity(self.model, self.data, q)

        # Get current foot position and velocity
        frame_id = self.foot_frame_ids[leg_id]
        p = self.data.oMf[frame_id].translation
        

        # Get Jacobian (Translational)
        J_full = pin.getFrameJacobian(self.model, self.data, 
                                 self.foot_frame_ids[leg_id], 
                                 pin.ReferenceFrame.LOCAL)[:3, :] 
        
        # Get columns for this leg's joints only
        joint_ids = self.leg_joint_indices[leg_id]
        J = J_full[:, joint_ids]  # Extract just this leg's columns
        v_current = J @ v[joint_ids]  # Use just this leg's velocities

        # Get mass matrix for just this leg
        M_leg = self.data.M[joint_ids][:, joint_ids]
    

        # Compute operational space inertia matrix
        Lambda = np.linalg.inv(J @ np.linalg.inv(M_leg) @ J.T)

        # Compute feedback term
        feedback = J.T @ (self.Kp_swing @ (p_ref - p) + self.Kd_swing @ (v_ref - v_current))

        # Compute feedforward term
        J_dot_q_dot = pin.getFrameClassicalAcceleration(self.model, self.data, 
                                              self.foot_frame_ids[leg_id]).linear
        
        C = self.data.C
        G = self.data.g

        # Get the Coriolis and gravity terms for just this leg
        C_leg = C[joint_ids, :][:, joint_ids]  # Extract leg's portion
        G_leg = G[joint_ids]
        
        # Compute feedforward with leg-specific terms
        tau_ff = J.T @ Lambda @ (a_ref - J_dot_q_dot) + C_leg @ v[joint_ids] + G_leg

        tau = feedback + tau_ff

        return np.clip(tau, -self.tau_max, self.tau_max)
    
    def compute_stance_torques(self, leg_id: str, q: np.ndarray, force: np.ndarray):
        """Compute torques for stance leg
        
            Returns:
                torque: [3 x 1] vector of torques for each joint of the stance leg
        """
        # print(f"\nForce Mapper Debug for {leg_id}:")
        # print(f"Input force components: {force}")  # Check force ordering
        # print(f"Pinocchio frame ID: {self.foot_frame_ids[leg_id]}")
        
        # Update kinematics
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get rotation matrix from base to world
        R = self.get_rotation_matrix(q)
        
        
        # Get Jacobian
        frame_id = self.foot_frame_ids[leg_id]
        J_full = pin.getFrameJacobian(
            self.model, self.data,
            frame_id,
            pin.ReferenceFrame.LOCAL
        )[:3, :]
        # print(f"Full Jacobian shape: {J_full.shape}")
        
        # Get columns for this leg
        joint_ids = self.leg_joint_indices[leg_id]
        J = J_full[:, joint_ids]
        # print(f"Leg Jacobian shape: {J.shape}")
        # print(f"Leg joint indices: {joint_ids}")
        
        # Compute torques
        tau = J.T @ R.T @ force
    
        return np.clip(tau, -self.tau_max, self.tau_max)
    
    def compute_leg_torques(self, leg_id, q, v, mode, **kwargs):
        """
        Compute torques based on leg mode
        mode: 'swing' or 'stance'
        kwargs: contains necessary parameters for each mode
        """

        if mode == 'swing':
            required_args = ['p_ref', 'v_ref', 'a_ref']
            if not all(arg in kwargs for arg in required_args):
                raise ValueError(f"Swing mode requires {required_args}")
            
            tau = self.compute_swing_torques(
                leg_id, q, v,
                kwargs['p_ref'],
                kwargs['v_ref'],
                kwargs['a_ref']
            )
            
            return tau
        
        elif mode == 'stance':
            required_args = ['force']
            if not all(arg in kwargs for arg in required_args):
                raise ValueError(f"Stance mode requires {required_args}")
                
            tau = self.compute_stance_torques(
                leg_id, q,
                kwargs['force']
            )
        
            return tau
            
        else:
            raise ValueError(f"Unknown mode: {mode}")