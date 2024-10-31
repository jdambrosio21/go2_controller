import numpy as np
import pinocchio as pin

class ForceMapper:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Control gains
        self.Kp = np.diag([100, 100, 100])  # Position gains
        self.Kd = np.diag([10, 10, 10])     # Velocity gains

        # Store foot frame ids
        self.foot_frames = {
            'FL': self.model.getFramedId("FL_foot"),
            'FR': self.model.getFrameId("FR_foot"),
            'RL': self.model.getFrameId("RL_foot"),
            'RR': self.model.getFrameId("RR_foot")
        }

    def compute_swing_torques(self, leg_id, q, v, p_ref, v_ref, a_ref):
        """Compute torques for swing leg control"""

        # Update model
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.crba(self.model, self.data, q)  # Compute mass matrix
        pin.computeCoriolisMatrix(self.model, self.data, q, v)
        pin.computeGeneralizedGravity(self.model, self.data, q)

        # Get Jacobian (Translational)
        J = pin.getFrameJacobian(self.model, self.data, 
                                 self.foot_frames[leg_id], 
                                 pin.ReferenceFrame.LOCAL)[:3, :] 
        
        # Get current foot position and velocity
        p = pin.getFramePosition(self.model, self.data, self.foot_frames[leg_id])
        v_current = J @ v

        # Compute operational space inertia matrix
        Lambda = np.linalg.inv(J @ np.linalg.inv(self.data.M) @ J.T)

        # Compute feedback term
        feedback = J.T @ (self.Kp @ (p_ref - p) + self.Kd @ (v_ref - v_current))

        # Compute feedforward term
        J_dot_q_dot = pin.getFrameAcceleration(self.model, self.data, 
                                              self.foot_frames[leg_id]).linear
        C = self.data.C
        G = self.data.g
        
        tau_ff = J.T @ Lambda @ (a_ref - J_dot_q_dot) + C @ v + G
        
        # Total torque
        tau = feedback + tau_ff
        return tau
    
    def compute_stance_torques(self, leg_id, q, force, R):
        """Computes torques for stance leg control"""

        # Update model
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get Jacobian
        J = pin.getJointJacobian(self.model, self.data, 
                                 self.foot_frames[leg_id], 
                                 pin.ReferenceFrame.LOCAL)[:3, :]
        
        # Compute torques
        tau = J.T @ R.T @ force
        return tau
    
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
            
            return self.compute_swing_torques(
                leg_id, q, v,
                kwargs['p_ref'],
                kwargs['v_ref'],
                kwargs['a_ref']
            )
        
        elif mode == 'stance':
            required_args = ['force', 'R']
            if not all(arg in kwargs for arg in required_args):
                raise ValueError(f"Stance mode requires {required_args}")
                
            return self.compute_stance_torques(
                leg_id, q,
                kwargs['force'],
                kwargs['R']
            )
            
        else:
            raise ValueError(f"Unknown mode: {mode}")