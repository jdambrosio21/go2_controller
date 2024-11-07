import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

class Quadruped:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Store hip frame ID's
        self.leg_frame_ids = {
            0: self.model.getFrameId("FL_"),
            1: self.model.getFrameId("FR_"),
            2: self.model.getFrameId("BL_"),
            3: self.model.getFrameId("BR_")
        }

        self.mass = pin.computeTotalMass(self.model, self.data)
        self.inertia = self.data.Ig

    def get_hip_position(self, q: np.ndarray, leg: int):
        """
            Gets hip position of robot's leg in world frame

            Args:
                q: Robot configuration vector
                leg: Leg index (0:FL, 1:FR, 2:BL, 3:BR)
            
            Returns:
                3D Position of hip in world frame
        """
        # Update kinematics 
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get hip frame position
        hip_frame_id = self.leg_frame_ids[leg] + "hip"
        hip_position = self.data.oMf[hip_frame_id].translation

        return hip_position
    
    def get_foot_positions(self, q: np.ndarray, leg: int):
        """
            Gets foot positions in world frame
        """
        # Update kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get foot frame positions
        foot_frame_id = self.leg_frame_ids[leg] + "foot"
        foot_position = self.data.oMf[foot_frame_id].translation

        return foot_position
    
   
        
