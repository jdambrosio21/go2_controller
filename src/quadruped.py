import numpy as np
import pinocchio as pin
#from pinocchio import casadi as cpin

class Quadruped:
    def __init__(self, urdf_path):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        # Store hip frame ID's
        # Store the correct frame IDs for hips and feet
        self.leg_frame_ids = {
            "hip": {
                "FL": self.model.getFrameId("FL_hip"),
                "FR": self.model.getFrameId("FR_hip"),
                "RL": self.model.getFrameId("RL_hip"),
                "RR": self.model.getFrameId("RR_hip")
            },
            "foot": {
                "FL": self.model.getFrameId("FL_foot"),
                "FR": self.model.getFrameId("FR_foot"),
                "RL": self.model.getFrameId("RL_foot"),
                "RR": self.model.getFrameId("RR_foot")
            }
        }
        q = pin.neutral(self.model)
        v = pin.utils.zero(self.model.nv)

        self.mass = pin.computeTotalMass(self.model, self.data)
        pin.ccrba(self.model, self.data, q, v)
        self.inertia = self.data.Ig.inertia

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
        hip_frame_id = self.leg_frame_ids["hip"][leg]
        hip_position = self.data.oMf[hip_frame_id].translation

        return hip_position
    
    def get_foot_positions(self, q: np.ndarray) -> np.ndarray:
        """
            Get all feet positions in world frame

            Args:
                q: Full state vector [19] (base + joints)
            
            Returns:
                foot_positions: [4,3] array of foot positions
        """
        # Update kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get all foot positions
        foot_positions = np.zeros((4, 3))
        for i, leg in enumerate(["FR", "FL", "RR", "RL"]):
            foot_frame_id = self.leg_frame_ids["foot"][leg]
            foot_positions[i] = self.data.oMf[foot_frame_id].translation

        return foot_positions
    
    def quat_to_rpy(self, quat: np.ndarray):
        """
        Converts quaternion to RPY Euler angles.

        Args:
            quat: Quaternion (x, y, z, w)
        
        Returns:
            rpy: numpy array of Roll, Pitch, and Yaw Angles
        """
        quat = quat.astype(float)  # Ensure the quaternion is in float
        quaternion = pin.Quaternion(quat[3], quat[0], quat[1], quat[2])  # Note the order: w, x, y, z
        rpy = pin.rpy.matrixToRpy(quaternion.toRotationMatrix())
        return rpy

   
        
if __name__ == "__main__":
    robot = Quadruped("/home/parallels/go2_controller/robots/go2_description/xacro/go2_generated.urdf")    
    print("Inertia:", robot.inertia)
    print("Mass:", robot.mass)
    
    # Define a test configuration vector with appropriate size
    q = np.zeros(robot.model.nq)  # Initialize to zeros
    
    # Test get_hip_position for leg 0
    #print("Hip position of leg 0:", robot.get_hip_position(q, 0))
    
    # Test get_foot_positions for leg 0
    print("Foot position of leg:", robot.get_foot_positions(q))
    
    # Test quaternion to RPY conversion
    quat = np.array([0, 0, 0, 1])  # Identity quaternion
    print("RPY angles from quaternion:", robot.quat_to_rpy(quat))