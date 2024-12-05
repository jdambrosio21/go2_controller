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
                "RR": self.model.getFrameId("RR_hip"),
                    0: self.model.getFrameId("FL_hip"),
                    1: self.model.getFrameId("FR_hip"),
                    2: self.model.getFrameId("RL_hip"),
                    3: self.model.getFrameId("RR_hip")
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
        self.inertia_p = self.data.Ig.inertia
        self.inertia = np.zeros(3)
        self.inertia[0] = self.inertia_p[0, 0]
        self.inertia[1] = self.inertia_p[1, 1]
        self.inertia[2] = self.inertia_p[2, 2]


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

        hip_position = np.zeros(3)

        # Get hip frame position
        hip_frame_id = self.leg_frame_ids["hip"][leg]
        hip_position[:2]= self.data.oMf[hip_frame_id].translation[:2]

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
    def print_dof_ordering(self):
        """
        Prints the DoF ordering and their indices in the configuration (q) and velocity (v) vectors.
        """
        print("Degrees of Freedom (DoF) ordering and index in Pinocchio:")
        print("Joint Name        | Index in q | Index in v | DoF (nv)")
        print("-------------------------------------------------------")
        for i, joint in enumerate(self.model.joints):
            joint_name = self.model.names[i]
            print(f"{joint_name:<16} | {joint.idx_q:<10} | {joint.idx_v:<10} | {joint.nv:<3}")

    def debug_frames(self, q: np.ndarray):
        """Debug frame transforms for each leg"""
        # Update kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get base frame transform
        base_frame_id = self.model.getFrameId('floating_base')
        base_transform = self.data.oMf[base_frame_id]
        print("\nBase Frame Transform:")
        print(f"Position: {base_transform.translation}")
        print(f"Rotation:\n{base_transform.rotation}")

        # Debug each leg's frames
        for leg in ["FL", "FR", "RL", "RR"]:
            print(f"\n{leg} Leg Frames:")
            # Hip frame
            hip_frame_id = self.leg_frame_ids["hip"][leg]
            hip_transform = self.data.oMf[hip_frame_id]
            print(f"Hip position (world): {hip_transform.translation}")
            print(f"Hip rotation (world to hip):\n{hip_transform.rotation}")
            
            # Foot frame
            foot_frame_id = self.leg_frame_ids["foot"][leg]
            foot_transform = self.data.oMf[foot_frame_id]
            print(f"Foot position (world): {foot_transform.translation}")
            print(f"Foot rotation (world to foot):\n{foot_transform.rotation}")

            # Calculate hip to foot vector in world frame
            hip_to_foot = foot_transform.translation - hip_transform.translation
            print(f"Hip to foot vector (world): {hip_to_foot}")
   
        
if __name__ == "__main__":
    robot = Quadruped("/home/parallels/go2_controller/robots/go2_description/xacro/go2_generated.urdf") 
    robot.print_dof_ordering()
    print("Inertia:", robot.inertia)
    print("Mass:", robot.mass)
    
    # Define a test configuration vector with appropriate size
    q = np.zeros(robot.model.nq)  # Initialize to zeros
    
    # Test get_hip_position for leg 0
    print("Hip position of leg 0:", robot.get_hip_position(q, 0))
    print("Hip position of leg 1:", robot.get_hip_position(q, 1))
    print("Hip position of leg 2:", robot.get_hip_position(q, 2))
    print("Hip position of leg 3:", robot.get_hip_position(q, 3))
    
    # Test get_foot_positions for leg 0
    print("Foot position of leg:", robot.get_foot_positions(q))
    
    # Test quaternion to RPY conversion
    quat = np.array([0, 0, 0, 1])  # Identity quaternion
    print("RPY angles from quaternion:", robot.quat_to_rpy(quat))

    # Debug frame transforms
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)

    print("\n=== Base Frame ===")
    base_frame_id = robot.model.getFrameId('base')
    base_transform = robot.data.oMf[base_frame_id]
    print(f"Base Position: {base_transform.translation.WORLD}")
    print(f"Base Rotation:\n{base_transform.rotation}")

    # Print transforms for each leg
    for leg in ["FL", "FR", "RL", "RR"]:
        print(f"\n=== {leg} Leg ===")
        # Hip
        hip_frame_id = robot.leg_frame_ids["hip"][leg]
        hip_transform = robot.data.oMf[hip_frame_id]
        print(f"Hip position: {hip_transform.translation}")
        
        # Foot
        foot_frame_id = robot.leg_frame_ids["foot"][leg]
        foot_transform = robot.data.oMf[foot_frame_id]
        print(f"Foot position: {foot_transform.translation}")
        
        # Hip to foot vector
        hip_to_foot = foot_transform.translation - hip_transform.translation
        print(f"Hip to foot vector: {hip_to_foot}")
        print(f"Hip to foot distance: {np.linalg.norm(hip_to_foot)}")