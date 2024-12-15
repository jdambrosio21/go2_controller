import numpy as np
from scipy.interpolate import CubicSpline
from numpy.typing import NDArray

class FootSwingTrajectory:
    """Utility to generate foot swing trajectories"""
    def __init__(self,
                 p0: NDArray,      # Initial position
                 pf: NDArray,      # Final position
                 height: float):   # Swing height
        """Initialize trajectory generator
        
        Args:
            p0: Initial foot position
            pf: Final foot position
            height: Maximum swing height
        """
        self.p0 = p0
        self.pf = pf
        self.height = height
        
        # Current state
        self.p = np.zeros(3)  # Position
        self.v = np.zeros(3)  # Velocity
        self.a = np.zeros(3)  # Acceleration

        # Create interpolators for XY motion
        self.times = np.array([0.0, 0.5, 1.0])
        self.xy_waypoints = np.array([
            [p0[0], p0[1]],               # Start
            [(p0[0] + pf[0])/2, (p0[1] + pf[1])/2],  # Mid
            [pf[0], pf[1]]                # End
        ])
        
        # Z waypoints with apex at midpoint
        self.z_waypoints = np.array([
            p0[2],              # Start
            p0[2] + height,     # Mid (apex)
            pf[2]               # End
        ])
        
        # Create splines
        self.xy_spline = CubicSpline(self.times, self.xy_waypoints, bc_type='clamped')
        self.z_spline = CubicSpline(self.times, self.z_waypoints, bc_type='clamped')

    def compute_swing_trajectory(self, phase: float, swing_time: float):
        """Compute foot swing trajectory 
        
        Args:
            phase: How far along we are in the swing (0 to 1)
            swing_time: How long the swing should take (seconds)
        """
        # Get XY trajectory
        xy = self.xy_spline(phase)
        xy_vel = self.xy_spline(phase, 1) / swing_time
        xy_acc = self.xy_spline(phase, 2) / (swing_time * swing_time)
        
        # Get Z trajectory
        z = self.z_spline(phase)
        z_vel = self.z_spline(phase, 1) / swing_time
        z_acc = self.z_spline(phase, 2) / (swing_time * swing_time)
        
        # Update states
        self.p = np.array([xy[0], xy[1], z])
        self.v = np.array([xy_vel[0], xy_vel[1], z_vel])
        self.a = np.array([xy_acc[0], xy_acc[1], z_acc])

        #print(f"Trajectory params: start_z={self.p0[2]}, end_z={self.pf[2]}, height={self.height}")