import numpy as np
from numpy.typing import NDArray

def cubic_bezier(p0: NDArray, pf: NDArray, t: float) -> NDArray:
    """Cubic Bezier curve interpolation"""
    # Control points for cubic Bezier
    p1 = p0 + (pf - p0) * 0.333  # First control point at 1/3
    p2 = p0 + (pf - p0) * 0.667  # Second control point at 2/3
    
    # Cubic Bezier formula
    return (1 - t)**3 * p0 + \
           3 * (1 - t)**2 * t * p1 + \
           3 * (1 - t) * t**2 * p2 + \
           t**3 * pf

def cubic_bezier_first_derivative(p0: NDArray, pf: NDArray, t: float) -> NDArray:
    """First derivative of cubic Bezier curve"""
    p1 = p0 + (pf - p0) * 0.333
    p2 = p0 + (pf - p0) * 0.667
    
    return 3 * (1 - t)**2 * (p1 - p0) + \
           6 * (1 - t) * t * (p2 - p1) + \
           3 * t**2 * (pf - p2)

def cubic_bezier_second_derivative(p0: NDArray, pf: NDArray, t: float) -> NDArray:
    """Second derivative of cubic Bezier curve"""
    p1 = p0 + (pf - p0) * 0.333
    p2 = p0 + (pf - p0) * 0.667
    
    return 6 * (1 - t) * (p2 - 2*p1 + p0) + \
           6 * t * (pf - 2*p2 + p1)

class FootSwingTrajectory:
    """Utility to generate foot swing trajectories"""
    def __init__(self, p0: NDArray, pf: NDArray, height: float):
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

    def compute_swing_trajectory(self, phase: float, swing_time: float):
        """Compute foot swing trajectory with split z-trajectory like MIT"""
        # XY motion
        xy =cubic_bezier(self.p0[0:2], self.pf[0:2], phase)
        xy_vel = cubic_bezier_first_derivative(self.p0[0:2], self.pf[0:2], phase) / swing_time
        xy_acc = cubic_bezier_second_derivative(self.p0[0:2], self.pf[0:2], phase) / (swing_time * swing_time)
        
        # Split Z trajectory
        if phase < 0.5:
            # First half - go from start to apex
            t = phase * 2.0  # Rescale phase to [0,1] 
            z0 = np.array([self.p0[2]]) 
            z1 = np.array([self.p0[2] + self.height])
            
            z = cubic_bezier(z0, z1, t)[0]
            z_vel = cubic_bezier_first_derivative(z0, z1, t)[0] * 2.0 / swing_time
            z_acc = cubic_bezier_second_derivative(z0, z1, t)[0] * 4.0 / (swing_time * swing_time)
            
        else:
            # Second half - go from apex to final
            t = phase * 2.0 - 1.0  # Rescale phase to [0,1]
            z0 = np.array([self.p0[2] + self.height])
            z1 = np.array([self.pf[2]])
            
            z = cubic_bezier(z0, z1, t)[0]
            z_vel = cubic_bezier_first_derivative(z0, z1, t)[0] * 2.0 / swing_time
            z_acc = cubic_bezier_second_derivative(z0, z1, t)[0] * 4.0 / (swing_time * swing_time)

        self.p = np.array([xy[0], xy[1], z]) 
        self.v = np.array([xy_vel[0], xy_vel[1], z_vel])
        self.a = np.array([xy_acc[0], xy_acc[1], z_acc])
