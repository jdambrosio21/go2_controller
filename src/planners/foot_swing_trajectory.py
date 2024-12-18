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
        """Compute foot swing trajectory using Bezier curves like MIT Cheetah
        Args:
            phase: How far along we are in the swing (0 to 1)
            swing_time: How long the swing should take (seconds)
        """
        # Compute XY motion using single Bezier
        self.p[0:2] = cubic_bezier(self.p0[0:2], self.pf[0:2], phase)
        self.v[0:2] = cubic_bezier_first_derivative(self.p0[0:2], self.pf[0:2], phase) / swing_time
        self.a[0:2] = cubic_bezier_second_derivative(self.p0[0:2], self.pf[0:2], phase) / (swing_time * swing_time)

        # Split Z motion into two phases like MIT implementation
        if phase < 0.5:
            # First half: go from start to apex
            t = phase * 2  # Rescale phase to [0,1] for first half
            z0 = np.array([self.p0[2]])
            z1 = np.array([self.p0[2] + self.height])
            
            self.p[2] = cubic_bezier(z0, z1, t)[0]
            self.v[2] = cubic_bezier_first_derivative(z0, z1, t)[0] * 2 / swing_time  # Factor of 2 from chain rule
            self.a[2] = cubic_bezier_second_derivative(z0, z1, t)[0] * 4 / (swing_time * swing_time)  # Factor of 4
            
        else:
            # Second half: go from apex to end
            t = phase * 2 - 1  # Rescale phase to [0,1] for second half
            z0 = np.array([self.p0[2] + self.height])
            z1 = np.array([self.pf[2]])
            
            self.p[2] = cubic_bezier(z0, z1, t)[0]
            self.v[2] = cubic_bezier_first_derivative(z0, z1, t)[0] * 2 / swing_time
            self.a[2] = cubic_bezier_second_derivative(z0, z1, t)[0] * 4 / (swing_time * swing_time)